"""FastAPI app served by the HF Space Docker container.

Three surfaces:
  - `GET /api/health`: liveness + memory info, polled by the frontend to
    show a "Ready/Waking/Offline" badge and detect cold starts.
  - `POST /api/analysis/run`: execute a `CustomRequest` (same shape as the
    frontend's analysis_worker.CustomRequest). Returns a `CustomResultMsg`.
  - `WS /ws/bridge/{session_id}`: browser tunnel for local-folder zarrs.
  - `GET /api/data/{session_id}/{path:path}`: serve per-session output
    bytes (e.g. synthesized zarrs from _TG_NEW_LAYER).

Architecturally the backend mirrors the Pyodide worker's responsibilities:
open each requested layer, materialize as numpy, hand Python globals
(layers, DataFrames) to the user code, collect the _TG_* outputs, convert
them to the JSON the frontend expects.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import platform
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from browser_store import TunnelSession, read_zarr_scale
from sandbox import SandboxResult, run_sandboxed

log = logging.getLogger("tourguide")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# --- App setup --------------------------------------------------------------

APP_VERSION = "0.1.0"
SESSION_ROOT = Path(os.environ.get("TG_SESSION_ROOT", "/tmp/tourguide-sessions"))
SESSION_ROOT.mkdir(parents=True, exist_ok=True)
SESSION_TTL_SECONDS = 10 * 60

# Caps exposed to the frontend via /health — match the plan.
MAX_CONCURRENT_ANALYSES = 2
ANALYSIS_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)
QUEUE_DEPTH = 0  # updated inside the handler; read by /health

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="tourguide-analysis", version=APP_VERSION)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(_request: Request, exc: RateLimitExceeded) -> Response:
    return Response(
        content=json.dumps({"error": "rate_limit", "detail": str(exc.detail)}),
        status_code=429,
        media_type="application/json",
    )


# Allow the Cloudflare Pages origin plus local dev. In prod we could tighten
# this; since this is a research tool the allow-list is intentionally loose.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tourguide-8j4.pages.dev",
        "http://localhost:5173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:4173",
    ],
    allow_origin_regex=r"https://.*\.pages\.dev",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --- Tunnel session registry ------------------------------------------------

TUNNELS: Dict[str, TunnelSession] = {}


def get_tunnel(session_id: str) -> Optional[TunnelSession]:
    return TUNNELS.get(session_id)


# --- Request / response schemas ---------------------------------------------


class LayerSpec(BaseModel):
    varName: str
    url: str
    scalePath: str = ""
    axesOrder: List[str] = Field(default_factory=list)
    voxelNm: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    offsetNm: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class TableSpec(BaseModel):
    name: str
    columns: List[str]
    rows: List[List[Any]]


class CustomRequestBody(BaseModel):
    layers: List[LayerSpec] = Field(default_factory=list)
    tables: List[TableSpec] = Field(default_factory=list)
    code: str
    timeoutMs: int = 60000
    sessionId: Optional[str] = None


# --- Health -----------------------------------------------------------------


def _mem_gb_total() -> float:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 2)
    except (ValueError, OSError):
        return 0.0


def _mem_gb_free() -> float:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 1024**3, 2)
    except (ValueError, OSError):
        return 0.0


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "version": APP_VERSION,
        "python": platform.python_version(),
        "mem_gb_total": _mem_gb_total(),
        "mem_gb_free": _mem_gb_free(),
        "queue_depth": QUEUE_DEPTH,
        "max_concurrent": MAX_CONCURRENT_ANALYSES,
    }


# --- Tunnel WS --------------------------------------------------------------


@app.websocket("/ws/bridge/{session_id}")
async def ws_bridge(ws: WebSocket, session_id: str) -> None:
    await ws.accept()
    session = TunnelSession(session_id)
    TUNNELS[session_id] = session

    async def send(msg: Dict[str, Any]) -> None:
        await ws.send_text(json.dumps(msg))

    session.attach(send)
    log.info("tunnel open (session=%s)", session_id)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("tunnel: bad json from %s: %s", session_id, raw[:200])
                continue
            if msg.get("type") == "response":
                session.resolve(
                    int(msg.get("req_id", -1)),
                    msg.get("bytes_b64"),
                    bool(msg.get("found", True)),
                )
    except WebSocketDisconnect:
        pass
    finally:
        session.close("disconnected")
        TUNNELS.pop(session_id, None)
        log.info("tunnel closed (session=%s)", session_id)


# --- Analysis run ------------------------------------------------------------


def _is_local_url(url: str) -> bool:
    # Local-folder URLs are always origin-relative paths under /local-data/
    # or may be fully-qualified with a same-origin hostname that the backend
    # obviously can't reach; either way, we tunnel them through the browser.
    return ("/local-data/" in url) or url.startswith("/local-data/") or url.startswith("local-data/")


async def _load_layer(layer: LayerSpec, tunnel: Optional[TunnelSession]) -> np.ndarray:
    """Resolve a LayerSpec to a numpy ndarray.

    Remote (https://, s3://, gs://) → zarr-python + fsspec.
    Local-folder → BrowserStore tunnel via the browser.
    """
    if _is_local_url(layer.url):
        if tunnel is None or not tunnel.is_ready():
            raise HTTPException(status_code=400, detail=f"layer '{layer.varName}' is local but no browser tunnel is attached")

        # The tunnel's peer is the user's browser. It resolves paths relative
        # to its own origin, so we must pass just the PATH (no scheme, no
        # host) through the WS. Otherwise `new URL(path, window.origin)`
        # resolves wrong and Cloudflare Pages' SPA fallback returns HTML for
        # a non-existent route — which looks like "empty .zarray" to the
        # zarr reader and fails with a bewildering JSONDecodeError.
        from urllib.parse import urlparse
        parsed = urlparse(layer.url)
        base_path = parsed.path or "/"
        if not base_path.endswith("/"):
            base_path += "/"
        async def read_at_scale(path: str) -> Optional[bytes]:
            full = _join_url(base_path, path)
            log.debug("tunnel read %s", full)
            return await tunnel.read(full)
        return await read_zarr_scale(read_at_scale, layer.scalePath)

    # Remote path.
    import zarr
    import fsspec  # noqa: WPS433

    url = layer.url
    raw_proto = url.split("://", 1)[0] if "://" in url else "file"
    # fsspec's HTTP backend covers both http and https under the "http" key.
    proto = "http" if raw_proto in ("http", "https") else raw_proto
    fs_kwargs: Dict[str, Any] = {}
    if proto == "s3":
        fs_kwargs["anon"] = True  # CellMap buckets are public
    try:
        fs = fsspec.filesystem(proto, **fs_kwargs)
        mapper = fs.get_mapper(url)
        arr = zarr.open(mapper, path=layer.scalePath, mode="r")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"failed to open {url} (path={layer.scalePath!r}): {type(exc).__name__}: {exc}",
        )
    # Reduce leading non-spatial dims to 0; mirror frontend's Pyodide path.
    sel = tuple(slice(None) if a in ("x", "y", "z") else 0 for a in layer.axesOrder) if layer.axesOrder else tuple(
        slice(None) for _ in range(arr.ndim)
    )
    if len(sel) != arr.ndim:
        sel = tuple(slice(None) if i >= arr.ndim - 3 else 0 for i in range(arr.ndim))
    try:
        return np.asarray(arr[sel])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"failed to read {url} at {layer.scalePath!r} shape={arr.shape}: {type(exc).__name__}: {exc}",
        )


def _join_url(base: str, rel: str) -> str:
    base = base.rstrip("/")
    rel = rel.lstrip("/")
    return f"{base}/{rel}" if base else rel


# --- Python globals prep (mirrors analysis_worker.ts) -----------------------


def _layer_to_globals(layer: LayerSpec, arr: np.ndarray) -> Dict[str, Any]:
    """Build the per-layer sub-dict the user code sees in `layers["name"]`."""
    spatial_axes = [a for a in layer.axesOrder if a in ("x", "y", "z")]
    if len(spatial_axes) != arr.ndim:
        # Fallback: ZYX convention if axis names don't line up with array rank.
        spatial_axes = ["z", "y", "x"][-arr.ndim:]
    axis_scale = {"x": layer.voxelNm[0], "y": layer.voxelNm[1], "z": layer.voxelNm[2]}
    axis_offset = {"x": layer.offsetNm[0], "y": layer.offsetNm[1], "z": layer.offsetNm[2]}
    return {
        "array": arr,
        "spacing": tuple(axis_scale.get(a, 1.0) for a in spatial_axes),
        "offsets": tuple(axis_offset.get(a, 0.0) for a in spatial_axes),
        "axes": list(spatial_axes),
    }


def _build_globals(
    layers: Dict[str, Dict[str, Any]],
    tables: List[TableSpec],
) -> Dict[str, Any]:
    import pandas as pd  # noqa: WPS433

    g: Dict[str, Any] = {
        "layers": layers,
        "_TG_TABLE": None,
        "_TG_TABLE_NAME": None,
        "_TG_PLOT": None,
        "_TG_FLY": None,
        "_TG_NARRATION": None,
        "_TG_STDOUT": [],
        "_TG_ANNOTATIONS": None,
        "_TG_HIGHLIGHT": None,
        "_TG_ADD_SOURCE_LAYER": None,
        "_TG_NEW_LAYER": None,
    }
    for name, info in layers.items():
        g[name] = info["array"]
    for t in tables:
        df = pd.DataFrame(t.rows, columns=t.columns)
        g[f"df_{t.name}"] = df
    return g


# --- Output encoding --------------------------------------------------------


def _encode_table(tg_table: Any, name: Optional[str]) -> Optional[Dict[str, Any]]:
    if tg_table is None:
        return None
    import pandas as pd  # noqa: WPS433

    df = tg_table if isinstance(tg_table, pd.DataFrame) else pd.DataFrame(tg_table)
    df = df.where(df.notnull(), None)
    return {
        "name": str(name) if name else "custom_result",
        "columns": list(df.columns),
        "rows": df.values.tolist(),
    }


def _encode_fly(tg_fly: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(tg_fly, dict):
        return None
    pos = tg_fly.get("pos")
    if not pos or len(pos) < 3:
        return None
    return {
        "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
        "segmentId": str(tg_fly["segment_id"]) if tg_fly.get("segment_id") is not None else None,
        "layer": str(tg_fly["layer"]) if tg_fly.get("layer") is not None else None,
    }


def _encode_annotations(ann: Any) -> Optional[Dict[str, Any]]:
    if ann is None:
        return None
    if isinstance(ann, list):
        ann = {"layer_name": "custom_points", "points": ann}
    points: List[Dict[str, Any]] = []
    for p in ann.get("points") or []:
        if isinstance(p, (list, tuple)):
            points.append({"pos": [float(p[0]), float(p[1]), float(p[2])]})
        elif isinstance(p, dict):
            pos = p.get("pos") or []
            if len(pos) < 3:
                continue
            entry = {"pos": [float(pos[0]), float(pos[1]), float(pos[2])]}
            if p.get("id") is not None:
                entry["id"] = str(p["id"])
            if p.get("description") is not None:
                entry["description"] = str(p["description"])
            points.append(entry)
    return {"layerName": str(ann.get("layer_name", "custom_points")), "points": points}


def _encode_highlight(h: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(h, dict) or not h.get("layer"):
        return None
    return {"layer": str(h["layer"]), "ids": [str(i) for i in (h.get("ids") or [])]}


def _encode_add_source(s: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(s, dict) or not s.get("source"):
        return None
    return {
        "source": str(s["source"]),
        "name": str(s.get("name", "new_layer")),
        "type": "image" if str(s.get("type", "image")) == "image" else "segmentation",
    }


def _encode_new_layer_and_write(
    session_id: str,
    nl: Any,
    layers: Dict[str, Dict[str, Any]],
    request: Request,
) -> Optional[Dict[str, Any]]:
    """For _TG_NEW_LAYER: write a minimal OME-NGFF zarr under the session dir,
    return a spec the frontend can feed to `addSourceLayer`.
    """
    if nl is None or "array" not in nl:
        return None
    arr = nl["array"]
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    # Neuroglancer-friendly dtype coercion (same rule the Pyodide worker uses).
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    elif arr.dtype in (np.float16, np.float64):
        arr = arr.astype(np.float32)
    arr = np.ascontiguousarray(arr)

    # Default spacing/offsets/axes to first input layer if not given.
    first_layer = next(iter(layers.values())) if layers else None
    spacing = list(nl.get("spacing") or (first_layer["spacing"] if first_layer else [1.0] * arr.ndim))
    offsets = list(nl.get("offsets") or (first_layer["offsets"] if first_layer else [0.0] * arr.ndim))
    axes = list(nl.get("axes") or (first_layer["axes"] if first_layer else ["z", "y", "x"][-arr.ndim:]))

    layer_id = f"{str(nl.get('name', 'new_layer')).replace('/', '_')}-{uuid.uuid4().hex[:6]}"
    session_dir = SESSION_ROOT / session_id / layer_id
    session_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "uint8": "|u1", "int8": "|i1",
        "uint16": "<u2", "int16": "<i2",
        "uint32": "<u4", "int32": "<i4",
        "uint64": "<u8", "int64": "<i8",
        "float32": "<f4", "float64": "<f8",
    }
    zarr_dtype = dtype_map.get(str(arr.dtype))
    if zarr_dtype is None:
        raise HTTPException(status_code=400, detail=f"unsupported dtype for new layer: {arr.dtype}")

    (session_dir / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (session_dir / ".zattrs").write_text(json.dumps({
        "multiscales": [{
            "version": "0.4",
            "axes": [{"name": n, "type": "space", "unit": "nanometer"} for n in axes],
            "datasets": [{
                "path": "s0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": spacing},
                    {"type": "translation", "translation": offsets},
                ],
            }],
        }],
    }))
    (session_dir / "s0").mkdir(exist_ok=True)
    (session_dir / "s0" / ".zarray").write_text(json.dumps({
        "zarr_format": 2,
        "shape": list(arr.shape),
        "chunks": list(arr.shape),
        "dtype": zarr_dtype,
        "compressor": None,
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": "/",
    }))
    chunk_rel = "/".join(["0"] * arr.ndim)
    (session_dir / "s0" / chunk_rel).parent.mkdir(parents=True, exist_ok=True)
    (session_dir / "s0" / chunk_rel).write_bytes(arr.tobytes(order="C"))

    base = str(request.url).rsplit("/api/", 1)[0]
    return {
        "synthesizedId": f"{session_id}/{layer_id}",
        "name": str(nl.get("name", "new_layer")),
        "type": "image" if str(nl.get("type", "segmentation")) == "image" else "segmentation",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        # The frontend will build `zarr://<base>/api/data/<synthesizedId>/`
        # and add a layer pointing at it.
        "serveUrl": f"{base}/api/data/{session_id}/{layer_id}/",
    }


# --- Serve synthesized zarrs -------------------------------------------------


@app.get("/api/data/{session_id}/{layer_id}/{path:path}")
async def data(session_id: str, layer_id: str, path: str) -> Response:
    # No dot-segments, no absolute paths.
    if ".." in path.split("/") or path.startswith("/"):
        raise HTTPException(status_code=400, detail="invalid path")
    full = SESSION_ROOT / session_id / layer_id / path
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404)
    return Response(content=full.read_bytes(), media_type="application/octet-stream")


async def _cleanup_session_later(session_id: str) -> None:
    await asyncio.sleep(SESSION_TTL_SECONDS)
    shutil.rmtree(SESSION_ROOT / session_id, ignore_errors=True)


# --- Run endpoint ------------------------------------------------------------


@app.post("/api/analysis/run")
async def run_analysis(request: Request, body: CustomRequestBody, background: BackgroundTasks) -> Dict[str, Any]:
    # Rate limiting intentionally not applied here — slowapi's @limiter
    # decorator interferes with FastAPI's body/dependency inspection and
    # makes pydantic bodies look like query params. For a single-user or
    # small-group Space the semaphore + cold-start delay are already
    # effective throttles. Revisit if the Space sees abuse.
    global QUEUE_DEPTH  # noqa: PLW0603
    session_id = body.sessionId or uuid.uuid4().hex
    tunnel = get_tunnel(session_id) if any(_is_local_url(l.url) for l in body.layers) else None
    t0 = time.monotonic()

    import traceback
    QUEUE_DEPTH += 1
    try:
        async with ANALYSIS_SEMAPHORE:
            # Load each layer into a numpy array.
            try:
                layers_info: Dict[str, Dict[str, Any]] = {}
                for layer in body.layers:
                    arr = await _load_layer(layer, tunnel)
                    layers_info[layer.varName] = _layer_to_globals(layer, arr)
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("layer load failed")
                return {
                    "kind": "error",
                    "message": f"failed to load layer: {type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }

            # Build the globals dict that the user code will see.
            g = _build_globals(layers_info, body.tables)
            timeout_s = max(5, body.timeoutMs // 1000)

            # Run user code sandboxed. fork keeps numpy arrays COW-shared.
            sandbox_result: SandboxResult = run_sandboxed(
                body.code,
                g,
                timeout_s=timeout_s,
            )
            if not sandbox_result.ok:
                return {
                    "kind": "error",
                    "message": sandbox_result.error or "analysis failed",
                    "traceback": sandbox_result.traceback or "",
                }
            outputs = sandbox_result.outputs

            out_msg: Dict[str, Any] = {"kind": "customResult"}
            table = _encode_table(outputs.get("_TG_TABLE"), outputs.get("_TG_TABLE_NAME"))
            if table:
                out_msg["table"] = table
            if outputs.get("_TG_NARRATION"):
                out_msg["narration"] = str(outputs["_TG_NARRATION"])
            stdout = outputs.get("_TG_STDOUT")
            if stdout:
                out_msg["stdout"] = "\n".join(str(s) for s in stdout)
            fly = _encode_fly(outputs.get("_TG_FLY"))
            if fly:
                out_msg["fly"] = fly
            ann = _encode_annotations(outputs.get("_TG_ANNOTATIONS"))
            if ann:
                out_msg["annotations"] = ann
            hi = _encode_highlight(outputs.get("_TG_HIGHLIGHT"))
            if hi:
                out_msg["highlight"] = hi
            add_src = _encode_add_source(outputs.get("_TG_ADD_SOURCE_LAYER"))
            if add_src:
                out_msg["addSourceLayer"] = add_src
            new_layer = _encode_new_layer_and_write(session_id, outputs.get("_TG_NEW_LAYER"), layers_info, request)
            if new_layer:
                out_msg["newLayer"] = new_layer
                # Ensure this session gets cleaned up eventually.
                background.add_task(_cleanup_session_later, session_id)

            # matplotlib figure capture: if the user code called plt.*, we
            # saved a figure in their subprocess; since that process is dead
            # now, there's no figure to capture. For the remote path we
            # require the user to explicitly base64-encode figures into
            # _TG_PLOT_PNG (string). If present, forward it.
            plot_png = outputs.get("_TG_PLOT_PNG")
            if plot_png:
                out_msg["plotPngDataUrl"] = f"data:image/png;base64,{plot_png}"

            elapsed = round(time.monotonic() - t0, 2)
            log.info("analysis done in %.2fs (session=%s, layers=%d)", elapsed, session_id, len(body.layers))
            return out_msg
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("unexpected failure in /api/analysis/run")
        return {
            "kind": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        QUEUE_DEPTH -= 1


# --- Dev entrypoint ----------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
