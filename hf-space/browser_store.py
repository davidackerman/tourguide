"""Browser tunnel: read zarr chunks from the user's browser over a WebSocket.

Normal case: the browser has the zarr (e.g. a folder the user picked with
FileSystemDirectoryHandle). We can't reach that folder from the cloud, but
the browser's service worker can serve it. So the flow is:

    Browser                         HF Space
    -------                         --------
    opens /ws/bridge/<sid>  ←──→   accepts connection
                                    when it wants byte `path`:
                                      sends {type: "request", req_id, path}
    fetches path via SW,
    sends {type: "response",
           req_id, bytes_b64}   ──→ resolves the future keyed by req_id
                                   returns bytes to zarr loader

We don't try to implement zarr-python's sync Store interface. Instead we
have a small async loader that reads `.zarray` + all needed chunks and
reassembles into a numpy array directly. Uses numcodecs to decompress
(blosc / gzip / zstd / raw). That's all we need for OME-Zarr inputs.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import logging
from dataclasses import dataclass
from math import ceil
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np

log = logging.getLogger("browser_store")


@dataclass
class PendingRequest:
    future: asyncio.Future
    path: str


class TunnelSession:
    """Holds state for a single browser WS connection.

    Each analysis request typically owns one session. The session maps
    request ids (minted server-side) to Futures resolved when the browser
    replies. All `read(path)` calls are awaited by the loader.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._next_id = 0
        self._pending: Dict[int, PendingRequest] = {}
        self._send: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
        self._closed = False
        self._cache: Dict[str, Optional[bytes]] = {}

    def attach(self, send_coro: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        self._send = send_coro

    def is_ready(self) -> bool:
        return self._send is not None and not self._closed

    async def read(self, path: str, *, timeout: float = 30.0) -> Optional[bytes]:
        """Fetch bytes for `path` from the browser. Returns None for 404."""
        if path in self._cache:
            return self._cache[path]
        if not self.is_ready():
            raise RuntimeError("tunnel not connected")
        req_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[Optional[bytes]] = loop.create_future()
        self._pending[req_id] = PendingRequest(future=fut, path=path)
        assert self._send is not None
        await self._send({"type": "request", "req_id": req_id, "path": path})
        try:
            data = await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise
        self._cache[path] = data
        return data

    def resolve(self, req_id: int, data_b64: Optional[str], found: bool) -> None:
        pending = self._pending.pop(req_id, None)
        if pending is None:
            log.warning("tunnel response for unknown req_id=%s (session=%s)", req_id, self.session_id)
            return
        if not found:
            pending.future.set_result(None)
            return
        try:
            bytes_ = base64.b64decode(data_b64 or "") if data_b64 else b""
        except Exception as exc:  # noqa: BLE001
            pending.future.set_exception(exc)
            return
        pending.future.set_result(bytes_)

    def close(self, reason: str = "closed") -> None:
        self._closed = True
        for p in list(self._pending.values()):
            if not p.future.done():
                p.future.set_exception(RuntimeError(f"tunnel {reason}"))
        self._pending.clear()


# ----------------------------------------------------------------------------
# Minimal zarr v2 reader over any async `read(path) -> bytes | None` coro.
# ----------------------------------------------------------------------------


async def read_zarr_scale(
    read: Callable[[str], Awaitable[Optional[bytes]]],
    scale_path: str,
) -> np.ndarray:
    """Fetch the array at `scale_path` using the given async `read` function.

    `scale_path` is the subpath under the zarr root (e.g. "s0"). The caller
    has already resolved any OME-NGFF multiscale nav; we just have to pull
    `.zarray` + each chunk and stitch them.
    """
    zarray_raw = await read(f"{scale_path}/.zarray" if scale_path else ".zarray")
    if zarray_raw is None:
        raise FileNotFoundError(f"no .zarray at {scale_path or '<root>'}")
    meta = json.loads(zarray_raw.decode("utf-8"))

    shape: List[int] = list(meta["shape"])
    chunks: List[int] = list(meta["chunks"])
    dtype = np.dtype(meta["dtype"])
    compressor_spec = meta.get("compressor")
    filters_spec = meta.get("filters") or []
    fill_value = meta.get("fill_value", 0)
    dim_sep = meta.get("dimension_separator", ".")
    order: str = meta.get("order", "C")

    # Allocate the output and fill with fill_value (usually 0).
    try:
        fv: Any = dtype.type(fill_value if fill_value is not None else 0)
    except Exception:  # noqa: BLE001
        fv = 0
    out = np.full(shape, fv, dtype=dtype)

    # Import numcodecs lazily so sandbox checks don't flag the import path.
    import numcodecs  # noqa: WPS433

    codec = numcodecs.get_codec(compressor_spec) if compressor_spec else None
    filters = [numcodecs.get_codec(f) for f in filters_spec] if filters_spec else []

    n_chunks_per_dim = [ceil(s / c) for s, c in zip(shape, chunks)]
    # Fetch chunks sequentially. Coalescing in parallel is possible but the
    # tunnel is already bottlenecked by the per-message WS round-trip, so
    # we avoid stampeding the browser. For big arrays this is the main
    # latency cost — noted in the plan.
    for idx in itertools.product(*(range(n) for n in n_chunks_per_dim)):
        key_parts = [str(i) for i in idx]
        chunk_key = (f"{scale_path}/" if scale_path else "") + dim_sep.join(key_parts)
        raw = await read(chunk_key)
        if raw is None:
            # Chunk missing → leave the fill value in place.
            continue
        buf = codec.decode(raw) if codec else raw
        for f in reversed(filters):
            buf = f.decode(buf)
        chunk_arr = np.frombuffer(buf, dtype=dtype)
        expected = int(np.prod(chunks))
        if chunk_arr.size != expected:
            raise ValueError(
                f"chunk {chunk_key} has {chunk_arr.size} elements; expected {expected}"
            )
        chunk_arr = chunk_arr.reshape(chunks, order=order)
        # Compute the destination slab; last chunks along each axis may be
        # partial when shape isn't a multiple of chunks.
        dst_slices = tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(idx, chunks, shape)
        )
        src_slices = tuple(
            slice(0, s.stop - s.start) for s in dst_slices
        )
        out[dst_slices] = chunk_arr[src_slices]

    return out
