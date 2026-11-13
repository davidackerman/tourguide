"""Synchronous Tourguide Workspace session client.

Talks to the local bridge over HTTP (the same /op contract the MCP adapter
and the browser transport use). Synchronous for notebook/script ergonomics.

Auth: the bridge requires a bearer token. It is read from
TOURGUIDE_BRIDGE_TOKEN (or TG_BRIDGE_TOKEN), else from the token file the
bridge writes in the OS temp dir (`tourguide-bridge-<port>.token`), else pass
`token=` explicitly.
"""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx

from .schemas import WorkspaceError

DEFAULT_BRIDGE_URL = "http://127.0.0.1:7723"


def token_file_for(bridge_url: str) -> Path:
    port = urlparse(bridge_url).port or 7723
    return Path(tempfile.gettempdir()) / f"tourguide-bridge-{port}.token"


def discover_token(bridge_url: str) -> str | None:
    env = os.environ.get("TOURGUIDE_BRIDGE_TOKEN") or os.environ.get("TG_BRIDGE_TOKEN")
    if env:
        return env
    try:
        raw = token_file_for(bridge_url).read_text().strip()
    except OSError:
        return None
    if not raw:
        return None
    if raw.startswith("{"):  # {token, viewToken}
        try:
            return json.loads(raw).get("token") or None
        except json.JSONDecodeError:
            return None
    return raw


class TourguideSession:
    def __init__(
        self,
        bridge_url: str = DEFAULT_BRIDGE_URL,
        op_timeout: float = 35.0,
        token: str | None = None,
    ):
        self.bridge_url = bridge_url.rstrip("/")
        self.token = token or discover_token(self.bridge_url)
        self._http = httpx.Client(timeout=op_timeout)
        self.record: dict[str, Any] | None = None

    # --- connection ----------------------------------------------------------

    @classmethod
    def attach(
        cls,
        bridge_url: str = DEFAULT_BRIDGE_URL,
        wait: float = 15.0,
        token: str | None = None,
    ) -> "TourguideSession":
        """Attach to a running Tourguide workspace session, waiting up to
        `wait` seconds for a tab to connect. Raises if none appears."""
        s = cls(bridge_url, token=token)
        deadline = time.time() + wait
        while True:
            sess = s._running_session()
            if sess:
                s.record = sess
                return s
            if time.time() >= deadline:
                raise WorkspaceError(
                    f"no running Tourguide session at {bridge_url}. Start the stack "
                    "(`cd web-app && npm run workspace:preview`) and open the workspace "
                    "URL it prints (it carries the bridge token)."
                )
            time.sleep(0.5)

    def _headers(self) -> dict[str, str]:
        if self.token is None:
            self.token = discover_token(self.bridge_url)
        return {"authorization": f"Bearer {self.token}"} if self.token else {}

    def _check(self, r: httpx.Response) -> None:
        if r.status_code == 401:
            self.token = None
            raise WorkspaceError(
                "bridge rejected the request (401). Pass token=..., set TOURGUIDE_BRIDGE_TOKEN, "
                f"or make {token_file_for(self.bridge_url)} readable."
            )
        r.raise_for_status()

    def health(self) -> dict[str, Any]:
        r = self._http.get(f"{self.bridge_url}/health")
        r.raise_for_status()
        return r.json()

    def sessions(self) -> list[dict[str, Any]]:
        r = self._http.get(f"{self.bridge_url}/sessions", headers=self._headers())
        self._check(r)
        return r.json()

    def _running_session(self) -> dict | None:
        try:
            running = [s for s in self.sessions() if s.get("status") == "running"]
        except WorkspaceError:
            raise
        except Exception:
            return None
        if not running:
            return None
        running.sort(key=lambda s: s.get("createdAt", ""), reverse=True)
        return running[0]

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "TourguideSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- core call -----------------------------------------------------------

    def call(self, op: str, params: dict[str, Any] | None = None) -> Any:
        request = {"id": str(uuid.uuid4()), "op": op, "params": params, "source": "python_sdk"}
        try:
            r = self._http.post(f"{self.bridge_url}/op", json=request, headers=self._headers())
        except httpx.HTTPError as e:
            raise WorkspaceError(f"could not reach Tourguide bridge: {e}") from e
        self._check(r)
        env = r.json()
        if not env.get("ok"):
            raise WorkspaceError((env.get("error") or {}).get("message", "workspace op failed"))
        return env.get("result")

    # --- session / viewer ----------------------------------------------------

    def get_session(self) -> dict:
        return self.call("get_session")

    def load_descriptor(self, descriptor: dict, wait: bool = True) -> dict:
        return self.call("load_descriptor", {"descriptor": descriptor, "wait": wait})

    def wait_for_ready(self, timeout_ms: int = 30000) -> dict:
        return self.call("wait_for_ready", {"timeoutMs": timeout_ms})

    def screenshot(self, path: str | Path | None = None, max_width: int | None = None) -> bytes:
        """PNG bytes of the current view; also written to `path` if given."""
        params: dict[str, Any] = {}
        if max_width is not None:
            params["maxWidth"] = max_width
        res = self.call("screenshot", params)
        data = base64.b64decode(res["png"])
        if path is not None:
            Path(path).write_bytes(data)
        return data

    def get_viewer_state(self) -> dict:
        return self.call("get_viewer_state")

    def set_viewer_state(self, state: dict) -> dict:
        return self.call("set_viewer_state", {"state": state})

    def get_selection(self) -> dict:
        return self.call("get_selection")

    def select_segments(self, layer: str, segment_ids: Iterable[Any]) -> dict:
        return self.call("select_segments", {"layer": layer, "segmentIds": [str(i) for i in segment_ids]})

    def fly_to(self, position: list[float], segment_id: str | None = None, layer: str | None = None) -> dict:
        params: dict[str, Any] = {"position": list(position)}
        if segment_id is not None:
            params["segmentId"] = str(segment_id)
        if layer is not None:
            params["layer"] = layer
        return self.call("fly_to", params)

    def fly_to_segment(self, layer: str, segment_id: Any, table: str | None = None, select: bool = True) -> dict:
        params: dict[str, Any] = {"layer": layer, "segmentId": str(segment_id), "select": select}
        if table is not None:
            params["table"] = table
        return self.call("fly_to_segment", params)

    def add_layer(self, layer: dict) -> dict:
        return self.call("add_layer", {"layer": layer})

    def add_annotations(self, annotations: list[dict], layer_name: str | None = None, replace: bool = False) -> dict:
        params: dict[str, Any] = {"annotations": annotations, "replace": replace}
        if layer_name is not None:
            params["layerName"] = layer_name
        return self.call("add_annotations", params)

    # --- tables --------------------------------------------------------------

    def list_tables(self) -> dict:
        return self.call("list_tables")

    def get_table_schema(self, table: str) -> dict:
        return self.call("get_table_schema", {"table": table})

    def run_sql(self, sql: str) -> dict:
        return self.call("run_sql", {"sql": sql})

    def ingest_table(self, name: str, columns: list[str], rows: list[list]) -> dict:
        """Push a table you computed into Tourguide (the core artifact-sink op).
        Include 'object_id' + 'com_x_nm'/'com_y_nm'/'com_z_nm' for click-to-fly."""
        return self.call("ingest_table", {"name": name, "columns": columns, "rows": rows})

    def ingest_dataframe(self, name: str, df: Any) -> dict:
        """ingest_table for a pandas DataFrame (NaN -> NULL, numpy -> python)."""
        import math

        cols = [str(c) for c in df.columns]
        rows = []
        for rec in df.itertuples(index=False, name=None):
            row = []
            for v in rec:
                if hasattr(v, "item"):
                    v = v.item()
                if isinstance(v, float) and math.isnan(v):
                    v = None
                row.append(v)
            rows.append(row)
        return self.ingest_table(name, cols, rows)

    def show_table(self, sql: str, name: str | None = None) -> dict:
        params: dict[str, Any] = {"sql": sql}
        if name is not None:
            params["name"] = name
        return self.call("show_table", params)

    # --- plots ---------------------------------------------------------------

    def show_plot(
        self,
        png: bytes | str,
        title: str | None = None,
        kind: str | None = None,
        source_table: str | None = None,
    ) -> dict:
        """Display a PNG you rendered: raw bytes, base64 text, or a data URL."""
        if isinstance(png, (bytes, bytearray)):
            png = base64.b64encode(bytes(png)).decode("ascii")
        params: dict[str, Any] = {"png": png}
        for k, v in (("title", title), ("kind", kind), ("sourceTable", source_table)):
            if v is not None:
                params[k] = v
        return self.call("show_plot", params)

    def show_plot_file(self, path: str | Path, title: str | None = None, **kw: Any) -> dict:
        p = Path(path)
        return self.show_plot(p.read_bytes(), title=title if title is not None else p.stem, **kw)

    def show_figure(self, fig: Any, title: str | None = None, dpi: int = 120, **kw: Any) -> dict:
        """Display a matplotlib Figure (rendered here, shown in Tourguide)."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        return self.show_plot(buf.getvalue(), title=title, **kw)

    # --- events --------------------------------------------------------------

    def get_recent_events(self, since_seq: int = 0, types: list[str] | None = None) -> dict:
        return self._events(since_seq, 0, types)

    def wait_for_user_action(
        self, since_seq: int = 0, timeout_ms: int = 60000, types: list[str] | None = None
    ) -> dict:
        """Block until the person at the screen selects segments / moves the
        camera / loads a dataset (or timeout). Returns {events, latestSeq}."""
        types = types or ["selection_changed", "position_changed", "dataset_changed"]
        return self._events(since_seq, min(timeout_ms, 120_000), types)

    def _events(self, since: int, wait_ms: int, types: list[str] | None) -> dict:
        params: dict[str, Any] = {"since": since, "wait": wait_ms}
        if types:
            params["types"] = ",".join(types)
        r = self._http.get(
            f"{self.bridge_url}/events",
            params=params,
            headers=self._headers(),
            timeout=wait_ms / 1000 + 10,
        )
        self._check(r)
        return r.json()

    # --- saved states / recording --------------------------------------------

    def save_session_state(self, name: str | None = None) -> dict:
        return self.call("save_session_state", {"name": name} if name else {})

    def restore_session_state(self, id: str) -> dict:
        return self.call("restore_session_state", {"id": id})

    def list_saved_states(self) -> dict:
        return self.call("list_saved_states")

    def start_recording(self) -> dict:
        return self.call("start_recording")

    def stop_recording(self) -> dict:
        return self.call("stop_recording")

    def add_narration_note(self, text: str, position: list[float] | None = None, segment_id: str | None = None) -> dict:
        params: dict[str, Any] = {"text": text}
        if position is not None:
            params["position"] = position
        if segment_id is not None:
            params["segmentId"] = segment_id
        return self.call("add_narration_note", params)

    def export_session_summary(self) -> dict:
        return self.call("export_session_summary")
