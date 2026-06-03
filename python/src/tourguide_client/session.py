"""Synchronous Tourguide Workspace session client.

Talks to the local bridge over HTTP (the same /op contract the MCP adapter
and the browser transport use). Synchronous for notebook/script ergonomics.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import httpx

from .schemas import WorkspaceError

DEFAULT_BRIDGE_URL = "http://localhost:7723"


class TourguideSession:
    def __init__(self, bridge_url: str = DEFAULT_BRIDGE_URL, op_timeout: float = 35.0):
        self.bridge_url = bridge_url.rstrip("/")
        self._http = httpx.Client(timeout=op_timeout)
        self.record: dict[str, Any] | None = None

    # --- connection ----------------------------------------------------------

    @classmethod
    def attach(
        cls, bridge_url: str = DEFAULT_BRIDGE_URL, wait: float = 15.0
    ) -> "TourguideSession":
        """Attach to a running Tourguide workspace session, waiting up to
        `wait` seconds for a tab to connect. Raises if none appears."""
        s = cls(bridge_url)
        deadline = time.time() + wait
        while True:
            sess = s._running_session()
            if sess:
                s.record = sess
                return s
            if time.time() >= deadline:
                raise WorkspaceError(
                    f"no running Tourguide session at {bridge_url}. Open "
                    "http://localhost:5173/?mode=workspace (with the bridge "
                    "running: `npm run bridge`)."
                )
            time.sleep(0.5)

    def health(self) -> dict[str, Any]:
        r = self._http.get(f"{self.bridge_url}/health")
        r.raise_for_status()
        return r.json()

    def sessions(self) -> list[dict[str, Any]]:
        r = self._http.get(f"{self.bridge_url}/sessions")
        r.raise_for_status()
        return r.json()

    def _running_session(self) -> dict | None:
        try:
            running = [s for s in self.sessions() if s.get("status") == "running"]
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
            r = self._http.post(f"{self.bridge_url}/op", json=request)
        except httpx.HTTPError as e:
            raise WorkspaceError(f"could not reach Tourguide bridge: {e}") from e
        if r.status_code >= 400:
            raise WorkspaceError(f"bridge /op {r.status_code}: {r.text}")
        env = r.json()
        if not env.get("ok"):
            raise WorkspaceError((env.get("error") or {}).get("message", "workspace op failed"))
        return env.get("result")

    # --- convenience wrappers (snake_case -> Workspace API) ------------------

    def get_session(self) -> dict:
        return self.call("get_session")

    def get_viewer_state(self) -> dict:
        return self.call("get_viewer_state")

    def set_viewer_state(self, state: dict) -> dict:
        return self.call("set_viewer_state", {"state": state})

    def get_selection(self) -> dict:
        return self.call("get_selection")

    def select_segments(self, layer: str, segment_ids: list[str]) -> dict:
        return self.call("select_segments", {"layer": layer, "segmentIds": segment_ids})

    def fly_to(self, position: list[float], segment_id: str | None = None, layer: str | None = None) -> dict:
        params: dict[str, Any] = {"position": position}
        if segment_id is not None:
            params["segmentId"] = segment_id
        if layer is not None:
            params["layer"] = layer
        return self.call("fly_to", params)

    def add_layer(self, layer: dict) -> dict:
        return self.call("add_layer", {"layer": layer})

    def add_annotations(self, annotations: list[dict], layer_name: str | None = None) -> dict:
        params: dict[str, Any] = {"annotations": annotations}
        if layer_name is not None:
            params["layerName"] = layer_name
        return self.call("add_annotations", params)

    def load_descriptor(self, descriptor: dict) -> dict:
        return self.call("load_descriptor", {"descriptor": descriptor})

    def list_tables(self) -> dict:
        return self.call("list_tables")

    def get_table_schema(self, table: str) -> dict:
        return self.call("get_table_schema", {"table": table})

    def run_sql(self, sql: str) -> dict:
        return self.call("run_sql", {"sql": sql})

    def show_table(self, sql: str, name: str | None = None) -> dict:
        params: dict[str, Any] = {"sql": sql}
        if name is not None:
            params["name"] = name
        return self.call("show_table", params)

    def show_plot(
        self,
        code: str | None = None,
        question: str | None = None,
        title: str | None = None,
        kind: str | None = None,
        source_table: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {}
        for k, v in (
            ("code", code),
            ("question", question),
            ("title", title),
            ("kind", kind),
            ("sourceTable", source_table),
        ):
            if v is not None:
                params[k] = v
        return self.call("show_plot", params)

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
