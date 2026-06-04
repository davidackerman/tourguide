"""Session facade used by the MCP tools.

Holds the client + launcher and the currently attached session record. Tools
call `session.call(op, params)`; if no session is attached yet they get a
clear nudge to run `launch_or_attach` first.
"""

from __future__ import annotations

import os
import urllib.parse
import webbrowser
from typing import Any

from .client import WorkspaceClient, WorkspaceError
from .launcher import Launcher, LauncherConfig, _wait_for

# Viewer ops that, in python-viewer mode, are served by the in-process
# Neuroglancer viewer directly instead of being relayed to a browser tab.
_VIEWER_OPS = frozenset({
    "get_viewer_state", "set_viewer_state", "fly_to", "select_segments",
    "add_layer", "get_selection", "get_session",
})


def _trim_session_urls(result: dict[str, Any]) -> None:
    """Drop the '#!{...}' viewer-state fragment from any session URL in a
    launch_or_attach result (a single record or an ambiguous {sessions:[…]}),
    so thousands of chars of encoded state never reach the agent's context."""
    def trim(rec: Any) -> None:
        if isinstance(rec, dict) and isinstance(rec.get("url"), str):
            rec["url"] = rec["url"].split("#!", 1)[0]

    trim(result)
    for s in result.get("sessions", []) or []:
        trim(s)


class WorkspaceSession:
    def __init__(self, config: LauncherConfig | None = None):
        self.config = config or LauncherConfig()
        self.client = WorkspaceClient(self.config.bridge_url)
        self.launcher = Launcher(self.client, self.config)
        self.record: dict[str, Any] | None = None
        # Spike: TG_VIEWER=python drives an in-process Neuroglancer viewer
        # directly (no bridge → browser relay for viewer ops).
        self.ng = None
        if os.environ.get("TG_VIEWER", "bridge").lower() == "python":
            from .ng_viewer import NgViewer

            self.ng = NgViewer()

    async def launch_or_attach(
        self, new: bool = False, session: str | None = None
    ) -> dict[str, Any]:
        # Python-viewer mode: the viewer lives in this process. Co-start the web
        # app (for tables/plots) and open it EMBEDDING the in-process viewer
        # (?ngViewer=…). Viewer ops go straight to the viewer; workspace ops
        # (ingest_table/show_plot/run_sql) still relay through the bridge to the
        # embedding page. Falls back to the bare viewer URL if the web app can't
        # be started.
        if self.ng is not None:
            viewer_url = self.ng.url()
            open_url = viewer_url
            try:
                await self.launcher.ensure_bridge()
                await self.launcher.ensure_webapp()
                ws = self.config.workspace_url
                sep = "&" if "?" in ws else "?"
                open_url = f"{ws}{sep}ngViewer={urllib.parse.quote(viewer_url, safe='')}"
            except WorkspaceError:
                pass  # no web app — viewer still works, just no tables/plots
            if self.config.auto_open:
                try:
                    webbrowser.open(open_url)
                except Exception:
                    pass
            # Wait (briefly) for the embedding tab to register, so workspace ops
            # have a session to route to. Not fatal if it doesn't.
            await _wait_for(self.launcher._running_session, timeout=20.0)
            self.record = {
                "viewer": "python",
                "mode": "python-embedded",
                "viewerUrl": viewer_url,
                "url": open_url,
            }
            return self.record
        result = await self.launcher.launch_or_attach(new=new, session=session)
        # An ambiguous result (several tabs open, no choice made) is a prompt
        # for the agent to pick — don't pin it as the bound tab.
        if not result.get("ambiguous"):
            self.record = result
        # Strip the giant '#!{...}' Neuroglancer state out of session URLs
        # before they reach the agent — it can be thousands of chars per record
        # and the agent never needs it (it's the encoded viewer state).
        _trim_session_urls(result)
        # NB: deliberately do NOT advertise the LAN workspace URL as a "share"
        # here — opening it gives a fresh BLANK workspace, not this view. To
        # share the actual view use share_view (a Neuroglancer link that
        # carries the state) or export_session (a portable file).
        return result

    @property
    def session_id(self) -> str | None:
        return self.record.get("sessionId") if self.record else None

    def _viewer_call(self, op: str, p: dict[str, Any]) -> Any:
        """Serve a viewer op from the in-process Python Neuroglancer viewer."""
        ng = self.ng
        if op == "set_viewer_state":
            ng.set_state(p["state"]); return {"ok": True}
        if op == "get_viewer_state":
            return ng.get_state()
        if op == "fly_to":
            ng.fly_to(p["position"])
            if p.get("layer") and p.get("segmentId"):
                ng.select_segments(p["layer"], [p["segmentId"]])
            return {"position": p["position"]}
        if op == "select_segments":
            return ng.select_segments(p["layer"], p.get("segmentIds", []))
        if op == "add_layer":
            ng.add_layer(p["layer"]); return {"ok": True}
        if op == "get_selection":
            return ng.get_selection()
        if op == "get_session":
            st = ng.get_state()
            return {
                "mode": "python-viewer",
                "viewerUrl": ng.url(),
                "viewer": {
                    "layers": [{"name": l.get("name"), "type": l.get("type")}
                               for l in st.get("layers", [])],
                    "position": st.get("position"),
                },
            }
        raise WorkspaceError(f"viewer op {op!r} not supported in python-viewer mode")

    async def call(self, op: str, params: dict[str, Any] | None = None) -> Any:
        # Python-viewer mode: serve viewer ops directly (workspace ops like
        # ingest_table/show_plot still need the web app — out of scope here).
        if self.ng is not None and op in _VIEWER_OPS:
            return self._viewer_call(op, params or {})
        # Lazily attach so the agent can call any tool without ceremony, but
        # surface a precise error if nothing is connectable.
        if self.record is None:
            try:
                await self.launch_or_attach()
            except WorkspaceError:
                # Fall through: the op itself will raise the precise reason
                # (e.g. "no running Tourguide session") from the bridge.
                pass
        # Route to the bound tab so this caller keeps driving the same one even
        # if other tabs are open; without a pin the bridge errors on ambiguity
        # rather than guessing.
        return await self.client.call(op, params, session=self.session_id)
