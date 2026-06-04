"""Session facade used by the MCP tools.

Holds the client + launcher and the currently attached session record. Tools
call `session.call(op, params)`; if no session is attached yet they get a
clear nudge to run `launch_or_attach` first.
"""

from __future__ import annotations

from typing import Any

from .client import WorkspaceClient, WorkspaceError
from .launcher import Launcher, LauncherConfig


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

    async def launch_or_attach(
        self, new: bool = False, session: str | None = None
    ) -> dict[str, Any]:
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

    async def call(self, op: str, params: dict[str, Any] | None = None) -> Any:
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
