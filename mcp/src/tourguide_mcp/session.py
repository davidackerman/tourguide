"""Session facade used by the MCP tools.

Holds the client + launcher and the currently attached session record. Tools
call `session.call(op, params)`; if no session is attached yet they get a
clear nudge to run `launch_or_attach` first.
"""

from __future__ import annotations

from typing import Any

from .client import WorkspaceClient, WorkspaceError
from .launcher import Launcher, LauncherConfig


class WorkspaceSession:
    def __init__(self, config: LauncherConfig | None = None):
        self.config = config or LauncherConfig()
        self.client = WorkspaceClient(self.config.bridge_url)
        self.launcher = Launcher(self.client, self.config)
        self.record: dict[str, Any] | None = None

    async def launch_or_attach(self) -> dict[str, Any]:
        self.record = await self.launcher.launch_or_attach()
        return self.record

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
        return await self.client.call(op, params)
