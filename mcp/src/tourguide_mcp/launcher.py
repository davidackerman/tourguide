"""Launch / attach semantics for Tourguide.

Load-bearing behavior (see the agent-workspace plan):
  1. If the bridge is healthy and a session is running, attach.
  2. If the bridge is down, start it (when a web-app dir is configured).
  3. If no session exists, open the workspace URL in a browser and wait.
  4. Pick the most recently created session for v0 (the bridge does this).
  5. On reconnect failure, raise a clear error so the caller can relaunch.

Config via environment:
  TOURGUIDE_BRIDGE_URL     default http://localhost:7723
  TOURGUIDE_WORKSPACE_URL  default http://localhost:5173/?mode=workspace
  TOURGUIDE_WEBAPP_DIR     path to web-app/ (enables auto-starting the bridge)
  TOURGUIDE_AUTO_OPEN      "1" (default) to open a browser when no session
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import webbrowser
from dataclasses import dataclass

from .client import WorkspaceClient, WorkspaceError

DEFAULT_BRIDGE_URL = "http://localhost:7723"
DEFAULT_WORKSPACE_URL = "http://localhost:5173/?mode=workspace"


@dataclass
class LauncherConfig:
    bridge_url: str = os.environ.get("TOURGUIDE_BRIDGE_URL", DEFAULT_BRIDGE_URL)
    workspace_url: str = os.environ.get("TOURGUIDE_WORKSPACE_URL", DEFAULT_WORKSPACE_URL)
    webapp_dir: str | None = os.environ.get("TOURGUIDE_WEBAPP_DIR")
    auto_open: bool = os.environ.get("TOURGUIDE_AUTO_OPEN", "1") != "0"


async def _wait_for(predicate, timeout: float, interval: float = 0.5):
    """Poll an async predicate until it returns a truthy value or timeout."""
    elapsed = 0.0
    while elapsed < timeout:
        result = await predicate()
        if result:
            return result
        await asyncio.sleep(interval)
        elapsed += interval
    return None


class Launcher:
    def __init__(self, client: WorkspaceClient, config: LauncherConfig | None = None):
        self.client = client
        self.config = config or LauncherConfig()
        self._bridge_proc: subprocess.Popen | None = None

    async def ensure_bridge(self, timeout: float = 20.0) -> None:
        if await self.client.is_healthy():
            return
        if not self.config.webapp_dir:
            raise WorkspaceError(
                f"Tourguide bridge is not reachable at {self.config.bridge_url} and "
                "TOURGUIDE_WEBAPP_DIR is not set, so it can't be auto-started. "
                "Start it manually: `cd web-app && npm run bridge`."
            )
        # Spawn `npm run bridge` detached; it self-reports readiness via /health.
        self._bridge_proc = subprocess.Popen(
            ["npm", "run", "bridge"],
            cwd=self.config.webapp_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ok = await _wait_for(self.client.is_healthy, timeout=timeout)
        if not ok:
            raise WorkspaceError(
                f"started the bridge but it never became healthy at {self.config.bridge_url}."
            )

    async def launch_or_attach(self, wait_for_session: float = 30.0) -> dict:
        """Return the attached session record, launching a tab if needed."""
        await self.ensure_bridge()

        running = await self._running_session()
        if running:
            return running

        # No session yet — open the workspace URL so a tab connects.
        if self.config.auto_open:
            try:
                webbrowser.open(self.config.workspace_url)
            except Exception:
                pass  # headless / no browser — caller may open it manually

        session = await _wait_for(self._running_session, timeout=wait_for_session)
        if not session:
            raise WorkspaceError(
                "no Tourguide workspace session connected. Open "
                f"{self.config.workspace_url} in a browser (it must be able to "
                f"reach the bridge at {self.config.bridge_url}), then retry."
            )
        return session

    async def _running_session(self) -> dict | None:
        try:
            sessions = await self.client.sessions()
        except Exception:
            return None
        running = [s for s in sessions if s.get("status") == "running"]
        if not running:
            return None
        # Most recently created — matches the bridge's relay target (v0).
        running.sort(key=lambda s: s.get("createdAt", ""), reverse=True)
        return running[0]
