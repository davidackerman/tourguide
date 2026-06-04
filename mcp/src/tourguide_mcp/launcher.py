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
from urllib.parse import urlparse

import httpx

from .client import WorkspaceClient, WorkspaceError

DEFAULT_BRIDGE_URL = "http://localhost:7723"
DEFAULT_WORKSPACE_URL = "http://localhost:5173/?mode=workspace"


@dataclass
class LauncherConfig:
    bridge_url: str = os.environ.get("TOURGUIDE_BRIDGE_URL", DEFAULT_BRIDGE_URL)
    workspace_url: str = os.environ.get("TOURGUIDE_WORKSPACE_URL", DEFAULT_WORKSPACE_URL)
    webapp_dir: str | None = os.environ.get("TOURGUIDE_WEBAPP_DIR")
    auto_open: bool = os.environ.get("TOURGUIDE_AUTO_OPEN", "1") != "0"
    # "preview" serves the production build (renders data correctly);
    # "dev" uses the Vite dev server (fast/hot-reload, but its worker/codec
    # handling leaves Neuroglancer image chunks black on some setups).
    webapp_mode: str = os.environ.get("TOURGUIDE_WEBAPP_MODE", "preview")


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
        self._webapp_proc: subprocess.Popen | None = None

    async def ensure_bridge(self, timeout: float = 20.0) -> None:
        if await self.client.is_healthy():
            return
        if not self.config.webapp_dir:
            raise WorkspaceError(
                f"Tourguide bridge is not reachable at {self.config.bridge_url} and "
                "TOURGUIDE_WEBAPP_DIR is not set, so it can't be auto-started. "
                "Start it manually: `cd web-app && npm run bridge`."
            )
        # Spawn `npm run bridge` in its own session so it outlives this MCP
        # process (e.g. when the client restarts the server); it self-reports
        # readiness via /health.
        self._bridge_proc = subprocess.Popen(
            ["npm", "run", "bridge"],
            cwd=self.config.webapp_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        ok = await _wait_for(self.client.is_healthy, timeout=timeout)
        if not ok:
            raise WorkspaceError(
                f"started the bridge but it never became healthy at {self.config.bridge_url}."
            )

    async def _webapp_reachable(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(self.config.workspace_url)
                return r.status_code < 500
        except Exception:
            return False

    async def ensure_webapp(self, timeout: float = 240.0) -> None:
        """Start the web app if it isn't already serving. Idempotent: if a
        server is already up at the workspace URL we skip (covers a manually
        run server and repeated launch calls).

        Defaults to the production build (`npm run build` + `npm run preview`)
        because the Vite dev server can leave Neuroglancer image chunks black
        on some setups. Set TOURGUIDE_WEBAPP_MODE=dev to use the dev server."""
        if await self._webapp_reachable():
            return
        if not self.config.webapp_dir:
            raise WorkspaceError(
                f"Tourguide web app is not reachable at {self.config.workspace_url} "
                "and TOURGUIDE_WEBAPP_DIR is not set, so it can't be auto-started. "
                "Start it manually: `cd web-app && npm run preview` (after `npm run build`)."
            )
        # Pin the port so the URL we open matches the server we start.
        port = str(urlparse(self.config.workspace_url).port or 5173)
        if self.config.webapp_mode == "dev":
            cmd = ["npm", "run", "dev", "--", "--port", port, "--strictPort"]
        else:
            # Build once, then serve the static output via Vite preview.
            build = await asyncio.create_subprocess_exec(
                "npm", "run", "build",
                cwd=self.config.webapp_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if await build.wait() != 0:
                raise WorkspaceError(
                    "`npm run build` failed, so the preview server can't start. "
                    "Run it manually in web-app to see the error, or set "
                    "TOURGUIDE_WEBAPP_MODE=dev to use the dev server instead."
                )
            cmd = ["npm", "run", "preview", "--", "--port", port, "--strictPort"]
        self._webapp_proc = subprocess.Popen(
            cmd,
            cwd=self.config.webapp_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        ok = await _wait_for(self._webapp_reachable, timeout=timeout)
        if not ok:
            raise WorkspaceError(
                f"started the web app but it never came up at {self.config.workspace_url}. "
                f"Is port {port} free? (a stale server there will block --strictPort)."
            )

    async def launch_or_attach(self, wait_for_session: float = 45.0) -> dict:
        """Return the attached session record, launching the whole stack if
        needed: bridge -> web app -> a workspace tab."""
        await self.ensure_bridge()

        running = await self._running_session()
        if running:
            return running

        # No session yet — make sure the web app is up, then open a tab.
        await self.ensure_webapp()
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
