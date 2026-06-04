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
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from .client import WorkspaceClient, WorkspaceError

DEFAULT_BRIDGE_URL = "http://localhost:7723"
DEFAULT_WORKSPACE_URL = "http://localhost:5173/?mode=workspace"


def _detect_webapp_dir() -> str | None:
    """Find the sibling web-app/ from the installed package location, so the
    MCP server can auto-start it without TOURGUIDE_WEBAPP_DIR being set or any
    assumption about the client's working directory. This file lives at
    <repo>/mcp/src/tourguide_mcp/launcher.py → <repo>/web-app."""
    env = os.environ.get("TOURGUIDE_WEBAPP_DIR")
    if env:
        return env
    from pathlib import Path

    candidate = Path(__file__).resolve().parents[3] / "web-app"
    return str(candidate) if (candidate / "package.json").exists() else None


@dataclass
class LauncherConfig:
    bridge_url: str = os.environ.get("TOURGUIDE_BRIDGE_URL", DEFAULT_BRIDGE_URL)
    workspace_url: str = os.environ.get("TOURGUIDE_WORKSPACE_URL", DEFAULT_WORKSPACE_URL)
    webapp_dir: str | None = field(default_factory=_detect_webapp_dir)
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

    def lan_url(self) -> str | None:
        """The workspace URL with this machine's LAN IP swapped in, so it can
        be shared with others on the same network (vite preview + the bridge
        already bind all interfaces, and the page derives its bridge URL from
        the host it was opened on). Returns None if no LAN IP is found."""
        import socket

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # No packets sent; just selects the primary outbound interface.
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
            finally:
                s.close()
        except Exception:
            return None
        if not ip or ip.startswith("127."):
            return None
        parsed = urlparse(self.config.workspace_url)
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{ip}{port}{parsed.path}{('?' + parsed.query) if parsed.query else ''}"

    async def ensure_deps(self) -> None:
        """Install web-app npm deps if missing. Idempotent and cheap: once
        node_modules exists this is a no-op, so a fresh clone "just works"
        (the launcher shells out to `npm run …`, which needs deps installed)
        without the user having to run `npm install` by hand first."""
        if not self.config.webapp_dir:
            return
        from pathlib import Path

        if (Path(self.config.webapp_dir) / "node_modules").is_dir():
            return
        install = await asyncio.create_subprocess_exec(
            "npm", "install",
            cwd=self.config.webapp_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if await install.wait() != 0:
            raise WorkspaceError(
                "`npm install` failed in the web app, so it can't be started. "
                "Run it manually in web-app to see the error."
            )

    async def ensure_bridge(self, timeout: float = 20.0) -> None:
        if await self.client.is_healthy():
            return
        if not self.config.webapp_dir:
            raise WorkspaceError(
                f"Tourguide bridge is not reachable at {self.config.bridge_url} and "
                "TOURGUIDE_WEBAPP_DIR is not set, so it can't be auto-started. "
                "Start it manually: `cd web-app && npm run bridge`."
            )
        await self.ensure_deps()
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

    async def _reachable_mode(self) -> str | None:
        """Probe the workspace URL and classify what's serving it:
          "dev"     — a Vite dev server (injects the /@vite/client module),
          "preview" — the production build served by `vite preview`,
          None      — nothing reachable.
        The dev server renders Neuroglancer image chunks black on some setups,
        so we treat it as "not what we want" even though it's reachable."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(self.config.workspace_url)
                if r.status_code >= 500:
                    return None
                return "dev" if "/@vite/client" in r.text else "preview"
        except Exception:
            return None

    async def _webapp_reachable(self) -> bool:
        return await self._reachable_mode() is not None

    def _kill_port(self, port: str) -> None:
        """Kill whatever is listening on `port` (e.g. a stray dev server we
        didn't start). Best-effort; uses lsof so it works regardless of which
        process/terminal started the squatter."""
        try:
            out = subprocess.run(
                ["lsof", "-tiTCP:" + port, "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        except Exception:
            return
        for pid in {p for p in out.split() if p.strip()}:
            try:
                subprocess.run(["kill", pid], timeout=5)
            except Exception:
                pass

    async def _is_port_free(self, port: str) -> bool:
        try:
            out = subprocess.run(
                ["lsof", "-tiTCP:" + port, "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            return not out.strip()
        except Exception:
            return True

    async def ensure_webapp(self, timeout: float = 240.0) -> None:
        """Start the web app if it isn't already serving the right thing.

        Idempotent for a *preview* server: if the production build is already
        up at the workspace URL we attach to it (no needless rebuild). But if a
        Vite *dev* server is squatting on the port, we kill it and start preview
        anyway — otherwise it renders Neuroglancer image chunks black, and its
        mere reachability would otherwise defeat the preview default. Set
        TOURGUIDE_WEBAPP_MODE=dev to opt into the dev server on purpose."""
        port = str(urlparse(self.config.workspace_url).port or 5173)
        mode = await self._reachable_mode()
        if mode == self.config.webapp_mode:
            return  # already serving exactly what we want — attach.
        if not self.config.webapp_dir:
            # Can't auto-start (and won't evict a server we can't replace).
            if mode is not None:
                raise WorkspaceError(
                    f"A Vite {mode} server is on {self.config.workspace_url}, but "
                    f"TOURGUIDE_WEBAPP_MODE={self.config.webapp_mode} is wanted and "
                    "TOURGUIDE_WEBAPP_DIR is not set, so it can't be rebuilt. "
                    "Stop it and run `cd web-app && npm run preview` (after `npm run build`)."
                )
            raise WorkspaceError(
                f"Tourguide web app is not reachable at {self.config.workspace_url} "
                "and TOURGUIDE_WEBAPP_DIR is not set, so it can't be auto-started. "
                "Start it manually: `cd web-app && npm run preview` (after `npm run build`)."
            )
        if mode is not None:
            # Wrong server on the port (typically a dev server when we want
            # preview). Evict it so the build below can claim the port.
            self._kill_port(port)
            await _wait_for(lambda: self._is_port_free(port), timeout=10.0)
        # Pin the port so the URL we open matches the server we start.
        await self.ensure_deps()
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

        async def _fresh_session() -> dict | None:
            for s in await self._live_sessions():
                if s["sessionId"] not in before:
                    return s
            return None

        opened = await _wait_for(_fresh_session, timeout=wait_for_session)
        if not opened:
            raise WorkspaceError(
                "no Tourguide workspace session connected. Open "
                f"{self.config.workspace_url} in a browser (it must be able to "
                f"reach the bridge at {self.config.bridge_url}), then retry."
            )
        return opened

    @staticmethod
    def _ambiguous(live: list[dict]) -> dict:
        return {
            "ambiguous": True,
            "sessions": [
                {"sessionId": s["sessionId"], "label": s.get("label"), "url": s.get("url")}
                for s in live
            ],
            "message": (
                "Multiple workspace tabs are open. Ask which to drive, then call "
                "launch_or_attach(session=<sessionId>) — or launch_or_attach(new=True) "
                "for a fresh dedicated tab."
            ),
        }

    async def _find_live(self, session_id: str) -> dict | None:
        for s in await self._live_sessions():
            if s["sessionId"] == session_id:
                return s
        return None

    # A live tab pongs the bridge every ~20s, refreshing lastSeenAt. Treat a
    # "running" session whose lastSeenAt is older than this as not actually
    # live — it's a tab that died without a clean WS close and the bridge
    # hasn't pruned it yet. This closes the gap between a tab vanishing and
    # the bridge's heartbeat noticing, so we never attach to a phantom.
    LIVENESS_WINDOW_S = 45.0

    async def _live_sessions(self) -> list[dict]:
        """Workspace tabs that are genuinely live, newest first. Only tabs with
        a fresh heartbeat count: the bridge and this launcher share one machine
        clock, so lastSeenAt freshness is a reliable liveness signal — a
        "running" record without it is a tab that died ungracefully, and
        attaching to it gives a blank/loading page."""
        try:
            sessions = await self.client.sessions()
        except Exception:
            return []
        live = [s for s in sessions if s.get("status") == "running" and self._seen_recently(s)]
        live.sort(key=lambda s: s.get("createdAt", ""), reverse=True)
        return live

    async def _running_session(self) -> dict | None:
        live = await self._live_sessions()
        return live[0] if live else None

    def _seen_recently(self, session: dict) -> bool:
        from datetime import datetime, timezone

        ts = session.get("lastSeenAt")
        if not ts:
            return False
        try:
            seen = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return False
        age = (datetime.now(timezone.utc) - seen).total_seconds()
        return age <= self.LIVENESS_WINDOW_S
