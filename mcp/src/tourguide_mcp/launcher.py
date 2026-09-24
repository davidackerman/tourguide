"""Launch / attach semantics for Tourguide.

Load-bearing behavior (see the agent-workspace plan):
  1. If the bridge is healthy and a session is running, attach.
  2. If the bridge is down, start it (when a web-app dir is configured).
  3. If no session exists, open the workspace URL in a browser and wait.
  4. Pick the most recently created session for v0 (the bridge does this).
  5. On reconnect failure, raise a clear error so the caller can relaunch.

Config via environment:
  TOURGUIDE_BRIDGE_URL     default http://127.0.0.1:7723
  TOURGUIDE_WORKSPACE_URL  default http://localhost:5173/?mode=workspace
  TOURGUIDE_WEBAPP_DIR     path to web-app/ (auto-detected from this repo)
  TOURGUIDE_WEBAPP_MODE    "preview" (default; production build) or "dev"
  TOURGUIDE_AUTO_OPEN      "1" (default) to open a browser when no session
  TOURGUIDE_BRIDGE_TOKEN   bearer token to use / pass to a bridge we start
                           (default: generated per launch; read back from the
                           bridge's token file when attaching)
  TOURGUIDE_LOG_DIR        where bridge/webapp logs go (default: <tmp>/tourguide-logs)
  TG_BRIDGE_HOST / TG_HOST passed through to the bridge / Vite; default loopback.
                           Set both to 0.0.0.0 to share sessions on the LAN.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import subprocess
import tempfile
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from .client import WorkspaceClient, WorkspaceError

DEFAULT_BRIDGE_URL = "http://127.0.0.1:7723"
DEFAULT_WORKSPACE_URL = "http://localhost:5173/?mode=workspace"
LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def _default_log_dir() -> Path:
    d = Path(os.environ.get("TOURGUIDE_LOG_DIR") or Path(tempfile.gettempdir()) / "tourguide-logs")
    d.mkdir(parents=True, exist_ok=True)
    try:
        d.chmod(0o700)  # private: logs are on a possibly shared machine
    except OSError:
        pass
    return d


def _tail(path: Path, n: int = 15) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "(no log)"


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
    token: str | None = os.environ.get("TOURGUIDE_BRIDGE_TOKEN") or os.environ.get("TG_BRIDGE_TOKEN")
    log_dir: Path = field(default_factory=_default_log_dir)


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
        if self.config.token:
            self.client.set_token(self.config.token)

    @property
    def bridge_log(self) -> Path:
        return self.config.log_dir / "bridge.log"

    @property
    def webapp_log(self) -> Path:
        return self.config.log_dir / "webapp.log"

    def lan_exposed(self) -> bool:
        """True when the bridge/web app were asked to bind a non-loopback
        address (TG_BRIDGE_HOST / TG_HOST), so LAN share links can work."""
        return (
            os.environ.get("TG_BRIDGE_HOST", "127.0.0.1") not in LOOPBACK
            and os.environ.get("TG_HOST", "localhost") not in LOOPBACK
        )

    def workspace_url_with_token(self, view: bool = False) -> str:
        """The workspace URL plus bridgeToken (and bridgePort if non-default)
        so the tab can authenticate to the bridge. view=True uses the weaker
        view token (read-only viewer links)."""
        u = urlparse(self.config.workspace_url)
        q = dict(parse_qsl(u.query))
        q.setdefault("mode", "workspace")
        bport = urlparse(self.config.bridge_url).port or 7723
        if bport != 7723:
            q["bridgePort"] = str(bport)
        token = self.client.view_token if view else self.client.token
        if token:
            q["bridgeToken"] = token
        return urlunparse(u._replace(query=urlencode(q)))

    def lan_url(self) -> str | None:
        """The workspace URL with this machine's LAN IP swapped in, for sharing
        with others on the same network. Only meaningful when the bridge and
        web app are bound to a non-loopback address (lan_exposed()). Returns
        None if no LAN IP is found."""
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
        log = open(self.webapp_log, "ab")
        install = await asyncio.create_subprocess_exec(
            "npm", "install",
            cwd=self.config.webapp_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if await install.wait() != 0:
            raise WorkspaceError(
                "`npm install` failed in the web app, so it can't be started.\n"
                f"Log ({self.webapp_log}):\n{_tail(self.webapp_log, 30)}"
            )

    async def ensure_bridge(self, timeout: float = 20.0) -> None:
        if await self.client.is_healthy():
            # Attaching to a bridge someone else started: pick up its token.
            health = await self.client.health()
            if health.get("auth") and self.client.token is None:
                raise WorkspaceError(
                    f"a bridge is running at {self.config.bridge_url} but its token could not be "
                    "found. Set TOURGUIDE_BRIDGE_TOKEN to the value it was started with."
                )
            return
        if not self.config.webapp_dir:
            raise WorkspaceError(
                f"Tourguide bridge is not reachable at {self.config.bridge_url} and "
                "TOURGUIDE_WEBAPP_DIR is not set, so it can't be auto-started. "
                "Start it manually: `cd web-app && npm run bridge`."
            )
        await self.ensure_deps()
        token = self.config.token or secrets.token_urlsafe(24)
        self.config.token = token
        self.client.set_token(token)
        parsed = urlparse(self.config.bridge_url)
        env = {
            **os.environ,
            "TG_BRIDGE_TOKEN": token,
            "TG_BRIDGE_PORT": str(parsed.port or 7723),
            "TG_BRIDGE_HOST": os.environ.get("TG_BRIDGE_HOST") or parsed.hostname or "127.0.0.1",
        }
        # Spawn `npm run bridge` in its own session so it outlives this MCP
        # process (e.g. when the client restarts the server); stdout/err go to
        # a log file so a failure is diagnosable instead of silent. The token
        # travels in the environment, never on the command line or in logs.
        log = open(self.bridge_log, "ab")
        self._bridge_proc = subprocess.Popen(
            ["npm", "run", "bridge"],
            cwd=self.config.webapp_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        ok = await _wait_for(self.client.is_healthy, timeout=timeout)
        if not ok:
            raise WorkspaceError(
                f"started the bridge but it never became healthy at {self.config.bridge_url}.\n"
                f"Log ({self.bridge_log}):\n{_tail(self.bridge_log)}"
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
        log = open(self.webapp_log, "ab")
        if self.config.webapp_mode == "dev":
            cmd = ["npm", "run", "dev", "--", "--port", port, "--strictPort"]
        else:
            # Build once, then serve the static output via Vite preview.
            build = await asyncio.create_subprocess_exec(
                "npm", "run", "build",
                cwd=self.config.webapp_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if await build.wait() != 0:
                raise WorkspaceError(
                    "`npm run build` failed, so the preview server can't start.\n"
                    f"Log ({self.webapp_log}):\n{_tail(self.webapp_log, 40)}"
                )
            cmd = ["npm", "run", "preview", "--", "--port", port, "--strictPort"]
        self._webapp_proc = subprocess.Popen(
            cmd,
            cwd=self.config.webapp_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ok = await _wait_for(self._webapp_reachable, timeout=timeout)
        if not ok:
            raise WorkspaceError(
                f"started the web app but it never came up at {self.config.workspace_url}. "
                f"Is port {port} free? (a stale server there will block --strictPort).\n"
                f"Log ({self.webapp_log}):\n{_tail(self.webapp_log)}"
            )

    async def launch_or_attach(
        self,
        wait_for_session: float = 45.0,
        new: bool = False,
        session: str | None = None,
    ) -> dict:
        """Bind this caller to a workspace tab, launching the stack as needed.

        Selection — never silently guesses among multiple tabs:
          - `session` given  → attach to that specific tab (error if it's gone).
          - `new=True`       → open a fresh, dedicated tab and bind to it.
          - exactly one live → attach to it.
          - none live        → open one and attach.
          - several live     → return an `{ambiguous: True, sessions: [...]}`
                               payload so the caller asks the user which to use
                               (or to pass `new=True`).
        """
        await self.ensure_bridge()

        if session is not None:
            chosen = await self._find_live(session)
            if not chosen:
                raise WorkspaceError(
                    f"workspace tab '{session}' is not connected. "
                    "Run launch_or_attach without a session to see what's open."
                )
            return chosen

        if not new:
            live = await self._live_sessions()
            if len(live) == 1:
                return live[0]
            if len(live) > 1:
                return self._ambiguous(live)

        # new=True, or nothing live: open a fresh tab and bind to it. Track the
        # tabs that already exist so we return the newly-opened one, not an
        # existing one (important when `new` and others are already open).
        before = {s["sessionId"] for s in await self._live_sessions()}
        await self.ensure_webapp()
        url = self.workspace_url_with_token()
        if self.config.auto_open:
            try:
                webbrowser.open(url)
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
                "no Tourguide workspace session connected. Open this URL in a browser "
                f"on this machine (it carries the bridge token), then retry:\n  {url}"
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
        live = [
            s for s in sessions
            if s.get("status") == "running" and self._seen_recently(s) and not s.get("readOnly")
        ]
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
