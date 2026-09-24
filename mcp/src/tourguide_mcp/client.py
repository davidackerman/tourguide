"""HTTP client for the Tourguide Workspace API bridge.

POST a WorkspaceRequest to /op, get a WorkspaceResponse back. The MCP adapter
and the Python SDK both build on this.

Auth: the bridge requires a bearer token. It comes from TOURGUIDE_BRIDGE_TOKEN
(or TG_BRIDGE_TOKEN), else from the token file the bridge writes in the OS
temp dir (`tourguide-bridge-<port>.token`, JSON {token, viewToken}).
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


class WorkspaceError(RuntimeError):
    """A Workspace operation returned ok:false, or the bridge was unreachable."""


def token_file_for(bridge_url: str) -> Path:
    port = urlparse(bridge_url).port or 7723
    return Path(tempfile.gettempdir()) / f"tourguide-bridge-{port}.token"


def read_token_file(bridge_url: str) -> dict[str, str]:
    """{token, viewToken} from the bridge's token file, or {} if absent.
    Accepts the older plain-string format too."""
    try:
        raw = token_file_for(bridge_url).read_text().strip()
    except OSError:
        return {}
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            d = json.loads(raw)
            return {k: v for k, v in d.items() if isinstance(v, str)}
        except json.JSONDecodeError:
            return {}
    return {"token": raw}


def discover_token(bridge_url: str) -> str | None:
    env = os.environ.get("TOURGUIDE_BRIDGE_TOKEN") or os.environ.get("TG_BRIDGE_TOKEN")
    if env:
        return env
    return read_token_file(bridge_url).get("token")


def discover_view_token(bridge_url: str) -> str | None:
    env = os.environ.get("TOURGUIDE_BRIDGE_VIEW_TOKEN") or os.environ.get("TG_BRIDGE_VIEW_TOKEN")
    if env:
        return env
    return read_token_file(bridge_url).get("viewToken")


class WorkspaceClient:
    def __init__(
        self,
        base_url: str,
        source: str = "mcp",
        op_timeout: float = 35.0,
        token: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.source = source
        self.op_timeout = op_timeout
        self._token = token

    @property
    def token(self) -> str | None:
        # Re-discover lazily: the bridge may have been (re)started after this
        # client was created, writing a fresh token file.
        if self._token is None:
            self._token = discover_token(self.base_url)
        return self._token

    def set_token(self, token: str | None) -> None:
        self._token = token

    @property
    def view_token(self) -> str | None:
        return discover_view_token(self.base_url)

    def _headers(self) -> dict[str, str]:
        t = self.token
        return {"authorization": f"Bearer {t}"} if t else {}

    def _raise_for_auth(self, r: httpx.Response) -> None:
        if r.status_code == 401:
            self._token = None  # force re-discovery next time
            raise WorkspaceError(
                "bridge rejected the request (401). Set TOURGUIDE_BRIDGE_TOKEN or make sure "
                f"{token_file_for(self.base_url)} is readable."
            )

    async def events(
        self, since: int = 0, wait_ms: int = 0, types: list[str] | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"since": since, "wait": wait_ms}
        if types:
            params["types"] = ",".join(types)
        async with httpx.AsyncClient(timeout=wait_ms / 1000 + 10) as c:
            r = await c.get(f"{self.base_url}/events", params=params, headers=self._headers())
            self._raise_for_auth(r)
            r.raise_for_status()
            return r.json()

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{self.base_url}/health")
            r.raise_for_status()
            return r.json()

    async def is_healthy(self) -> bool:
        try:
            h = await self.health()
            return bool(h.get("ok"))
        except Exception:
            return False

    async def sessions(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{self.base_url}/sessions", headers=self._headers())
            self._raise_for_auth(r)
            r.raise_for_status()
            return r.json()

    async def share_state(self, state: Any) -> str:
        """Store a viewer state on the bridge for a short LAN link; returns its
        id (the recipient's browser fetches it back via /share-state/<id>)."""
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.post(f"{self.base_url}/share-state", json=state, headers=self._headers())
        self._raise_for_auth(r)
        if r.status_code >= 400:
            raise WorkspaceError(f"share-state failed: HTTP {r.status_code}")
        body = r.json()
        if not body.get("ok") or not body.get("id"):
            raise WorkspaceError("share-state: bridge returned no id")
        return body["id"]

    async def call(
        self, op: str, params: dict[str, Any] | None = None, session: str | None = None
    ) -> Any:
        """Issue one operation, returning its result or raising WorkspaceError.

        `session` pins the op to a specific workspace tab (by sessionId). When
        omitted the bridge routes to the sole live tab, or errors if several
        are open rather than guessing."""
        request: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "op": op,
            "params": params,
            "source": self.source,
        }
        if session is not None:
            request["session"] = session
        try:
            async with httpx.AsyncClient(timeout=self.op_timeout) as c:
                r = await c.post(f"{self.base_url}/op", json=request, headers=self._headers())
        except httpx.HTTPError as e:
            raise WorkspaceError(
                f"could not reach Tourguide bridge at {self.base_url}: {e}. "
                "Call launch_or_attach first (it starts the bridge and web app)."
            ) from e
        self._raise_for_auth(r)
        if r.status_code >= 400:
            raise WorkspaceError(f"bridge /op {r.status_code}: {r.text}")
        env = r.json()
        if not env.get("ok"):
            msg = (env.get("error") or {}).get("message", "workspace op failed")
            raise WorkspaceError(msg)
        return env.get("result")
