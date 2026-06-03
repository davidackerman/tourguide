"""HTTP client for the Tourguide Workspace API bridge.

Mirrors web-app/src/workspace_api/transport_http.ts exactly: POST a
WorkspaceRequest to /op, get a WorkspaceResponse back. The MCP adapter and
the Python SDK both build on this.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx


class WorkspaceError(RuntimeError):
    """A Workspace operation returned ok:false, or the bridge was unreachable."""


class WorkspaceClient:
    def __init__(self, base_url: str, source: str = "mcp", op_timeout: float = 35.0):
        self.base_url = base_url.rstrip("/")
        self.source = source
        self.op_timeout = op_timeout

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
            r = await c.get(f"{self.base_url}/sessions")
            r.raise_for_status()
            return r.json()

    async def call(self, op: str, params: dict[str, Any] | None = None) -> Any:
        """Issue one operation, returning its result or raising WorkspaceError."""
        request = {
            "id": str(uuid.uuid4()),
            "op": op,
            "params": params,
            "source": self.source,
        }
        try:
            async with httpx.AsyncClient(timeout=self.op_timeout) as c:
                r = await c.post(f"{self.base_url}/op", json=request)
        except httpx.HTTPError as e:
            raise WorkspaceError(
                f"could not reach Tourguide bridge at {self.base_url}: {e}. "
                "Is the bridge running (npm run bridge) and a workspace tab open?"
            ) from e
        if r.status_code >= 400:
            raise WorkspaceError(f"bridge /op {r.status_code}: {r.text}")
        env = r.json()
        if not env.get("ok"):
            msg = (env.get("error") or {}).get("message", "workspace op failed")
            raise WorkspaceError(msg)
        return env.get("result")
