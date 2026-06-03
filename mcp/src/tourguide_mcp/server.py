"""tourguide-mcp — stdio MCP server proxying to the Tourguide Workspace API.

Run:  uv run tourguide-mcp        (or: tourguide-mcp once installed)

The server launches/attaches to a Tourguide workspace session via the local
bridge and exposes high-level workspace tools to any MCP-capable agent.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .launcher import LauncherConfig
from .session import WorkspaceSession
from .tools import register_tools


def build_server() -> FastMCP:
    mcp = FastMCP("tourguide")
    session = WorkspaceSession(LauncherConfig())
    register_tools(mcp, session)
    return mcp


def main() -> None:
    build_server().run()  # stdio transport by default


if __name__ == "__main__":
    main()
