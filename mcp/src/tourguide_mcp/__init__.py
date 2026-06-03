"""Tourguide MCP adapter.

A thin proxy from the Model Context Protocol to the Tourguide Workspace API.
The durable artifact is the Workspace API (HTTP + WebSocket served by the
local bridge); this package is just the first adapter. The Python SDK and any
future adapter speak the same operations.
"""

__version__ = "0.1.0"
