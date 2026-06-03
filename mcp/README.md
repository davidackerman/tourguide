# tourguide-mcp

MCP adapter for the **Tourguide Workspace API**. Lets any MCP-capable agent
(Claude Desktop, Claude Code, Cursor, …) launch or attach to a running
Tourguide session and drive its visual workspace: viewer, selections,
tables, plots, saved states, recording, and narration notes.

This adapter is deliberately **thin** — it proxies to the local Workspace API
bridge over HTTP. The durable artifact is the Workspace API, not MCP.

```
agent ──MCP/stdio──► tourguide-mcp ──HTTP /op──► bridge ──WS──► Tourguide tab
```

## Install & run

```bash
cd mcp
uv sync
uv run tourguide-mcp
```

The server speaks MCP over stdio. Point your MCP client at it, e.g. Claude
Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tourguide": {
      "command": "uv",
      "args": ["run", "tourguide-mcp"],
      "cwd": "/path/to/tourguide/mcp",
      "env": {
        "TOURGUIDE_WEBAPP_DIR": "/path/to/tourguide/web-app"
      }
    }
  }
}
```

## Configuration (environment)

| Var | Default | Meaning |
| --- | --- | --- |
| `TOURGUIDE_BRIDGE_URL` | `http://localhost:7723` | Workspace API bridge |
| `TOURGUIDE_WORKSPACE_URL` | `http://localhost:5173/?mode=workspace` | tab to open when launching |
| `TOURGUIDE_WEBAPP_DIR` | _(unset)_ | path to `web-app/`; enables auto-starting the bridge |
| `TOURGUIDE_AUTO_OPEN` | `1` | open a browser tab when no session is connected |

## Launch / attach

`launch_or_attach`:

1. Bridge healthy + a session running → **attach** to the most recent session.
2. Bridge down → start it (`npm run bridge`) when `TOURGUIDE_WEBAPP_DIR` is set.
3. No session → open the workspace URL and wait for a tab to connect.
4. Still nothing after the wait → clear error telling you what to open.

The Vite dev server (`npm run dev`) must be running for the workspace URL to
load; the adapter starts the *bridge* but not Vite.

## Tools

`launch_or_attach`, `get_session`, `load_descriptor`, `get_viewer_state`,
`set_viewer_state`, `get_selection`, `select_segments`, `fly_to`, `add_layer`,
`add_annotations`, `list_tables`, `get_table_schema`, `run_sql`, `show_table`,
`show_plot`, `save_session_state`, `restore_session_state`, `list_saved_states`,
`start_recording`, `stop_recording`, `add_narration_note`,
`export_session_summary`.

These map 1:1 to Workspace API operations. Prefer the semantic viewer tools
(`select_segments`, `fly_to`, `add_layer`); `set_viewer_state` is the escape
hatch for raw Neuroglancer blobs.
