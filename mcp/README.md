# tourguide-mcp

MCP adapter for the **Tourguide Workspace API**. Lets any MCP-capable agent
(Claude Code, Cursor, Claude Desktop, …) launch or attach to a running
Tourguide session and drive its visual workspace: viewer, selections,
tables, plots, annotations, saved states, and events.

The adapter proxies to the local Workspace API bridge over HTTP. What it does
itself: reads files the agent wrote (`ingest_table(path=)`,
`show_plot(png_path=)`) so big payloads don't pass through the model, returns
`screenshot` as an image, runs the built-in analysis recipes (`measure`,
`run_recipe`, `meshify`) in the agent-side `analysis/` env, and builds share
links.

```
agent ──MCP/stdio──► tourguide-mcp ──HTTP /op (bearer token)──► bridge ──WS──► Tourguide tab
```

## Install & run

```bash
cd mcp
uv sync
uv run tourguide-mcp        # stdio MCP server
```

**Claude Code / Cursor:** the repo's [`.mcp.json`](../.mcp.json) already
registers it. Nothing to configure.

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "tourguide": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/tourguide/mcp", "tourguide-mcp"]
    }
  }
}
```

The web-app directory is auto-detected from the repo layout; set
`TOURGUIDE_WEBAPP_DIR` only if you moved it.

## Launch / attach

`launch_or_attach`:

1. Bridge healthy + a session running → **attach** to the most recent session
   (token read from the bridge's token file).
2. Bridge down → start it with a fresh bearer token (`npm run bridge`).
3. Web app down → `npm run build` + `npm run preview` (production build, so
   image data renders; `TOURGUIDE_WEBAPP_MODE=dev` for the dev server).
4. No session → open `…/?mode=workspace&bridgeToken=…` in a browser and wait.
5. Still nothing → clear error with the URL to open and the log paths.

Bridge / web-app stdout goes to `tourguide-logs/` under the OS temp dir
(`TOURGUIDE_LOG_DIR` to change), mode 0700. Tokens are never written to logs.

With several workspace tabs open, `launch_or_attach` returns
`{ambiguous: true, sessions: […]}` instead of guessing; call it again with
`session_id=` or `new=True`.

## Configuration (environment)

| Var | Default | Meaning |
| --- | --- | --- |
| `TOURGUIDE_BRIDGE_URL` | `http://127.0.0.1:7723` | Workspace API bridge |
| `TOURGUIDE_WORKSPACE_URL` | `http://localhost:5173/?mode=workspace` | tab to open when launching |
| `TOURGUIDE_WEBAPP_DIR` | _(auto)_ | path to `web-app/` |
| `TOURGUIDE_WEBAPP_MODE` | `preview` | `preview` (prod build) or `dev` |
| `TOURGUIDE_AUTO_OPEN` | `1` | open a browser tab when no session is connected |
| `TOURGUIDE_BRIDGE_TOKEN` | _(generated)_ | bearer token; else read from the bridge's token file |
| `TOURGUIDE_LOG_DIR` | `<tmp>/tourguide-logs` | where launch logs go |
| `TG_BRIDGE_HOST` / `TG_HOST` | loopback | set both to `0.0.0.0` to share sessions on the LAN |
| `TG_NEUROGLANCER_URL` | `https://neuroglancer-demo.appspot.com` | host for `share_view` links |

## Tools

Session: `launch_or_attach`, `get_session`, `load_descriptor`, `load_url`,
`wait_for_ready`.
Seeing: `screenshot`.
Viewer: `fly_to`, `fly_to_segment`, `select_segments`, `get_selection`,
`add_layer`, `add_annotations`, `get_viewer_state`, `set_viewer_state`.
Compute (agent-side env): `measure`, `run_recipe`, `list_recipes`, `meshify`.
Tables: `ingest_table`, `run_sql`, `show_table`, `list_tables`, `get_table_schema`.
Plots: `show_plot`.
Events: `get_recent_events`, `wait_for_user_action`.
Sharing: `share_session`, `share_view`, `export_session`.
State: `save_session_state`, `restore_session_state`, `list_saved_states`,
`start_recording`, `stop_recording`, `add_narration_note`,
`export_session_summary`.

Prefer the semantic viewer tools; `set_viewer_state` is the escape hatch for
raw Neuroglancer blobs. Prefer `path=` / `png_path=` for anything larger than a
few hundred rows or a tiny image. `share_session` links carry the read-only
view token; when the bridge is loopback-bound (the default) the tool says so
and the link only works on this machine.
