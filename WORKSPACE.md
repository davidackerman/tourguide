# Tourguide Workspace mode

Tourguide is becoming a persistent **visual workspace** controlled by external
agents (Claude Desktop, Claude Code, Cursor, a local script, …) rather than a
chat app with its own LLM plumbing.

```
Agent owns conversation/reasoning.
Tourguide owns visual state.
```

The durable artifact is the **Workspace API**. MCP is the first adapter, not
the product.

## Architecture

```
External agent
  └─ MCP adapter / Python SDK / HTTP client
       └─ Workspace API  (HTTP + WebSocket, served by the local bridge)
            └─ Tourguide web app (?mode=workspace)
                 └─ Neuroglancer view · tables · plots · saved states
                    · annotations · narration notes · recording · action history
```

A browser tab can't host a server, but the live viewer/DB/plots live in the
browser. So a small Node **bridge** is the hub: the workspace tab connects to
it over WebSocket and registers a session; agents POST operations over HTTP
and subscribe to live events (action history, connection status) over
WebSocket. The bridge relays op-requests to the tab, which executes them
against the live viewer/DB and returns results. The browser is the source of
truth; the bridge keeps only lightweight session metadata.

| Layer | Location |
| --- | --- |
| Protocol contract | `web-app/src/workspace_api/protocol.ts` |
| Handlers (execute ops) | `web-app/src/workspace_api/handlers.ts` |
| Saved states / plots / recording | `web-app/src/workspace_api/session_state.ts` |
| Browser transports | `web-app/src/workspace_api/{transport_ws,transport_http}.ts` |
| Browser bridge wiring | `web-app/src/workspace_api/bridge.ts` |
| Bridge server (Node) | `web-app/bridge/server.mjs` |
| MCP adapter | `mcp/` (`tourguide-mcp`) |
| Python SDK | `python/` (`tourguide_client`) |

## Modes

- `?mode=workspace` — agent-driven: chat composer + AI provider chrome hidden;
  shows agent connection status + the **Agent Actions** history panel. Viewer,
  tables, plots, saved states, annotations, recording and local-folder support
  all remain.
- `?mode=chat` — legacy chat (unchanged).

The default stays `chat` during early phases (`DEFAULT_MODE` in
`web-app/src/mode.ts`); flip it to `workspace` once the MCP adapter is in daily
use, then deprecate chat.

## Run it

### Zero terminals (recommended)

Configure your MCP client with `TOURGUIDE_WEBAPP_DIR` (see `mcp/README.md`),
then just tell the agent to *"attach to Tourguide."* `launch_or_attach` starts
the bridge **and** the Vite dev server **and** opens the workspace tab — you
launch nothing by hand.

### One terminal

```bash
cd web-app && npm install
npm run workspace:preview   # builds, then serves the bridge + preview together
# then open http://localhost:5173/?mode=workspace
```

> **dev vs preview:** `npm run workspace` (Vite dev server) is faster and
> hot-reloads, but on some setups the Vite **dev** server mis-handles
> Neuroglancer's chunk-decoder workers and **image data renders black**
> (metadata still resolves). The **production build** (`workspace:preview`,
> or `npm run build && npm run preview`) renders correctly. Use dev for UI
> hacking, preview for actually looking at data. The MCP launcher defaults to
> the preview build for this reason (`TOURGUIDE_WEBAPP_MODE=dev` to override).

### Two terminals (explicit)

```bash
# 1. web app
cd web-app && npm install
npm run dev            # http://localhost:5173

# 2. bridge (separate terminal)
npm run bridge         # http://localhost:7723

# 3. open the workspace
#    http://localhost:5173/?mode=workspace
#    -> the tab connects to the bridge; status dot turns green

# 4a. drive it from the terminal
node bridge/test_client.mjs op get_session
node bridge/test_client.mjs op fly_to '{"position":[12000,8000,4000],"layer":"mito"}'

# 4b. or via the MCP adapter
cd ../mcp && uv sync && uv run tourguide-mcp

# 4c. or the Python SDK
cd ../python && uv pip install -e .
python -c "from tourguide_client import TourguideSession; print(TourguideSession.attach().get_session())"
```

For an MCP client (e.g. Claude Desktop), point it at `uv run tourguide-mcp`
with `cwd` = `mcp/` and `TOURGUIDE_WEBAPP_DIR` = `web-app/` so it can
auto-start the bridge. See `mcp/README.md`.

## Operations / tool surface

`launch_or_attach`, `get_session`, `load_descriptor`, `get_viewer_state`,
`set_viewer_state`, `get_selection`, `select_segments`, `fly_to`, `add_layer`,
`add_annotations`, `list_tables`, `get_table_schema`, `run_sql`, `ingest_table`,
`show_table`, `show_plot`, `save_session_state`, `restore_session_state`,
`list_saved_states`, `start_recording`, `stop_recording`, `add_narration_note`,
`export_session_summary`.

Prefer the semantic viewer ops (`select_segments`, `fly_to`, `add_layer`);
`set_viewer_state` is the escape hatch for raw Neuroglancer blobs.

### The agent computes; the workspace displays

The agent (Claude, a script, …) runs analysis **in its own environment** —
full Python, full RAM, the real data — and the workspace is the **sink** for
the results and the **source** of state. Tourguide is not a compute runtime.

- `get_session` hands back each layer's **data source URL** + voxel size, so
  the agent can read the zarr/n5 directly and compute (regionprops, meshing,
  …) itself.
- **`ingest_table(name, columns, rows)`** pushes a computed table in — it shows
  in the structured browser with click-to-fly (include `object_id` +
  `com_x_nm`/`com_y_nm`/`com_z_nm`).
- **`show_plot(png=…)`** displays a figure the agent already rendered — no
  in-browser matplotlib needed.

So *"measure mito"* is: `get_session` → read `mito_seg`'s source → compute
regionprops locally → `ingest_table(...)`. No Pyodide, no HuggingFace, no AI
key. (The legacy in-browser Pyodide / cloud-backend compute paths remain for
the deprecated chat mode and as fallbacks, but the agent flow doesn't use
them.)

## Status by phase

- **Phase 1 — Workspace mode**: done. `?mode=workspace` hides chat UI, keeps
  viewer/tables/plots/saved states; legacy chat at `?mode=chat`.
- **Phase 2 — Workspace API**: done. HTTP `/op` + WS event stream; Agent
  Actions panel; validated end-to-end with a simulated browser.
- **Phase 3 — Launch/attach + MCP**: done. `tourguide-mcp` (22 tools);
  deterministic most-recent-session selection; reconnect/relaunch errors.
- **Phase 4 — Tables and plots**: handlers implemented (`list_tables`,
  `get_table_schema`, `run_sql`, `show_table`, `show_plot` via matplotlib
  code or an AI `question`).
- **Phase 5 — Recording and export**: workspace-side recording status,
  narration notes, and `export_session_summary`. Full autonomous
  screenshot→vision→TTS narration is intentionally **future work** (see the
  narration decision in the plan).

## Verifying in a real browser

Automated transport tests (`web-app/bridge/test_client.mjs` and a simulated
browser) and `npm run build` cover the wire contract and bundling. To confirm
the live tab end to end:

1. `npm run bridge` and `npm run dev`.
2. Open `http://localhost:5173/?mode=workspace`; confirm the connection dot
   turns green and `node bridge/test_client.mjs sessions` lists the session.
3. Load a dataset, then `node bridge/test_client.mjs op get_session` — the
   summary should reflect the live layers; `op fly_to ...` should move the
   camera and append an entry to the Agent Actions panel.

## Legacy `neuroglancer-mcp`

The standalone `neuroglancer-mcp` wrapper is not the center of this
architecture. Preserve its useful helpers as library code; prefer the
Tourguide Workspace API as the primary agent-facing surface and reduce the
standalone wrapper over time.
