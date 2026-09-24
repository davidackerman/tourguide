# Tourguide Workspace mode

Tourguide is a persistent **visual workspace** controlled by external agents
(Claude Code, Cursor, Claude Desktop, a local script, …) rather than a chat
app with its own LLM plumbing.

```
Agent owns conversation/reasoning/compute.
Tourguide owns visual state.
```

The durable artifact is the **Workspace API**. MCP is the first adapter, not
the product.

## Architecture

```
External agent
  └─ MCP adapter / Python SDK / HTTP client        (bearer token)
       └─ Workspace API  (HTTP + WebSocket, served by the local bridge on 127.0.0.1)
            └─ Tourguide web app (?mode=workspace — the default)
                 └─ Neuroglancer view · tables · plots · annotations · saved states
                    · narration notes · recording · action history · user events
```

A browser tab can't host a server, but the live viewer/DB/plots live in the
browser. So a small Node **bridge** is the hub: the workspace tab connects to
it over WebSocket and registers a session; agents POST operations over HTTP
and read live events over WebSocket (`/agent`) or long-poll (`GET /events`).
The bridge relays op-requests to the tab, which executes them against the
live viewer/DB and returns results. The browser is the source of truth; the
bridge keeps session metadata, disk-backed saved states and per-session
snapshots (`~/.tourguide/`), agent-computed artifacts, and a short event ring
buffer.

| Layer | Location |
| --- | --- |
| Protocol contract | `web-app/src/workspace_api/protocol.ts` |
| Handlers (execute ops) | `web-app/src/workspace_api/handlers.ts` |
| Saved states / plots / recording | `web-app/src/workspace_api/session_state.ts` |
| Browser transport + user events + persistence | `web-app/src/workspace_api/{transport_ws,bridge}.ts` |
| Bridge server (Node) + disk store | `web-app/bridge/{server,state_store}.mjs` |
| MCP adapter | `mcp/` (`tourguide-mcp`) |
| Python SDK | `python/` (`tourguide_client`) |
| Analysis env + recipes (run by the MCP process) | `analysis/` |

## Security model

The bridge can read every table and push arbitrary layers, so it is locked
down by default:

- **Loopback only.** Binds `127.0.0.1` (`TG_BRIDGE_HOST=0.0.0.0` to widen).
  Vite likewise serves on `localhost` (`TG_HOST`). Set both to share on the LAN.
- **Bearer token.** Required on `/op`, `/sessions`, `/events`, `POST
  /share-state`, `/viewer-fly` and both WS paths. `TG_BRIDGE_TOKEN` or
  generated per start; written 0600 as JSON `{token, viewToken}` to
  `<tmpdir>/tourguide-bridge-<port>.token`. The tab gets it via
  `?bridgeToken=` (the launcher and `npm run workspace` supply it); the MCP
  adapter and SDK read the file. `TG_BRIDGE_NO_AUTH=1` disables (single-user
  machines only). Tokens are never printed to stdout or logs.
- **View token.** A second, weaker token that only lets a tab register as a
  read-only viewer (`?view=1`). `share_session` links carry this one, never
  the full token. Read-only is enforced server-side: a viewer never persists,
  never answers ops, and is never an op target.
- **Origin allowlist.** Requests with an `Origin` are accepted from loopback,
  this machine's own addresses (a LAN-served page), and
  `TG_BRIDGE_ALLOWED_ORIGINS` (default: the hosted Tourguide page). CORS
  echoes allowed origins only.
- **Unauthenticated, read-only by design:** `GET /health`, `GET /artifacts/*`
  (Neuroglancer fetches chunks itself and cannot attach a token) and `GET
  /share-state/<id>` (unguessable ids). They expose only agent-computed
  derived data, and only off-machine when you opt into a LAN bind.
- **Read-only queries.** `run_sql` / `show_table` accept a single SELECT-shaped
  statement; `ingest_table` is the write path.
- **No third-party calls in workspace mode.** No LLM, no Pyodide CDN, no
  analysis backend, no share upload. Legacy chat mode (`?mode=chat`) still has
  those, all opt-in via Settings.

## Modes

- `?mode=workspace` (**default**) — agent-driven: chat composer, AI provider
  chrome, in-browser compute buttons and the upload-based Share button are
  hidden; shows agent connection status + the **Agent Actions** history panel,
  docked plots, share links, and Download / Load workspace.
- `?mode=chat` — legacy chat (in-app LLM with a user-supplied key). Deprecated;
  logs a warning.

## Run it

### Zero terminals (recommended)

Open the repo in Claude Code / Cursor (the project `.mcp.json` registers the
server) and say *"attach to Tourguide."* `launch_or_attach` installs deps if
needed, builds the app, starts the bridge with a fresh token, opens the
workspace tab, and attaches. Launch logs go to `<tmpdir>/tourguide-logs/`
(`TOURGUIDE_LOG_DIR`), mode 0700.

### One terminal

```bash
cd web-app && npm install
npm run workspace:preview   # builds, then serves bridge + preview; prints the tokened URL
```

> **dev vs preview:** `npm run workspace` (Vite dev server) hot-reloads but on
> some setups renders Neuroglancer image chunks black. The production build
> (`workspace:preview`) renders correctly. The MCP launcher defaults to
> preview and evicts a squatting dev server.

### Drive it by hand

```bash
node bridge/test_client.mjs sessions
node bridge/test_client.mjs op get_session
node bridge/test_client.mjs op fly_to '{"position":[12000,8000,4000],"layer":"mito_seg"}'
node bridge/test_client.mjs screenshot view.png
node bridge/test_client.mjs watch          # live events
```

The CLI reads the token from the bridge's token file.

## Sessions, sharing and hosting

The bridge runs on **your machine**; the page can be served from anywhere and
still be driven by your local agent. The page connects *out* to the bridge.

- **Hosted page → local bridge.** Serve the static build anywhere (the live
  build is on Cloudflare Pages) and open it with
  `?bridge=localhost:7723&bridgeToken=<token>`; the page connects to your local
  bridge (localhost is mixed-content-exempt, the bridge answers Chrome's
  Private-Network-Access preflight, and the hosted origin is on the allowlist).
- **Unique, addressable sessions.** Each workspace carries a `?session=<id>`
  (minted on first open). Reopening the link returns to the same session; the
  bridge keys routing on the id, so concurrent tabs don't collide.
  `launch_or_attach` refuses to guess among several open tabs.
- **Reopen restores everything.** The page auto-saves a snapshot (viewer state
  + table data + plot images) keyed by session id (`~/.tourguide/session-states`).
- **Read-only share links.** `?session=<id>&view=1&bridgeToken=<viewToken>`
  registers under a fresh id and never writes back. `share_session` builds this
  link (short `?state=<id>` form). Viewers need only a browser.
- **Portable files.** `export_session` / the panel's Download button write the
  snapshot to a file; `restore_session_state(path=…)` / Load applies it into
  the recipient's own copy.
- **Agent-computed layers.** The agent writes a result (a zarr, meshes from
  `meshify`) to `~/.tourguide/artifacts` (`TG_ARTIFACTS_DIR`); the bridge serves
  it at `/artifacts/…` and `add_layer` renders it. Layer URLs are rewritten to
  the bridge host so a LAN peer fetches from the host machine when LAN-bound.

## Operations

| Op | Purpose |
| --- | --- |
| `launch_or_attach`, `get_session`, `load_descriptor`, `wait_for_ready` | session; `get_session` includes each layer's `source`, `localPath`, and a `viewer.ready` flag |
| `screenshot` | PNG of the current view (base64) — the agent's eyes |
| `fly_to`, `fly_to_segment`, `select_segments`, `get_selection` | camera + selection; `fly_to_segment` looks the centroid up in the layer's table |
| `add_layer`, `add_annotations` | layers; native point / line / bbox annotations, append or replace |
| `list_tables`, `get_table_schema`, `run_sql`, `show_table` | read tables |
| `ingest_table` | push agent-computed rows (merges by `object_id` on re-ingest) |
| `show_plot` | display an agent-rendered PNG |
| `show_share_link` | show a clickable link in the panel (used by the share tools) |
| `save_session_state`, `restore_session_state`, `list_saved_states` | disk-backed bookmarks (`~/.tourguide/saved-states`); restore reloads the dataset if it differs |
| `start_recording`, `stop_recording`, `add_narration_note`, `export_session_summary` | narration / export |
| `get_viewer_state`, `set_viewer_state` | raw Neuroglancer escape hatch |

The MCP adapter adds, on top: file-path variants (`ingest_table(path=)`,
`show_plot(png_path=)`, `set_viewer_state(path=)`, `add_annotations(path=)`),
`load_url`, the compute tools (`measure`, `run_recipe`, `list_recipes`,
`meshify` — run in the agent-side `analysis/` env), the share tools
(`share_session`, `share_view`, `export_session`), and the event tools
(`get_recent_events`, `wait_for_user_action`). The SDK adds
`ingest_dataframe` and `show_figure`.

## Events

The tab publishes, and the bridge buffers (last 500, with `seq`):

- `selection_changed` — the person changed visible segments
- `position_changed` — the camera moved (debounced, rounded nm)
- `dataset_changed` — a different descriptor was loaded
- `action` — an op ran (with a summary); `connection_status`; `heartbeat`

Agent-driven changes are suppressed for ~1.5 s so an agent isn't told about
its own `fly_to`. `GET /events?since=<seq>&wait=<ms>&types=a,b` long-polls.

## Descriptors and local data

A descriptor YAML lists layers with `source` URLs. For data on disk, a
`folders:` block makes the loader prompt for a directory pick (served to
Neuroglancer via a service worker). Add a matching `paths:` block so agents
get a real path:

```yaml
folders:
  data: "pick the jrc_hela-2 folder"
paths:
  data: /nrs/cellmap/data/jrc_hela-2
layers:
  - name: mito_seg
    type: segmentation
    source: n5://data/jrc_hela-2.n5/labels/mito_seg
```

`get_session` then reports `localPath: /nrs/cellmap/data/jrc_hela-2/jrc_hela-2.n5/labels/mito_seg`.

## Verifying

`npm run test:smoke` starts a bridge (with a token) and Vite, drives headless
Chromium to the workspace, and asserts: connection, auth rejections (401 /
403), `get_session`, `ingest_table` → `run_sql`, read-only enforcement,
`show_plot` with a PNG, `wait_for_ready`, `/events`, and the Agent Actions
panel.
