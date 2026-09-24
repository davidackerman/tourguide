# Tourguide

A 3D microscopy **visual workspace** — Neuroglancer plus tables, plots, saved
states and annotations — that a coding agent drives over a local
**Workspace API**. Ships as an MCP server you drop into Claude Code, Cursor,
or Claude Desktop.

> **You (the agent) own reasoning and compute. Tourguide owns visual state.**

The agent reads the data and runs analysis in its own environment, then pushes
results into the workspace: tables with click-to-fly, figures, meshes,
annotations, camera moves. Tourguide runs no LLM and holds no API keys. By
default nothing leaves your machine except reads of the data sources you point
it at, and every request to the bridge needs a per-launch token.

```
Claude Code / Cursor / Claude Desktop
  └─ tourguide-mcp   (stdio; this repo's .mcp.json)
       └─ local bridge  (127.0.0.1:7723, bearer token, origin allowlist)
            └─ Tourguide tab  (?mode=workspace — the default)
                 └─ Neuroglancer · tables · plots · meshes · annotations · saved states
```

## Install

```bash
git clone <this repo> && cd tourguide
cd web-app && npm install && cd ..
cd mcp && uv sync && cd ..
uv sync --project analysis          # the agent's pre-loaded measurement env
```

Open the repo in Claude Code or Cursor: [`.mcp.json`](.mcp.json) registers the
`tourguide` MCP server automatically. For Claude Desktop see
[`mcp/README.md`](mcp/README.md).

Then just say **"attach to Tourguide"**. `launch_or_attach` builds the web app,
starts the bridge with a fresh token, opens the workspace tab, and attaches.
No terminals.

## The loop

```
get_session                         → layer source URLs (zarr/n5/precomputed) + voxel size
measure(source=…)                   → volume + centroid per object, ingested as a table
   (or your own Python → ingest_table(name, path="mito.csv"))
fly_to_segment("mito_seg", 4312)    → camera + selection via the table centroid
meshify(source=…, segment_ids=[…])  → 3D meshes for a label volume that has none
show_plot(png_path="hist.png")      → a figure you rendered
screenshot()                        → you SEE the view and decide what's next
wait_for_user_action()              → the human clicks something; you respond
share_session() / export_session()  → hand the view or the whole session to a colleague
```

Full guide for agents: [`CLAUDE.md`](CLAUDE.md). Architecture, security model
and the wire contract: [`WORKSPACE.md`](WORKSPACE.md). Python SDK for scripts
and notebooks: [`python/`](python/). Recipes and the analysis environment:
[`analysis/`](analysis/).

## Tools

| Group | Tools |
| --- | --- |
| Session | `launch_or_attach` (multi-tab aware), `get_session`, `load_descriptor`, `load_url`, `wait_for_ready` |
| Seeing | `screenshot` (returns an image) |
| Viewer | `fly_to`, `fly_to_segment`, `select_segments`, `get_selection`, `add_layer`, `add_annotations` (point / line / bbox), `get_viewer_state`, `set_viewer_state` |
| Compute (runs in the agent's env) | `measure`, `run_recipe`, `list_recipes`, `meshify` |
| Tables | `ingest_table` (inline or `path=` to CSV / JSON), `run_sql` (read-only), `show_table`, `list_tables`, `get_table_schema` |
| Plots | `show_plot` (`png_path=` or inline PNG) |
| Events | `get_recent_events`, `wait_for_user_action` |
| Sharing | `share_session` (short Tourguide link, view token), `share_view` (Neuroglancer link), `export_session` (portable file) |
| State | `save_session_state`, `restore_session_state`, `list_saved_states`, `start_recording`, `stop_recording`, `add_narration_note`, `export_session_summary` |

## Privacy and security

- **No keys.** Workspace mode has no LLM integration. Your agent brings its own.
- **Loopback only by default.** The bridge binds `127.0.0.1` and the web app
  `localhost`. Set `TG_BRIDGE_HOST=0.0.0.0` and `TG_HOST=0.0.0.0` to share
  sessions on the LAN.
- **Token auth.** Every bridge request that reads workspace state or drives
  the viewer needs a bearer token generated per launch (written 0600 to
  `tourguide-bridge-<port>.token` in the OS temp dir, never printed). Share
  links carry a weaker **view token** that only permits a read-only viewer.
- **Origin allowlist.** Browser requests are accepted only from loopback, this
  machine's own addresses, or the hosted Tourguide page. A random web page in
  the same browser can't drive your workspace.
- **Read-only SQL.** `run_sql` refuses writes; `ingest_table` is the only way in.
- **Read-only viewers are server-enforced.** A `?view=1` tab can look but
  never persists or answers ops.
- **No beacons.** The page contacts no analysis backend or share server unless
  you configure one in Settings (legacy chat mode only).

## Sharing and hosting

The bridge runs on **your machine**; the page can be served from anywhere
(the live static build is on Cloudflare Pages) and still be driven by your
local agent via `?bridge=localhost:7723&bridgeToken=…`. Each workspace has an
addressable `?session=<id>`; reopening it restores layers, tables and plots
from `~/.tourguide/session-states`. `share_session` produces a short link with
the view token; `export_session` writes a portable file a colleague can load
into their own copy.

## Legacy

Two earlier designs are preserved but not maintained:

- **Chat mode** (`?mode=chat`, deprecated): the browser itself calls an LLM
  with a key you paste, runs Python via Pyodide or an optional Hugging Face
  Space, and can upload share links. The analysis backend is now empty by
  default and uploads require a confirm. See
  [`web-app/README.md`](web-app/README.md) and [`hf-space/`](hf-space/).
- **Sidecar server** (Python, narration, movies): moved to
  [`legacy/`](legacy/) with its docs in [`legacy/docs/`](legacy/docs/).

## Develop

```bash
cd web-app
npm run workspace:preview   # build + bridge + preview; prints the tokened URL
npm run test:smoke          # headless end-to-end (Playwright)
npx tsc --noEmit            # type-check
```

## License

GPL-3.0. See [LICENSE](LICENSE).
