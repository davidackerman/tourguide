# Tourguide — agent guide

Tourguide is a 3D microscopy **visual workspace** (Neuroglancer + tables +
plots + annotations + saved states) that an external agent drives over the
**Workspace API**. The principle:

> **You (the agent) own reasoning and compute. Tourguide owns visual state.**

You read the data and run analysis in **your own environment**, then push the
results into the workspace. Tourguide is an artifact *sink*, not a compute
runtime — don't ask it to run your Python. It holds no API keys and contacts
no servers other than the data sources you point it at.

## Driving the workspace (the `tourguide` MCP server)

This repo ships a project-scoped MCP server in [`.mcp.json`](.mcp.json) — a
coding agent (Claude Code, Cursor) picks it up automatically. The tools:

- `launch_or_attach` — **call first.** Builds + starts the web app and the
  bridge if needed (with a per-launch auth token), opens the tab, attaches.
- `get_session` — what's loaded: layers with **data source URL**, **on-disk
  `localPath`** when known, **voxel size**; selection; camera; `ready` flag;
  tables; plots.
- `screenshot` — **your eyes.** Returns a PNG of the current view as an image.
  Use it after every visual change to check what the user is seeing.
- `wait_for_ready` — block until layers finish loading (after fly_to / load).
- `measure(source=…)` — **the predesigned measurement**: volume + centroid
  per object, run in the agent-side `analysis/` env and ingested as a table.
  `run_recipe` / `list_recipes` for other recipes (yours go in
  `~/.tourguide/recipes`). `meshify(source=…, segment_ids=[…])` builds 3D
  meshes for a label volume that ships none.
- `ingest_table(name, path="mito.csv")` — **push a table you computed** (CSV /
  JSON on disk; rows never pass through your tokens). Include `object_id` and
  `com_x_nm`/`com_y_nm`/`com_z_nm` for click-to-fly. Inline `columns`+`rows`
  for tiny tables only.
- `show_plot(png_path="hist.png")` — display a figure **you rendered**.
  Inline `png=` for tiny images only.
- `load_url(url)` — apply a Neuroglancer link in one step.
- `share_session` / `share_view` / `export_session` — hand the view (short
  Tourguide link with a read-only view token / plain Neuroglancer link) or the
  whole session (portable file) to a colleague. Never paste long URLs in chat.
- `fly_to_segment(layer, id)` — camera to an object via its table centroid.
  `fly_to(position)`, `select_segments`, `add_layer`, `add_annotations`
  (native point / line / bbox) — drive the viewer.
- `run_sql` (read-only) / `show_table` — query tables already in Tourguide.
- `wait_for_user_action` / `get_recent_events` — **react to the human.** Blocks
  until they select segments, move the camera, or load a dataset.
- `save_session_state` / `restore_session_state` — bookmark and return.
- `set_viewer_state` — escape hatch for raw Neuroglancer blobs.

## The measure-and-show loop (do it this way)

1. `get_session` → read the target segmentation layer's **source URL** (or
   `localPath`) and **voxel size**.
2. In your **own** Python, open that zarr/n5 (`tensorstore`/`zarr`), compute
   (e.g. `cc3d.statistics` or `skimage.measure.regionprops_table` for volume +
   centroid per object). Write a CSV. Install what you need; you have a shell.
3. `ingest_table("mito", path="mito.csv")` — or skip 2–3 with `measure(source=…)`.
4. `fly_to_segment("mito_seg", <largest id>)` → `screenshot()` to confirm →
   `show_plot_file("volumes.png")` for the distribution.

Result: the table appears in Tourguide with click-to-fly, the user sees the
object, you see what they see. No Pyodide, no cloud backend, no AI key.

## Interactive mode (the human points, you explain)

```
loop:
  ev = wait_for_user_action(since_seq=last, timeout_ms=60000)
  for selection_changed → run_sql on that object_id, screenshot, narrate
  last = ev.latestSeq
```

## Gotchas

- **Tabs are addressable.** Each workspace has a `?session=<id>`; with several
  open, `launch_or_attach` returns `{ambiguous, sessions}` — ask which, then
  pass `session_id` (or `new=True`). Prefer reusing the bound tab.
- **Big payloads go via files.** Rows and PNGs passed inline become your output
  tokens. Use `path=` / `png_path=` for anything non-trivial.
- **Use the pre-loaded env.** `uv run --project analysis python …` has
  tensorstore, zarr, cc3d, scikit-image, dask, pandas, matplotlib. Don't
  pip-install.
- **Security is handled.** The bridge is loopback-only with a per-launch
  token; `launch_or_attach` deals with it. Never paste tokens into chat.
- **Local-folder layers.** Their `source` is a browser-only `/local-data/` URL.
  Use `localPath` from `get_session`; if it's missing, ask the user for the
  directory (or have them add a `paths:` block to the descriptor YAML).
- **Preview, not dev, for viewing data.** `launch_or_attach` already defaults to
  the production build; the Vite dev server can render image chunks black.
- **Coordinates are world nm, x/y/z.** Everything the API takes or returns.

## Repo layout / commands

- `web-app/` — the workspace web app (Vite + Neuroglancer). Workspace API in
  `web-app/src/workspace_api/`; the bridge in `web-app/bridge/`.
- `mcp/` — the `tourguide-mcp` adapter (uv project; thin proxy to the bridge).
- `python/` — `tourguide_client` SDK (same ops, synchronous, pandas/matplotlib
  helpers).
- `analysis/` — the agent's pre-loaded compute env + built-in recipes
  (`measure_objects`, `contact_sites`), run by the MCP process.
- `hf-space/` — optional cloud backend for **legacy chat mode only**.
- `legacy/` — the old Python sidecar server and its docs. Not maintained.

```bash
cd web-app && npm install
npm run workspace:preview     # one terminal: build + bridge + preview  (or let launch_or_attach do it)
npm run test:smoke            # headless end-to-end check
cd ../mcp && uv sync          # the MCP server deps
```

More detail: [WORKSPACE.md](WORKSPACE.md). General "MCP for any app" guide:
[MCP_FOR_ANY_APP.md](MCP_FOR_ANY_APP.md).
