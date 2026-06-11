# Tourguide — agent guide

Tourguide is a 3D microscopy **visual workspace** (Neuroglancer + tables +
plots + saved states) that an external agent drives over the **Workspace API**.
The principle:

> **You (the agent) own reasoning and compute. Tourguide owns visual state.**

You read the data and run analysis in **your own environment**, then push the
results into the workspace. Tourguide is an artifact *sink*, not a compute
runtime — don't ask it to run your Python.

## Driving the workspace (the `tourguide` MCP server)

This repo ships a project-scoped MCP server in [`.mcp.json`](.mcp.json) — a
coding agent (Claude Code, Cursor) picks it up automatically. It exposes
high-level tools; the ones you'll use most:

- `launch_or_attach` — **call first.** Starts the web app + bridge if needed
  (production build, so image data renders) and attaches to the open tab.
- `get_session` — what's loaded: layers (with **data source URLs + voxel
  size**), current selection, camera, tables, plots.
- `ingest_table(name, columns, rows)` — **push a table you computed.** Shows in
  the structured browser with click-to-fly — include `object_id` and
  `com_x_nm`/`com_y_nm`/`com_z_nm`.
- `show_plot(png=…)` — display a figure **you rendered** (base64/data-url PNG).
- `select_segments(layer, segment_ids)`, `fly_to(position, layer?)`,
  `add_layer(layer)`, `add_annotations(...)` — drive the viewer.
- `run_sql` / `show_table` — query tables already in Tourguide.
- `set_viewer_state` — escape hatch for raw Neuroglancer blobs.

## The measure-and-show loop (do it this way)

Don't look for a Tourguide "measure" tool — there isn't one, by design. Instead:

1. `get_session` → read the target segmentation layer's **source URL + voxel
   size**.
2. In your **own** Python, open that zarr/n5 (`tensorstore`/`zarr`), compute
   (e.g. `cc3d.statistics` or `skimage.measure.regionprops` for volume +
   centroid per object). Install what you need; you have a real shell.
3. `ingest_table("mito", ["object_id","volume_nm_3","com_x_nm","com_y_nm","com_z_nm"], rows)`.
4. Optionally `fly_to` the largest object, `select_segments`, or
   `show_plot(png=…)` of the distribution.

Result: the table appears in Tourguide, click-to-fly works, no Pyodide / no
cloud backend / no AI key involved.

## Gotchas

- **One workspace tab.** The bridge routes to the most-recently-created session;
  a second open tab will silently steal your commands. Keep one.
- **Use a coding agent for compute.** Claude Desktop has no local Python — it
  can only call these tools. Claude Code / Cursor can run the analysis.
- **Preview, not dev, for viewing data.** `launch_or_attach` already defaults to
  the production build; the Vite dev server can render image chunks black.

## Repo layout / commands

- `web-app/` — the workspace web app (Vite + Neuroglancer). Workspace API in
  `web-app/src/workspace_api/`; the bridge in `web-app/bridge/`.
- `mcp/` — the `tourguide-mcp` adapter (uv project; thin proxy to the bridge).
- `python/` — `tourguide_client` SDK (same ops, synchronous).

```bash
cd web-app && npm install
npm run workspace:preview     # one terminal: build + bridge + preview  (or let launch_or_attach do it)
npm run test:smoke            # headless end-to-end check
cd ../mcp && uv sync          # the MCP server deps
```

More detail: [WORKSPACE.md](WORKSPACE.md). General "MCP for any app" guide:
[MCP_FOR_ANY_APP.md](MCP_FOR_ANY_APP.md).
