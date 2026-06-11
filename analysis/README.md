# tourguide-analysis — the agent's pre-loaded compute env

A ready-to-use Python environment so the agent never has to `pip install`
mid-task. It holds the standard stack for measuring microscopy segmentations
and handing results back to a Tourguide workspace:

- **Array IO over the cloud:** `tensorstore`, `zarr`, `numcodecs`, `s3fs`,
  `fsspec`, `aiohttp` — open the zarr/n5/precomputed source a layer points at,
  straight from S3/GCS/http, without downloading it whole.
- **Measurement:** `connected-components-3d` (cc3d), `scikit-image`, `scipy`,
  `numpy`.
- **Out-of-core:** `dask` for volumes too big to load at once.
- **Results:** `pandas`, `matplotlib`, `trimesh`.

## Use it

```bash
uv sync --project analysis          # one-time (CI / fresh clone); the MCP also does this
uv run --project analysis python my_measure.py
```

The `tourguide` MCP server points the agent here automatically and tells it to
run compute in this env rather than installing libraries each time.

## The measure-and-show pattern

1. `get_session` → the target layer's source URL + voxel size.
2. Open it here (`tensorstore`/`zarr`), compute per object — `cc3d.statistics`
   gives volume + centroid + bbox in one pass. For a volume too big to load
   whole, read it in blocks and accumulate. A downsampled scale is fine for a
   quick volume/centroid distribution; surface area is resolution-sensitive
   (use native res or the published meshes via `trimesh`).
3. Write `results.csv`, then `ingest_table(name, path="results.csv")` — pass
   the **path**, not the rows, so the data never round-trips through the
   agent's tokens.
