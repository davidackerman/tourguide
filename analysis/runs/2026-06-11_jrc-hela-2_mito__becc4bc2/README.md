# Eval: jrc_hela-2 — mitochondria morphometrics

- **Date:** 2026-06-11
- **Tourguide session:** `becc4bc2-151f-4663-b5df-77eed0882c10` (label `workspace-3`)
- **Dataset:** jrc_hela-2 (Janelia COSEM)
- **Image:** `zarr://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8/`
- **Segmentation:** `n5://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/labels/mito_seg`

## What was done

1. **Measure** — `measure` recipe over the `mito_seg` source.
   Ran at **scale s4** (64×64×83.84 nm/voxel, ~30M voxels). → `tables/mito.csv`
   (421 objects: object_id, volume_nm_3, voxel_count, com_x/y/z_nm).
2. **Meshify** — in-house zmesh, all 421 objects, **scale s4** → live layer
   `mito_mesh_mesh`, artifacts in `~/.tourguide/artifacts/mito_mesh-b4b22d/`
   (symlinked here as `meshes/`).
3. **Plot** — `scripts/plot_dist.py` → `plots/mito_dist.png`
   (volume hist, equiv-diameter hist, cumulative-volume, volume-vs-depth).
4. **Fly to + select** the largest object **176** in `mito_mesh_mesh`.

## Key results (n=421, scale s4)

| metric | value |
|---|---|
| volume median / mean / max (µm³) | 0.139 / 0.327 / 6.302 |
| equiv-diameter median / max (µm) | 0.64 / 2.29 |
| largest object | **176** — 6.30 µm³, COM (27464, 1399, 21612) nm |
| top 5 by volume (µm³) | 176 (6.30), 138 (5.25), 303 (3.90), 203 (3.64), 200 (3.33) |

## Reproduce

```bash
# from this run folder
uv run --project /Users/ackermand/Documents/programming/tourguide/analysis \
    python scripts/plot_dist.py
```

Measure + meshify were run via the Tourguide MCP `measure` / `meshify` tools on
the source URLs above (pass `scale=s2/s1/s0` for finer). `meshes/` and `data/`
are regenerable and git-ignored; `scripts/`, `tables/`, `plots/`, this README
are tracked.
