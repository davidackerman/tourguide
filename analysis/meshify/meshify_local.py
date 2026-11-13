"""In-house Tourguide meshify — generate Neuroglancer meshes with zmesh in the
pre-loaded analysis env (NO external mesh-n-bone install) and serve them via the
bridge, so the agent can add them as a layer.

Reuses measure_objects' source/axis handling (so meshes land exactly where
`measure` reports objects) and serve_layer.write_mesh_layer (zmesh → precomputed
seg + legacy single-res mesh into ~/.tourguide/artifacts, served by the bridge).
Reads the source DIRECTLY from S3 — no staging. Picks a coarse/fast scale by
default (like `measure`); pass --scale for finer or --ids to mesh specific
objects (bbox-restricted, so a finer scale is affordable).

    uv run --project analysis python analysis/meshify/meshify_local.py \
        --source n5://s3://.../labels/mito_seg --ids 176 --name mito --out result.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

_ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ANALYSIS / "recipes"))
sys.path.insert(0, str(_ANALYSIS))
import tensorstore as ts
import measure_objects as mo
from meshify.serve_layer import write_mesh_layer


def main() -> None:
    ap = argparse.ArgumentParser(description="Mesh a segmentation with zmesh, serve via the bridge.")
    ap.add_argument("--source", required=True, help="layer source URL (n5://s3://… / zarr://s3://…)")
    ap.add_argument("--name", default="mesh", help="layer/base name")
    ap.add_argument("--ids", default=None, help="comma-separated object ids (default: all)")
    ap.add_argument("--scale", default=None, help="force a multiscale level, e.g. s2 (default: coarse/fast)")
    ap.add_argument("--max-mem-gb", type=float, default=0.5, help="scale-pick memory budget")
    ap.add_argument("--bridge", default="localhost:7723")
    ap.add_argument("--out", default=None, help="write JSON result here (also printed)")
    args = ap.parse_args()

    fmt, bucket, group = mo.parse_source(args.source)
    want = [int(i) for i in args.ids.split(",")] if args.ids else None

    def open_scale(path):
        return ts.open({"driver": fmt, "kvstore": mo.s3_kv(bucket, f"{group}/{path}", True)}).result()

    if want:
        # Two-pass: locate the objects at a coarse scale, then mesh just their
        # bbox at the FINEST scale that fits the budget for that small region —
        # the same ROI-unlocks-finer trick `measure` uses. (A whole-volume pick
        # would settle for a coarse scale like s4; one object's bbox affords s1/s2.)
        loc_path, loc_scale, translate, axes, _ = mo.pick_scale(fmt, bucket, group, True, None, 0.5e9)
        if not axes or sorted(axes) != ["x", "y", "z"]:
            sys.exit(f"meshify_local: unexpected transform axes {axes!r}; needs a custom recipe.")
        arr_axes = list(reversed(axes))
        sc_loc = {ax: loc_scale[i] for i, ax in enumerate(axes)}
        tr = {ax: translate[i] for i, ax in enumerate(axes)}
        vol_loc = np.asarray(open_scale(loc_path).read().result())
        keep = np.isin(vol_loc, want)
        if not keep.any():
            sys.exit(f"meshify_local: none of ids {want} present at {loc_path}.")
        idx = np.array(np.where(keep))
        lo_l, hi_l = idx.min(axis=1), idx.max(axis=1) + 1
        pad = 2
        roi = {arr_axes[i]: ((lo_l[i] - pad) * sc_loc[arr_axes[i]] + tr[arr_axes[i]],
                             (hi_l[i] + pad) * sc_loc[arr_axes[i]] + tr[arr_axes[i]])
               for i in range(3)}
        scale_path, scale, translate, axes, _ = mo.pick_scale(
            fmt, bucket, group, True, args.scale, args.max_mem_gb * 1e9, roi=roi)
        sc = {ax: scale[i] for i, ax in enumerate(axes)}
        tr = {ax: translate[i] for i, ax in enumerate(axes)}
        arr = open_scale(scale_path)
        starts, stops = mo.roi_voxel_bounds(arr.shape, arr_axes, sc, tr, roi)
        sub = np.asarray(arr[tuple(slice(starts[i], stops[i]) for i in range(3))].read().result())
        sub = np.where(np.isin(sub, want), sub, 0)
        start = {arr_axes[i]: starts[i] for i in range(3)}
    else:
        # All objects: budget against the whole volume (coarse/fast, like measure).
        scale_path, scale, translate, axes, _ = mo.pick_scale(
            fmt, bucket, group, True, args.scale, args.max_mem_gb * 1e9)
        if not axes or sorted(axes) != ["x", "y", "z"]:
            sys.exit(f"meshify_local: unexpected transform axes {axes!r}; needs a custom recipe.")
        arr_axes = list(reversed(axes))
        sc = {ax: scale[i] for i, ax in enumerate(axes)}
        tr = {ax: translate[i] for i, ax in enumerate(axes)}
        sub = np.asarray(open_scale(scale_path).read().result())
        start = {arr_axes[i]: 0 for i in range(3)}

    perm = [arr_axes.index("z"), arr_axes.index("y"), arr_axes.index("x")]
    labels_zyx = np.ascontiguousarray(np.transpose(sub, perm))
    spacing_zyx = [sc["z"], sc["y"], sc["x"]]
    offsets_zyx = [start["z"] * sc["z"] + tr["z"],
                   start["y"] * sc["y"] + tr["y"],
                   start["x"] * sc["x"] + tr["x"]]

    res = write_mesh_layer(labels_zyx, name=args.name, spacing_zyx=spacing_zyx,
                           offsets_zyx=offsets_zyx, ids=want, bridge=args.bridge)
    res["scale"] = scale_path
    res["tool"] = "zmesh (in-house)"
    blob = json.dumps(res, indent=2)
    print(blob)
    if args.out:
        Path(args.out).write_text(blob)


if __name__ == "__main__":
    main()
