#!/usr/bin/env python
"""Predesigned recipe: measure every object in a segmentation layer.

Default measurement pipeline so the agent doesn't re-write analysis code each
task. Given a segmentation source URL (from get_session) it computes, per
object: voxel count, physical volume, centroid (nm), and bounding box — then
writes a CSV ready for `ingest_table(name, path=...)` (object_id +
com_x_nm/com_y_nm/com_z_nm give click-to-fly).

    uv run --project analysis python recipes/measure_objects.py \
        "n5://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/labels/mito_seg" \
        --out mito.csv

Scale: by default the finest multiscale level whose voxel count fits
--max-voxels (so it stays tractable); override with --scale sN. A downsampled
scale is accurate for volume + centroid distributions; surface area is
resolution-sensitive and is NOT computed here (use the published meshes for
that).

ROI: pass --roi X0,Y0,Z0,X1,Y1,Z1 (world nm) to restrict to a sub-cube. Only
that cube is read AND the scale budget is computed against the ROI's voxel
count — so a small region unlocks a FINER scale than the whole volume could
afford. Centroids are reported in absolute world coords. Objects crossing the
ROI boundary are measured only within the cube (clipped).

Supports COSEM-style N5 multiscale over public S3. For other sources/layouts,
ask for a custom recipe.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from urllib.parse import urlparse

import cc3d
import numpy as np
import tensorstore as ts


def parse_source(url: str) -> tuple[str, str, str]:
    """('n5'|'zarr', bucket, key) from e.g. n5://s3://bucket/key/to/group."""
    fmt = "n5"
    for prefix in ("n5://", "zarr://", "zarr2://", "zarr3://"):
        if url.startswith(prefix):
            fmt = "zarr" if prefix.startswith("zarr") else "n5"
            url = url[len(prefix):]
            break
    if url.startswith("precomputed://"):
        sys.exit("measure_objects: precomputed sources aren't supported; point at the "
                 "segmentation's n5/zarr volume instead.")
    p = urlparse(url)
    if p.scheme != "s3":
        sys.exit(f"measure_objects: only s3:// sources are supported here (got {p.scheme!r}); "
                 "ask for a custom recipe for http/gcs/local.")
    return fmt, p.netloc, p.path.lstrip("/")


def s3_kv(bucket: str, key: str, anon: bool) -> dict:
    kv = {"driver": "s3", "bucket": bucket, "path": key.rstrip("/") + "/"}
    if anon:
        kv["aws_credentials"] = {"type": "anonymous"}
    return kv


def read_json(bucket: str, key: str, anon: bool) -> dict | None:
    """Read one JSON object out of the kvstore (e.g. attributes.json)."""
    store = ts.KvStore.open(s3_kv(bucket, key.rsplit("/", 1)[0], anon)).result()
    res = store.read(key.rsplit("/", 1)[1]).result()
    if res.state != "value":
        return None
    return json.loads(bytes(res.value))


_DTYPE_BYTES = {"uint8": 1, "int8": 1, "uint16": 2, "int16": 2, "uint32": 4,
                "int32": 4, "uint64": 8, "int64": 8, "float32": 4, "float64": 8}


def pick_scale(fmt, bucket, group_key, anon, want_scale, max_bytes, work_mult=4.0, roi=None):
    """Pick the finest multiscale level whose in-memory working set fits a
    MEMORY budget (like the web app) — accounts for dtype bytes and the
    operation's working multiplier, not just voxel count. Returns
    (scale_path, scale, translate, axes, est_bytes).
    """
    attrs = read_json(bucket, f"{group_key}/attributes.json", anon) or {}
    datasets = (attrs.get("multiscales") or [{}])[0].get("datasets")
    if not datasets:
        sys.exit("measure_objects: no multiscale metadata at the source; ask for a custom recipe.")
    chosen = None
    for ds in datasets:
        path = ds["path"]
        tf = ds.get("transform", {})
        axes = tf.get("axes")              # e.g. ['z', 'y', 'x']
        scale = tf.get("scale")            # nm per voxel, aligned with axes
        translate = tf.get("translate", [0, 0, 0])
        meta = read_json(bucket, f"{group_key}/{path}/attributes.json", anon) or {}
        dims = meta.get("dimensions")
        if not dims:
            continue
        bpv = _DTYPE_BYTES.get(str(meta.get("dataType", "uint16")), 2)
        # With an ROI, budget against the sub-cube's voxel count at THIS scale
        # (not the full array) — a small ROI then unlocks a finer scale that
        # wouldn't fit for the whole volume. That's the point of the feature.
        if roi and axes and sorted(axes) == ["x", "y", "z"]:
            sc_d = {ax: scale[i] for i, ax in enumerate(axes)}
            tr_d = {ax: translate[i] for i, ax in enumerate(axes)}
            rs, re = roi_voxel_bounds(dims, list(reversed(axes)), sc_d, tr_d, roi)
            n_vox = int(np.prod([re[i] - rs[i] for i in range(len(dims))]))
        else:
            n_vox = int(np.prod(dims))
        est = n_vox * bpv * work_mult
        rec = (path, scale, translate, axes, est)
        if want_scale:
            if path == want_scale:
                return rec
        elif est <= max_bytes:
            return rec
        chosen = rec  # remember coarsest as fallback
    if want_scale:
        sys.exit(f"measure_objects: scale {want_scale!r} not found.")
    return chosen  # everything exceeded the budget → coarsest level


def parse_roi(spec: str | None) -> dict | None:
    """--roi 'X0,Y0,Z0,X1,Y1,Z1' (world nm, x/y/z, min/max in any order) ->
    {'x': (lo, hi), 'y': (lo, hi), 'z': (lo, hi)}."""
    if not spec:
        return None
    try:
        v = [float(x) for x in spec.split(",")]
    except ValueError:
        sys.exit("measure_objects: --roi must be 6 numbers X0,Y0,Z0,X1,Y1,Z1 (world nm).")
    if len(v) != 6:
        sys.exit("measure_objects: --roi needs exactly 6 values: X0,Y0,Z0,X1,Y1,Z1 (world nm).")
    x0, y0, z0, x1, y1, z1 = v
    return {"x": (min(x0, x1), max(x0, x1)),
            "y": (min(y0, y1), max(y0, y1)),
            "z": (min(z0, z1), max(z0, z1))}


def roi_voxel_bounds(shape, arr_axes, sc, tr, roi):
    """Voxel [start, stop) per ARRAY axis for an ROI given in world nm by axis
    name. `shape` is array order (parallel to arr_axes); sc/tr map axis name ->
    nm-per-voxel / translate. Clamped to the volume; unconstrained axes span the
    full extent. With roi=None this is the whole volume (start all-zero), so the
    caller's centroid offset and slicing stay no-ops."""
    starts, stops = [], []
    for i, ax in enumerate(arr_axes):
        n = int(shape[i])
        if roi and ax in roi:
            lo, hi = roi[ax]
            s = max(0, int(np.floor((lo - tr[ax]) / sc[ax])))
            e = min(n, int(np.ceil((hi - tr[ax]) / sc[ax])))
            if e <= s:                       # degenerate / fully clamped → ≥1 voxel
                s = min(max(s, 0), n - 1)
                e = s + 1
        else:
            s, e = 0, n
        starts.append(s)
        stops.append(e)
    return starts, stops


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure objects in a segmentation layer.")
    ap.add_argument("source", help="layer source URL, e.g. n5://s3://bucket/path/to/seg")
    ap.add_argument("--out", default="objects.csv", help="output CSV path")
    ap.add_argument("--scale", default=None, help="force a multiscale level, e.g. s2")
    ap.add_argument("--max-mem-gb", type=float, default=0.5,
                    help="memory budget for the in-memory working set; auto-picks "
                         "the finest scale that fits. Small default = fast/coarse "
                         "(like the web app); raise it (or --scale) for finer.")
    ap.add_argument("--roi", default=None,
                    help="restrict to a sub-cube in WORLD nm: X0,Y0,Z0,X1,Y1,Z1. "
                         "Only this cube loads AND the scale budget is computed "
                         "against it, so a small ROI unlocks a finer scale. "
                         "Centroids stay in absolute world coords.")
    ap.add_argument("--no-anon", action="store_true", help="use AWS credentials instead of anonymous")
    args = ap.parse_args()
    anon = not args.no_anon
    roi = parse_roi(args.roi)

    fmt, bucket, group_key = parse_source(args.source)
    scale_path, scale, translate, axes, est_bytes = pick_scale(
        fmt, bucket, group_key, anon, args.scale, args.max_mem_gb * 1e9, work_mult=4.0, roi=roi)
    if not axes or sorted(axes) != ["x", "y", "z"]:
        sys.exit(f"measure_objects: unexpected transform axes {axes!r}; ask for a custom recipe.")
    # Map physical axes BY NAME, never by position. The transform lists
    # axes/scale/translate together (e.g. axes=['z','y','x']); the array's axis
    # order is the reverse of that (n5/COSEM convention) — so centroid axis i
    # is the physical axis reversed(axes)[i]. (Verified: getting this wrong
    # swapped x and z and put centroids out of bounds.)
    sc = {ax: scale[i] for i, ax in enumerate(axes)}
    tr = {ax: translate[i] for i, ax in enumerate(axes)}
    arr_axes = list(reversed(axes))                  # array axis 0,1,2 -> name
    voxel_vol = sc["x"] * sc["y"] * sc["z"]

    spec = {"driver": fmt, "kvstore": s3_kv(bucket, f"{group_key}/{scale_path}", anon)}
    arr = ts.open(spec).result()
    # ROI -> voxel slice at this scale (full volume when roi is None, so starts
    # is [0,0,0] and the read/offset below are no-ops). `starts` is also added
    # back to centroids so reported positions stay in absolute world coords.
    starts, stops = roi_voxel_bounds(arr.shape, arr_axes, sc, tr, roi)
    sub_shape = [stops[i] - starts[i] for i in range(len(arr_axes))]
    mvox = int(np.prod(sub_shape)) / 1e6
    roi_note = "" if roi is None else " (ROI sub-cube)"
    # "chose scale" line — surfaced to the user so they know the resolution
    # (and can ask for finer). Picked by MEMORY budget, like the web app.
    print(f"chose scale {scale_path} ({sc['x']:g}x{sc['y']:g}x{sc['z']:g} nm/voxel, "
          f"{mvox:.0f}M voxels{roi_note}, ~{est_bytes/1e9:.1f}GB working set) — "
          f"fits the memory budget; pass scale=s2/s1/s0 for finer.",
          file=sys.stderr)
    if roi is None:
        labels = np.asarray(arr.read().result())
    else:
        labels = np.asarray(
            arr[starts[0]:stops[0], starts[1]:stops[1], starts[2]:stops[2]].read().result())

    stats = cc3d.statistics(labels.astype(np.uint32, copy=False))
    counts = stats["voxel_counts"]          # [n_labels+1]
    centroids = stats["centroids"]          # [n_labels+1, 3] in array-axis order

    rows = []
    for obj_id in range(1, len(counts)):
        n = int(counts[obj_id])
        if n == 0:
            continue
        cen = centroids[obj_id]
        # Add the ROI origin back (starts is [0,0,0] without an ROI) so the
        # local sub-cube centroid becomes an absolute-volume voxel index.
        pos = {arr_axes[i]: starts[i] + cen[i] for i in range(len(arr_axes))}  # name -> voxel idx
        rows.append([
            obj_id,
            round(n * voxel_vol, 3),                          # volume_nm_3
            n,                                                # voxel_count
            round(pos["x"] * sc["x"] + tr["x"], 1),           # com_x_nm
            round(pos["y"] * sc["y"] + tr["y"], 1),           # com_y_nm
            round(pos["z"] * sc["z"] + tr["z"], 1),           # com_z_nm
        ])
    rows.sort(key=lambda r: r[1], reverse=True)      # largest first

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object_id", "volume_nm_3", "voxel_count", "com_x_nm", "com_y_nm", "com_z_nm"])
        w.writerows(rows)
    print(f"wrote {len(rows)} objects to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
