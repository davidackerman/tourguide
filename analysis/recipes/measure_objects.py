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


def pick_scale(fmt, bucket, group_key, anon, want_scale, max_voxels):
    """Return (scale_path, scale, translate, axes) for a tractable level.
    `scale`/`translate`/`axes` come from the COSEM multiscale transform and are
    aligned with each other (axes[i] names scale[i]); axes is e.g. ['z','y','x'].
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
        dims = (read_json(bucket, f"{group_key}/{path}/attributes.json", anon) or {}).get("dimensions")
        if not dims:
            continue
        voxels = int(np.prod(dims))
        if want_scale:
            if path == want_scale:
                return path, scale, translate, axes
        elif voxels <= max_voxels:
            return path, scale, translate, axes
        chosen = (path, scale, translate, axes)  # remember coarsest as fallback
    if want_scale:
        sys.exit(f"measure_objects: scale {want_scale!r} not found.")
    return chosen  # everything exceeded the budget → coarsest level


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure objects in a segmentation layer.")
    ap.add_argument("source", help="layer source URL, e.g. n5://s3://bucket/path/to/seg")
    ap.add_argument("--out", default="objects.csv", help="output CSV path")
    ap.add_argument("--scale", default=None, help="force a multiscale level, e.g. s2")
    ap.add_argument("--max-voxels", type=float, default=5e8,
                    help="auto-pick the finest level under this voxel budget")
    ap.add_argument("--no-anon", action="store_true", help="use AWS credentials instead of anonymous")
    args = ap.parse_args()
    anon = not args.no_anon

    fmt, bucket, group_key = parse_source(args.source)
    scale_path, scale, translate, axes = pick_scale(
        fmt, bucket, group_key, anon, args.scale, args.max_voxels)
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
    print(f"measuring {group_key} @ {scale_path}  shape={tuple(arr.shape)}  "
          f"array_axes={arr_axes}  voxel(x,y,z)={sc['x']}x{sc['y']}x{sc['z']} nm",
          file=sys.stderr)
    labels = np.asarray(arr.read().result())

    stats = cc3d.statistics(labels.astype(np.uint32, copy=False))
    counts = stats["voxel_counts"]          # [n_labels+1]
    centroids = stats["centroids"]          # [n_labels+1, 3] in array-axis order

    rows = []
    for obj_id in range(1, len(counts)):
        n = int(counts[obj_id])
        if n == 0:
            continue
        cen = centroids[obj_id]
        pos = {arr_axes[i]: cen[i] for i in range(len(arr_axes))}  # name -> voxel idx
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
