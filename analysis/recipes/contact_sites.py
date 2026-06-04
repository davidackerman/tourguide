#!/usr/bin/env python
"""Predesigned recipe: membrane contact sites between two segmentations.

Finds where two organelle segmentations come within a contact distance of each
other (e.g. ER--mitochondria contacts) and reports each contact site: volume,
centroid (nm), and the two object IDs it bridges. Writes a CSV ready for
ingest_table (contact_id + com_x/y/z_nm give click-to-fly).

    uv run --project analysis python recipes/contact_sites.py \
        "n5://s3://.../labels/er_seg" --other "n5://s3://.../labels/mito_seg" \
        --max-dist-nm 100 --out contacts.csv

Contact = voxels within --max-dist-nm of BOTH segmentations' surfaces; connected
components of that zone are the sites. Resolution-limited by the chosen scale
(--scale / --max-voxels) — coarse scales merge nearby contacts. COSEM-style N5
over public S3.
"""

from __future__ import annotations

import argparse
import csv
import sys
from urllib.parse import urlparse

import cc3d
import numpy as np
import tensorstore as ts
from scipy import ndimage


def parse_source(url: str):
    fmt = "n5"
    for p in ("n5://", "zarr://", "zarr2://", "zarr3://"):
        if url.startswith(p):
            fmt = "zarr" if p.startswith("zarr") else "n5"
            url = url[len(p):]
            break
    u = urlparse(url)
    if u.scheme != "s3":
        sys.exit(f"contact_sites: only s3:// sources supported (got {u.scheme!r})")
    return fmt, u.netloc, u.path.lstrip("/")


def s3_kv(bucket, key, anon):
    kv = {"driver": "s3", "bucket": bucket, "path": key.rstrip("/") + "/"}
    if anon:
        kv["aws_credentials"] = {"type": "anonymous"}
    return kv


def read_json(bucket, key, anon):
    store = ts.KvStore.open(s3_kv(bucket, key.rsplit("/", 1)[0], anon)).result()
    res = store.read(key.rsplit("/", 1)[1]).result()
    return None if res.state != "value" else __import__("json").loads(bytes(res.value))


def pick_scale(bucket, group_key, anon, want_scale, max_voxels):
    attrs = read_json(bucket, f"{group_key}/attributes.json", anon) or {}
    datasets = (attrs.get("multiscales") or [{}])[0].get("datasets") or []
    chosen = None
    for ds in datasets:
        tf = ds.get("transform", {})
        dims = (read_json(bucket, f"{group_key}/{ds['path']}/attributes.json", anon) or {}).get("dimensions")
        if not dims:
            continue
        rec = (ds["path"], tf.get("scale"), tf.get("translate", [0, 0, 0]), tf.get("axes"))
        if want_scale:
            if ds["path"] == want_scale:
                return rec
        elif int(np.prod(dims)) <= max_voxels:
            return rec
        chosen = rec
    if want_scale:
        sys.exit(f"contact_sites: scale {want_scale!r} not found")
    return chosen


def open_mask(fmt, bucket, group_key, scale_path, anon):
    spec = {"driver": fmt, "kvstore": s3_kv(bucket, f"{group_key}/{scale_path}", anon)}
    return np.asarray(ts.open(spec).result())


def main():
    ap = argparse.ArgumentParser(description="Contact sites between two segmentations.")
    ap.add_argument("source", help="first segmentation source URL (e.g. er_seg)")
    ap.add_argument("--other", required=True, help="second segmentation source URL (e.g. mito_seg)")
    ap.add_argument("--out", default="contacts.csv")
    ap.add_argument("--scale", default=None, help="force a multiscale level, e.g. s3")
    ap.add_argument("--max-voxels", type=float, default=4e7, help="auto-pick finest level under this")
    ap.add_argument("--max-dist-nm", type=float, default=100.0, help="contact distance threshold (nm)")
    ap.add_argument("--min-voxels", type=int, default=2, help="drop contact sites smaller than this")
    ap.add_argument("--no-anon", action="store_true")
    args = ap.parse_args()
    anon = not args.no_anon

    _, b1, k1 = parse_source(args.source)
    fmt2, b2, k2 = parse_source(args.other)
    fmt1 = "n5" if args.source.startswith("n5") else "zarr"
    sp1, scale, translate, axes = pick_scale(b1, k1, anon, args.scale, args.max_voxels)
    sp2, *_ = pick_scale(b2, k2, anon, args.scale, args.max_voxels)
    if not axes or sorted(axes) != ["x", "y", "z"]:
        sys.exit(f"contact_sites: unexpected axes {axes!r}")

    a = open_mask(fmt1, b1, k1, sp1, anon)
    b = open_mask(fmt2, b2, k2, sp2, anon)
    if a.shape != b.shape:
        sys.exit(f"contact_sites: scale mismatch {a.shape} vs {b.shape}; pass --scale to align")

    sc = {ax: scale[i] for i, ax in enumerate(axes)}          # nm/voxel by name
    tr = {ax: translate[i] for i, ax in enumerate(axes)}
    arr_axes = list(reversed(axes))                            # array axis -> name (x,y,z)
    sampling = [sc[ax] for ax in arr_axes]                     # nm spacing per array axis
    voxel_vol = sc["x"] * sc["y"] * sc["z"]
    print(f"contact: shapes {a.shape} axes={arr_axes} voxel(x,y,z)={sc['x']}x{sc['y']}x{sc['z']} nm "
          f"thresh={args.max_dist_nm}nm", file=sys.stderr)

    mask_a = a > 0
    mask_b = b > 0
    # Distance (in nm) to nearest voxel of each segmentation, plus the index of
    # that nearest voxel (to recover which object is involved).
    dist_a, idx_a = ndimage.distance_transform_edt(~mask_a, sampling=sampling, return_indices=True)
    dist_b, idx_b = ndimage.distance_transform_edt(~mask_b, sampling=sampling, return_indices=True)
    contact = (dist_a <= args.max_dist_nm) & (dist_b <= args.max_dist_nm)
    n_contact = int(contact.sum())
    if n_contact == 0:
        sys.exit("contact_sites: no contacts within threshold (try a larger --max-dist-nm or finer --scale)")

    labels = cc3d.connected_components(contact.astype(np.uint32), connectivity=26)
    stats = cc3d.statistics(labels)
    counts = stats["voxel_counts"]
    centroids = stats["centroids"]

    # Nearest object id of each segmentation at every voxel.
    nearest_a = a[tuple(idx_a)]
    nearest_b = b[tuple(idx_b)]

    rows = []
    for cid in range(1, len(counts)):
        n = int(counts[cid])
        if n < args.min_voxels:
            continue
        m = labels == cid
        # the object pair this contact bridges = most common nearest ids in the zone
        aid = int(np.bincount(nearest_a[m].ravel()).argmax())
        bid = int(np.bincount(nearest_b[m].ravel()).argmax())
        cen = centroids[cid]
        pos = {arr_axes[i]: cen[i] for i in range(3)}
        rows.append([
            cid,
            round(n * voxel_vol, 1),                       # volume_nm_3
            n,                                             # voxel_count
            round(pos["x"] * sc["x"] + tr["x"], 1),        # com_x_nm
            round(pos["y"] * sc["y"] + tr["y"], 1),        # com_y_nm
            round(pos["z"] * sc["z"] + tr["z"], 1),        # com_z_nm
            aid,                                           # object_id_a (e.g. er)
            bid,                                           # object_id_b (e.g. mito)
        ])
    rows.sort(key=lambda r: r[1], reverse=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contact_id", "volume_nm_3", "voxel_count",
                    "com_x_nm", "com_y_nm", "com_z_nm", "object_id_a", "object_id_b"])
        w.writerows(rows)
    print(f"wrote {len(rows)} contact sites to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
