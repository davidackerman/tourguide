"""Reference: measure a segmentation in YOUR environment, show it in Tourguide.

This is the canonical coding-agent loop — the agent (or you) computes locally
and pushes the result in. No Tourguide-side compute.

Prereqs: a workspace tab open with a segmentation layer loaded, the bridge
running (or let the MCP `launch_or_attach` start everything). Plus, in *this*
environment:  pip install tourguide-client tensorstore numpy scikit-image
(zmesh/cc3d are faster but Linux-only; skimage works everywhere.)

    python measure_segmentation.py mito_seg
"""

from __future__ import annotations

import sys

from tourguide_client import TourguideSession


def main(layer_name: str) -> None:
    s = TourguideSession.attach()                      # finds the running tab

    # 1. Ask Tourguide what's loaded — including each layer's data source URL.
    session = s.get_session()
    layers = {l["name"]: l for l in session["viewer"]["layers"]}
    if layer_name not in layers:
        sys.exit(f"layer {layer_name!r} not found; have: {list(layers)}")
    layer = layers[layer_name]
    source = layer.get("source")
    voxel = session.get("descriptor", {}).get("voxelSizeNm")  # [x, y, z] nm
    if not source or not voxel:
        sys.exit("layer has no resolvable source/voxel size to read")
    print(f"reading {layer_name} from {source}  (voxel {voxel} nm)")

    # 2. Read + compute IN THIS ENVIRONMENT. Swap in tensorstore/zarr/cc3d as
    #    appropriate for the source (n5/zarr/precomputed). Sketch with skimage:
    import numpy as np  # noqa: F401
    from skimage.measure import regionprops_table

    arr = load_array(source)                            # <- you implement per source type
    props = regionprops_table(arr, properties=("label", "area", "centroid"))
    vx, vy, vz = voxel
    rows = []
    for i in range(len(props["label"])):
        # regionprops centroid is (z, y, x) for a zyx array; adjust to your axes.
        cz, cy, cx = props["centroid-0"][i], props["centroid-1"][i], props["centroid-2"][i]
        rows.append([
            int(props["label"][i]),
            float(props["area"][i] * vx * vy * vz),     # volume_nm_3
            float(cx * vx), float(cy * vy), float(cz * vz),
        ])

    # 3. Push the result into Tourguide — appears in the browser, click-to-fly.
    cols = ["object_id", "volume_nm_3", "com_x_nm", "com_y_nm", "com_z_nm"]
    res = s.ingest_table(layer_name, cols, rows)
    print(f"ingested {res['rowCount']} rows as table '{res['tableId']}'")

    # 4. Fly to the biggest one.
    rows.sort(key=lambda r: r[1], reverse=True)
    if rows:
        biggest = rows[0]
        s.fly_to([biggest[2], biggest[3], biggest[4]], layer=layer_name, segment_id=str(biggest[0]))
        print(f"flew to object {biggest[0]} ({biggest[1]/1e9:.2f} um^3)")


def load_array(source: str):
    """Open the layer's source as a numpy array. Implement per source scheme —
    e.g. tensorstore for n5/zarr/precomputed. Sketch raises so you fill it in."""
    raise NotImplementedError(
        "open the zarr/n5 at `source` (e.g. with tensorstore) and return a numpy array"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mito_seg")
