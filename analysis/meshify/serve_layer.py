"""Local parity for the HF Space's mesh / derived-layer pipeline.

Writes a Neuroglancer **precomputed** segmentation directory (volume chunk +
legacy single-resolution meshes + inline segment_properties) into the bridge's
served artifacts dir (``~/.tourguide/artifacts`` by default, served at
``/artifacts/...`` with CORS). Returns the ``precomputed://`` source URL so the
caller can ``add_layer`` it over MCP — no HF Space, no Pyodide.

This is the local port of ``_encode_new_mesh_layer_and_write`` in
``hf-space/app.py`` (same on-disk layout, same axis conventions), so anything the
backend "Generate 3D meshes" toggle did now runs from the local analysis env.
"""
from __future__ import annotations
import json, os, uuid
from pathlib import Path
import numpy as np


def artifacts_dir() -> Path:
    d = (os.environ.get("TG_ARTIFACTS_DIR") or "").strip()
    base = Path(d) if d else (Path.home() / ".tourguide" / "artifacts")
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_mesh_layer(
    labels: np.ndarray,
    name: str,
    spacing_zyx,            # voxel size in nm, array-axis order (z, y, x)
    offsets_zyx=(0.0, 0.0, 0.0),  # world origin of labels[0,0,0] in nm (z, y, x)
    ids=None,              # which non-zero ids to mesh; default = all
    reduction_factor: int = 10,
    max_error: float = 8.0,
    bridge: str = "localhost:7723",
    write_volume: bool = True,  # also write the seg slab so it shows in 2D cross-sections
):
    """Mesh ``labels`` with zmesh and write a precomputed seg+mesh dir.

    Returns a dict with the ``precomputed://`` source URL, the written ids, and
    the on-disk path. Mirrors the HF writer's binary formats exactly.
    """
    import zmesh

    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(f"labels must be 3D (Z,Y,X), got {labels.shape}")
    if labels.dtype == np.bool_:
        labels = labels.astype(np.uint8)
    if str(labels.dtype) not in ("uint8", "uint16", "uint32", "uint64"):
        labels = labels.astype(np.uint32)
    labels = np.ascontiguousarray(labels)

    spacing_zyx = [float(v) for v in list(spacing_zyx)[:3]]
    offsets_zyx = [float(v) for v in list(offsets_zyx)[:3]]
    if ids is None:
        ids = [int(i) for i in np.unique(labels) if i != 0]
    else:
        ids = [int(i) for i in ids]

    layer_id = f"{name.replace('/', '_')}-{uuid.uuid4().hex[:6]}"
    out_dir = artifacts_dir() / layer_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- meshes (legacy single-resolution) ---
    mesher = zmesh.Mesher(spacing_zyx)
    mesher.mesh(labels)
    meshes_dir = out_dir / "mesh"
    meshes_dir.mkdir(exist_ok=True)
    (meshes_dir / "info").write_text(json.dumps({"@type": "neuroglancer_legacy_mesh"}))
    offset_xyz = np.array([offsets_zyx[2], offsets_zyx[1], offsets_zyx[0]], dtype=np.float64)
    written = []
    for seg_id in ids:
        try:
            mesh = mesher.get(seg_id, normals=False,
                              reduction_factor=reduction_factor, max_error=max_error)
        except Exception:
            continue
        if mesh.vertices.shape[0] == 0:
            continue
        # zmesh vertices: (N,3) in (z,y,x) nm relative to array origin → x,y,z + world offset
        verts_xyz = (mesh.vertices.astype(np.float64)[:, [2, 1, 0]] + offset_xyz).astype(np.float32)
        faces = mesh.faces.astype(np.uint32, copy=False)
        frag = (np.array([verts_xyz.shape[0]], dtype="<u4").tobytes()
                + np.ascontiguousarray(verts_xyz).tobytes()
                + np.ascontiguousarray(faces).tobytes())
        (meshes_dir / f"{seg_id}:0:0").write_bytes(frag)
        (meshes_dir / f"{seg_id}:0").write_text(json.dumps({"fragments": [f"{seg_id}:0:0"]}))
        written.append(seg_id)

    # --- segment_properties (so the Segments panel is pre-populated) ---
    sp_dir = out_dir / "segment_properties"
    sp_dir.mkdir(exist_ok=True)
    (sp_dir / "info").write_text(json.dumps({
        "@type": "neuroglancer_segment_properties",
        "inline": {"ids": [str(i) for i in written],
                   "properties": [{"id": "label", "type": "label",
                                   "values": [f"{name}_{i}" for i in written]}]},
    }))

    # --- precomputed info (+ optional volume chunk) ---
    sz, sy, sx = labels.shape
    size_xyz = [int(sx), int(sy), int(sz)]
    res_xyz = [spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]]
    vox_off_xyz = [int(round(offsets_zyx[2] / spacing_zyx[2])) if spacing_zyx[2] else 0,
                   int(round(offsets_zyx[1] / spacing_zyx[1])) if spacing_zyx[1] else 0,
                   int(round(offsets_zyx[0] / spacing_zyx[0])) if spacing_zyx[0] else 0]
    info = {
        "type": "segmentation",
        "data_type": str(labels.dtype),
        "num_channels": 1,
        "scales": [{"key": "s0", "size": size_xyz, "resolution": res_xyz,
                    "voxel_offset": vox_off_xyz, "chunk_sizes": [size_xyz],
                    "encoding": "raw"}],
        "mesh": "mesh",
        "segment_properties": "segment_properties",
    }
    (out_dir / "info").write_text(json.dumps(info))
    s0 = out_dir / "s0"
    s0.mkdir(exist_ok=True)
    if write_volume:
        # Single whole-volume chunk. NG raw is x-fastest; numpy (Z,Y,X) C-order
        # bytes already match. Chunk filename uses global voxel ranges.
        ox, oy, oz = vox_off_xyz
        fname = f"{ox}-{ox+sx}_{oy}-{oy+sy}_{oz}-{oz+sz}"
        (s0 / fname).write_bytes(np.ascontiguousarray(labels).tobytes())

    serve = f"http://{bridge}/artifacts/{layer_id}"
    return {
        "source": f"precomputed://{serve}",
        "serveUrl": serve + "/",
        "layer_id": layer_id,
        "out_dir": str(out_dir),
        "mesh_ids": [str(i) for i in written],
        "shape_zyx": [int(sz), int(sy), int(sx)],
        "has_volume": bool(write_volume),
    }
