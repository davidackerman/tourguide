"""Tourguide meshify runner — generate Neuroglancer meshes with mesh-n-bone
and serve them through the bridge, so the agent can add them as a layer.

This runs in the **mesh-n-bone environment** (which has mesh-n-bone + tensorstore),
NOT the Tourguide analysis env. mesh-n-bone v0.5.0+ reads the volume directly from
S3, so NO local staging is needed — point it at the layer's source URL. This runner
handles n5 label volumes on s3:// (the scale pick reads N5 attributes.json).

It picks a coarse/fast multiscale level by default (like the `measure` recipe),
meshes (single-res by default), writes the precomputed mesh dir into the bridge's
served artifacts dir, and prints a JSON result the MCP `meshify` tool turns into
an add_layer call.

    <mnb-env-python> meshify_runner.py \
        --source n5://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/labels/mito_seg \
        --ids 176 --name mito --out result.json
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile, uuid
from pathlib import Path

import numpy as np
import tensorstore as ts

_DTYPE_BYTES = {"uint8": 1, "int8": 1, "uint16": 2, "int16": 2, "uint32": 4,
                "int32": 4, "uint64": 8, "int64": 8, "float32": 4, "float64": 8}


def _strip_prefix(url: str) -> tuple[str, str]:
    """Return (fmt, bare_url) — drop n5:///zarr*:// and report the format."""
    for p, fmt in (("n5://", "n5"), ("zarr3://", "zarr3"), ("zarr2://", "zarr"),
                   ("zarr://", "zarr"), ("precomputed://", "precomputed")):
        if url.startswith(p):
            return fmt, url[len(p):]
    return "n5", url  # COSEM default


def _s3_parts(bare: str) -> tuple[str, str]:
    """('bucket', 'key') from s3://bucket/key."""
    assert bare.startswith("s3://"), f"only s3:// supported here, got {bare!r}"
    netloc, _, key = bare[len("s3://"):].partition("/")
    return netloc, key.rstrip("/")


def _read_json_s3(bucket: str, key: str) -> dict | None:
    store = ts.KvStore.open({"driver": "s3", "bucket": bucket,
                             "path": key.rsplit("/", 1)[0] + "/",
                             "aws_credentials": {"type": "anonymous"}}).result()
    res = store.read(key.rsplit("/", 1)[1]).result()
    return json.loads(bytes(res.value)) if res.state == "value" else None


def pick_scale(bucket: str, group_key: str, want: str | None, max_bytes: float) -> str:
    """Finest multiscale level whose working set fits the budget (coarse=fast),
    mirroring measure_objects. Returns a scale key like 's4'."""
    attrs = _read_json_s3(bucket, f"{group_key}/attributes.json") or {}
    datasets = (attrs.get("multiscales") or [{}])[0].get("datasets")
    if not datasets:
        raise SystemExit("meshify: no multiscale metadata at the source.")
    coarsest = None
    for ds in datasets:
        path = ds["path"]
        meta = _read_json_s3(bucket, f"{group_key}/{path}/attributes.json") or {}
        dims = meta.get("dimensions")
        if not dims:
            continue
        bpv = _DTYPE_BYTES.get(str(meta.get("dataType", "uint16")), 2)
        est = int(np.prod(dims)) * bpv * 4.0
        if want:
            if path == want:
                return path
        elif est <= max_bytes:
            return path
        coarsest = path
    if want:
        raise SystemExit(f"meshify: scale {want!r} not found.")
    return coarsest


def main() -> None:
    ap = argparse.ArgumentParser(description="Mesh a segmentation with mesh-n-bone, serve via the bridge.")
    ap.add_argument("--source", required=True, help="layer source URL (n5://s3://… etc.)")
    ap.add_argument("--name", default="mesh", help="layer/base name")
    ap.add_argument("--ids", default=None, help="comma-separated object ids to mesh (default: all)")
    ap.add_argument("--scale", default=None, help="force a multiscale level, e.g. s2 (default: coarse/fast)")
    ap.add_argument("--roi", default=None, help="z0:z1,y0:y1,x0:x1 in voxels (mesh-n-bone roi)")
    ap.add_argument("--num-lods", type=int, default=1, help="LODs (1 = single-res, default)")
    ap.add_argument("--max-mem-gb", type=float, default=0.5, help="scale-pick memory budget")
    # Same resolution as the bridge and serve_layer.artifacts_dir(), so the
    # bridge serves what we write.
    ap.add_argument("--artifacts-dir", default=(os.environ.get("TG_ARTIFACTS_DIR") or "").strip()
                    or str(Path.home() / ".tourguide" / "artifacts"))
    ap.add_argument("--bridge", default="localhost:7723")
    ap.add_argument("--mnb", default="mesh-n-bone", help="mesh-n-bone CLI command")
    ap.add_argument("--out", default=None, help="write JSON result here (also printed)")
    args = ap.parse_args()

    fmt, bare = _strip_prefix(args.source)
    bucket, group_key = _s3_parts(bare)
    scale = pick_scale(bucket, group_key, args.scale, args.max_mem_gb * 1e9)
    input_path = f"s3://{bucket}/{group_key}/{scale}"   # mesh-n-bone reads s3 directly

    layer_id = f"{args.name.replace('/', '_')}_mesh-{uuid.uuid4().hex[:6]}"
    art = Path(args.artifacts_dir) / layer_id
    art.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tg_meshify_") as tmp:
        out_dir = os.path.join(tmp, "out")
        cfg_dir = os.path.join(tmp, "cfg")
        os.makedirs(cfg_dir, exist_ok=True)
        run_cfg = {
            "input_path": input_path,
            "output_directory": out_dir,
            "num_workers": 1,
            "target_reduction": 0.5,
            "n_smoothing_iter": 2,
            "do_simplification": True,
            "check_mesh_validity": False,
            "do_multires": True,
            "num_lods": args.num_lods,
            "multires_strategy": "decimate",
            "decimation_factor": 2,
            "target_faces_per_lod0_chunk": 25000,
            "delete_decimated_meshes": True,
            "sharded": False,   # unsharded = simplest to serve over the bridge
        }
        if args.ids:
            run_cfg["target_ids"] = [int(i) for i in args.ids.split(",")]
        if args.roi:
            run_cfg["roi"] = args.roi.split(",")
        import yaml
        Path(cfg_dir, "run-config.yaml").write_text(yaml.safe_dump(run_cfg))
        Path(cfg_dir, "dask-config.yaml").write_text(_DASK_CONFIG)

        proc = subprocess.run([args.mnb, "meshify", cfg_dir, "-n", "1"],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
            raise SystemExit(f"meshify failed (exit {proc.returncode})")

        multires = os.path.join(out_dir, "multires")
        if not os.path.isdir(multires):
            raise SystemExit(f"meshify produced no multires dir at {multires}")
        shutil.copytree(multires, art / "mesh")

    # mesh ids = numeric files in the mesh dir
    mesh_ids = sorted((p.name for p in (art / "mesh").iterdir() if p.name.isdigit()), key=int)
    result = {
        "layer_id": layer_id,
        "name": args.name,
        "scale": scale,
        "mesh_ids": mesh_ids,
        # The MCP tool builds a segmentation layer combining the ORIGINAL volume
        # source (coordinate space + voxels) with this bridge-served mesh source.
        "volume_source": args.source,
        "mesh_source": f"precomputed://http://{args.bridge}/artifacts/{layer_id}/mesh",
        "artifact_dir": str(art),
    }
    blob = json.dumps(result, indent=2)
    print(blob)
    if args.out:
        Path(args.out).write_text(blob)


_DASK_CONFIG = """\
jobqueue:
  local:
    ncpus: 1
    processes: 1
    cores: 1
    log-directory: job-logs
    name: dask-worker
distributed:
  scheduler:
    work-stealing: true
  worker:
    memory:
      target: 0.0
      spill: 0.0
      pause: 0.0
      terminate: 0.0
  admin:
    log-format: '[%(asctime)s] %(levelname)s %(message)s'
    tick:
      interval: 20ms
      limit: 3h
"""


if __name__ == "__main__":
    main()
