/// <reference lib="webworker" />

// Browser-side segmentation analysis. Runs inside a dedicated Web Worker so
// the heavy work (Pyodide startup, zarr chunk fetches, regionprops) never
// blocks the main thread / viewer.
//
// Pipeline:
//   1. Open the zarr group at the user's layer source URL.
//   2. Walk OME-NGFF multiscales metadata to find available scales.
//   3. Read the selected scale's full array (TypedArray) via zarrita.
//   4. Hand that TypedArray + shape + spacing to Pyodide.
//   5. Run scipy.ndimage.label (if needed) + skimage.measure.regionprops_table.
//   6. Post results back as columnar rows matching the existing CSV schema
//      (object_id, volume, position_x/y/z, ...) so the rest of the app can
//      consume them with no changes.

import * as zarr from "zarrita";

// Pin to same Pyodide version as plot.ts so both features share a cache of
// wheels when the browser revisits.
const PYODIDE_VERSION = "0.27.0";
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

interface PyodideProxy {
  loadPackage(pkgs: string[]): Promise<void>;
  runPythonAsync(code: string): Promise<unknown>;
  globals: { set(name: string, value: unknown): void; get(name: string): unknown };
}

// Messages — in
export interface InspectRequest {
  kind: "inspect";
  url: string; // layer source URL, stripped of "zarr://" etc.
  defaultVoxelNm: [number, number, number]; // fallback if attrs lack scale
}
export interface AnalyzeRequest {
  kind: "analyze";
  url: string;
  scalePath: string; // e.g. "s4" or "4" or "" for the root
  axesOrder: string[]; // names like ["z","y","x"] — assumed
  voxelNm: [number, number, number]; // physical spacing in nm (x,y,z)
  offsetNm: [number, number, number]; // world-space origin of voxel (0,0,0), nm (x,y,z)
  // If the selected array has >3 spatial dims (e.g. includes t or c), we
  // reduce the non-spatial dims by taking index 0. voxelNm still applies to
  // x,y,z regardless.
  maxVoxels: number; // safety cap; caller should have warned user.
  connectivity: 1 | 2 | 3; // for scipy.ndimage.label
  alreadyLabeled: boolean; // skip label() if values are already unique ids
}
export interface CancelRequest { kind: "cancel" }
export type IncomingMsg = InspectRequest | AnalyzeRequest | CancelRequest;

// Messages — out
export interface ProgressMsg { kind: "progress"; message: string; phase?: string }
export interface InspectResultMsg {
  kind: "inspectResult";
  isMultiscale: boolean;
  axes: { name: string; type?: string }[];
  // One per available scale, coarsest last:
  scales: {
    path: string;
    shape: number[];
    voxelNm: [number, number, number];
    offsetNm: [number, number, number];
    downsample: [number, number, number]; // factor relative to finest
    approxBytes: number;
  }[];
}
export interface AnalyzeResultMsg {
  kind: "analyzeResult";
  columns: string[];
  rows: (number | string)[][]; // JSON-safe
  shape: number[]; // spatial shape analyzed
  voxelNm: [number, number, number];
  labelCount: number;
}
export interface ErrorMsg { kind: "error"; message: string; where?: string }
export type OutgoingMsg = ProgressMsg | InspectResultMsg | AnalyzeResultMsg | ErrorMsg;

declare const self: DedicatedWorkerGlobalScope;

let pyodide: PyodideProxy | null = null;
let cancelled = false;

function emit(msg: OutgoingMsg): void {
  self.postMessage(msg);
}

function progress(message: string, phase?: string): void {
  emit({ kind: "progress", message, phase });
}

async function ensurePyodide(): Promise<PyodideProxy> {
  if (pyodide) return pyodide;
  progress("Fetching Pyodide runtime (~6 MB, one-time) …", "pyodide-init");
  // Dynamic import from CDN. @vite-ignore stops the build from trying to
  // resolve the URL at bundle time.
  const mod: any = await import(/* @vite-ignore */ `${PYODIDE_BASE}pyodide.mjs`);
  pyodide = await mod.loadPyodide({ indexURL: PYODIDE_BASE });
  progress("Installing numpy + scipy + scikit-image …", "pyodide-pkgs");
  await pyodide!.loadPackage(["numpy", "scipy", "scikit-image"]);
  return pyodide!;
}

function bytesForDtype(dtype: string): number {
  // zarrita dtype strings: "<u1", "<u2", "<u4", "<u8", "<i4", "<f4" etc.
  const m = /([0-9]+)$/.exec(dtype);
  return m ? parseInt(m[1], 10) : 1;
}

interface MultiscaleInfo {
  isMultiscale: boolean;
  axes: { name: string; type?: string }[];
  datasets: {
    path: string;
    arr: zarr.Array<zarr.DataType, any>;
    scale: [number, number, number]; // xyz nm per voxel (effective, incl. group transforms)
    offset: [number, number, number]; // xyz nm, world-space origin of voxel (0,0,0)
  }[];
}

// Extract scale + translation from an OME-NGFF coordinateTransformations
// array. Returns an axis→value map for each, in the axes' order.
function extractTransforms(
  cts: any[] | undefined,
  axes: { name: string; type?: string }[],
): { scale: Record<string, number>; translation: Record<string, number> } {
  const scale: Record<string, number> = {};
  const translation: Record<string, number> = {};
  if (!cts) return { scale, translation };
  for (const t of cts) {
    if (t?.type === "scale" && Array.isArray(t.scale)) {
      axes.forEach((a, i) => {
        if (t.scale[i] !== undefined) scale[a.name] = t.scale[i];
      });
    } else if (t?.type === "translation" && Array.isArray(t.translation)) {
      axes.forEach((a, i) => {
        if (t.translation[i] !== undefined) translation[a.name] = t.translation[i];
      });
    }
  }
  return { scale, translation };
}

function pickXYZ(
  map: Record<string, number>,
  fallback: [number, number, number],
): [number, number, number] {
  return [
    map.x !== undefined ? map.x : fallback[0],
    map.y !== undefined ? map.y : fallback[1],
    map.z !== undefined ? map.z : fallback[2],
  ];
}

async function walkMultiscales(
  url: string,
  defaultVoxelNm: [number, number, number],
): Promise<MultiscaleInfo> {
  const store = new zarr.FetchStore(url);
  const node = await zarr.open(store);
  if (node.kind === "array") {
    return {
      isMultiscale: false,
      axes: inferAxesFromShape(node.shape),
      datasets: [{ path: "", arr: node, scale: defaultVoxelNm, offset: [0, 0, 0] }],
    };
  }
  // group — look for multiscales
  const attrs = node.attrs as any;
  const ms = attrs?.multiscales as any[] | undefined;
  if (!ms || ms.length === 0) {
    throw new Error("Group has no OME-Zarr multiscales metadata. Open the data directly at the array path, or pick a source with multiscales.");
  }
  const topMs = ms[0];
  const axes: { name: string; type?: string }[] = (topMs.axes ?? []).map((a: any) =>
    typeof a === "string" ? { name: a } : { name: a.name, type: a.type },
  );
  // Some older OME-NGFFs don't include axes; fall back to inferred.
  const effectiveAxes = axes.length ? axes : inferAxesFromShape((await zarr.open(node.resolve(topMs.datasets[0].path), { kind: "array" })).shape);
  // Group-level transforms apply *after* per-dataset ones per OME-NGFF spec.
  const groupT = extractTransforms(topMs.coordinateTransformations, effectiveAxes);
  const datasets: MultiscaleInfo["datasets"] = [];
  for (const ds of topMs.datasets) {
    const arr = await zarr.open(node.resolve(ds.path), { kind: "array" });
    const dsT = extractTransforms(ds.coordinateTransformations, effectiveAxes);
    // Combine: world = groupScale * (dsScale * index + dsTranslation) + groupTranslation
    //        = (groupScale * dsScale) * index + (groupScale * dsTranslation + groupTranslation)
    const combinedScale: Record<string, number> = {};
    const combinedTrans: Record<string, number> = {};
    for (const a of effectiveAxes) {
      const n = a.name;
      const ds_s = dsT.scale[n] ?? 1;
      const ds_t = dsT.translation[n] ?? 0;
      const g_s = groupT.scale[n] ?? 1;
      const g_t = groupT.translation[n] ?? 0;
      combinedScale[n] = ds_s * g_s;
      combinedTrans[n] = g_s * ds_t + g_t;
    }
    const scale = pickXYZ(combinedScale, defaultVoxelNm);
    const offset = pickXYZ(combinedTrans, [0, 0, 0]);
    // If neither dataset nor group had any scale info, scale will fall back
    // to defaultVoxelNm. Offset will be 0.
    datasets.push({ path: ds.path, arr, scale, offset });
  }
  return { isMultiscale: true, axes: effectiveAxes, datasets };
}

function inferAxesFromShape(shape: number[]): { name: string; type?: string }[] {
  // Best-effort: last three dims are z,y,x; any leading dims are t/c.
  const out: { name: string; type?: string }[] = [];
  const spatial = ["z", "y", "x"];
  const extra = ["t", "c"];
  const nExtra = shape.length - 3;
  for (let i = 0; i < nExtra; i++) {
    out.push({ name: extra[i] ?? `d${i}`, type: extra[i] === "t" ? "time" : "channel" });
  }
  for (let i = 0; i < Math.min(3, shape.length); i++) {
    const axis = spatial[spatial.length - Math.min(3, shape.length) + i];
    out.push({ name: axis, type: "space" });
  }
  return out;
}

async function handleInspect(msg: InspectRequest): Promise<void> {
  const info = await walkMultiscales(msg.url, msg.defaultVoxelNm);
  console.log(
    "[analysis] inspect",
    info.datasets.map((d) => ({
      path: d.path,
      shape: d.arr.shape,
      scale_xyz_nm: d.scale,
      offset_xyz_nm: d.offset,
    })),
  );
  // Compute downsample factors and approx bytes for each dataset.
  const finest = info.datasets[0];
  const scales = info.datasets.map((ds) => {
    const ds0Shape = finest.arr.shape;
    const dsShape = ds.arr.shape;
    const downsample: [number, number, number] = [
      ds0Shape.length >= 1 ? (ds0Shape[ds0Shape.length - 1] / dsShape[dsShape.length - 1]) : 1,
      ds0Shape.length >= 2 ? (ds0Shape[ds0Shape.length - 2] / dsShape[dsShape.length - 2]) : 1,
      ds0Shape.length >= 3 ? (ds0Shape[ds0Shape.length - 3] / dsShape[dsShape.length - 3]) : 1,
    ];
    const nvox = dsShape.reduce((a, b) => a * b, 1);
    const approxBytes = nvox * bytesForDtype(ds.arr.dtype as string);
    return {
      path: ds.path,
      shape: [...dsShape],
      voxelNm: ds.scale,
      offsetNm: ds.offset,
      downsample,
      approxBytes,
    };
  });
  emit({ kind: "inspectResult", isMultiscale: info.isMultiscale, axes: info.axes, scales });
}

async function handleAnalyze(msg: AnalyzeRequest): Promise<void> {
  progress("Opening zarr array …", "open");
  console.log("[analysis] analyze", { url: msg.url, scalePath: msg.scalePath, axesOrder: msg.axesOrder });
  const store = new zarr.FetchStore(msg.url);
  const root = await zarr.open(store);
  const arr = msg.scalePath
    ? await zarr.open(root.kind === "group" ? root.resolve(msg.scalePath) : root, { kind: "array" })
    : (root.kind === "array" ? root : await zarr.open(root.resolve(""), { kind: "array" }));

  const shape = arr.shape;
  console.log("[analysis] opened array", {
    shape,
    dtype: arr.dtype,
    spacingZYX: undefined, // filled in below
    offsetZYX: undefined,
    axesOrder: msg.axesOrder,
    voxelNmXYZ: msg.voxelNm,
    offsetNmXYZ: msg.offsetNm,
  });
  // Build a selection that reduces non-spatial dims to 0. msg.axesOrder tells
  // us which axes are x/y/z; everything else gets pinned to index 0.
  const axisNames = msg.axesOrder;
  const selection: (null | number)[] = axisNames.map((a) =>
    a === "x" || a === "y" || a === "z" ? null : 0,
  );
  // If the array has fewer or more dims than axisNames (mismatch), fall back
  // to: last 3 dims are spatial.
  let spatialSelection: (null | number)[] = selection;
  if (selection.length !== shape.length) {
    spatialSelection = shape.map((_, i) => (i >= shape.length - 3 ? null : 0));
  }
  const spatialShape = shape.filter((_, i) => spatialSelection[i] === null);
  const nVox = spatialShape.reduce((a, b) => a * b, 1);
  if (nVox > msg.maxVoxels) {
    throw new Error(`Selected scale has ${nVox.toLocaleString()} voxels (cap ${msg.maxVoxels.toLocaleString()}). Pick a coarser scale.`);
  }

  progress(`Reading ${nVox.toLocaleString()} voxels …`, "read");
  console.log("[analysis] reading", { spatialSelection, spatialShape, nVox });
  const result = await zarr.get(arr, spatialSelection as any);
  if (cancelled) return;
  console.log("[analysis] read done", {
    resultShape: (result as any).shape,
    dataLen: (result as any).data?.length,
    dtype: (result as any).data?.constructor?.name,
  });

  // Figure out which array axes remained after we reduced non-spatial dims.
  // skimage operates on the array's native axis order, so `spacing` and
  // `centroid-i` indexing must follow that same order. We compute per-axis
  // scale/offset here by matching each surviving axis name to its entry in
  // voxelNm/offsetNm (which are given as [x, y, z]).
  const survivingAxes = msg.axesOrder.filter((_a, i) => spatialSelection[i] === null);
  // When selection fell back to "last 3 dims", synthesize axis names.
  const axesForPython =
    survivingAxes.length === spatialShape.length
      ? survivingAxes
      : ["z", "y", "x"].slice(-spatialShape.length); // best-effort
  const axisScaleMap: Record<string, number> = {
    x: msg.voxelNm[0],
    y: msg.voxelNm[1],
    z: msg.voxelNm[2],
  };
  const axisOffsetMap: Record<string, number> = {
    x: msg.offsetNm[0],
    y: msg.offsetNm[1],
    z: msg.offsetNm[2],
  };
  const spacing = axesForPython.map((a) => axisScaleMap[a] ?? 1);
  const offsets = axesForPython.map((a) => axisOffsetMap[a] ?? 0);
  console.log("[analysis] spacing/offsets by array axis", {
    axesForPython,
    spacing,
    offsets,
    voxelNmXYZ: msg.voxelNm,
    offsetNmXYZ: msg.offsetNm,
  });

  progress("Loading Python runtime …", "python");
  const py = await ensurePyodide();
  if (cancelled) return;

  // Hand data to Pyodide. The TypedArray is copied into WASM memory.
  py.globals.set("_tg_data", result.data);
  py.globals.set("_tg_shape", result.shape);
  py.globals.set("_tg_spacing", spacing);
  py.globals.set("_tg_offset", offsets);
  py.globals.set("_tg_axes", axesForPython);
  py.globals.set("_tg_already_labeled", msg.alreadyLabeled);
  py.globals.set("_tg_connectivity", msg.connectivity);

  progress("Running regionprops …", "compute");
  console.log("[analysis] handing off to python …", { spacing, offsets, axesForPython, spatialShape });
  // Python kernel: reshape, label if needed, regionprops_table, serialize.
  const code = `
import numpy as np
import json

shape = tuple(int(x) for x in list(_tg_shape))
spacing = tuple(float(x) for x in list(_tg_spacing))         # nm per voxel, in array-axis order
offsets = tuple(float(x) for x in list(_tg_offset))          # nm origin, in array-axis order
axes = [str(a) for a in list(_tg_axes)]                      # e.g. ["z","y","x"] or ["x","y","z"]
already_labeled = bool(_tg_already_labeled)
connectivity = int(_tg_connectivity)

arr = np.asarray(_tg_data).reshape(shape)
# skimage can't take uint64 labels directly — regionprops wants intp-ish
if arr.dtype == np.uint64:
    # If any labels exceed int64 max, we'd lose info, but that's astronomically rare.
    arr = arr.astype(np.int64, copy=False)

if already_labeled:
    labels = arr.astype(np.int64, copy=False)
    n_labels = int(labels.max()) if labels.size else 0
else:
    from scipy import ndimage as ndi
    structure = ndi.generate_binary_structure(arr.ndim, connectivity)
    labels, n_labels = ndi.label(arr > 0, structure=structure)

from skimage.measure import regionprops_table

# props requested: label (object_id), area (voxel count), bbox, centroid, equivalent_diameter
props = ("label", "area", "bbox", "centroid", "equivalent_diameter_area")
tbl = regionprops_table(labels, spacing=spacing, properties=props)

# Build output columns matching existing CSV schema.
voxel_volume_nm3 = float(spacing[0] * spacing[1] * spacing[2])
label_col = tbl["label"].tolist()
# skimage with spacing: "area" is volume in physical units (nm^3) already when spacing is set.
volume_col = tbl["area"].tolist()
# centroid-i / bbox-i correspond to array axis i. The 'axes' list tells us
# which world-space axis that is (x, y, or z). Add the per-axis offset so
# positions are in world-space nm (matching Neuroglancer's frame).
def axis_column(prefix, world_axis):
    idx = axes.index(world_axis) if world_axis in axes else None
    if idx is None:
        # Axis not in the data (e.g., 2D slice). Return zeros.
        return [0.0] * len(label_col)
    off = offsets[idx]
    return [float(v) + off for v in tbl[f"{prefix}-{idx}"].tolist()]

cx = axis_column("centroid", "x")
cy = axis_column("centroid", "y")
cz = axis_column("centroid", "z")

# skimage's bbox stays in array indices even when spacing is set. Convert
# to physical nm ourselves: bbox_nm = array_idx * spacing + offset.
def bbox_pair(world_axis):
    if world_axis not in axes:
        return [0.0] * len(label_col), [0.0] * len(label_col)
    idx = axes.index(world_axis)
    s = spacing[idx]
    off = offsets[idx]
    lo = [float(v) * s + off for v in tbl[f"bbox-{idx}"].tolist()]
    hi = [float(v) * s + off for v in tbl[f"bbox-{idx + len(axes)}"].tolist()]
    return lo, hi

bx0, bx1 = bbox_pair("x")
by0, by1 = bbox_pair("y")
bz0, bz1 = bbox_pair("z")
diam = tbl["equivalent_diameter_area"].tolist()

# Also keep raw voxel count for sanity.
n_vox = [int(v / voxel_volume_nm3 + 0.5) for v in volume_col]

columns = [
    "object_id",
    "volume",
    "position_x",
    "position_y",
    "position_z",
    "bbox_min_x",
    "bbox_min_y",
    "bbox_min_z",
    "bbox_max_x",
    "bbox_max_y",
    "bbox_max_z",
    "equivalent_diameter",
    "n_voxels",
]
rows = []
for i in range(len(label_col)):
    rows.append([
        int(label_col[i]),
        float(volume_col[i]),
        float(cx[i]), float(cy[i]), float(cz[i]),
        float(bx0[i]), float(by0[i]), float(bz0[i]),
        float(bx1[i]), float(by1[i]), float(bz1[i]),
        float(diam[i]),
        int(n_vox[i]),
    ])

_tg_columns = columns
_tg_rows = rows
_tg_nlabels = int(n_labels)
`;
  try {
    await py.runPythonAsync(code);
  } catch (err) {
    // Surface Python tracebacks clearly instead of letting them drop as
    // "RuntimeError: null" on the main thread.
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[analysis] Python traceback:", err);
    throw new Error(`Python analysis failed: ${msg}`);
  }
  if (cancelled) return;

  const columns = (py.globals.get("_tg_columns") as any).toJs({ create_proxies: false }) as string[];
  const rows = (py.globals.get("_tg_rows") as any).toJs({ create_proxies: false }) as (number | string)[][];
  const labelCount = py.globals.get("_tg_nlabels") as number;
  console.log("[analysis] python done", { nRows: rows.length, labelCount, columns });

  emit({
    kind: "analyzeResult",
    columns,
    rows,
    shape: spatialShape,
    voxelNm: msg.voxelNm,
    labelCount,
  });
}

self.addEventListener("message", (ev: MessageEvent<IncomingMsg>) => {
  const msg = ev.data;
  if (msg.kind === "cancel") {
    cancelled = true;
    return;
  }
  cancelled = false;
  const handler = msg.kind === "inspect" ? handleInspect(msg) : handleAnalyze(msg);
  handler.catch((err: unknown) => {
    const message = err instanceof Error ? err.message : String(err);
    emit({ kind: "error", message, where: msg.kind });
  });
});
