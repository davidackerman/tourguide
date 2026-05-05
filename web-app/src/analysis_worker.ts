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
export interface CustomRequest {
  kind: "custom";
  // Each layer gets read at the given scale and bound to a Python variable.
  layers: {
    varName: string; // identifier-safe Python name, e.g. "mito"
    url: string;
    scalePath: string;
    axesOrder: string[];
    voxelNm: [number, number, number];
    offsetNm: [number, number, number];
  }[];
  // DataFrames already in the sql.js DB that should be exposed to Python as
  // df_<organelle_class> (already the make_plot convention).
  tables: { name: string; columns: string[]; rows: (number | string | null)[][] }[];
  code: string; // user- or LLM-written Python
  timeoutMs: number;
}
export interface CancelRequest { kind: "cancel" }
export type IncomingMsg = InspectRequest | AnalyzeRequest | CustomRequest | CancelRequest;

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
export interface CustomResultMsg {
  kind: "customResult";
  // Optional output channels — any may be absent. The UI wires them through.
  table?: {
    name: string;
    columns: string[];
    rows: (number | string | null)[][];
  };
  plotPngDataUrl?: string;
  fly?: { pos: [number, number, number]; segmentId?: string; layer?: string };
  narration?: string;
  stdout?: string;
  annotations?: {
    layerName: string;
    points: { pos: [number, number, number]; id?: string; description?: string }[];
  };
  highlight?: { layer: string; ids: string[] };
  addSourceLayer?: {
    source: string;
    name: string;
    type: "image" | "segmentation";
  };
  newLayer?: {
    // Worker already wrote the synthesized zarr to IndexedDB under this id.
    // Main thread just has to add a layer pointing at the relative URL.
    synthesizedId: string;
    name: string;
    type: "image" | "segmentation";
    shape: number[];
    dtype: string;
  };
}
export interface ErrorMsg { kind: "error"; message: string; where?: string; traceback?: string }
export type OutgoingMsg =
  | ProgressMsg
  | InspectResultMsg
  | AnalyzeResultMsg
  | CustomResultMsg
  | ErrorMsg;

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
  let host = node;
  let attrs = host.attrs as any;
  // OME-NGFF 0.4 puts `multiscales` at the top of attributes; OME 0.5
  // (zarr v3) nests it under an `ome` namespace. Accept both.
  let ms = (attrs?.multiscales ?? attrs?.ome?.multiscales) as any[] | undefined;
  // Path prefix to prepend to per-dataset paths so downstream consumers
  // (analyze, assembleZarr) can resolve them from the *original* user-typed
  // URL. Stays empty unless we auto-descend below.
  let pathPrefix = "";
  // Paintera-converted zarrs put the multiscales under a `data/` subgroup
  // and use the parent group for label-block metadata. Auto-descend so the
  // user can paste the natural `<root>/cells/` path instead of having to
  // know the convention and append `/data/` themselves.
  if (!ms || ms.length === 0) {
    try {
      const dataNode = await zarr.open(node.resolve("data"), { kind: "group" });
      const dataAttrs = dataNode.attrs as any;
      const dataMs = (dataAttrs?.multiscales ?? dataAttrs?.ome?.multiscales) as any[] | undefined;
      if (dataMs && dataMs.length > 0) {
        host = dataNode;
        attrs = dataAttrs;
        ms = dataMs;
        pathPrefix = "data/";
      } else {
        // Native Paintera: data/.zattrs has Paintera/N5 conventions
        // (resolution, offset, scales) but no OME-NGFF multiscales. Fake one.
        const synth = await synthesizeMultiscales(dataNode, dataAttrs);
        if (synth) {
          host = dataNode;
          attrs = dataAttrs;
          ms = [synth];
          pathPrefix = "data/";
        }
      }
    } catch {
      // no `data/` subgroup; fall through
    }
  }
  // Also handle the case where the URL itself is a non-OME multiscale group
  // (children s0, s1, ... ; `resolution` / `pixelResolution` / `scales` in
  // .zattrs). Common in raw N5 / converted Paintera roots without a data/
  // wrapper.
  if (!ms || ms.length === 0) {
    const synth = await synthesizeMultiscales(node, attrs);
    if (synth) ms = [synth];
  }
  if (!ms || ms.length === 0) {
    throw new Error("Group has no OME-Zarr multiscales metadata. Open the data directly at the array path (e.g. <group>/s0), or pick a source with multiscales.");
  }
  const topMs = ms[0];
  const axes: { name: string; type?: string }[] = (topMs.axes ?? []).map((a: any) =>
    typeof a === "string" ? { name: a } : { name: a.name, type: a.type },
  );
  // Some older OME-NGFFs don't include axes; fall back to inferred.
  const effectiveAxes = axes.length ? axes : inferAxesFromShape((await zarr.open(host.resolve(topMs.datasets[0].path), { kind: "array" })).shape);
  // Group-level transforms apply *after* per-dataset ones per OME-NGFF spec.
  const groupT = extractTransforms(topMs.coordinateTransformations, effectiveAxes);
  const datasets: MultiscaleInfo["datasets"] = [];
  for (const ds of topMs.datasets) {
    const arr = await zarr.open(host.resolve(ds.path), { kind: "array" });
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
    datasets.push({ path: pathPrefix + ds.path, arr, scale, offset });
  }
  return { isMultiscale: true, axes: effectiveAxes, datasets };
}

// Build a synthetic OME-NGFF `multiscales` entry for groups that follow N5 /
// Paintera / cellmap conventions instead. Probes for child arrays at sN/<int>
// and reads scale + offset from per-array and group-level attrs.
//
// Per-scale attrs (preferred — these are the absolute physical scale and
// translation for that array):
//   - transform: { axes: ["z","y","x"], scale: [sz,sy,sx], translate: [...] }
//     (cellmap convention; axes give the order of scale/translate).
//   - resolution: [rx, ry, rz]   (Paintera/N5, xyz absolute at this scale)
//   - offset:     [ox, oy, oz]   (Paintera/N5, xyz absolute at this scale)
// Group-level fallbacks (used when per-array info is missing):
//   - resolution / pixelResolution.dimensions / scales[0] for a base xyz
//     resolution at s0; per-scale `downsamplingFactors` multiplies it.
// Returns null if no scale-level child arrays were found.
async function synthesizeMultiscales(
  group: any,
  groupAttrs: any,
): Promise<{ axes: any[]; datasets: any[]; coordinateTransformations?: any[] } | null> {
  const childNames: string[] = [];
  for (let i = 0; i < 16; i++) childNames.push(`s${i}`);
  for (let i = 0; i < 16; i++) childNames.push(String(i));
  type ScaleHit = {
    path: string;
    arr: any;
    childAttrs: any;
    scaleAxisOrder: number[] | null; // [sz, sy, sx] or null if unknown
    transAxisOrder: number[] | null; // [tz, ty, tx] or null if unknown
    dsFactorsXYZ: number[] | null;   // relative downsampling [dx,dy,dz]
  };
  const hits: ScaleHit[] = [];
  for (const name of childNames) {
    try {
      const arr = await zarr.open(group.resolve(name), { kind: "array" });
      const childAttrs = (arr.attrs as any) || {};
      hits.push({
        path: name,
        arr,
        childAttrs,
        scaleAxisOrder: extractAxisOrderedScale(childAttrs, arr.shape.length),
        transAxisOrder: extractAxisOrderedTranslation(childAttrs, arr.shape.length),
        dsFactorsXYZ: Array.isArray(childAttrs.downsamplingFactors)
          ? (childAttrs.downsamplingFactors as number[])
          : null,
      });
    } catch {
      /* missing scale level — keep probing */
    }
  }
  if (hits.length === 0) return null;
  const ndim = hits[0].arr.shape.length;
  const inferredAxes = inferAxesFromShape(hits[0].arr.shape);
  // Group-level fallback resolution/offset (xyz).
  const groupResXYZ: number[] | undefined = Array.isArray(groupAttrs?.resolution)
    ? groupAttrs.resolution
    : Array.isArray(groupAttrs?.pixelResolution?.dimensions)
      ? groupAttrs.pixelResolution.dimensions
      : Array.isArray(groupAttrs?.scales?.[0])
        ? groupAttrs.scales[0]
        : undefined;
  const groupOffsetXYZ: number[] | undefined = Array.isArray(groupAttrs?.offset)
    ? groupAttrs.offset
    : Array.isArray(groupAttrs?.translate)
      ? groupAttrs.translate
      : undefined;
  const datasets = hits.map((h) => {
    // Pick scale (in array-axis order, zyx-ish): prefer per-array, then
    // group base * downsamplingFactors, else fallback to ratio inferred
    // later from shapes (we just emit [1,1,...] here).
    let scaleOrdered: number[] | null = h.scaleAxisOrder;
    if (!scaleOrdered && groupResXYZ && groupResXYZ.length >= 3) {
      const ds = h.dsFactorsXYZ ?? [1, 1, 1];
      const xyz = [groupResXYZ[0] * ds[0], groupResXYZ[1] * ds[1], groupResXYZ[2] * ds[2]];
      scaleOrdered = padToNdim(xyz.slice().reverse(), ndim, 1);
    }
    let transOrdered: number[] | null = h.transAxisOrder;
    if (!transOrdered && groupOffsetXYZ && groupOffsetXYZ.length >= 3) {
      transOrdered = padToNdim([...groupOffsetXYZ].slice(0, 3).reverse(), ndim, 0);
    }
    const ct: any[] = [
      { type: "scale", scale: scaleOrdered ?? padToNdim([1, 1, 1], ndim, 1) },
    ];
    if (transOrdered) ct.push({ type: "translation", translation: transOrdered });
    return { path: h.path, coordinateTransformations: ct };
  });
  return { axes: inferredAxes, datasets };
}

// Read an array's absolute physical scale and return it in array-axis order
// (matching arr.shape). Looks at:
//   - transform.scale + transform.axes (cellmap convention; axes can be in
//     any order — we permute to match the array's leading axes).
//   - resolution: [rx, ry, rz] (xyz; reverse for zyx-ordered arrays).
// Returns null if no usable scale info is present.
function extractAxisOrderedScale(attrs: any, ndim: number): number[] | null {
  const t = attrs?.transform;
  if (t && Array.isArray(t.scale) && Array.isArray(t.axes)) {
    const ordered = orderByAxes(t.scale, t.axes, ndim, 1);
    if (ordered) return ordered;
  }
  if (Array.isArray(attrs?.resolution) && attrs.resolution.length >= 3) {
    return padToNdim([...attrs.resolution].slice(0, 3).reverse(), ndim, 1);
  }
  if (Array.isArray(attrs?.pixelResolution?.dimensions) && attrs.pixelResolution.dimensions.length >= 3) {
    return padToNdim([...attrs.pixelResolution.dimensions].slice(0, 3).reverse(), ndim, 1);
  }
  return null;
}

function extractAxisOrderedTranslation(attrs: any, ndim: number): number[] | null {
  const t = attrs?.transform;
  if (t && Array.isArray(t.translate) && Array.isArray(t.axes)) {
    const ordered = orderByAxes(t.translate, t.axes, ndim, 0);
    if (ordered) return ordered;
  }
  if (Array.isArray(attrs?.offset) && attrs.offset.length >= 3) {
    return padToNdim([...attrs.offset].slice(0, 3).reverse(), ndim, 0);
  }
  if (Array.isArray(attrs?.translate) && attrs.translate.length >= 3) {
    return padToNdim([...attrs.translate].slice(0, 3).reverse(), ndim, 0);
  }
  return null;
}

// Build an ndim-long vector indexed by the spatial axes z/y/x (last 3 dims)
// from a value array + the axes labels declared in the same metadata. Any
// non-spatial leading dims get the fill value.
function orderByAxes(
  values: number[],
  axes: string[],
  ndim: number,
  fill: number,
): number[] | null {
  if (values.length !== axes.length) return null;
  const map: Record<string, number> = {};
  for (let i = 0; i < axes.length; i++) map[String(axes[i]).toLowerCase()] = values[i];
  const spatial = ["z", "y", "x"];
  const out: number[] = new Array(ndim).fill(fill);
  const nSpatial = Math.min(3, ndim);
  for (let i = 0; i < nSpatial; i++) {
    const axis = spatial[spatial.length - nSpatial + i];
    const idx = ndim - nSpatial + i;
    out[idx] = map[axis] ?? fill;
  }
  return out;
}

function padToNdim(arr: number[], ndim: number, fill: number): number[] {
  const out = arr.slice();
  while (out.length < ndim) out.unshift(fill);
  return out.slice(out.length - ndim);
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

// --- Synthesized-zarr IndexedDB writer --------------------------------------
// Python can emit a numpy array as `_TG_NEW_LAYER`. We encode it as an
// uncompressed zarr v2 with a single chunk and a minimal OME-NGFF .zattrs,
// and stash every file as a record in IndexedDB. The service worker then
// serves requests under `/synthesized/<id>/<path>` from the same store, so
// Neuroglancer can consume it with a plain `zarr://<origin>/synthesized/<id>/`
// URL — exact same mechanism as local-folder loading, just a different route.

const SYNTH_DB = "tourguide-synthesized";
const SYNTH_STORE = "files";

function openSynthDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(SYNTH_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(SYNTH_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function synthPut(key: string, value: Uint8Array | string): Promise<void> {
  const db = await openSynthDB();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(SYNTH_STORE, "readwrite");
    tx.objectStore(SYNTH_STORE).put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

function numpyToZarrDtype(dtype: string): string | null {
  const map: Record<string, string> = {
    uint8: "|u1",
    int8: "|i1",
    uint16: "<u2",
    int16: "<i2",
    uint32: "<u4",
    int32: "<i4",
    uint64: "<u8",
    int64: "<i8",
    float32: "<f4",
    float64: "<f8",
    // bool intentionally omitted — Python side coerces to uint8 so NG can
    // render it. Letting it through as |b1 makes NG fail to parse the zarr.
  };
  return map[dtype] ?? null;
}

async function writeSynthesizedZarr(params: {
  id: string;
  bytes: Uint8Array;
  shape: number[];
  dtype: string; // numpy dtype name, e.g. "uint8"
  axes: string[]; // e.g. ["z", "y", "x"] (array-axis order)
  spacing: number[]; // nm per voxel, array-axis order
  offsets: number[]; // nm origin, array-axis order
}): Promise<void> {
  const zarrDtype = numpyToZarrDtype(params.dtype);
  if (!zarrDtype) throw new Error(`Unsupported dtype for synthesized layer: ${params.dtype}`);

  const zgroup = JSON.stringify({ zarr_format: 2 });
  const zattrs = JSON.stringify({
    multiscales: [
      {
        version: "0.4",
        name: params.id,
        axes: params.axes.map((n) => ({ name: n, type: "space", unit: "nanometer" })),
        datasets: [
          {
            path: "s0",
            coordinateTransformations: [
              { type: "scale", scale: params.spacing },
              { type: "translation", translation: params.offsets },
            ],
          },
        ],
      },
    ],
  });
  const zarray = JSON.stringify({
    zarr_format: 2,
    shape: params.shape,
    // Single-chunk layout — simplest, and our synthesized volumes are small
    // (bounded by the same 32M-voxel analysis cap, so ~tens of MB max).
    chunks: params.shape,
    dtype: zarrDtype,
    compressor: null,
    fill_value: 0,
    order: "C",
    filters: null,
    dimension_separator: "/",
  });

  const prefix = `${params.id}/`;
  const enc = new TextEncoder();
  await synthPut(`${prefix}.zgroup`, enc.encode(zgroup));
  await synthPut(`${prefix}.zattrs`, enc.encode(zattrs));
  await synthPut(`${prefix}s0/.zarray`, enc.encode(zarray));
  // Single chunk at origin — with dimension_separator="/" that's `s0/0/0/0...`.
  const chunkKey = `${prefix}s0/${params.shape.map(() => "0").join("/")}`;
  await synthPut(chunkKey, params.bytes);
}

// --- Custom (arbitrary Python) mode -----------------------------------------
// Loads N zarr layers as numpy arrays, exposes them + the sql.js DataFrames
// + numpy/scipy/skimage/pandas/matplotlib to user- or LLM-written Python,
// then returns any of: table, plot, fly-to, narration.

async function readLayerArray(layerSpec: CustomRequest["layers"][number]): Promise<{
  data: ArrayBufferView;
  shape: number[];
  spacing: number[];
  offsets: number[];
  axes: string[];
}> {
  const store = new zarr.FetchStore(layerSpec.url);
  const root = await zarr.open(store);
  const arr = layerSpec.scalePath
    ? await zarr.open(root.kind === "group" ? root.resolve(layerSpec.scalePath) : root, { kind: "array" })
    : (root.kind === "array" ? root : await zarr.open(root.resolve(""), { kind: "array" }));
  const fullShape = arr.shape;
  const axisNames = layerSpec.axesOrder;
  let sel: (null | number)[] = axisNames.map((a) =>
    a === "x" || a === "y" || a === "z" ? null : 0,
  );
  if (sel.length !== fullShape.length) {
    sel = fullShape.map((_, i) => (i >= fullShape.length - 3 ? null : 0));
  }
  const surviving = axisNames.filter((_a, i) => sel[i] === null);
  const axesForPython = surviving.length === fullShape.filter((_, i) => sel[i] === null).length
    ? surviving
    : ["z", "y", "x"].slice(-fullShape.filter((_, i) => sel[i] === null).length);
  const axisScaleMap: Record<string, number> = {
    x: layerSpec.voxelNm[0], y: layerSpec.voxelNm[1], z: layerSpec.voxelNm[2],
  };
  const axisOffsetMap: Record<string, number> = {
    x: layerSpec.offsetNm[0], y: layerSpec.offsetNm[1], z: layerSpec.offsetNm[2],
  };
  const spacing = axesForPython.map((a) => axisScaleMap[a] ?? 1);
  const offsets = axesForPython.map((a) => axisOffsetMap[a] ?? 0);
  const result = await zarr.get(arr, sel as any);
  return {
    data: (result as any).data,
    shape: (result as any).shape,
    spacing,
    offsets,
    axes: axesForPython,
  };
}

async function handleCustom(msg: CustomRequest): Promise<void> {
  progress("Loading layers …", "load");
  const loaded: Record<string, Awaited<ReturnType<typeof readLayerArray>>> = {};
  for (const layer of msg.layers) {
    progress(`Reading ${layer.varName} (${layer.scalePath || "root"}) …`, "read");
    loaded[layer.varName] = await readLayerArray(layer);
    if (cancelled) return;
  }

  progress("Loading Python runtime …", "python");
  const py = await ensurePyodide();
  // Custom mode needs pandas + matplotlib in addition to the regionprops set.
  await py.loadPackage(["pandas", "matplotlib"]);
  if (cancelled) return;

  // Hand each array to Python with metadata.
  py.globals.set("_tg_layer_names", msg.layers.map((l) => l.varName));
  py.globals.set("_tg_tables_json", JSON.stringify(msg.tables));
  for (const [name, info] of Object.entries(loaded)) {
    py.globals.set(`__tg_${name}_data`, info.data);
    py.globals.set(`__tg_${name}_shape`, info.shape);
    py.globals.set(`__tg_${name}_spacing`, info.spacing);
    py.globals.set(`__tg_${name}_offsets`, info.offsets);
    py.globals.set(`__tg_${name}_axes`, info.axes);
  }
  py.globals.set("_tg_user_code", msg.code);
  py.globals.set("_tg_timeout_ms", msg.timeoutMs);

  progress("Running custom analysis …", "compute");
  const setupCode = `
import numpy as np, pandas as pd, json, io, base64, traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
try:
    import skimage
    from skimage import measure as _sk_measure
except Exception:
    _sk_measure = None

# Reconstruct each loaded layer as a dict {array, spacing, offsets, axes}.
layers = {}
for _name in list(_tg_layer_names):
    _arr = np.asarray(globals()[f'__tg_{_name}_data']).reshape(
        tuple(int(x) for x in list(globals()[f'__tg_{_name}_shape']))
    )
    layers[_name] = {
        "array": _arr,
        "spacing": tuple(float(x) for x in list(globals()[f'__tg_{_name}_spacing'])),
        "offsets": tuple(float(x) for x in list(globals()[f'__tg_{_name}_offsets'])),
        "axes":    [str(a) for a in list(globals()[f'__tg_{_name}_axes'])],
    }
    # Expose the bare array under its var name (the common case).
    globals()[_name] = _arr

# Build DataFrames from sql.js tables sent across the wire.
_tg_tables = json.loads(_tg_tables_json)
for _t in _tg_tables:
    _df = pd.DataFrame(_t["rows"], columns=_t["columns"])
    globals()[f"df_{_t['name']}"] = _df

# Output channels. User code sets any of these.
_TG_TABLE = None       # pandas DataFrame
_TG_TABLE_NAME = None  # display name for the table
_TG_PLOT = None        # True if you called plt.<stuff>; we'll grab the current figure
_TG_FLY = None         # {"pos": [x,y,z], "segment_id": str, "layer": str}
_TG_NARRATION = None   # string
_TG_STDOUT = []        # captured via print()
_TG_ANNOTATIONS = None # {"layer_name": "my_points", "points": [{"pos": [x,y,z], "id": "...", "description": "..."}, ...]}
_TG_HIGHLIGHT = None   # {"layer": "<existing ng layer name>", "ids": [1, 2, 3]}
_TG_ADD_SOURCE_LAYER = None  # {"source": "zarr://...", "name": "new_layer", "type": "segmentation"|"image"}
_TG_NEW_LAYER = None   # {"array": ndarray, "name": "...", "type": "segmentation"|"image",
                       #  "spacing": (sz,sy,sx) nm, "offsets": (oz,oy,ox) nm, "axes": ["z","y","x"]}
                       # - spacing/offsets/axes default to the first selected input layer's values.
_TG_NEW_MESH_LAYER = None  # HF-only: {"labels": ndarray, "name": "...", "spacing": ..., "offsets": ..., "ids": [...]?}
                           # - in the local Pyodide runtime zmesh is not available; this var is read by
                           #   the HF backend only. Setting it here is a no-op.

# Capture print() output without touching sys.stdout (workers don't have one).
import builtins as _b
_real_print = _b.print
def _captured_print(*args, **kwargs):
    _TG_STDOUT.append(" ".join(str(a) for a in args))
    _real_print(*args, **kwargs)
_b.print = _captured_print
`;
  await py.runPythonAsync(setupCode);

  // Run user code inside a try/except so we get tracebacks, not opaque errors.
  const userCode = msg.code;
  const runCode = `
_tg_err = None
try:
${indent(userCode, 4)}
except Exception as e:
    _tg_err = traceback.format_exc()
`;
  try {
    await py.runPythonAsync(runCode);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(`Python failed: ${message}`);
  }
  const pyErr = py.globals.get("_tg_err") as string | null;
  if (pyErr) {
    emit({ kind: "error", message: String(pyErr).split("\n").slice(-3).join("\n"), where: "custom", traceback: String(pyErr) });
    return;
  }

  // Collect output channels.
  const outMsg: CustomResultMsg = { kind: "customResult" };

  // Table — convert a pandas DataFrame into {columns, rows}.
  await py.runPythonAsync(`
_tg_out_cols = None
_tg_out_rows = None
_tg_out_name = None
if _TG_TABLE is not None:
    _df = _TG_TABLE
    if not isinstance(_df, pd.DataFrame):
        _df = pd.DataFrame(_df)
    _tg_out_cols = list(_df.columns)
    _tg_out_rows = _df.where(pd.notnull(_df), None).values.tolist()
    _tg_out_name = str(_TG_TABLE_NAME) if _TG_TABLE_NAME else "custom_result"
`);
  const tblCols = py.globals.get("_tg_out_cols");
  if (tblCols) {
    outMsg.table = {
      name: String(py.globals.get("_tg_out_name")),
      columns: (tblCols as any).toJs({ create_proxies: false }) as string[],
      rows: (py.globals.get("_tg_out_rows") as any).toJs({ create_proxies: false }) as any[][],
    };
  }

  // Plot — if user drew something, save the current figure as PNG.
  await py.runPythonAsync(`
_tg_out_png = None
if plt.get_fignums() or _TG_PLOT is not None:
    _buf = io.BytesIO()
    plt.savefig(_buf, format="png", bbox_inches="tight", dpi=120)
    plt.close("all")
    _tg_out_png = base64.b64encode(_buf.getvalue()).decode("ascii")
`);
  const b64 = py.globals.get("_tg_out_png");
  if (b64) outMsg.plotPngDataUrl = `data:image/png;base64,${b64}`;

  // Fly target.
  await py.runPythonAsync(`
_tg_out_fly = None
if _TG_FLY is not None:
    _f = _TG_FLY
    if isinstance(_f, dict):
        _tg_out_fly = {"pos": [float(x) for x in list(_f.get("pos") or [])], "segment_id": str(_f.get("segment_id", "")) or None, "layer": str(_f.get("layer", "")) or None}
`);
  const flyPy = py.globals.get("_tg_out_fly");
  if (flyPy) {
    const f = (flyPy as any).toJs({ create_proxies: false, dict_converter: Object.fromEntries }) as {
      pos: number[];
      segment_id?: string;
      layer?: string;
    };
    if (f.pos && f.pos.length >= 3) {
      outMsg.fly = {
        pos: [f.pos[0], f.pos[1], f.pos[2]],
        segmentId: f.segment_id,
        layer: f.layer,
      };
    }
  }

  // Narration + stdout.
  const narr = py.globals.get("_TG_NARRATION");
  if (narr) outMsg.narration = String(narr);
  const stdoutPy = py.globals.get("_TG_STDOUT");
  if (stdoutPy) {
    const lines = (stdoutPy as any).toJs({ create_proxies: false }) as string[];
    if (lines.length) outMsg.stdout = lines.join("\n");
  }

  // Annotations.
  await py.runPythonAsync(`
_tg_out_ann = None
if _TG_ANNOTATIONS is not None:
    a = _TG_ANNOTATIONS
    if isinstance(a, list):
        a = {"layer_name": "custom_points", "points": a}
    pts = []
    for p in (a.get("points") or []):
        if isinstance(p, (list, tuple)):
            pts.append({"pos": [float(p[0]), float(p[1]), float(p[2])], "id": None, "description": None})
        else:
            pos = p.get("pos")
            pts.append({
                "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                "id": (str(p.get("id")) if p.get("id") is not None else None),
                "description": (str(p.get("description")) if p.get("description") is not None else None),
            })
    _tg_out_ann = {"layer_name": str(a.get("layer_name", "custom_points")), "points": pts}
`);
  const annPy = py.globals.get("_tg_out_ann");
  if (annPy) {
    const a = (annPy as any).toJs({
      create_proxies: false,
      dict_converter: Object.fromEntries,
    }) as {
      layer_name: string;
      points: { pos: number[]; id?: string | null; description?: string | null }[];
    };
    outMsg.annotations = {
      layerName: a.layer_name,
      points: a.points.map((p) => ({
        pos: [p.pos[0], p.pos[1], p.pos[2]] as [number, number, number],
        id: p.id ?? undefined,
        description: p.description ?? undefined,
      })),
    };
  }

  // Highlight segments.
  await py.runPythonAsync(`
_tg_out_hi = None
if _TG_HIGHLIGHT is not None:
    h = _TG_HIGHLIGHT
    _tg_out_hi = {"layer": str(h.get("layer", "")), "ids": [str(i) for i in (h.get("ids") or [])]}
`);
  const hiPy = py.globals.get("_tg_out_hi");
  if (hiPy) {
    const h = (hiPy as any).toJs({
      create_proxies: false,
      dict_converter: Object.fromEntries,
    }) as { layer: string; ids: string[] };
    if (h.layer) outMsg.highlight = { layer: h.layer, ids: h.ids };
  }

  // Add a layer from an existing (remote) zarr/n5/precomputed source.
  await py.runPythonAsync(`
_tg_out_src = None
if _TG_ADD_SOURCE_LAYER is not None:
    s = _TG_ADD_SOURCE_LAYER
    _tg_out_src = {"source": str(s.get("source", "")), "name": str(s.get("name", "new_layer")), "type": str(s.get("type", "image"))}
`);
  const srcPy = py.globals.get("_tg_out_src");
  if (srcPy) {
    const s = (srcPy as any).toJs({
      create_proxies: false,
      dict_converter: Object.fromEntries,
    }) as { source: string; name: string; type: string };
    if (s.source) {
      outMsg.addSourceLayer = {
        source: s.source,
        name: s.name,
        type: s.type === "segmentation" ? "segmentation" : "image",
      };
    }
  }

  // New layer backed by a synthesized zarr (the interesting one).
  // We extract bytes + shape + dtype from Python, then write a minimal
  // OME-NGFF zarr v2 into IndexedDB for the service worker to serve.
  await py.runPythonAsync(`
_tg_new_layer_spec = None
if _TG_NEW_LAYER is not None:
    nl = _TG_NEW_LAYER
    arr = nl.get("array") if isinstance(nl, dict) else None
    if arr is None:
        raise ValueError("_TG_NEW_LAYER must be a dict with an 'array' key")
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    # Default spacing/offsets/axes to the first selected input layer's values.
    _default = layers[list(_tg_layer_names)[0]] if list(_tg_layer_names) else None
    _spacing = nl.get("spacing")
    if _spacing is None and _default is not None: _spacing = _default["spacing"]
    _offsets = nl.get("offsets")
    if _offsets is None and _default is not None: _offsets = _default["offsets"]
    _axes = nl.get("axes")
    if _axes is None and _default is not None: _axes = _default["axes"]
    if _axes is None: _axes = ["z", "y", "x"][-arr.ndim:]
    if _spacing is None: _spacing = [1.0] * arr.ndim
    if _offsets is None: _offsets = [0.0] * arr.ndim
    # Neuroglancer doesn't handle bool zarrs — and float16 / float64 are also
    # awkward for its layer renderers. Coerce to NG-friendly dtypes.
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    elif arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    elif arr.dtype == np.float16:
        arr = arr.astype(np.float32)
    arr_c = np.ascontiguousarray(arr)
    _tg_new_layer_spec = {
        "bytes": arr_c.tobytes(order="C"),
        "shape": list(arr_c.shape),
        "dtype": str(arr_c.dtype),
        "axes": list(_axes),
        "spacing": [float(x) for x in list(_spacing)],
        "offsets": [float(x) for x in list(_offsets)],
        "name": str(nl.get("name", "new_layer")),
        "type": str(nl.get("type", "segmentation")),
    }
`);
  const newLayerPy = py.globals.get("_tg_new_layer_spec");
  if (newLayerPy) {
    const spec = (newLayerPy as any).toJs({
      create_proxies: false,
      dict_converter: Object.fromEntries,
    }) as {
      bytes: Uint8Array;
      shape: number[];
      dtype: string;
      axes: string[];
      spacing: number[];
      offsets: number[];
      name: string;
      type: string;
    };
    const id = `${spec.name.replace(/[^a-zA-Z0-9_-]/g, "_")}-${Math.random()
      .toString(36)
      .slice(2, 8)}`;
    await writeSynthesizedZarr({
      id,
      bytes: spec.bytes,
      shape: spec.shape,
      dtype: spec.dtype,
      axes: spec.axes,
      spacing: spec.spacing,
      offsets: spec.offsets,
    });
    outMsg.newLayer = {
      synthesizedId: id,
      name: spec.name,
      type: spec.type === "image" ? "image" : "segmentation",
      shape: spec.shape,
      dtype: spec.dtype,
    };
  }

  emit(outMsg);
}

// Helper: indent a block of code by `n` spaces so it fits inside a try/except.
function indent(code: string, n: number): string {
  const pad = " ".repeat(n);
  return code
    .split("\n")
    .map((l) => pad + l)
    .join("\n");
}

self.addEventListener("message", (ev: MessageEvent<IncomingMsg>) => {
  const msg = ev.data;
  if (msg.kind === "cancel") {
    cancelled = true;
    return;
  }
  cancelled = false;
  const handler =
    msg.kind === "inspect" ? handleInspect(msg) :
    msg.kind === "analyze" ? handleAnalyze(msg) :
    handleCustom(msg);
  handler.catch((err: unknown) => {
    const message = err instanceof Error ? err.message : String(err);
    emit({ kind: "error", message, where: msg.kind });
  });
});
