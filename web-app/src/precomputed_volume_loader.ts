// Neuroglancer precomputed segmentation VOLUME reader.
//
// Reads the voxel grid (not just meshes / skeletons) out of a precomputed
// segmentation directory so python_on_layers can run regionprops / CC /
// contact-area / arbitrary-segmentation code on hemibrain, FAFB, MICrONS,
// etc. Until this existed, those datasets only worked through the mesh
// or skeleton path — fine for known segment IDs, useless for whole-
// dataset sweeps like "regionprops on every body".
//
// Scope of this implementation (deliberately small — covers the FlyEM /
// cellmap segmentation case):
//
//   • encoding: "compressed_segmentation" only. (raw + jpeg are different
//     decoders; segmentations almost never use them.)
//   • data_type: "uint64" only. (uint32/16/8 segmentations exist but are
//     rare; can be added later by widening the decoder type.)
//   • num_channels: 1.
//   • Sharded OR unsharded chunk layout. Sharded uses our sharded reader
//     (precomputed_sharded.ts); unsharded uses path-based chunk URLs.
//   • Loads a FULL chosen scale into one big BigUint64Array. The caller
//     picks the scale by budget — coarse scales fit in Pyodide's WASM
//     heap; fine scales don't.
//
// We reuse neuroglancer's `decodeChannel` + `encodeZIndexCompressed3d`
// rather than reimplementing — same pattern as Draco / murmurhash. Both
// are small, pure, and already in the bundle through other imports.

import { decodeChannel } from "neuroglancer/unstable/sliceview/compressed_segmentation/decode_common.js";
import { encodeZIndexCompressed3d } from "neuroglancer/unstable/util/zorder.js";

import { precomputedToHttps } from "./precomputed_info.js";
import {
  fetchShardedChunk,
  parseSharding,
  type ShardingSpec,
} from "./precomputed_sharded.js";

export interface PrecomputedScale {
  key: string;
  // Resolution in nm per axis, x/y/z (matches the info file's `resolution`).
  resolutionNm: [number, number, number];
  // Voxel extent at this scale, x/y/z.
  size: [number, number, number];
  // Chunk size in voxels, x/y/z. Almost always [64,64,64].
  chunkSize: [number, number, number];
  // Voxel offset (origin) — added to every chunk position. Usually [0,0,0].
  voxelOffset: [number, number, number];
  encoding: "compressed_segmentation" | "raw" | "jpeg";
  // compressed_segmentation only: block size inside chunks, x/y/z.
  // Almost always [8,8,8].
  compressedSegmentationBlockSize?: [number, number, number];
  // When present, chunks are packed into shard files; the sharded reader
  // handles lookup. When absent, chunk paths are
  // <base>/<key>/<x0>-<x1>_<y0>-<y1>_<z0>-<z1>.
  sharding?: ShardingSpec;
  // Total bytes if loaded fully into memory at the volume's dtype. Used
  // by pickScaleForBudget to choose the finest scale that fits.
  approxBytes: number;
}

export interface PrecomputedVolumeInfo {
  dataType: "uint64";
  numChannels: 1;
  scales: PrecomputedScale[];
}

// Parse a precomputed segmentation info JSON into our scale list. Returns
// null when the info isn't a recognized segmentation (different @type,
// unsupported dtype, no scales). Scales whose sharding spec we can't
// understand are skipped silently — coarser fallbacks may still work.
export function parsePrecomputedVolumeInfo(
  raw: Record<string, unknown>,
): PrecomputedVolumeInfo | null {
  if (String(raw["@type"] ?? "") !== "neuroglancer_multiscale_volume") return null;
  if (String(raw["type"] ?? "") !== "segmentation") return null;
  if (String(raw["data_type"] ?? "") !== "uint64") return null;
  if (Number(raw["num_channels"] ?? 1) !== 1) return null;
  const scalesRaw = raw["scales"];
  if (!Array.isArray(scalesRaw)) return null;
  const scales: PrecomputedScale[] = [];
  for (const s of scalesRaw) {
    if (!s || typeof s !== "object") continue;
    const sr = s as Record<string, unknown>;
    const sizeArr = sr["size"] as unknown;
    const chunkSizesArr = sr["chunk_sizes"] as unknown;
    const resArr = sr["resolution"] as unknown;
    if (
      !Array.isArray(sizeArr) ||
      sizeArr.length !== 3 ||
      !Array.isArray(chunkSizesArr) ||
      chunkSizesArr.length === 0 ||
      !Array.isArray((chunkSizesArr as unknown[])[0]) ||
      !Array.isArray(resArr) ||
      resArr.length !== 3
    ) {
      continue;
    }
    const chunkSize = (chunkSizesArr as number[][])[0] as number[];
    if (chunkSize.length !== 3) continue;
    const voxelOffsetArr = sr["voxel_offset"] as unknown;
    const voxelOffset: [number, number, number] = Array.isArray(voxelOffsetArr) && voxelOffsetArr.length === 3
      ? [Number(voxelOffsetArr[0]), Number(voxelOffsetArr[1]), Number(voxelOffsetArr[2])]
      : [0, 0, 0];
    const encoding = String(sr["encoding"] ?? "raw") as PrecomputedScale["encoding"];
    const blockSizeArr = sr["compressed_segmentation_block_size"] as unknown;
    const blockSize: [number, number, number] | undefined =
      Array.isArray(blockSizeArr) && blockSizeArr.length === 3
        ? [Number(blockSizeArr[0]), Number(blockSizeArr[1]), Number(blockSizeArr[2])]
        : undefined;
    let sharding: ShardingSpec | undefined;
    try {
      const parsed = parseSharding(sr["sharding"]);
      if (parsed) sharding = parsed;
    } catch {
      // Unsupported sharding variant (e.g. unknown hash) — skip this
      // scale rather than failing the whole probe. Coarser scales
      // may still be loadable.
      continue;
    }
    const sx = Number((sizeArr as number[])[0]);
    const sy = Number((sizeArr as number[])[1]);
    const sz = Number((sizeArr as number[])[2]);
    scales.push({
      key: String(sr["key"]),
      resolutionNm: [Number((resArr as number[])[0]), Number((resArr as number[])[1]), Number((resArr as number[])[2])],
      size: [sx, sy, sz],
      chunkSize: [Number(chunkSize[0]), Number(chunkSize[1]), Number(chunkSize[2])],
      voxelOffset,
      encoding,
      compressedSegmentationBlockSize: blockSize,
      sharding,
      approxBytes: sx * sy * sz * 8, // uint64 = 8 bytes
    });
  }
  if (scales.length === 0) return null;
  return { dataType: "uint64", numChannels: 1, scales };
}

// Pick the finest scale that fits in the byte budget. Returns null when
// even the coarsest scale exceeds budget (shouldn't happen for typical
// datasets — even hemibrain's coarsest is ~30 KB).
export function pickScaleForBudget(
  info: PrecomputedVolumeInfo,
  budgetBytes: number,
): PrecomputedScale | null {
  // Scales are stored finest-first. Walk fine→coarse, take the first
  // one that fits. The caller can also pass a smaller budget to force
  // coarser scales.
  for (const s of info.scales) {
    if (s.approxBytes <= budgetBytes) return s;
  }
  return null;
}

export interface LoadedScale {
  // Flat uint64 array in C-order with axes (z, y, x): index = z*(sy*sx) + y*sx + x.
  data: BigUint64Array;
  // Shape in (z, y, x) order — matches the data flat-layout.
  shape: [number, number, number];
  // Per-axis voxel size in nm, (z, y, x) order.
  spacingNm: [number, number, number];
  // World-nm offset of voxel (0,0,0), (z, y, x) order.
  offsetNm: [number, number, number];
  // The scale we picked, so the caller can log/surface it.
  scale: PrecomputedScale;
}

// Read a full scale of a precomputed segmentation volume into one
// BigUint64Array. Fetches chunks in parallel up to `concurrency`
// (default 8) — for sharded scales the minishard indices are cached so
// repeated fetches cost only one ranged HTTP each.
export async function loadPrecomputedScale(
  baseUrl: string,
  scale: PrecomputedScale,
  onProgress?: (msg: string) => void,
  concurrency = 8,
): Promise<LoadedScale> {
  if (scale.encoding !== "compressed_segmentation") {
    throw new Error(
      `Precomputed volume reader only supports 'compressed_segmentation' encoding; got '${scale.encoding}' at scale ${scale.key}.`,
    );
  }
  if (!scale.compressedSegmentationBlockSize) {
    throw new Error(
      `Scale ${scale.key} is compressed_segmentation but missing 'compressed_segmentation_block_size'.`,
    );
  }
  const httpsBase = precomputedToHttps(baseUrl).replace(/\/$/, "");
  const scaleBase = `${httpsBase}/${scale.key}`;
  const [sx, sy, sz] = scale.size;
  const [csx, csy, csz] = scale.chunkSize;
  const [bsx, bsy, bsz] = scale.compressedSegmentationBlockSize;
  const gridShape: [number, number, number] = [
    Math.ceil(sx / csx),
    Math.ceil(sy / csy),
    Math.ceil(sz / csz),
  ];
  const numChunks = gridShape[0] * gridShape[1] * gridShape[2];
  // Bits needed to encode any chunk grid coord (used for z-order key).
  const xBits = Math.max(1, Math.ceil(Math.log2(Math.max(1, gridShape[0]))));
  const yBits = Math.max(1, Math.ceil(Math.log2(Math.max(1, gridShape[1]))));
  const zBits = Math.max(1, Math.ceil(Math.log2(Math.max(1, gridShape[2]))));

  // Allocate once. C-order with axes (z, y, x).
  const out = new BigUint64Array(sx * sy * sz);

  // Enumerate every chunk's grid coords so we can map+await with bounded
  // concurrency below.
  const tasks: { cx: number; cy: number; cz: number; idx: number }[] = [];
  for (let cz = 0; cz < gridShape[2]; cz++) {
    for (let cy = 0; cy < gridShape[1]; cy++) {
      for (let cx = 0; cx < gridShape[0]; cx++) {
        tasks.push({ cx, cy, cz, idx: tasks.length });
      }
    }
  }

  let completed = 0;
  let lastReportedAt = 0;
  const reportProgress = (): void => {
    if (!onProgress) return;
    const now = Date.now();
    // Throttle to ~10 updates/sec so we don't drown the UI thread.
    if (now - lastReportedAt < 100) return;
    lastReportedAt = now;
    onProgress(
      `Reading volume chunks: ${completed}/${numChunks} (scale ${scale.key})`,
    );
  };

  const fetchAndDecodeOne = async (t: { cx: number; cy: number; cz: number }): Promise<void> => {
    const { cx, cy, cz } = t;
    const x0 = cx * csx;
    const y0 = cy * csy;
    const z0 = cz * csz;
    const vx = Math.min(csx, sx - x0);
    const vy = Math.min(csy, sy - y0);
    const vz = Math.min(csz, sz - z0);
    let chunkBytes: ArrayBuffer | null = null;
    try {
      if (scale.sharding) {
        const key = encodeZIndexCompressed3d(xBits, yBits, zBits, cx, cy, cz);
        const got = await fetchShardedChunk(scaleBase, key, scale.sharding);
        chunkBytes = got?.data ?? null;
      } else {
        const x1 = x0 + vx;
        const y1 = y0 + vy;
        const z1 = z0 + vz;
        const url = `${scaleBase}/${x0}-${x1}_${y0}-${y1}_${z0}-${z1}`;
        const res = await fetch(url);
        if (res.ok) chunkBytes = await res.arrayBuffer();
        else if (res.status !== 404) {
          throw new Error(`chunk fetch ${res.status} at ${url}`);
        }
      }
    } finally {
      completed += 1;
      reportProgress();
    }
    if (!chunkBytes) return; // missing chunk → leave as zeros (background)
    // compressed_segmentation header: for single-channel data, the first
    // u32 is the byte offset (in u32 words) where this channel's data
    // starts. Pass it as the baseOffset to decodeChannel.
    const dataU32 = new Uint32Array(chunkBytes);
    if (dataU32.length < 1) return;
    const baseOffset = dataU32[0];
    const chunkOut = new BigUint64Array(vx * vy * vz);
    decodeChannel(
      chunkOut,
      dataU32,
      baseOffset,
      [vx, vy, vz],
      [bsx, bsy, bsz],
    );
    // Stitch chunk into the global array. Chunk-local layout (from
    // decodeChannel) is x-fastest within (z, y), so each (z, y) row is
    // a vx-long contiguous run we can copy with one set() call.
    // BigUint64Array.set works the same way as Uint32Array.set.
    for (let z = 0; z < vz; z++) {
      for (let y = 0; y < vy; y++) {
        const srcStart = (z * vy + y) * vx;
        const dstStart = ((z0 + z) * sy + (y0 + y)) * sx + x0;
        out.set(chunkOut.subarray(srcStart, srcStart + vx), dstStart);
      }
    }
  };

  // Bounded-concurrency executor — N parallel workers each pull tasks
  // from the head of the queue until empty.
  let queueIdx = 0;
  const workers: Promise<void>[] = [];
  const workerCount = Math.min(concurrency, tasks.length);
  for (let w = 0; w < workerCount; w++) {
    workers.push((async (): Promise<void> => {
      while (true) {
        const myIdx = queueIdx++;
        if (myIdx >= tasks.length) return;
        await fetchAndDecodeOne(tasks[myIdx]);
      }
    })());
  }
  await Promise.all(workers);
  if (onProgress) {
    onProgress(`Read ${completed} chunks (${(out.length / 1e6).toFixed(1)}M voxels at ${scale.key})`);
  }
  return {
    data: out,
    shape: [sz, sy, sx],
    spacingNm: [scale.resolutionNm[2], scale.resolutionNm[1], scale.resolutionNm[0]],
    offsetNm: [
      scale.voxelOffset[2] * scale.resolutionNm[2],
      scale.voxelOffset[1] * scale.resolutionNm[1],
      scale.voxelOffset[0] * scale.resolutionNm[0],
    ],
    scale,
  };
}
