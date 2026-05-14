// Neuroglancer precomputed multi-LOD Draco mesh reader, unsharded variant.
//
// Format reference:
//
//   neuroglancer_multilod_draco (unsharded)
//     <base>/info                — JSON describing the mesh format
//        @type: "neuroglancer_multilod_draco"
//        vertex_quantization_bits: 10 | 16 (vertex coords are u32 in
//           [0, 2^bits - 1], remapped to the fragment bbox)
//        transform: number[12], optional row-major 3x4 (native → nm)
//        lod_scale_multiplier: number (default 1), affects LOD selection
//           only — irrelevant for analysis-by-finest-LOD
//     <base>/<seg_id>.index      — binary manifest (per-segment)
//     <base>/<seg_id>            — binary fragment data (per-segment),
//        concatenated draco-encoded chunks in LOD-major order
//
//   Manifest binary layout (.index):
//     chunk_shape:           3 × f32 LE
//     grid_origin:           3 × f32 LE
//     num_stored_lods:       u32 LE
//     stored_lod_scales:     num_stored_lods × f32 LE
//     vertex_offsets:        num_stored_lods × 3 × f32 LE
//     num_fragments_per_lod: num_stored_lods × u32 LE
//     for each lod:
//       fragment_positions: num_fragments_per_lod[lod] × 3 × u32 LE,
//          laid out column-major: x[0..N], y[0..N], z[0..N]
//       fragment_data_sizes: num_fragments_per_lod[lod] × u32 LE
//
//   Fragment data layout:
//     chunks concatenated in (lod=0, lod=1, ...) order, within each LOD
//     in the order their fragment_positions appear in the manifest.
//     A chunk with data_size=0 is empty (segment doesn't touch that
//     cell at this LOD).
//
//   Dequantization (per LOD `lod`, per fragment at grid pos (fx,fy,fz)):
//     scale = 1 << lod                       // 1 at LOD 0
//     fragmentOrigin = grid_origin
//                    + (fx,fy,fz) * chunk_shape * scale
//                    + vertex_offsets[lod]
//     fragmentShape  = chunk_shape * scale
//     world_native   = fragmentOrigin
//                    + (quant_pos / (2^bits - 1)) * fragmentShape
//     world_nm       = transform * (world_native, 1)
//
// We always pull the FINEST LOD (typically lod=0) for analysis. If lod=0
// has zero fragments for this segment we fall through to lod=1, etc.
// Coarser-LOD-only segments are rare but exist for tiny objects.
//
// Sharded variant is NOT supported here. The shard reader is significantly
// more complex (multi-level index, gzip/raw shards, minishard hash); the
// agent.ts gate keeps us out of it.

// We bring in neuroglancer's bundled draco wasm + JS wrapper rather than
// adding draco3d as a dependency. The package only exposes its internals
// via the `neuroglancer/unstable/*` subpath; the underlying wasm is
// loaded relative to the JS module via import.meta.url (vite picks up
// the asset automatically through the bundler's URL-import handling).
import { decodeDracoPartitioned } from "neuroglancer/unstable/mesh/draco/index.js";

import type { ParsedMesh } from "./precomputed_mesh_loader.js";
import { concatMeshes } from "./precomputed_mesh_loader.js";
import { precomputedToHttps } from "./precomputed_info.js";
import {
  fetchShardedChunk,
  fetchRawShardBytes,
  type ShardingSpec,
} from "./precomputed_sharded.js";

export interface MultilodFragment {
  // Grid-cell position of this fragment at its LOD.
  gridX: number;
  gridY: number;
  gridZ: number;
  // Byte offset/length within the fragment data file (LOD-cumulative).
  offset: number;
  length: number;
}

export interface MultilodManifest {
  chunkShape: [number, number, number];
  gridOrigin: [number, number, number];
  // vertexOffsets[lod] = [dx, dy, dz] applied AFTER chunk-grid translation.
  vertexOffsets: [number, number, number][];
  lodScales: number[]; // per-LOD scale factor (rendering hint, not spatial)
  fragmentsByLod: MultilodFragment[][];
}

// Parse the `.index` manifest file.
export function parseMultilodManifest(buf: ArrayBuffer): MultilodManifest {
  if (buf.byteLength < 28 || buf.byteLength % 4 !== 0) {
    throw new Error(`Invalid .index size: ${buf.byteLength}`);
  }
  const dv = new DataView(buf);
  let p = 0;
  const chunkShape: [number, number, number] = [
    dv.getFloat32(p, true),
    dv.getFloat32(p + 4, true),
    dv.getFloat32(p + 8, true),
  ];
  p += 12;
  const gridOrigin: [number, number, number] = [
    dv.getFloat32(p, true),
    dv.getFloat32(p + 4, true),
    dv.getFloat32(p + 8, true),
  ];
  p += 12;
  const numLods = dv.getUint32(p, true);
  p += 4;
  // Header fixed-size area beyond this point: numLods × (4 + 12 + 4)
  if (buf.byteLength < p + numLods * 20) {
    throw new Error(`.index truncated: ${numLods} LODs but only ${buf.byteLength} bytes`);
  }
  const lodScales: number[] = [];
  for (let i = 0; i < numLods; i++) {
    lodScales.push(dv.getFloat32(p, true));
    p += 4;
  }
  const vertexOffsets: [number, number, number][] = [];
  for (let i = 0; i < numLods; i++) {
    vertexOffsets.push([
      dv.getFloat32(p, true),
      dv.getFloat32(p + 4, true),
      dv.getFloat32(p + 8, true),
    ]);
    p += 12;
  }
  const numFragmentsPerLod: number[] = [];
  let totalFragments = 0;
  for (let i = 0; i < numLods; i++) {
    const n = dv.getUint32(p, true);
    numFragmentsPerLod.push(n);
    totalFragments += n;
    p += 4;
  }
  if (buf.byteLength !== p + totalFragments * 16) {
    throw new Error(
      `.index size mismatch: header expects ${p + totalFragments * 16} bytes for ` +
        `${numLods} LODs × ${totalFragments} fragments, got ${buf.byteLength}`,
    );
  }
  const fragmentsByLod: MultilodFragment[][] = [];
  let cumOffset = 0;
  for (let lod = 0; lod < numLods; lod++) {
    const n = numFragmentsPerLod[lod];
    // Column-major: x[0..N], y[0..N], z[0..N], size[0..N]
    const xs = new Uint32Array(buf, p, n);
    const ys = new Uint32Array(buf, p + n * 4, n);
    const zs = new Uint32Array(buf, p + n * 8, n);
    const sizes = new Uint32Array(buf, p + n * 12, n);
    p += n * 16;
    const frags: MultilodFragment[] = [];
    for (let i = 0; i < n; i++) {
      const length = sizes[i];
      frags.push({
        gridX: xs[i],
        gridY: ys[i],
        gridZ: zs[i],
        offset: cumOffset,
        length,
      });
      cumOffset += length;
    }
    fragmentsByLod.push(frags);
  }
  return { chunkShape, gridOrigin, vertexOffsets, lodScales, fragmentsByLod };
}

// Pick the finest LOD that actually has fragments for this segment.
// LOD 0 is the finest; we step up if it's empty.
function pickFinestLod(manifest: MultilodManifest): number {
  for (let lod = 0; lod < manifest.fragmentsByLod.length; lod++) {
    if (manifest.fragmentsByLod[lod].some((f) => f.length > 0)) {
      return lod;
    }
  }
  return -1;
}

export interface MultilodOptions {
  // Required: vertex quantization bits from the mesh info (10 or 16).
  vertexQuantizationBits: number;
  // Optional 3x4 row-major transform from native coords → nm.
  transform?: number[];
  // When present, the segment's manifest + fragments live inside
  // packed shard files rather than per-segment <id> / <id>.index
  // files. The loader will route through the sharded reader.
  sharding?: ShardingSpec;
}

// Fetch + decode one segment's multilod_draco mesh. Returns a single
// ParsedMesh in nm world coords, faces rebased across concatenated
// fragments. Picks the finest non-empty LOD.
//
// Two on-disk layouts:
//
//   Unsharded:
//     <base>/<seg_id>.index — manifest (raw bytes)
//     <base>/<seg_id>       — concatenated fragment data
//   Sharded:
//     <base>/<hex>.shard    — packs many segments' manifests + fragments
//     The minishard index points at the segment's manifest blob
//     (gzip-decoded). The fragment data sits immediately BEFORE the
//     manifest in the same shard file (raw, NOT gzipped), at addresses
//     given by manifest offsets. We range-fetch from the shard file.
export async function fetchMultilodMesh(
  base: string,
  segmentId: string,
  opts: MultilodOptions,
): Promise<ParsedMesh> {
  const httpsBase = precomputedToHttps(base).replace(/\/$/, "");

  // Variables that differ between sharded / unsharded paths:
  //   manifest      — parsed .index contents
  //   fragmentSrc   — { kind: "file", url } | { kind: "shard", url, manifestRawEnd }
  //                   For "shard", manifestRawEnd is the absolute byte
  //                   offset in the shard file where the manifest begins
  //                   (equivalently, where the fragment data ends).
  let manifestBuf: ArrayBuffer;
  let fragmentSrc:
    | { kind: "file"; url: string }
    | { kind: "shard"; url: string; manifestRawEnd: number };

  if (opts.sharding) {
    const key = BigInt(segmentId);
    const got = await fetchShardedChunk(httpsBase, key, opts.sharding);
    if (!got) {
      throw new Error(
        `Sharded multilod segment ${segmentId} not present in any minishard at ${httpsBase}.`,
      );
    }
    manifestBuf = got.data;
    fragmentSrc = {
      kind: "shard",
      url: got.location.shardUrl,
      manifestRawEnd: got.location.rawOffset,
    };
  } else {
    const indexUrl = `${httpsBase}/${segmentId}.index`;
    const dataUrl = `${httpsBase}/${segmentId}`;
    const indexRes = await fetch(indexUrl);
    if (!indexRes.ok) {
      throw new Error(`multilod .index ${indexRes.status} at ${indexUrl}`);
    }
    manifestBuf = await indexRes.arrayBuffer();
    fragmentSrc = { kind: "file", url: dataUrl };
  }

  const manifest = parseMultilodManifest(manifestBuf);

  const lod = pickFinestLod(manifest);
  if (lod < 0) {
    throw new Error(`Segment ${segmentId} has no mesh fragments at any LOD.`);
  }
  const fragments = manifest.fragmentsByLod[lod].filter((f) => f.length > 0);
  if (fragments.length === 0) {
    throw new Error(`Segment ${segmentId} has only empty fragments at LOD ${lod}.`);
  }

  // Range-fetch covers all chosen-LOD fragments in one request — fragment
  // offsets within the file are cumulative across all LODs, so we use the
  // first fragment's offset and the last fragment's end.
  //
  // Manifest offsets are RELATIVE to the start of the fragment data:
  //   - unsharded: that's start of <seg_id>, so absolute = offset
  //   - sharded:   that's manifestRawEnd - fullDataSize in the shard
  //     file. We compute fullDataSize as the cumulative length across
  //     ALL LODs (matches neuroglancer's downloadFragment math), then
  //     adjust each fragment's offset/length by that base.
  const fullDataSize = manifest.fragmentsByLod
    .flat()
    .reduce((s, f) => s + f.length, 0);
  const baseInSource =
    fragmentSrc.kind === "shard" ? fragmentSrc.manifestRawEnd - fullDataSize : 0;
  const rangeStart = baseInSource + fragments[0].offset;
  const rangeEnd =
    baseInSource +
    fragments[fragments.length - 1].offset +
    fragments[fragments.length - 1].length;
  let allBuf: ArrayBuffer;
  if (fragmentSrc.kind === "shard") {
    allBuf = await fetchRawShardBytes(fragmentSrc.url, rangeStart, rangeEnd);
  } else {
    const dataRes = await fetch(fragmentSrc.url, {
      headers: { Range: `bytes=${rangeStart}-${rangeEnd - 1}` },
    });
    if (!dataRes.ok && dataRes.status !== 206) {
      throw new Error(`multilod data ${dataRes.status} at ${fragmentSrc.url}`);
    }
    allBuf = await dataRes.arrayBuffer();
    if (dataRes.status === 200 && allBuf.byteLength > rangeEnd - rangeStart) {
      allBuf = allBuf.slice(rangeStart, rangeEnd);
    }
  }
  // After this point, slices are taken from allBuf starting at offset
  // `(frag.offset + baseInSource) - rangeStart` — i.e. each fragment's
  // absolute source position MINUS where we started the fetch.
  const sourceFetchStart = rangeStart;

  const scale = 1 << lod;
  const [cx, cy, cz] = manifest.chunkShape;
  const [gx, gy, gz] = manifest.gridOrigin;
  const [vox, voy, voz] = manifest.vertexOffsets[lod];
  const fragShapeX = cx * scale;
  const fragShapeY = cy * scale;
  const fragShapeZ = cz * scale;
  const qMax = (1 << opts.vertexQuantizationBits) - 1;

  // Decode each fragment SERIALLY. The neuroglancer draco wasm uses a
  // module-level decodeResult slot, so concurrent decodes would race.
  const parts: ParsedMesh[] = [];
  for (const frag of fragments) {
    const sliceStart = baseInSource + frag.offset - sourceFetchStart;
    const sliceEnd = sliceStart + frag.length;
    if (sliceEnd > allBuf.byteLength) {
      throw new Error(
        `multilod fragment range [${sliceStart}, ${sliceEnd}) exceeds buffer ${allBuf.byteLength}`,
      );
    }
    const fragBuf = new Uint8Array(allBuf, sliceStart, frag.length);
    const decoded = await decodeDracoPartitioned(
      fragBuf,
      opts.vertexQuantizationBits,
      false, // partition=false: we don't need the 8-octant subChunkOffsets
    );
    const numVertices = decoded.vertexPositions.length / 3;
    const numFaces = decoded.indices.length / 3;
    if (numVertices === 0 || numFaces === 0) continue;
    const fragOriginX = gx + frag.gridX * fragShapeX + vox;
    const fragOriginY = gy + frag.gridY * fragShapeY + voy;
    const fragOriginZ = gz + frag.gridZ * fragShapeZ + voz;
    const verts = new Float32Array(numVertices * 3);
    const q = decoded.vertexPositions; // Uint32Array, length 3*N
    for (let i = 0; i < numVertices; i++) {
      verts[i * 3] = fragOriginX + (q[i * 3] / qMax) * fragShapeX;
      verts[i * 3 + 1] = fragOriginY + (q[i * 3 + 1] / qMax) * fragShapeY;
      verts[i * 3 + 2] = fragOriginZ + (q[i * 3 + 2] / qMax) * fragShapeZ;
    }
    parts.push({
      vertices: verts,
      faces: new Uint32Array(decoded.indices), // copy out of decoder buffer
      numVertices,
      numFaces,
    });
  }

  if (parts.length === 0) {
    throw new Error(`Segment ${segmentId}: all LOD ${lod} fragments decoded to empty meshes.`);
  }

  const merged = concatMeshes(parts);
  // Apply optional info-level transform (native → nm) in place.
  if (opts.transform && opts.transform.length === 12) {
    applyTransform(merged.vertices, opts.transform);
  }
  return merged;
}

function applyTransform(verts: Float32Array, t: number[]): void {
  const isIdentity =
    t[0] === 1 &&
    t[1] === 0 &&
    t[2] === 0 &&
    t[3] === 0 &&
    t[4] === 0 &&
    t[5] === 1 &&
    t[6] === 0 &&
    t[7] === 0 &&
    t[8] === 0 &&
    t[9] === 0 &&
    t[10] === 1 &&
    t[11] === 0;
  if (isIdentity) return;
  const n = verts.length / 3;
  for (let i = 0; i < n; i++) {
    const x = verts[i * 3];
    const y = verts[i * 3 + 1];
    const z = verts[i * 3 + 2];
    verts[i * 3] = t[0] * x + t[1] * y + t[2] * z + t[3];
    verts[i * 3 + 1] = t[4] * x + t[5] * y + t[6] * z + t[7];
    verts[i * 3 + 2] = t[8] * x + t[9] * y + t[10] * z + t[11];
  }
}
