// Neuroglancer precomputed `neuroglancer_uint64_sharded_v1` reader.
//
// This is the format big segmentations (hemibrain, FAFB, MICrONS, ...)
// use to pack millions of per-segment chunks (mesh manifests, mesh
// fragments, skeletons, volume chunks) into a handful of `.shard`
// files for storage efficiency. Without it we couldn't analyze
// hemibrain-scale data at all.
//
// Spec reference (Google): the precomputed sharded format spec lives in
// the neuroglancer repo (datasource/precomputed/sharded.md). We
// implement just enough for the mesh / skeleton case:
//
//   sharding spec (from the info file's `sharding` field):
//     hash: "identity" | "murmurhash3_x86_128"
//     preshift_bits: how many low bits to strip from segment_id before
//                    hashing (so consecutive IDs land in different shards)
//     shard_bits: log2(num shard files) — shard files are named by hex
//                 of the top shard_bits of the hash
//     minishard_bits: log2(num minishards per shard) — chosen by the
//                 next minishard_bits of the hash
//     data_encoding / minishard_index_encoding: "raw" | "gzip"
//
//   shard file layout:
//     [shard_index]                       (16 * 2^minishard_bits bytes)
//     [data_section]                      (variable)
//
//   shard_index: 2^minishard_bits pairs of (start, end) u64 LE giving
//     byte offsets within data_section where each minishard's index
//     lives. start == end means "empty minishard".
//
//   minishard_index (at data_section[start..end], optionally gzipped):
//     N entries, laid out column-major as 3 columns of N u64 LE:
//       column 0: delta_id   — cumulative-sum gives segment IDs
//       column 1: delta_start— cumulative-sum gives data offsets
//                              (initial prevStart = shardIndexSize)
//       column 2: size       — chunk byte length
//
//   data chunk (at file offset start, length size, optionally gzipped):
//     the actual payload for the segment (a mesh manifest, a skeleton,
//     a volume chunk, ...). For multilod_draco mesh specifically, the
//     payload is JUST the `.index` manifest, and the fragment data is
//     packed into the same shard file IMMEDIATELY BEFORE the manifest
//     (raw, not under any chunk index entry). Callers that need
//     fragments use `shardUrl + rawOffset - fullDataSize` to address
//     them — see fetchShardedChunk's returned shardUrl/rawOffset.
//
// We deliberately reimplement the small bits of neuroglancer's
// sharded.js / hash.js logic that we need (murmurhash3 is reused from
// the package — small and pure). Pulling in their full ShardedKvStore
// would drag along the ChunkManager / RPC machinery, which is huge.

import { murmurHash3_x86_128Hash64Bits_Bigint } from "neuroglancer/unstable/util/hash.js";

export interface ShardingSpec {
  hash: "identity" | "murmurhash3_x86_128";
  preshiftBits: number;
  shardBits: number;
  minishardBits: number;
  minishardIndexEncoding: "raw" | "gzip";
  dataEncoding: "raw" | "gzip";
}

// Parse a `sharding` block from a precomputed info file into our
// internal shape. Throws on unknown hash / encoding; returns null
// when the input doesn't look like a sharding spec at all (so callers
// can do `parseSharding(info.sharding) ?? null` cheaply).
export function parseSharding(raw: unknown): ShardingSpec | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  if (r["@type"] !== "neuroglancer_uint64_sharded_v1") return null;
  const hashStr = String(r.hash ?? "identity");
  if (hashStr !== "identity" && hashStr !== "murmurhash3_x86_128") {
    throw new Error(`Unsupported sharding hash '${hashStr}'`);
  }
  const enc = (v: unknown, name: string): "raw" | "gzip" => {
    const s = String(v ?? "raw");
    if (s !== "raw" && s !== "gzip") {
      throw new Error(`Unsupported sharding ${name} encoding '${s}'`);
    }
    return s;
  };
  return {
    hash: hashStr,
    preshiftBits: Number(r.preshift_bits ?? 0),
    shardBits: Number(r.shard_bits ?? 0),
    minishardBits: Number(r.minishard_bits ?? 0),
    minishardIndexEncoding: enc(r.minishard_index_encoding, "minishard_index"),
    dataEncoding: enc(r.data_encoding, "data"),
  };
}

// Browser-native gzip decode via DecompressionStream — supported in
// Chrome 80+, Firefox 113+, Safari 16.4+. We use it for the (small)
// minishard index reads and the (medium) chunk reads.
async function ungzip(buf: ArrayBuffer): Promise<ArrayBuffer> {
  const stream = new Blob([buf])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"));
  return await new Response(stream).arrayBuffer();
}

// Range-fetch [start, end) bytes from a URL. We pass `end - 1` since
// HTTP Range is INCLUSIVE on both ends. GCS / S3 return 206 with
// exactly the requested bytes; a few CDNs ignore Range and return 200
// with the full file — handled by callers slicing from absolute start.
async function fetchRange(url: string, start: number, end: number): Promise<ArrayBuffer> {
  const res = await fetch(url, {
    headers: { Range: `bytes=${start}-${end - 1}` },
  });
  if (!res.ok && res.status !== 206) {
    throw new Error(`shard fetch ${res.status} at ${url} bytes=${start}-${end - 1}`);
  }
  const buf = await res.arrayBuffer();
  if (res.status === 200 && buf.byteLength > end - start) {
    // Server returned full file. Slice to the requested window.
    return buf.slice(start, end);
  }
  return buf;
}

function hashCode(key: bigint, sharding: ShardingSpec): bigint {
  const shifted = key >> BigInt(sharding.preshiftBits);
  if (sharding.hash === "identity") return shifted;
  // 0n is the standard seed used by neuroglancer.
  return murmurHash3_x86_128Hash64Bits_Bigint(0, shifted);
}

// Tiny per-process cache for already-fetched minishard indices —
// looking up many segments in a query will repeatedly hit the same
// (shard, minishard) pair. Bytes are decoded into absolute-offset
// arrays so the lookup is constant-time after the first fetch.
interface MinishardIndex {
  ids: BigUint64Array;      // sorted-by-occurrence; not necessarily ascending
  starts: BigUint64Array;   // absolute byte offsets in the shard file
  ends: BigUint64Array;     // absolute byte offsets in the shard file
}
const MINISHARD_CACHE = new Map<string, Promise<MinishardIndex | null>>();

export function clearShardedCache(): void {
  MINISHARD_CACHE.clear();
}

async function loadMinishardIndex(
  baseUrl: string,
  shard: bigint,
  minishard: bigint,
  sharding: ShardingSpec,
): Promise<MinishardIndex | null> {
  const shardHex = shard.toString(16).padStart(Math.ceil(sharding.shardBits / 4), "0");
  const shardUrl = `${baseUrl}/${shardHex}.shard`;
  const cacheKey = `${shardUrl}#${minishard.toString()}`;
  const cached = MINISHARD_CACHE.get(cacheKey);
  if (cached) return cached;
  const promise = (async (): Promise<MinishardIndex | null> => {
    // Shard index: one (start, end) u64 LE pair per minishard.
    // Read just the 16 bytes for THIS minishard.
    const shardIndexSize = BigInt(16) << BigInt(sharding.minishardBits);
    const headerStart = Number(minishard) * 16;
    const headerBuf = await fetchRange(shardUrl, headerStart, headerStart + 16);
    if (headerBuf.byteLength < 16) return null;
    const dv = new DataView(headerBuf);
    let miniStart = dv.getBigUint64(0, true);
    let miniEnd = dv.getBigUint64(8, true);
    if (miniStart === miniEnd) return null;
    // Offsets in the shard index are relative to the start of the
    // data section (= just after the shard index itself).
    miniStart += shardIndexSize;
    miniEnd += shardIndexSize;
    let miniBuf = await fetchRange(shardUrl, Number(miniStart), Number(miniEnd));
    if (sharding.minishardIndexEncoding === "gzip") {
      miniBuf = await ungzip(miniBuf);
    }
    if (miniBuf.byteLength % 24 !== 0) {
      throw new Error(
        `minishard index ${miniBuf.byteLength} bytes not divisible by 24`,
      );
    }
    const n = miniBuf.byteLength / 24;
    const cols = new BigUint64Array(miniBuf);
    // Decode delta-encoded columns into absolute id / start / end.
    // prevStart starts at shardIndexSize (per spec), matching what
    // neuroglancer's reference impl does.
    const ids = new BigUint64Array(n);
    const starts = new BigUint64Array(n);
    const ends = new BigUint64Array(n);
    let prevId = 0n;
    let prevStart = shardIndexSize;
    for (let i = 0; i < n; i++) {
      const id = prevId + cols[i];
      prevId = id;
      ids[i] = id;
      const start = prevStart + cols[n + i];
      const size = cols[2 * n + i];
      const end = start + size;
      starts[i] = start;
      ends[i] = end;
      prevStart = end;
    }
    return { ids, starts, ends };
  })();
  MINISHARD_CACHE.set(cacheKey, promise);
  return promise;
}

export interface ShardedLocation {
  shardUrl: string;     // absolute https url to the shard file
  rawOffset: number;    // byte position of chunk in the shard file (RAW, pre-decode)
  rawLength: number;    // byte length of chunk in the shard file (RAW, pre-decode)
}

// Look up where `key`'s chunk lives in the shard files. Returns null
// when the segment is not present (empty minishard or missing entry).
export async function locateShardedChunk(
  baseUrl: string,
  key: bigint,
  sharding: ShardingSpec,
): Promise<ShardedLocation | null> {
  const hc = hashCode(key, sharding);
  const totalBits = sharding.minishardBits + sharding.shardBits;
  const shardAndMinishard = hc & ((1n << BigInt(totalBits)) - 1n);
  const minishardMask = (1n << BigInt(sharding.minishardBits)) - 1n;
  const minishard = shardAndMinishard & minishardMask;
  const shard = shardAndMinishard >> BigInt(sharding.minishardBits);
  const idx = await loadMinishardIndex(baseUrl, shard, minishard, sharding);
  if (!idx) return null;
  for (let i = 0; i < idx.ids.length; i++) {
    if (idx.ids[i] === key) {
      const start = Number(idx.starts[i]);
      const end = Number(idx.ends[i]);
      const shardHex = shard.toString(16).padStart(Math.ceil(sharding.shardBits / 4), "0");
      return {
        shardUrl: `${baseUrl}/${shardHex}.shard`,
        rawOffset: start,
        rawLength: end - start,
      };
    }
  }
  return null;
}

// Fetch and decode (gzip if applicable) the chunk for `key`. Returns
// null when the segment isn't in the shard. Callers that need to
// address bytes OUTSIDE the returned chunk (e.g. fragment data that
// sits BEFORE the manifest in multilod sharded layout) should use
// `locateShardedChunk` instead and do raw byte-range fetches.
export async function fetchShardedChunk(
  baseUrl: string,
  key: bigint,
  sharding: ShardingSpec,
): Promise<{ data: ArrayBuffer; location: ShardedLocation } | null> {
  const location = await locateShardedChunk(baseUrl, key, sharding);
  if (!location) return null;
  let buf = await fetchRange(
    location.shardUrl,
    location.rawOffset,
    location.rawOffset + location.rawLength,
  );
  if (sharding.dataEncoding === "gzip") {
    buf = await ungzip(buf);
  }
  return { data: buf, location };
}

// Raw byte-range read from a shard file with no decoding. Used by the
// sharded multilod_draco loader to grab fragment payloads that sit
// just before the manifest in the same shard file.
export async function fetchRawShardBytes(
  shardUrl: string,
  start: number,
  end: number,
): Promise<ArrayBuffer> {
  return fetchRange(shardUrl, start, end);
}
