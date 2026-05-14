// Lightweight reader for Neuroglancer precomputed info files.
//
// NG's precomputed segmentation layout (e.g. hemibrain) bundles
// volume + meshes + skeletons in one directory:
//
//   <base>/info                       — top-level segmentation metadata
//   <base>/<scale>/<chunk>            — volume chunks
//   <base>/<info.mesh>/info           — mesh format metadata
//   <base>/<info.mesh>/<seg_id>:0     — mesh fragments (legacy / multilod)
//   <base>/<info.skeletons>/info      — skeleton format metadata
//   <base>/<info.skeletons>/<seg_id>  — skeleton binary
//
// Tourguide doesn't yet read precomputed VOLUMES into Python (no
// browser-side reader for the chunk format), but the mesh / skeleton
// sub-resources are accessible — we just need to discover where they
// live by reading the top-level info file. Without that probe, a
// layer pasted as `precomputed://gs://.../segmentation` looks like a
// volume-only layer with no bundled skeletons; with the probe, we
// learn there's a `skeletons` subdir and can route skeleton queries
// to it.

export interface PrecomputedSegmentationInfo {
  // Subpath (relative to base) for meshes — typically "mesh" or
  // "meshes". Absent / empty string when no meshes are bundled.
  mesh?: string;
  // Subpath for skeletons — typically "skeletons". Absent / empty
  // when no skeletons are bundled.
  skeletons?: string;
  // Subpath for segment_properties — typically "segment_properties".
  // When present, that subdir's info JSON enumerates all segment IDs
  // (cheaply, since it's just a JSON list — no shard decode required).
  segmentProperties?: string;
  // Raw info JSON for callers that need more (data_type, scales, ...).
  raw: Record<string, unknown>;
}

// Translate a Neuroglancer source URL into an https URL the browser
// can fetch directly. NG's precomputed:// scheme is followed by the
// actual transport (gs://, s3://, http://, https://):
//   precomputed://gs://bucket/path  → https://storage.googleapis.com/bucket/path
//   precomputed://s3://bucket/path  → https://s3.amazonaws.com/bucket/path
//   precomputed://https://...       → https://...
//   precomputed://http://...        → http://...
//   gs://bucket/path                → https://storage.googleapis.com/bucket/path
//   (bare https://...)              → unchanged
// CORS depends on the bucket's policy; public Janelia / cellmap /
// google buckets generally allow it. Private buckets won't.
export function precomputedToHttps(url: string): string {
  let u = url.trim();
  if (u.startsWith("precomputed://")) u = u.slice("precomputed://".length);
  if (u.startsWith("gs://")) {
    const path = u.slice("gs://".length);
    return `https://storage.googleapis.com/${path}`;
  }
  if (u.startsWith("s3://")) {
    const path = u.slice("s3://".length);
    return `https://s3.amazonaws.com/${path}`;
  }
  // http:// or https:// — leave alone. Anything else (custom scheme):
  // leave alone too; caller will get a fetch error which is fine.
  return u;
}

// In-memory cache so repeated describe_dataset / skeleton-resolver
// calls in the same session don't refetch the info JSON. Keyed by
// the http(s)-normalized base URL.
const INFO_CACHE = new Map<string, Promise<PrecomputedSegmentationInfo | null>>();

export function clearPrecomputedInfoCache(): void {
  INFO_CACHE.clear();
}

// Fetch the precomputed info file at `<base>/info` and return the
// discovered mesh / skeletons subpaths. Returns null when the base
// isn't a precomputed source, the fetch fails, or the JSON doesn't
// look like an NG segmentation info. Never throws — callers treat
// "no info" as "no bundled sub-resources".
export async function probePrecomputedInfo(
  base: string,
): Promise<PrecomputedSegmentationInfo | null> {
  if (!base) return null;
  // Strip any precomputed:// prefix + trailing slash, then map to
  // https for fetching.
  const httpsBase = precomputedToHttps(base).replace(/\/$/, "");
  if (!/^https?:\/\//i.test(httpsBase)) return null;
  const cached = INFO_CACHE.get(httpsBase);
  if (cached) return cached;
  const promise = (async (): Promise<PrecomputedSegmentationInfo | null> => {
    try {
      const res = await fetch(`${httpsBase}/info`);
      if (!res.ok) return null;
      const raw = (await res.json()) as Record<string, unknown>;
      // Sanity check: NG segmentations have @type =
      // "neuroglancer_multiscale_volume" with type "segmentation".
      // Some hand-written info files omit @type but still have a
      // `type: "segmentation"` field; accept either.
      const atType = String(raw["@type"] ?? "");
      const typeField = String(raw["type"] ?? "");
      if (
        atType !== "neuroglancer_multiscale_volume" &&
        atType !== "" &&
        typeField !== "segmentation" &&
        typeField !== "image"
      ) {
        return null;
      }
      const meshRaw = raw["mesh"];
      const skelRaw = raw["skeletons"];
      const propsRaw = raw["segment_properties"];
      return {
        mesh: typeof meshRaw === "string" && meshRaw ? meshRaw : undefined,
        skeletons: typeof skelRaw === "string" && skelRaw ? skelRaw : undefined,
        segmentProperties:
          typeof propsRaw === "string" && propsRaw ? propsRaw : undefined,
        raw,
      };
    } catch {
      return null;
    }
  })();
  INFO_CACHE.set(httpsBase, promise);
  return promise;
}

// Convenience: resolve the absolute URL of a bundled sub-resource.
// Returns null when the sub-resource isn't declared. The caller can
// then use this URL as a precomputed-skeleton or precomputed-mesh
// source for further reads.
export async function resolveBundledSubpath(
  base: string,
  kind: "mesh" | "skeletons",
): Promise<string | null> {
  const info = await probePrecomputedInfo(base);
  if (!info) return null;
  const sub = kind === "mesh" ? info.mesh : info.skeletons;
  if (!sub) return null;
  // Strip the precomputed:// prefix if present for the join, then
  // re-attach so downstream loaders that key on the prefix still
  // recognize it as precomputed.
  const trimmed = base.replace(/\/$/, "");
  return `${trimmed}/${sub}`;
}

// Mesh-specific info file (one level down from the segmentation, at
// <base>/<info.mesh>/info). Different shape from the segmentation
// info: declares the on-disk format the mesh chunks use.
//   @type: "neuroglancer_legacy_mesh"        — single fragment per
//          segment, simple binary (num_v u32 + vertices f32 +
//          indices u32). Easy to parse.
//   @type: "neuroglancer_multilod_draco"     — multi-LOD draco-
//          compressed fragments per segment, with manifests. Needs
//          a draco decoder.
//   sharding: { ... }                        — when present, all of
//          the above is packed into shard files for efficiency
//          (used by big datasets like hemibrain). Sharded readers
//          are even more complex.
export interface MeshInfo {
  atType: string;
  // For multilod_draco: vertex quantization bits, transform, etc.
  // Stored opaquely; the mesh loader reads what it needs.
  raw: Record<string, unknown>;
  isSharded: boolean;
}

const MESH_INFO_CACHE = new Map<string, Promise<MeshInfo | null>>();

// Cached probe of a precomputed segment_properties/info file. When the
// info is "inline" form (the usual case for hand-curated catalogs like
// hemibrain), the response includes an explicit `ids` list — perfect
// for cheap "how many neurons" / "what's segment X's body type" queries
// WITHOUT having to read any shard files. Returns null on miss or any
// network / parse failure.
export interface SegmentProperties {
  numSegments: number;       // total declared IDs
  ids: string[];             // full list when inline; capped at 50k to bound memory
  truncated: boolean;        // true when ids[] was truncated
  // Property labels declared in the info — useful for the agent to
  // know what attributes are queryable (e.g. ["status", "type"]).
  propertyLabels: string[];
}

const SEGMENT_PROPS_CACHE = new Map<string, Promise<SegmentProperties | null>>();

export function clearPrecomputedSegmentPropertiesCache(): void {
  SEGMENT_PROPS_CACHE.clear();
}

export async function fetchSegmentProperties(
  propsBase: string,
): Promise<SegmentProperties | null> {
  if (!propsBase) return null;
  const httpsBase = precomputedToHttps(propsBase).replace(/\/$/, "");
  if (!/^https?:\/\//i.test(httpsBase)) return null;
  const cached = SEGMENT_PROPS_CACHE.get(httpsBase);
  if (cached) return cached;
  const promise = (async (): Promise<SegmentProperties | null> => {
    try {
      const res = await fetch(`${httpsBase}/info`);
      if (!res.ok) return null;
      const raw = (await res.json()) as Record<string, unknown>;
      const inline = raw["inline"] as Record<string, unknown> | undefined;
      if (!inline || typeof inline !== "object") return null;
      const idsRaw = inline["ids"];
      if (!Array.isArray(idsRaw)) return null;
      const numSegments = idsRaw.length;
      const ID_CAP = 50_000;
      const truncated = numSegments > ID_CAP;
      const ids: string[] = (truncated ? idsRaw.slice(0, ID_CAP) : idsRaw)
        .map((v) => String(v));
      const propsRaw = inline["properties"];
      const propertyLabels: string[] = [];
      if (Array.isArray(propsRaw)) {
        for (const p of propsRaw as unknown[]) {
          if (p && typeof p === "object") {
            const id = (p as Record<string, unknown>)["id"];
            if (typeof id === "string") propertyLabels.push(id);
          }
        }
      }
      return { numSegments, ids, truncated, propertyLabels };
    } catch {
      return null;
    }
  })();
  SEGMENT_PROPS_CACHE.set(httpsBase, promise);
  return promise;
}

export async function probeMeshInfo(meshBase: string): Promise<MeshInfo | null> {
  if (!meshBase) return null;
  const httpsBase = precomputedToHttps(meshBase).replace(/\/$/, "");
  if (!/^https?:\/\//i.test(httpsBase)) return null;
  const cached = MESH_INFO_CACHE.get(httpsBase);
  if (cached) return cached;
  const promise = (async (): Promise<MeshInfo | null> => {
    try {
      const res = await fetch(`${httpsBase}/info`);
      if (!res.ok) return null;
      const raw = (await res.json()) as Record<string, unknown>;
      const atType = String(raw["@type"] ?? "");
      const isSharded = raw["sharding"] !== undefined && raw["sharding"] !== null;
      return { atType, raw, isSharded };
    } catch {
      return null;
    }
  })();
  MESH_INFO_CACHE.set(httpsBase, promise);
  return promise;
}
