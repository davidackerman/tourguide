// Neuroglancer precomputed skeleton fetch + parse, used by python_on_layers
// to expose skeletons as numpy arrays in the analysis worker.
//
// Format reference:
//   <base>/info — JSON {"@type": "neuroglancer_skeletons",
//                       "transform": [12 floats, optional row-major 3x4],
//                       "vertex_attributes": [...], "spatial_index": {...}}
//   <base>/<segment_id> — binary:
//       num_vertices: uint32 LE
//       num_edges:    uint32 LE
//       vertices:     float32[num_vertices, 3]   (in skeleton native coords)
//       edges:        uint32[num_edges, 2]
//       (vertex_attributes follow if declared in info)
//
// We always materialize vertices in nm world coords (apply info.transform
// if present) so the agent / user code can reason in physical units
// without remembering to transform. Edges stay as uint32 indices.

export interface SkeletonInfo {
  // Number of triplets returned in `vertices` per skeleton (always 3 here).
  vertexDim: 3;
  // Row-major 3x4 transform from skeleton native coords → nm. Identity
  // when info.transform is absent.
  transform: Float64Array;
  // Vertex attributes after the (vertices, edges) chunk — we don't read
  // these but we need their byte size to skip past on parse.
  attributeBytesPerVertex: number;
}

export interface ParsedSkeleton {
  vertices: Float32Array; // shape (num_vertices, 3) in nm, flat
  edges: Uint32Array; // shape (num_edges, 2), flat
  numVertices: number;
  numEdges: number;
}

// Strip neuroglancer-style scheme prefix, translate bucket protocols
// (gs:// / s3://) to fetchable https URLs, and drop any trailing
// slash so we can build subpath URLs cleanly.
//
// Examples:
//   precomputed://gs://bucket/path        → https://storage.googleapis.com/bucket/path
//   precomputed://https://janelia/.../sk  → https://janelia/.../sk
//   gs://bucket/path                       → https://storage.googleapis.com/bucket/path
//   https://janelia/.../skeleton          → unchanged
export function normalizeSkeletonBase(source: string): string {
  let u = source.replace(/^precomputed:\/\//, "");
  if (u.startsWith("gs://")) {
    u = `https://storage.googleapis.com/${u.slice("gs://".length)}`;
  } else if (u.startsWith("s3://")) {
    u = `https://s3.amazonaws.com/${u.slice("s3://".length)}`;
  }
  return u.replace(/\/$/, "");
}

export async function fetchSkeletonInfo(base: string): Promise<SkeletonInfo> {
  const url = `${base}/info`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`skeleton info ${res.status} at ${url}`);
  }
  const j = (await res.json()) as {
    "@type"?: string;
    transform?: number[];
    vertex_attributes?: { data_type?: string; num_components?: number }[];
  };
  if (j["@type"] && j["@type"] !== "neuroglancer_skeletons") {
    throw new Error(`Not a skeleton source (@type=${j["@type"]}) at ${url}`);
  }
  const transform = identityTransform();
  if (Array.isArray(j.transform) && j.transform.length === 12) {
    for (let i = 0; i < 12; i++) transform[i] = Number(j.transform[i]);
  }
  // Compute attribute footer bytes per vertex so the parse knows where
  // edges end. NG types: float32/uint8/uint32/etc — only float32 +
  // uint32 + uint16 + uint8 are common. Each attribute contributes
  // num_components × bytes_per_element per vertex.
  let attributeBytesPerVertex = 0;
  for (const a of j.vertex_attributes ?? []) {
    const nc = Number(a.num_components ?? 1);
    const bytes = bytesForDtype(String(a.data_type ?? "float32"));
    attributeBytesPerVertex += nc * bytes;
  }
  return { vertexDim: 3, transform, attributeBytesPerVertex };
}

export async function fetchSkeleton(
  base: string,
  segmentId: string,
  info: SkeletonInfo,
): Promise<ParsedSkeleton> {
  const url = `${base}/${segmentId}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`skeleton ${segmentId} ${res.status} at ${url}`);
  }
  const buf = await res.arrayBuffer();
  return parseSkeletonBuffer(buf, info);
}

export function parseSkeletonBuffer(
  buf: ArrayBuffer,
  info: SkeletonInfo,
): ParsedSkeleton {
  const view = new DataView(buf);
  if (buf.byteLength < 8) {
    throw new Error("Skeleton file too short for header");
  }
  const numVertices = view.getUint32(0, true);
  const numEdges = view.getUint32(4, true);
  const vBytes = numVertices * 12;
  const eBytes = numEdges * 8;
  const expected = 8 + vBytes + eBytes + numVertices * info.attributeBytesPerVertex;
  if (buf.byteLength < expected) {
    throw new Error(
      `Skeleton truncated: expected ${expected} bytes, got ${buf.byteLength} (V=${numVertices} E=${numEdges})`,
    );
  }
  // Slice into typed views — copy because the underlying buffer may be
  // larger than the chunk we care about, and we need standalone arrays
  // to bind to Python globals.
  const verticesNative = new Float32Array(buf.slice(8, 8 + vBytes));
  const edges = new Uint32Array(buf.slice(8 + vBytes, 8 + vBytes + eBytes));
  // Apply the transform → nm. We do this here once, on the JS side, so
  // Python sees clean nm coordinates and the model never has to think
  // about it.
  const verticesNm = applyTransform(verticesNative, info.transform);
  return {
    vertices: verticesNm,
    edges,
    numVertices,
    numEdges,
  };
}

// 3x4 row-major transform. Identity == [1,0,0,0, 0,1,0,0, 0,0,1,0].
function identityTransform(): Float64Array {
  const t = new Float64Array(12);
  t[0] = 1;
  t[5] = 1;
  t[10] = 1;
  return t;
}

function applyTransform(verts: Float32Array, t: Float64Array): Float32Array {
  // Identity skip — common case for Janelia/cellmap skeletons that are
  // already exported in nm. Avoids the multiply-add hot loop.
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
  if (isIdentity) return verts;
  const n = verts.length / 3;
  const out = new Float32Array(verts.length);
  for (let i = 0; i < n; i++) {
    const x = verts[i * 3];
    const y = verts[i * 3 + 1];
    const z = verts[i * 3 + 2];
    out[i * 3] = t[0] * x + t[1] * y + t[2] * z + t[3];
    out[i * 3 + 1] = t[4] * x + t[5] * y + t[6] * z + t[7];
    out[i * 3 + 2] = t[8] * x + t[9] * y + t[10] * z + t[11];
  }
  return out;
}

function bytesForDtype(dtype: string): number {
  switch (dtype.toLowerCase()) {
    case "float32":
      return 4;
    case "float64":
      return 8;
    case "uint8":
    case "int8":
      return 1;
    case "uint16":
    case "int16":
      return 2;
    case "uint32":
    case "int32":
      return 4;
    case "uint64":
    case "int64":
      return 8;
    default:
      // Unknown dtype: assume 4 bytes (float32 default). Safer to over-
      // assume than to under-skip and read garbage as the next field.
      return 4;
  }
}
