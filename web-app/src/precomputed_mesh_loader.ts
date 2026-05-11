// Neuroglancer precomputed mesh fetch + parse, used by python_on_layers
// to expose meshes as numpy arrays in the analysis worker.
//
// Currently supported on-disk format:
//
//   neuroglancer_legacy_mesh (unsharded)
//     <base>/info                       — { "@type": "neuroglancer_legacy_mesh" }
//     <base>/<seg_id>:0                 — JSON manifest, e.g. {"fragments": ["a","b"]}
//     <base>/<fragment_filename>        — binary:
//                                            num_vertices: u32 LE
//                                            vertices:     f32[num_vertices, 3]
//                                            faces:        u32[num_faces, 3]
//                                                          (fills the rest of the file)
//
// NOT YET supported:
//
//   neuroglancer_multilod_draco — multi-LOD, Draco-compressed fragments.
//     Needs a Draco decoder; left for a follow-up.
//   sharded variants — both legacy and multilod can be packed into shard
//     files for big datasets. Sharded readers add significant complexity.
//
// Vertex transform: legacy mesh vertices are typically already in world-nm
// for Janelia / cellmap convention (no per-mesh transform stored in info).
// If the dataset stores them in voxel units, the caller should apply the
// segmentation's spacing externally — we don't try to autodetect.

import { precomputedToHttps } from "./precomputed_info.js";

export interface ParsedMesh {
  vertices: Float32Array; // (N, 3) flat, world-nm-ish (source's native frame)
  faces: Uint32Array; // (M, 3) flat, into vertices
  numVertices: number;
  numFaces: number;
}

export function normalizeMeshBase(source: string): string {
  return precomputedToHttps(source).replace(/\/$/, "");
}

// Fetch one segment's legacy mesh by walking its manifest + fragments.
// Returns a single ParsedMesh with all fragments concatenated (face
// indices rebased so they point at the merged vertex array).
export async function fetchLegacyMesh(
  base: string,
  segmentId: string,
): Promise<ParsedMesh> {
  const httpsBase = normalizeMeshBase(base);
  const manifestUrl = `${httpsBase}/${segmentId}:0`;
  const manifestRes = await fetch(manifestUrl);
  if (!manifestRes.ok) {
    throw new Error(`mesh manifest ${manifestRes.status} at ${manifestUrl}`);
  }
  const manifest = (await manifestRes.json()) as { fragments?: unknown };
  if (!Array.isArray(manifest.fragments) || manifest.fragments.length === 0) {
    throw new Error(`mesh manifest at ${manifestUrl} has no fragments`);
  }
  // Fetch fragments in parallel — they're usually a handful per segment,
  // and serial fetches dominate latency on big mesh trees.
  const buffers = await Promise.all(
    (manifest.fragments as unknown[]).map(async (filename) => {
      const name = String(filename);
      const url = `${httpsBase}/${name}`;
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`mesh fragment ${name} ${res.status} at ${url}`);
      }
      return res.arrayBuffer();
    }),
  );
  const parts = buffers.map(parseLegacyMeshFragment);
  return concatMeshes(parts);
}

// Parse one legacy mesh fragment. Header is num_vertices (u32 LE) then
// vertices as float32 triples; the rest of the file is uint32 triples
// for the face indices.
export function parseLegacyMeshFragment(buf: ArrayBuffer): ParsedMesh {
  if (buf.byteLength < 4) {
    throw new Error("Legacy mesh fragment too short for header");
  }
  const dv = new DataView(buf);
  const numVertices = dv.getUint32(0, true);
  const vertBytes = numVertices * 3 * 4;
  if (4 + vertBytes > buf.byteLength) {
    throw new Error(
      `Legacy mesh fragment truncated: header says ${numVertices} vertices ` +
        `(needs ${vertBytes} bytes) but only ${buf.byteLength - 4} bytes available`,
    );
  }
  const indexBytes = buf.byteLength - 4 - vertBytes;
  if (indexBytes % 12 !== 0) {
    throw new Error(
      `Legacy mesh fragment index region (${indexBytes} bytes) is not a multiple of 12 ` +
        `(3 uint32 per triangle).`,
    );
  }
  const numFaces = indexBytes / 12;
  const vertices = new Float32Array(buf.slice(4, 4 + vertBytes));
  const faces = new Uint32Array(buf.slice(4 + vertBytes));
  return { vertices, faces, numVertices, numFaces };
}

// Concatenate multiple fragments into one ParsedMesh. Each fragment's
// face indices are rebased by the running vertex offset so they point
// at the right entries in the merged vertex array.
export function concatMeshes(parts: ParsedMesh[]): ParsedMesh {
  if (parts.length === 1) return parts[0];
  let totalV = 0;
  let totalF = 0;
  for (const p of parts) {
    totalV += p.numVertices;
    totalF += p.numFaces;
  }
  const vertices = new Float32Array(totalV * 3);
  const faces = new Uint32Array(totalF * 3);
  let vCursor = 0;
  let fCursor = 0;
  let vBase = 0;
  for (const p of parts) {
    vertices.set(p.vertices, vCursor * 3);
    for (let i = 0; i < p.faces.length; i++) {
      faces[fCursor * 3 + i] = p.faces[i] + vBase;
    }
    vCursor += p.numVertices;
    fCursor += p.numFaces;
    vBase += p.numVertices;
  }
  return { vertices, faces, numVertices: totalV, numFaces: totalF };
}
