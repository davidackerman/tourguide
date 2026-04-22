// Fetch and parse dataset metadata from a Neuroglancer source URL.
// Supports the three common scheme prefixes (zarr://, n5://, precomputed://)
// over HTTP(S) and public s3:// backends.

export interface DetectedMetadata {
  voxel_size_nm: [number, number, number];
  size_voxels?: [number, number, number];
  center_nm?: [number, number, number];
  source: string;
  kind: "zarr" | "n5" | "precomputed";
  via: string; // short description of what was parsed
}

const NM_PER = {
  nm: 1,
  nanometer: 1,
  nanometers: 1,
  "um": 1000,
  "µm": 1000,
  micrometer: 1000,
  micrometers: 1000,
  micron: 1000,
  microns: 1000,
  mm: 1e6,
  millimeter: 1e6,
  millimeters: 1e6,
  m: 1e9,
  meter: 1e9,
  meters: 1e9,
} as const;

function toNm(value: number, unit: string | undefined): number {
  const u = (unit ?? "nm").toLowerCase();
  const factor = (NM_PER as Record<string, number>)[u];
  if (factor === undefined) {
    console.warn(`Unknown unit ${unit}, assuming nm`);
    return value;
  }
  return value * factor;
}

function parseScheme(src: string): { kind: DetectedMetadata["kind"]; base: string } {
  const m = src.match(/^(zarr|n5|precomputed):\/\/(.+)$/i);
  if (!m) throw new Error(`Not a Neuroglancer source URL (missing zarr://, n5://, or precomputed:// prefix): ${src}`);
  let base = m[2].trim();
  // Public S3 virtual-hosted-style — works for most public buckets, including janelia-cosem-datasets.
  const s3 = base.match(/^s3:\/\/([^/]+)\/(.*)$/);
  if (s3) {
    base = `https://${s3[1]}.s3.amazonaws.com/${s3[2]}`;
  }
  // Strip trailing slashes for uniform URL building.
  base = base.replace(/\/+$/, "");
  return { kind: m[1].toLowerCase() as DetectedMetadata["kind"], base };
}

async function fetchJson(url: string): Promise<unknown> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText} at ${url}`);
  return res.json();
}

async function tryJson(url: string): Promise<unknown | null> {
  try {
    return await fetchJson(url);
  } catch {
    return null;
  }
}

interface ZarrMultiscaleDataset {
  path?: string;
  coordinateTransformations?: Array<{ type: string; scale?: number[] }>;
}
interface ZarrMultiscale {
  axes?: Array<{ name: string; type?: string; unit?: string }>;
  datasets?: ZarrMultiscaleDataset[];
  coordinateTransformations?: Array<{ type: string; scale?: number[] }>;
}

function parseMultiscalesVoxelSize(
  ms: ZarrMultiscale,
): { voxel_size_nm: [number, number, number]; via: string } | null {
  const ds0 = ms.datasets?.[0];
  if (!ds0) return null;
  const scale0 =
    ds0.coordinateTransformations?.find((t) => t.type === "scale")?.scale ??
    ms.coordinateTransformations?.find((t) => t.type === "scale")?.scale;
  if (!scale0) return null;
  const axes = ms.axes ?? [];
  // Keep only spatial axes (skip time/channel).
  const spatial: Array<{ index: number; name: string; unit?: string }> = [];
  for (let i = 0; i < scale0.length; i++) {
    const ax = axes[i];
    if (!ax || !ax.type || ax.type === "space") {
      spatial.push({ index: i, name: ax?.name ?? "xyz"[spatial.length] ?? "?", unit: ax?.unit });
    }
  }
  if (spatial.length < 3) return null;
  // Take the last three spatial axes as (x, y, z) in the order they appear.
  // Neuroglancer/OME-Zarr typically orders them as z, y, x; we surface the
  // values in x, y, z order by looking at axis names when available.
  let vx = scale0[spatial[spatial.length - 1].index];
  let vy = scale0[spatial[spatial.length - 2].index];
  let vz = scale0[spatial[spatial.length - 3].index];
  const unit = (axes.find((a) => a.type === "space" || !a.type)?.unit ?? "nm") as string;
  [vx, vy, vz] = [toNm(vx, unit), toNm(vy, unit), toNm(vz, unit)];
  return {
    voxel_size_nm: [vx, vy, vz],
    via: `multiscales[0].coordinateTransformations (${unit})`,
  };
}

async function detectZarr(base: string): Promise<DetectedMetadata> {
  // Try v3 first.
  const v3 = (await tryJson(`${base}/zarr.json`)) as
    | { attributes?: { multiscales?: ZarrMultiscale[]; ome?: { multiscales?: ZarrMultiscale[] } } }
    | null;
  if (v3) {
    const ms =
      v3.attributes?.multiscales?.[0] ??
      v3.attributes?.ome?.multiscales?.[0];
    if (ms) {
      const parsed = parseMultiscalesVoxelSize(ms);
      if (parsed) {
        return {
          voxel_size_nm: parsed.voxel_size_nm,
          source: `zarr://${base}`,
          kind: "zarr",
          via: `zarr v3: ${parsed.via}`,
        };
      }
    }
  }
  // Try v2.
  const zattrs = (await tryJson(`${base}/.zattrs`)) as { multiscales?: ZarrMultiscale[] } | null;
  if (zattrs?.multiscales?.[0]) {
    const parsed = parseMultiscalesVoxelSize(zattrs.multiscales[0]);
    if (parsed) {
      return {
        voxel_size_nm: parsed.voxel_size_nm,
        source: `zarr://${base}`,
        kind: "zarr",
        via: `zarr v2 .zattrs: ${parsed.via}`,
      };
    }
  }
  throw new Error(`zarr metadata found no parseable multiscales at ${base}`);
}

interface N5Attrs {
  pixelResolution?: { dimensions?: number[]; unit?: string };
  resolution?: number[];
  transform?: { scale?: number[]; units?: string[] };
  multiscales?: ZarrMultiscale[];
  dimensions?: number[];
}

async function detectN5(base: string): Promise<DetectedMetadata> {
  const attrs = (await tryJson(`${base}/attributes.json`)) as N5Attrs | null;
  if (!attrs) throw new Error(`No attributes.json at ${base}`);
  if (attrs.multiscales?.[0]) {
    const parsed = parseMultiscalesVoxelSize(attrs.multiscales[0]);
    if (parsed) {
      return {
        voxel_size_nm: parsed.voxel_size_nm,
        source: `n5://${base}`,
        kind: "n5",
        via: `n5 multiscales: ${parsed.via}`,
      };
    }
  }
  if (attrs.pixelResolution?.dimensions && attrs.pixelResolution.dimensions.length >= 3) {
    const dims = attrs.pixelResolution.dimensions;
    const unit = attrs.pixelResolution.unit ?? "nm";
    const [vx, vy, vz] = [toNm(dims[0], unit), toNm(dims[1], unit), toNm(dims[2], unit)];
    return {
      voxel_size_nm: [vx, vy, vz],
      source: `n5://${base}`,
      kind: "n5",
      via: `n5 pixelResolution (${unit})`,
    };
  }
  if (attrs.resolution && attrs.resolution.length >= 3) {
    const [vx, vy, vz] = attrs.resolution.slice(0, 3);
    return {
      voxel_size_nm: [vx, vy, vz],
      source: `n5://${base}`,
      kind: "n5",
      via: `n5 resolution (assumed nm)`,
    };
  }
  if (attrs.transform?.scale && attrs.transform.scale.length >= 3) {
    const units = attrs.transform.units ?? ["nm", "nm", "nm"];
    const s = attrs.transform.scale;
    return {
      voxel_size_nm: [toNm(s[0], units[0]), toNm(s[1], units[1]), toNm(s[2], units[2])],
      source: `n5://${base}`,
      kind: "n5",
      via: `n5 transform.scale (${units.join(",")})`,
    };
  }
  throw new Error(`n5 attributes.json at ${base} has no recognizable voxel-size field`);
}

interface PrecomputedInfo {
  scales?: Array<{ resolution?: number[]; size?: number[]; key?: string }>;
}

async function detectPrecomputed(base: string): Promise<DetectedMetadata> {
  const info = (await tryJson(`${base}/info`)) as PrecomputedInfo | null;
  if (!info || !info.scales || info.scales.length === 0) {
    throw new Error(`No precomputed info at ${base}/info`);
  }
  const s = info.scales[0];
  if (!s.resolution || s.resolution.length < 3) {
    throw new Error(`precomputed info at ${base} missing scales[0].resolution`);
  }
  const [vx, vy, vz] = s.resolution.slice(0, 3);
  const size = s.size && s.size.length >= 3 ? (s.size.slice(0, 3) as [number, number, number]) : undefined;
  const center = size ? ([(size[0] * vx) / 2, (size[1] * vy) / 2, (size[2] * vz) / 2] as [number, number, number]) : undefined;
  return {
    voxel_size_nm: [vx, vy, vz],
    size_voxels: size,
    center_nm: center,
    source: `precomputed://${base}`,
    kind: "precomputed",
    via: `precomputed scales[0].resolution (nm)`,
  };
}

export async function detectSourceMetadata(source: string): Promise<DetectedMetadata> {
  const { kind, base } = parseScheme(source);
  if (kind === "zarr") return detectZarr(base);
  if (kind === "n5") return detectN5(base);
  if (kind === "precomputed") return detectPrecomputed(base);
  throw new Error(`Unsupported source kind: ${kind}`);
}
