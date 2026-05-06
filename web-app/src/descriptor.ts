import yaml from "js-yaml";

export type LayerType = "image" | "segmentation";

export interface DatasetLayer {
  name: string;
  type: LayerType;
  source: string;
  organelle_class?: string;
  csv?: string;
}

export interface DatasetDescriptor {
  name: string;
  display_name: string;
  description?: string;
  // Required for the NG dim setup, but optional in the YAML — the
  // loader auto-fills this from the first OME-Zarr layer's metadata
  // when omitted. After loadDescriptorFromYaml + autofill, this is
  // always populated downstream.
  voxel_size_nm: [number, number, number];
  initial_position?: [number, number, number];
  projection_orientation?: [number, number, number, number];
  cross_section_scale?: number;
  projection_scale?: number;
  layers: DatasetLayer[];
  // Optional alias → "what folder to pick" hint. When set, the loader
  // prompts the user to pick each folder in turn and rewrites layer
  // sources prefixed by `<alias>/...` against the resulting baseUrl.
  // Lets a single YAML reference data spread across multiple disk
  // locations without forcing a single common-parent pick.
  folders?: Record<string, string>;
}

export interface CatalogEntry {
  name: string;
  url: string;
  thumbnail?: string;
}

export interface Catalog {
  version: number;
  datasets: CatalogEntry[];
}

export function parseDescriptor(text: string): DatasetDescriptor {
  const parsed = yaml.load(text) as unknown;
  return validateDescriptor(parsed);
}

export function validateDescriptor(value: unknown): DatasetDescriptor {
  if (!value || typeof value !== "object") {
    throw new Error("Descriptor must be a YAML object");
  }
  const d = value as Record<string, unknown>;
  // voxel_size_nm is no longer required at YAML time — the loader fills
  // it from the first OME-Zarr layer's coordinateTransformations when
  // omitted. We default-fill here so downstream code (descriptorToNgState)
  // always sees a 3-element array.
  if (!d.voxel_size_nm) d.voxel_size_nm = [1, 1, 1];
  // `name` and `display_name` are auto-filled when omitted — name comes
  // from the first layer's source URL tail, display_name falls back to
  // name. Lets the YAML really be just `folders + layers`.
  if (!d.name) {
    const firstLayer = (d.layers as Array<{ source?: string }> | undefined)?.[0];
    const tail = firstLayer?.source?.split("/").filter(Boolean).pop() ?? "dataset";
    d.name = tail.replace(/\.(zarr|n5|precomputed)$/i, "").replace(/[^A-Za-z0-9._-]/g, "_") || "dataset";
  }
  if (!d.display_name) d.display_name = d.name;
  if (!("layers" in d)) {
    throw new Error(`Descriptor missing required field: layers`);
  }
  const voxel = d.voxel_size_nm;
  if (!Array.isArray(voxel) || voxel.length !== 3) {
    throw new Error("voxel_size_nm must be an array of 3 numbers when set");
  }
  if (!Array.isArray(d.layers) || d.layers.length === 0) {
    throw new Error("layers must be a non-empty array");
  }
  // Auto-fill layer names from source URL tails when omitted, mirroring
  // what defaultLayerName / NG do — so a YAML can be just `type +
  // source` per layer with no explicit names.
  const used = new Set<string>();
  for (const layer of d.layers as DatasetLayer[]) {
    if (!layer.type || !layer.source) {
      throw new Error(
        `Each layer needs type and source — got ${JSON.stringify(layer)}`,
      );
    }
    if (layer.type !== "image" && layer.type !== "segmentation") {
      throw new Error(`Layer type must be image or segmentation — got ${layer.type}`);
    }
    if (!layer.name) {
      const tail = layer.source.split("/").filter(Boolean).pop() ?? layer.type;
      let derived = tail.replace(/\.(zarr|n5|precomputed)$/i, "").replace(/[^A-Za-z0-9._-]/g, "_") || layer.type;
      // Dedupe across layers in the same descriptor.
      let n = derived;
      let i = 2;
      while (used.has(n)) n = `${derived}-${i++}`;
      derived = n;
      layer.name = derived;
    }
    used.add(layer.name);
  }
  return d as unknown as DatasetDescriptor;
}
