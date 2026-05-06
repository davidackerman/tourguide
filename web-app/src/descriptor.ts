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
  const required = ["name", "display_name", "voxel_size_nm", "layers"];
  for (const field of required) {
    if (!(field in d)) {
      throw new Error(`Descriptor missing required field: ${field}`);
    }
  }
  const voxel = d.voxel_size_nm;
  if (!Array.isArray(voxel) || voxel.length !== 3) {
    throw new Error("voxel_size_nm must be an array of 3 numbers");
  }
  if (!Array.isArray(d.layers) || d.layers.length === 0) {
    throw new Error("layers must be a non-empty array");
  }
  for (const layer of d.layers as DatasetLayer[]) {
    if (!layer.name || !layer.type || !layer.source) {
      throw new Error(
        `Each layer needs name, type, and source — got ${JSON.stringify(layer)}`,
      );
    }
    if (layer.type !== "image" && layer.type !== "segmentation") {
      throw new Error(`Layer type must be image or segmentation — got ${layer.type}`);
    }
  }
  return d as unknown as DatasetDescriptor;
}
