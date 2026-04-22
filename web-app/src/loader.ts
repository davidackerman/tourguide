import yaml from "js-yaml";
import {
  type DatasetDescriptor,
  type DatasetLayer,
  type LayerType,
  parseDescriptor,
  validateDescriptor,
} from "./descriptor.js";

export interface PastedLayerInput {
  name: string;
  type: LayerType;
  source: string;
  organelle_class?: string;
  csv?: string;
}

export interface PastedDatasetInput {
  name: string;
  display_name?: string;
  description?: string;
  voxel_size_nm: [number, number, number];
  initial_position?: [number, number, number];
  layers: PastedLayerInput[];
}

export function buildDescriptorFromInput(input: PastedDatasetInput): DatasetDescriptor {
  const cleaned: PastedDatasetInput = {
    ...input,
    display_name: input.display_name?.trim() || input.name.trim(),
    layers: input.layers
      .filter((l) => l.name.trim() && l.source.trim())
      .map((l) => {
        const layer: DatasetLayer = {
          name: l.name.trim(),
          type: l.type,
          source: l.source.trim(),
        };
        if (l.organelle_class?.trim()) layer.organelle_class = l.organelle_class.trim();
        if (l.csv?.trim()) layer.csv = l.csv.trim();
        return layer;
      }),
  };
  return validateDescriptor(cleaned);
}

export function loadDescriptorFromYaml(text: string): DatasetDescriptor {
  return parseDescriptor(text);
}

export function descriptorToYaml(d: DatasetDescriptor): string {
  return yaml.dump(d, { lineWidth: 100 });
}
