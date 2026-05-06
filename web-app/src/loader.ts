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

// Rewrite layer sources / csv paths that are *relative* (no `://`) so
// they resolve against a picked-folder base URL. Lets a tourguide.yaml
// dropped inside a data folder say `source: cells/data` and have it
// turn into `zarr://<host>/local-data/<id>/cells/data/` at load time.
//
// Sources that already include a scheme (`zarr://https://…`,
// `precomputed://…`, etc.) pass through untouched, so a YAML that
// mixes local + remote layers still works.
export function resolveDescriptorAgainstFolder(
  d: DatasetDescriptor,
  baseUrl: string,
): DatasetDescriptor {
  const base = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
  const resolveSource = (source: string): string => {
    if (/^[a-z][a-z0-9+]*:\/\//i.test(source)) return source; // already absolute
    // Allow optional explicit kind prefix without authority, e.g. "n5:./data.n5/".
    const kindMatch = /^(zarr|n5|precomputed):(?:\/\/)?(.*)$/i.exec(source);
    let kind = "zarr";
    let path = source;
    if (kindMatch) {
      kind = kindMatch[1];
      path = kindMatch[2];
    } else {
      // Sniff kind from extension; default zarr.
      if (/\.n5\/?$/i.test(source) || /\.n5\//i.test(source)) kind = "n5";
      else if (/^precomputed\b/i.test(source) || /\/precomputed\//i.test(source)) kind = "precomputed";
    }
    const cleaned = path.replace(/^\.?\/+/, "");
    return `${kind}://${base}${cleaned}`;
  };
  const resolveCsv = (csv: string): string => {
    if (/^[a-z][a-z0-9+]*:\/\//i.test(csv)) return csv;
    return base + csv.replace(/^\.?\/+/, "");
  };
  return {
    ...d,
    layers: d.layers.map((l) => ({
      ...l,
      source: resolveSource(l.source),
      csv: l.csv ? resolveCsv(l.csv) : l.csv,
    })),
  };
}
