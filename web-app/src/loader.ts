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
// they resolve against picked-folder base URL(s). Two shapes:
//
// - Single base (back-compat): `resolveDescriptorAgainstFolder(d, baseUrl)`
//   rewrites every relative source against that one folder. Used when a
//   `tourguide.yaml` dropped at the root of a picked folder owns all
//   the layers.
//
// - Multi base (per-alias): `resolveDescriptorAgainstFolder(d, {em, seg})`
//   rewrites `<alias>/<subpath>` sources against the matching folder's
//   baseUrl. Lets a single YAML reference data spread across multiple
//   disk locations the user picked separately.
//
// Sources that already include a scheme (`zarr://https://…`, etc.)
// pass through untouched, so a YAML can freely mix local + remote.
export function resolveDescriptorAgainstFolder(
  d: DatasetDescriptor,
  bases: string | Record<string, string>,
): DatasetDescriptor {
  const baseMap: Record<string, string> =
    typeof bases === "string"
      ? { "": bases.endsWith("/") ? bases : bases + "/" }
      : Object.fromEntries(
          Object.entries(bases).map(([k, v]) => [k, v.endsWith("/") ? v : v + "/"]),
        );
  const splitAlias = (path: string): { base: string; rest: string } => {
    // Strip leading "./" before alias matching.
    const trimmed = path.replace(/^\.?\/+/, "");
    if ("" in baseMap) return { base: baseMap[""], rest: trimmed };
    // Multi-alias: longest-prefix match by alias.
    const aliases = Object.keys(baseMap).sort((a, b) => b.length - a.length);
    for (const a of aliases) {
      if (trimmed === a || trimmed.startsWith(a + "/")) {
        return { base: baseMap[a], rest: trimmed.slice(a.length).replace(/^\/+/, "") };
      }
    }
    throw new Error(`Layer source '${path}' has no matching folders alias (known: ${aliases.join(", ") || "(none)"})`);
  };
  const resolveSource = (source: string): string => {
    if (/^[a-z][a-z0-9+]*:\/\//i.test(source)) return source; // already absolute
    const kindMatch = /^(zarr|n5|precomputed):(?:\/\/)?(.*)$/i.exec(source);
    let kind = "zarr";
    let path = source;
    if (kindMatch) {
      kind = kindMatch[1];
      path = kindMatch[2];
    } else {
      if (/\.n5\/?$/i.test(source) || /\.n5\//i.test(source)) kind = "n5";
      else if (/^precomputed\b/i.test(source) || /\/precomputed\//i.test(source)) kind = "precomputed";
    }
    const { base, rest } = splitAlias(path);
    return `${kind}://${base}${rest}`;
  };
  const resolveCsv = (csv: string): string => {
    if (/^[a-z][a-z0-9+]*:\/\//i.test(csv)) return csv;
    const { base, rest } = splitAlias(csv);
    return base + rest;
  };
  // The `folders` block is a yaml-time directive only — strip it from
  // the descriptor we return so downstream code (permalink encode,
  // viewer, etc.) sees a clean descriptor with absolute sources.
  const { folders: _strip, ...rest } = d;
  void _strip;
  return {
    ...rest,
    layers: d.layers.map((l) => ({
      ...l,
      source: resolveSource(l.source),
      csv: l.csv ? resolveCsv(l.csv) : l.csv,
    })),
  };
}
