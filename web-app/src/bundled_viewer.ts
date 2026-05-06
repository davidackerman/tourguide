// Self-hosted Neuroglancer viewer. Replaces the iframe-based viewer.ts with
// a bundled version that lives on our own origin, so:
//   - Service workers can intercept its fetches (enables local-file loading).
//   - flyTo() updates state in-place — no full reload, smooth camera moves.
//   - State change events are observable for plot↔viewer cross-linking.
//
// Neuroglancer uses a conditional-imports system (package.json "imports" with
// custom conditions like "neuroglancer/datasource/zarr:enabled") to gate which
// datasources/kvstores/layers get bundled. Vite honors conditions for
// "exports" but not for self-imports, so every module falls through to the
// "disabled" variant unless we import the register files directly here.
// These imports do that registration at module load, before setupDefaultViewer
// reads the registries.

// Kvstores — where bytes come from (http, s3, gcs, gzip-wrapping, etc.).
import "neuroglancer/unstable/kvstore/http/register_frontend.js";
import "neuroglancer/unstable/kvstore/s3/register_frontend.js";
import "neuroglancer/unstable/kvstore/gcs/register.js";
import "neuroglancer/unstable/kvstore/gzip/register.js";
import "neuroglancer/unstable/kvstore/byte_range/register.js";
import "neuroglancer/unstable/kvstore/zip/register_frontend.js";
import "neuroglancer/unstable/kvstore/ocdbt/register_frontend.js";
import "neuroglancer/unstable/kvstore/icechunk/register_frontend.js";
import "neuroglancer/unstable/kvstore/middleauth/register_frontend.js";
import "neuroglancer/unstable/kvstore/middleauth/register_credentials_provider.js";
import "neuroglancer/unstable/kvstore/ngauth/register.js";
import "neuroglancer/unstable/kvstore/ngauth/register_credentials_provider.js";

// Data sources — formats we can read off those kvstores.
import "neuroglancer/unstable/datasource/zarr/register_default.js";
import "neuroglancer/unstable/datasource/n5/register_default.js";
import "neuroglancer/unstable/datasource/precomputed/register_default.js";
import "neuroglancer/unstable/datasource/render/register_default.js";
import "neuroglancer/unstable/datasource/nifti/register_default.js";
import "neuroglancer/unstable/datasource/obj/register_default.js";
import "neuroglancer/unstable/datasource/vtk/register_default.js";
import "neuroglancer/unstable/datasource/deepzoom/register_default.js";
import "neuroglancer/unstable/datasource/dvid/register_default.js";
import "neuroglancer/unstable/datasource/dvid/register_credentials_provider.js";

// Layer types — what rendering pipelines are available.
import "neuroglancer/unstable/layer/image/index.js";
import "neuroglancer/unstable/layer/segmentation/index.js";
import "neuroglancer/unstable/layer/annotation/index.js";
import "neuroglancer/unstable/layer/single_mesh/index.js";

import { setupDefaultViewer } from "neuroglancer/unstable/ui/default_viewer_setup.js";
import type { Viewer as NgViewer } from "neuroglancer/unstable/viewer.js";
import type { DatasetDescriptor } from "./descriptor.js";
import { descriptorToNgState } from "./viewer.js";

export class BundledViewer {
  private viewer: NgViewer | null = null;
  private container: HTMLElement;
  private currentState: ReturnType<typeof descriptorToNgState> | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  private ensureViewer(): NgViewer {
    if (this.viewer) return this.viewer;
    // Neuroglancer mounts inside a target element. We give it our container.
    this.viewer = setupDefaultViewer({ target: this.container });
    return this.viewer;
  }

  loadDescriptor(d: DatasetDescriptor): void {
    const viewer = this.ensureViewer();
    const state = descriptorToNgState(d);
    this.currentState = state;
    // Neuroglancer accepts a JSON state object via `state.restoreState`.
    // The shape we already build in viewer.ts (descriptorToNgState) matches
    // Neuroglancer's URL state format exactly.
    viewer.state.restoreState(state as unknown as Record<string, unknown>);
  }

  flyTo(
    position: [number, number, number],
    segmentId?: string,
    layerName?: string,
  ): void {
    const viewer = this.ensureViewer();
    if (!this.currentState) return;
    // NG's `navigationState.position.value` is in *output dim units*, not nm.
    // Even though we declare dimensions at 1 nm/unit, NG often inherits the
    // source zarr's native scale (e.g. z at 2.62 nm/unit). So we must:
    //   (1) read the live dim order and reshuffle our (x,y,z) input,
    //   (2) divide each value by the live per-dim scale to convert nm → units.
    // Input `position` is always in world-space nm (x, y, z).
    const cs = (viewer.navigationState as any).coordinateSpace?.value as
      | { names?: string[]; scales?: Float64Array | number[]; units?: string[] }
      | undefined;
    const names = cs?.names ?? ["x", "y", "z"];
    const scales = cs?.scales ?? [1e-9, 1e-9, 1e-9];
    const units = cs?.units ?? ["m", "m", "m"];
    const xyzNm: Record<string, number> = {
      x: position[0],
      y: position[1],
      z: position[2],
    };
    const orderedPos = names.map((n, i) => {
      const nmValue = xyzNm[n] ?? 0;
      const scale = Number(scales[i]); // base-unit per voxel
      const unit = units[i];
      // Convert world nm → base units of this dim, then divide by scale.
      // For unit "m" with scale 2.62e-9: nmValue * 1e-9 / 2.62e-9 = nmValue/2.62.
      // For unit "" (dimensionless) or unknown: assume already in dim units.
      const nmToBase =
        unit === "m" ? 1e-9 :
        unit === "µm" || unit === "um" || unit === "micrometer" ? 1e-3 :
        unit === "nm" || unit === "nanometer" ? 1 :
        unit === "" ? scale /* fallback: treat as already in nm */ :
        1;
      return scale > 0 ? (nmValue * nmToBase) / scale : nmValue;
    });
    console.log("[viewer] flyTo", {
      input_xyz_nm: position,
      ngDimNames: names,
      ngDimScales: Array.from(scales),
      ngDimUnits: units,
      written_in_dim_units: orderedPos,
    });
    viewer.navigationState.position.value = Float32Array.from(orderedPos);
    // Highlight a segment within a layer if requested.
    if (segmentId && layerName) {
      const layer = viewer.layerManager.getLayerByName(layerName);
      if (layer) {
        const userLayer = layer.layer as unknown as {
          displayState?: {
            segmentationGroupState?: {
              value?: { visibleSegments?: { add: (id: bigint) => void; clear: () => void } };
            };
          };
        };
        const visible = userLayer.displayState?.segmentationGroupState?.value?.visibleSegments;
        if (visible) {
          visible.clear();
          try {
            visible.add(BigInt(segmentId));
          } catch {
            console.warn(`Couldn't add segment ${segmentId} (not a valid bigint)`);
          }
        }
      }
    }
  }

  /** Returns the underlying Neuroglancer Viewer for advanced use (event hooks, etc). */
  getNgViewer(): NgViewer | null {
    return this.viewer;
  }

  // Snapshot the live Neuroglancer state (camera, selected segments,
  // layout, per-layer visibility) for permalink encoding. Returns null if
  // the viewer hasn't been mounted yet.
  getNgState(): Record<string, unknown> | null {
    if (!this.viewer) return null;
    return this.viewer.state.toJSON() as Record<string, unknown>;
  }

  // Apply a previously-captured NG state on top of the current one. Used
  // by permalink restoration to overlay camera/segments after the
  // descriptor has set up the layer scaffolding.
  applyNgState(state: Record<string, unknown>): void {
    const viewer = this.ensureViewer();
    viewer.state.restoreState(state);
    this.currentState = state as any;
  }

  // Add a Neuroglancer segmentation layer that only renders the mesh
  // subsource of a precomputed source — i.e. shows 3D meshes plus the
  // segment_properties list, but not the volume slab. Used for analysis
  // outputs where the user already has the source seg in another layer
  // and just wants meshes overlaid.
  addMeshOnlyLayer(spec: {
    name: string;
    source: string; // precomputed://<url>
    segments?: string[];
  }): void {
    this.addLayerFromSpec({
      type: "segmentation",
      name: spec.name,
      source: {
        url: spec.source,
        // Disable the volume subsource (the seg slab); keep mesh +
        // segment_properties enabled so the user sees the segments panel
        // and the 3D meshes only.
        enableDefaultSubsources: false,
        subsources: { mesh: true, segment_properties: true },
      },
      segments: spec.segments ?? [],
    });
  }

  // Merge extra layers into the current NG state (cheapest way to add a
  // new layer without rebuilding the whole viewer). Takes a partial layer
  // spec object.
  addLayerFromSpec(layer: Record<string, unknown>): void {
    const viewer = this.ensureViewer();
    const state = viewer.state.toJSON() as { layers?: Array<Record<string, unknown>> } & Record<string, unknown>;
    const layers = Array.isArray(state.layers) ? state.layers.slice() : [];
    // Drop any existing layer with the same name, so re-running an analysis
    // replaces rather than duplicates.
    const name = String(layer.name ?? "");
    const filtered = name ? layers.filter((l) => String(l.name) !== name) : layers;
    filtered.push(layer);
    state.layers = filtered;
    viewer.state.restoreState(state);
    this.currentState = state as any;
  }

  // Add an in-memory annotation layer with a list of points (world nm).
  addAnnotationLayer(layerName: string, points: { pos: [number, number, number]; id?: string; description?: string }[]): void {
    const annotations = points.map((p, i) => ({
      type: "point",
      id: p.id ?? `ann-${Date.now()}-${i}`,
      point: p.pos,
      description: p.description ?? "",
    }));
    this.addLayerFromSpec({
      type: "annotation",
      name: layerName,
      source: "local://annotations",
      annotations,
    });
  }

  // Change which segment IDs are visible in a segmentation layer.
  highlightSegments(layerName: string, ids: string[]): void {
    const viewer = this.ensureViewer();
    const layer = viewer.layerManager.getLayerByName(layerName);
    if (!layer) {
      console.warn(`[viewer] highlightSegments: layer '${layerName}' not found`);
      return;
    }
    const ul = layer.layer as unknown as {
      displayState?: {
        segmentationGroupState?: {
          value?: {
            visibleSegments?: { add: (id: bigint) => void; clear: () => void };
          };
        };
      };
    };
    const visible = ul.displayState?.segmentationGroupState?.value?.visibleSegments;
    if (!visible) {
      console.warn(`[viewer] highlightSegments: layer '${layerName}' has no visible-segments state`);
      return;
    }
    visible.clear();
    for (const id of ids) {
      try {
        visible.add(BigInt(id));
      } catch {
        /* skip invalid ids */
      }
    }
  }
}
