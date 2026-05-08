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
import { makeLayer } from "neuroglancer/unstable/layer/index.js";
import { layerDataSourceSpecificationFromJson } from "neuroglancer/unstable/layer/layer_data_source.js";
import type { DatasetDescriptor } from "./descriptor.js";
import { descriptorToNgState } from "./viewer.js";

function sameArray(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

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

  loadDescriptor(d: DatasetDescriptor, ngStateOverride?: Record<string, unknown>): void {
    // restoreState replaces declared layers but NG keeps its in-memory
    // navigation state (camera position, zoom, layout) across the call.
    // That's why a dataset switch leaves the camera pointing at the
    // *previous* dataset's center: NG never auto-fits because it
    // already has a position. The cleanest match for "user picked a
    // new dataset" semantics is to make NG see this as a fresh page
    // load — tear down the existing viewer and recreate it. NG's
    // standard auto-fit-to-layer-bounds then runs naturally and we
    // don't have to bolt on flyTo / setTimeout / state-reset hacks.
    const isFirstMount = !this.viewer;
    if (this.viewer) {
      try {
        (this.viewer as unknown as { dispose?: () => void }).dispose?.();
      } catch (err) {
        console.warn("[viewer] dispose failed:", (err as Error).message);
      }
      this.viewer = null;
      // Clear NG's mount before re-mounting so we don't end up with
      // two stacked viewers in the DOM.
      while (this.container.firstChild) this.container.removeChild(this.container.firstChild);
    }
    // On first mount, setupDefaultViewer (inside ensureViewer) reads the
    // URL hash and applies any '#!{...}' state to the viewer — that's how
    // a copy/pasted URL or a permalink gets its camera + selected segments
    // back. If we then call restoreState(descriptorState) we *clobber*
    // that with the descriptor's defaults (no position, no segments),
    // which is what made copy-pasting the URL bar look like a fresh demo
    // load. Skip the descriptor restore in that case and let NG own the
    // state — the descriptor's layer URLs are already in the hash anyway,
    // since the user copied from a tab on the same dataset.
    const viewer = this.ensureViewer();
    const hashHasNgState = window.location.hash.startsWith("#!");
    // When the caller passed an explicit ngStateOverride (paste-NG-state
    // load path), use it directly. Doing both — first restoreState with
    // descriptorToNgState (1nm/unit dimensions), then applyNgState with
    // the user's state (e.g. 4nm/unit) — would race: NG processes the
    // first state asynchronously and the second restoreState reading
    // halfway-applied dimensions caused position values to be
    // interpreted in the wrong unit, leaving the camera at the previous
    // dataset's coordinates.
    if (ngStateOverride) {
      this.currentState = ngStateOverride as unknown as ReturnType<typeof descriptorToNgState>;
      viewer.state.restoreState(ngStateOverride);
      return;
    }
    const state = descriptorToNgState(d);
    this.currentState = state;
    if (!(isFirstMount && hashHasNgState)) {
      viewer.state.restoreState(state as unknown as Record<string, unknown>);
    }
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

  // Attach a precomputed mesh source to an existing NG layer instead of
  // creating a separate mesh layer. Valid when the labeled volume the
  // meshes describe shares the same id space as the existing layer
  // (i.e. the user analyzed an already-labeled seg layer). Falls back
  // to false if the named layer doesn't exist or isn't a segmentation,
  // letting the caller add a standalone mesh layer instead.
  attachMeshSourceToLayer(spec: {
    layerName: string;
    meshSource: string; // precomputed://<url>
    segments?: string[];
  }): boolean {
    const viewer = this.ensureViewer();
    const managedLayer = viewer.layerManager.getLayerByName(spec.layerName);
    if (!managedLayer) return false;
    const userLayer = managedLayer.layer as
      | {
          type?: string;
          addDataSource?: (source: ReturnType<typeof layerDataSourceSpecificationFromJson>) => unknown;
          displayState?: {
            segmentationGroupState?: {
              value?: { visibleSegments?: { add: (id: bigint) => void } };
            };
          };
        }
      | null;
    if (!userLayer || userLayer.type !== "segmentation" || typeof userLayer.addDataSource !== "function") {
      return false;
    }
    const release = this.lockCamera();
    userLayer.addDataSource(layerDataSourceSpecificationFromJson({
      url: spec.meshSource,
      enableDefaultSubsources: false,
      subsources: { mesh: true, segment_properties: true },
    }));
    const visible = userLayer.displayState?.segmentationGroupState?.value?.visibleSegments;
    if (visible && spec.segments) {
      for (const id of spec.segments) {
        try {
          visible.add(BigInt(id));
        } catch {
          /* skip invalid ids */
        }
      }
    }
    release(10000);
    return true;
  }

  // Add or replace a single NG layer without restoring the whole viewer
  // state. A full `viewer.state.restoreState` also rewrites dimensions
  // and position; while async sources are resolving that can rebase the
  // camera through a temporary coordinate space and produce huge
  // position values. Layer-only mutation keeps global navigation stable.
  addLayerFromSpec(layer: Record<string, unknown>): void {
    const viewer = this.ensureViewer();
    const name = String(layer.name ?? "");
    if (!name) {
      console.warn("[viewer] addLayerFromSpec: layer spec has no name", layer);
      return;
    }
    const release = this.lockCamera();
    const existing = viewer.layerManager.getLayerByName(name);
    let index: number | undefined;
    if (existing) {
      index = viewer.layerManager.managedLayers.indexOf(existing);
      viewer.layerManager.removeManagedLayer(existing);
    }
    const managedLayer = makeLayer(viewer.layerSpecification, name, layer);
    viewer.layerSpecification.add(managedLayer, index);
    release(10000);
  }

  // Hold the camera at its current world position for `holdMs` after the
  // returned `release(holdMs)` is called. NG's coordinate space (per-axis
  // scale + dim order) can change when a new layer's data source
  // resolves — at that point the OLD numerical position values mean a
  // different physical location, so a naive snapshot-and-revert lands
  // somewhere wildly wrong (the user reported "extremely off + crazy
  // zoom", with displayed coords reaching trillions: NG-units × ratio).
  //
  // Capture the camera in **world nm** (xyz) before the layer add by
  // reading the live coord space and converting NG-units → nm. After
  // any drift (caught via the .changed signal AND a few timed retries
  // for paths that bypass the signal), re-apply by reading the *new*
  // coord space and converting nm → new-NG-units. This way the camera
  // lands at the same physical place even if the units shifted.
  private lockCamera(): (holdMs?: number) => void {
    const viewer = this.ensureViewer();
    const navState = viewer.navigationState as any;
    const cs0 = navState?.coordinateSpace?.value as
      | { names?: string[]; scales?: ArrayLike<number>; units?: string[] }
      | undefined;
    const positionNgBefore: number[] = navState?.position?.value
      ? Array.from(navState.position.value as Float32Array)
      : [];
    // Convert ng-units → world nm via the *current* coord space.
    const xyzNmBefore: Record<string, number> = {};
    if (cs0?.names && cs0?.scales && cs0?.units) {
      for (let i = 0; i < cs0.names.length; i++) {
        const name = cs0.names[i];
        const scale = Number(cs0.scales[i]);
        const unit = cs0.units[i];
        const baseToNm =
          unit === "m" ? 1e9
            : unit === "µm" || unit === "um" || unit === "micrometer" ? 1e3
            : unit === "nm" || unit === "nanometer" ? 1
            : unit === "" ? 1 / Math.max(scale, 1e-12)
            : 1;
        xyzNmBefore[name] = (positionNgBefore[i] ?? 0) * scale * baseToNm;
      }
    }
    const csScaleBefore = navState?.zoomFactor?.value as number | undefined;
    const projScaleBefore = navState?.depthRange?.value as number | undefined;
    let active = true;

    const restore = (): void => {
      if (!active) return;
      try {
        const cs = navState?.coordinateSpace?.value as typeof cs0;
        if (cs?.names && cs?.scales && cs?.units && navState?.position) {
          const ordered = cs.names.map((n: string, i: number) => {
            const nm = xyzNmBefore[n] ?? 0;
            const scale = Number(cs.scales![i]);
            const unit = cs.units![i];
            const nmToBase =
              unit === "m" ? 1e-9
                : unit === "µm" || unit === "um" || unit === "micrometer" ? 1e-3
                : unit === "nm" || unit === "nanometer" ? 1
                : unit === "" ? scale
                : 1;
            return scale > 0 ? (nm * nmToBase) / scale : nm;
          });
          const cur = navState.position.value as Float32Array | undefined;
          if (!cur || !sameArray(cur, ordered)) {
            navState.position.value = Float32Array.from(ordered);
          }
        }
        if (csScaleBefore !== undefined && navState?.zoomFactor && navState.zoomFactor.value !== csScaleBefore) {
          navState.zoomFactor.value = csScaleBefore;
        }
        if (projScaleBefore !== undefined && navState?.depthRange && navState.depthRange.value !== projScaleBefore) {
          navState.depthRange.value = projScaleBefore;
        }
      } catch {
        /* NG internals shifted; non-fatal */
      }
    };

    const subs: Array<() => void> = [];
    const sub = (signal: { add?: (cb: () => void) => void; remove?: (cb: () => void) => void }): void => {
      if (signal && typeof signal.add === "function") {
        signal.add(restore);
        subs.push(() => signal.remove?.(restore));
      }
    };
    sub(navState?.position?.changed);
    sub(navState?.zoomFactor?.changed);
    sub(navState?.depthRange?.changed);
    sub(navState?.coordinateSpace?.changed);

    return (holdMs = 2000): void => {
      restore();
      requestAnimationFrame(restore);
      setTimeout(restore, 100);
      setTimeout(restore, 400);
      setTimeout(restore, 1000);
      setTimeout(() => {
        restore();
        active = false;
        for (const u of subs) u();
      }, holdMs);
    };
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
  // Read the currently-visible segment IDs for a segmentation layer as a
  // sorted string list. Used by Custom Analysis's "↻ from layer" helper
  // to pre-fill skeleton segment-ID inputs from whatever the user has
  // selected in NG. Returns [] if the layer has no visible-segments state
  // (image layers, layers that haven't loaded yet, etc.).
  getVisibleSegments(layerName: string): string[] {
    const viewer = this.viewer;
    if (!viewer) return [];
    const layer = viewer.layerManager.getLayerByName(layerName);
    if (!layer) return [];
    const ul = layer.layer as unknown as {
      displayState?: {
        segmentationGroupState?: {
          value?: {
            visibleSegments?: Iterable<bigint> | { [Symbol.iterator]?: () => Iterator<bigint> };
          };
        };
      };
    };
    const visible = ul.displayState?.segmentationGroupState?.value?.visibleSegments;
    if (!visible) return [];
    const ids: string[] = [];
    try {
      for (const v of visible as Iterable<bigint>) {
        ids.push(typeof v === "bigint" ? v.toString() : String(v));
      }
    } catch {
      return [];
    }
    // Numeric sort when possible; segment IDs are conventionally numeric
    // bigints. Falls back to string compare for non-numeric ids.
    ids.sort((a, b) => {
      const an = Number(a);
      const bn = Number(b);
      if (Number.isFinite(an) && Number.isFinite(bn)) return an - bn;
      return a.localeCompare(b);
    });
    return ids;
  }

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
