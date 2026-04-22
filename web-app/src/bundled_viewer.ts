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
    // Update camera position (in nm, matching our state.dimensions of 1nm/unit).
    viewer.navigationState.position.value = Float32Array.from(position);
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
}
