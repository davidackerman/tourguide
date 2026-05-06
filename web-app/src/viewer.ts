import type { DatasetDescriptor } from "./descriptor.js";

const NEUROGLANCER_HOST = "https://neuroglancer-demo.appspot.com/";

interface NgDimensions {
  [axis: string]: [number, string];
}

interface NgLayer {
  type: "image" | "segmentation";
  name: string;
  source: string;
  visible?: boolean;
}

interface NgState {
  dimensions: NgDimensions;
  position?: [number, number, number];
  projectionOrientation?: [number, number, number, number];
  crossSectionScale?: number;
  projectionScale?: number;
  layers: NgLayer[];
  selectedLayer?: { visible: boolean; layer: string };
  layout: string;
}

// All app-side coordinates are in nanometers. We declare Neuroglancer's
// dimensions as 1 nm per unit on every axis so positions in nm map 1:1 into
// state.position. CSV position_x/y/z are already nm; descriptor.initial_position
// is nm; fly_to receives nm — everything matches.
export function descriptorToNgState(d: DatasetDescriptor): NgState {
  const [vx, vy, vz] = d.voxel_size_nm;
  const NM_PER_UNIT = 1e-9; // 1 unit == 1 nanometer
  const state: NgState = {
    dimensions: {
      x: [NM_PER_UNIT, "m"],
      y: [NM_PER_UNIT, "m"],
      z: [NM_PER_UNIT, "m"],
    },
    layers: d.layers.map((l) => ({
      type: l.type,
      name: l.name,
      source: l.source,
    })),
    layout: "4panel",
  };
  // Intentionally NOT forwarding d.initial_position into state.position.
  // The descriptor stores positions in world nm, but NG's runtime
  // coordinate space comes from the *layer's* source (OME-Zarr scale,
  // n5 resolution, etc.) — for a 4 nm voxel layer NG's units are 4 nm,
  // not 1 nm, so a position value of 16000 (meant as 16000 nm) gets
  // read as 16000 × 4 nm = 64000 nm. Off by the resolution factor.
  // flyTo handles the conversion at runtime; descriptor-init can't,
  // because the layer hasn't loaded yet and its scale is unknown.
  // NG's own auto-fit-to-bounds is the right default here — when the
  // first data source resolves, NG centers on the layer extent in its
  // own units automatically.
  if (d.projection_orientation) state.projectionOrientation = d.projection_orientation;
  // Cross-section default: NG's own default lands at ~1 voxel/pixel
  // (i.e. crossSectionScale ≈ voxel_size in NG units), which is way
  // too tight for typical EM volumes — the panel only shows a few
  // hundred nm of a multi-µm dataset. Seed a coarser default
  // (~64x voxel size) so the panels fit a usefully large slab on
  // first paint. NG's projection-fit handles the 3D camera on its
  // own. User can mousewheel-zoom from there.
  //
  // Caveat: this value is in NG's *runtime* unit (which inherits the
  // layer source's per-axis scale). At descriptor build time we don't
  // know NG's runtime unit, so we set a constant that's a multiple of
  // the descriptor's declared voxel size — close enough for typical
  // OME-Zarr datasets where NG's runtime unit ends up being the same
  // physical voxel size.
  const ZOOM_OUT_FACTOR = 4;
  const defaultCross = Math.max(vx, vy, vz) * ZOOM_OUT_FACTOR;
  state.crossSectionScale = d.cross_section_scale ?? defaultCross;
  if (d.projection_scale !== undefined) state.projectionScale = d.projection_scale;
  const firstImage = d.layers.find((l) => l.type === "image");
  if (firstImage) {
    state.selectedLayer = { visible: true, layer: firstImage.name };
  }
  return state;
}

export function ngStateToUrl(state: NgState): string {
  const json = JSON.stringify(state);
  return `${NEUROGLANCER_HOST}#!${encodeURIComponent(json)}`;
}

export class Viewer {
  private iframe: HTMLIFrameElement;
  private currentState: NgState | null = null;

  constructor(iframe: HTMLIFrameElement) {
    this.iframe = iframe;
  }

  loadDescriptor(d: DatasetDescriptor): void {
    const state = descriptorToNgState(d);
    this.currentState = state;
    this.iframe.src = ngStateToUrl(state);
  }

  flyTo(position: [number, number, number], segmentId?: string, layerName?: string): void {
    if (!this.currentState) return;
    const next: NgState = JSON.parse(JSON.stringify(this.currentState));
    next.position = position;
    if (segmentId && layerName) {
      const layer = next.layers.find((l) => l.name === layerName);
      if (layer) {
        const augmented = layer as NgLayer & { segments?: string[] };
        augmented.segments = [segmentId];
      }
    }
    this.currentState = next;
    this.iframe.src = ngStateToUrl(next);
  }
}
