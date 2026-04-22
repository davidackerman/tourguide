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
  if (d.initial_position) state.position = d.initial_position;
  if (d.projection_orientation) state.projectionOrientation = d.projection_orientation;
  // Default cross-section scale: ~1 voxel per pixel. Scale is in physical
  // units per pixel — with nm dimensions, that's just the voxel size in nm.
  const defaultScale = Math.max(vx, vy, vz);
  state.crossSectionScale = d.cross_section_scale ?? defaultScale;
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
