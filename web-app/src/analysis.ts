// Main-thread orchestrator for segmentation analysis. Owns the Worker
// lifecycle and exposes a small typed API (inspect / analyze / cancel) to the
// UI layer. Decoupled from how the results are rendered.

import type {
  OutgoingMsg,
  InspectResultMsg,
  AnalyzeResultMsg,
  CustomResultMsg,
  ProgressMsg,
  ErrorMsg,
  AnalyzeRequest,
  CustomRequest,
} from "./analysis_worker.js";

export interface LayerScaleInfo {
  path: string;
  shape: number[];
  voxelNm: [number, number, number];
  offsetNm: [number, number, number];
  downsample: [number, number, number];
  approxBytes: number;
}

export interface LayerInspection {
  isMultiscale: boolean;
  axes: { name: string; type?: string }[];
  scales: LayerScaleInfo[];
}

export interface AnalysisResult {
  columns: string[];
  rows: (number | string)[][];
  shape: number[];
  voxelNm: [number, number, number];
  labelCount: number;
}

export interface CustomAnalysisResult {
  table?: { name: string; columns: string[]; rows: (number | string | null)[][] };
  plotPngDataUrl?: string;
  fly?: { pos: [number, number, number]; segmentId?: string; layer?: string };
  narration?: string;
  stdout?: string;
  annotations?: {
    layerName: string;
    points: { pos: [number, number, number]; id?: string; description?: string }[];
  };
  highlight?: { layer: string; ids: string[] };
  addSourceLayer?: { source: string; name: string; type: "image" | "segmentation" };
  newLayer?: {
    synthesizedId: string;
    name: string;
    type: "image" | "segmentation";
    shape: number[];
    dtype: string;
  };
}

export type ProgressCallback = (message: string, phase?: string) => void;

// Strip Neuroglancer-style datasource prefixes so zarrita gets a plain URL.
// "zarr://https://..." → "https://...", "zarr2://./x" → "./x", etc.
export function normalizeZarrUrl(source: string): string {
  return source.replace(/^zarr(2|3)?:\/\//, "");
}

export function isZarrSource(source: string): boolean {
  return /^zarr(2|3)?:\/\//.test(source);
}

export class AnalysisClient {
  private worker: Worker | null = null;
  private pending: {
    resolve: (v: any) => void;
    reject: (err: Error) => void;
    onProgress?: ProgressCallback;
  } | null = null;

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    this.worker = new Worker(new URL("./analysis_worker.ts", import.meta.url), {
      type: "module",
    });
    this.worker.addEventListener("message", (ev: MessageEvent<OutgoingMsg>) => {
      this.handleMessage(ev.data);
    });
    this.worker.addEventListener("error", (ev: ErrorEvent) => {
      if (this.pending) {
        this.pending.reject(new Error(ev.message || "Worker error"));
        this.pending = null;
      }
    });
    return this.worker;
  }

  private handleMessage(msg: OutgoingMsg): void {
    if (!this.pending) return;
    switch (msg.kind) {
      case "progress": {
        const p = msg as ProgressMsg;
        this.pending.onProgress?.(p.message, p.phase);
        return;
      }
      case "inspectResult": {
        const m = msg as InspectResultMsg;
        this.pending.resolve({
          isMultiscale: m.isMultiscale,
          axes: m.axes,
          scales: m.scales,
        } as LayerInspection);
        this.pending = null;
        return;
      }
      case "analyzeResult": {
        const m = msg as AnalyzeResultMsg;
        this.pending.resolve({
          columns: m.columns,
          rows: m.rows,
          shape: m.shape,
          voxelNm: m.voxelNm,
          labelCount: m.labelCount,
        } as AnalysisResult);
        this.pending = null;
        return;
      }
      case "customResult": {
        const m = msg as CustomResultMsg;
        this.pending.resolve({
          table: m.table,
          plotPngDataUrl: m.plotPngDataUrl,
          fly: m.fly,
          narration: m.narration,
          stdout: m.stdout,
          annotations: m.annotations,
          highlight: m.highlight,
          addSourceLayer: m.addSourceLayer,
          newLayer: m.newLayer,
        } as CustomAnalysisResult);
        this.pending = null;
        return;
      }
      case "error": {
        const m = msg as ErrorMsg;
        this.pending.reject(new Error(m.message + (m.where ? ` (during ${m.where})` : "")));
        this.pending = null;
        return;
      }
    }
  }

  async inspect(
    url: string,
    defaultVoxelNm: [number, number, number],
    onProgress?: ProgressCallback,
  ): Promise<LayerInspection> {
    const worker = this.ensureWorker();
    if (this.pending) throw new Error("Analysis already in progress");
    return new Promise<LayerInspection>((resolve, reject) => {
      this.pending = { resolve, reject, onProgress };
      worker.postMessage({ kind: "inspect", url, defaultVoxelNm });
    });
  }

  async analyze(
    params: {
      url: string;
      scalePath: string;
      axesOrder: string[];
      voxelNm: [number, number, number];
      offsetNm: [number, number, number];
      maxVoxels: number;
      connectivity?: 1 | 2 | 3;
      alreadyLabeled?: boolean;
    },
    onProgress?: ProgressCallback,
  ): Promise<AnalysisResult> {
    const worker = this.ensureWorker();
    if (this.pending) throw new Error("Analysis already in progress");
    return new Promise<AnalysisResult>((resolve, reject) => {
      this.pending = { resolve, reject, onProgress };
      const req: AnalyzeRequest = {
        kind: "analyze",
        url: params.url,
        scalePath: params.scalePath,
        axesOrder: params.axesOrder,
        voxelNm: params.voxelNm,
        offsetNm: params.offsetNm,
        maxVoxels: params.maxVoxels,
        connectivity: params.connectivity ?? 1,
        alreadyLabeled: params.alreadyLabeled ?? true,
      };
      worker.postMessage(req);
    });
  }

  async customAnalyze(
    params: CustomRequest,
    onProgress?: ProgressCallback,
  ): Promise<CustomAnalysisResult> {
    const worker = this.ensureWorker();
    if (this.pending) throw new Error("Analysis already in progress");
    return new Promise<CustomAnalysisResult>((resolve, reject) => {
      this.pending = { resolve, reject, onProgress };
      worker.postMessage(params);
    });
  }

  cancel(): void {
    if (this.worker) this.worker.postMessage({ kind: "cancel" });
    if (this.pending) {
      this.pending.reject(new Error("Cancelled"));
      this.pending = null;
    }
  }

  terminate(): void {
    this.cancel();
    this.worker?.terminate();
    this.worker = null;
  }
}

// Byte-based budget instead of a flat voxel cap. Pyodide runs on WASM32 with
// a 4 GB hard ceiling; scipy.ndimage.label + regionprops typically allocate
// ~6–10× the input size in intermediates. A 1.5 GB input leaves enough
// headroom for the common single-layer case to finish without OOM.
// Over this threshold we *warn* rather than block — the user can opt in if
// they know their analysis is light (e.g. just a threshold or sum).
export const SAFE_INPUT_BYTES = 1.5 * 1024 * 1024 * 1024; // 1.5 GB

// Kept for backward compatibility; convert to a voxel count at uint8 just
// so any remaining callers don't break, but the UI should prefer the byte
// budget above.
export const DEFAULT_MAX_VOXELS = Math.floor(SAFE_INPUT_BYTES);
