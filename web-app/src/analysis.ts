// Main-thread orchestrator for segmentation analysis. Owns the Worker
// lifecycle and exposes a small typed API (inspect / analyze / cancel) to the
// UI layer. Decoupled from how the results are rendered.

import type {
  OutgoingMsg,
  InspectResultMsg,
  AnalyzeResultMsg,
  ProgressMsg,
  ErrorMsg,
  AnalyzeRequest,
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

// Heuristic: what's a "safe" voxel cap in the browser? uint64 at 64M voxels
// is 512MB, too much. uint32 at 64M is 256MB — also painful. Let's cap at
// 32M voxels; skimage will comfortably handle that in <5s on typical laptops.
export const DEFAULT_MAX_VOXELS = 32_000_000;
