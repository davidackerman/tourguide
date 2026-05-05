// Glue between the web-app and the optional HF Space analysis backend.
//
// Three responsibilities:
//   1. Health polling / cold-start handling (waitForBackendReady).
//   2. Browser-side of the WS tunnel (openBrowserTunnel): when the
//      backend asks for a local-folder path, we fetch it through the
//      service worker and post bytes back.
//   3. `/api/analysis/run` POST + response-shape adaptation into the
//      existing `CustomAnalysisResult`.
//
// This file is ONLY imported by code paths the user explicitly opts into
// (the "Run on backend" toggle). The default Cloudflare-only behavior of
// the app never touches it, which keeps the static-site deploy unaffected.

import type { CustomAnalysisResult } from "./analysis.js";

export interface BackendHealth {
  ok: boolean;
  version?: string;
  mem_gb_total?: number;
  mem_gb_free?: number;
  queue_depth?: number;
  max_concurrent?: number;
}

export type BackendStatus = "offline" | "waking" | "ready";

export async function fetchHealth(url: string, signal?: AbortSignal): Promise<BackendHealth | null> {
  try {
    const res = await fetch(new URL("api/health", ensureTrailingSlash(url)).toString(), {
      method: "GET",
      signal,
    });
    if (!res.ok) return null;
    const body = (await res.json()) as BackendHealth;
    return body;
  } catch {
    return null;
  }
}

export interface WaitForReadyOptions {
  onProgress?: (state: BackendStatus, msg: string) => void;
  maxMs?: number;
}

/** Poll /api/health with backoff until the backend responds. Handles the
 *  20-60 s HF Space cold-start window without giving up. */
export async function waitForBackendReady(url: string, opts: WaitForReadyOptions = {}): Promise<BackendHealth> {
  const { onProgress, maxMs = 90_000 } = opts;
  const started = Date.now();
  let delay = 500;
  let reportedWaking = false;

  while (true) {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), 5000);
    const h = await fetchHealth(url, ac.signal);
    clearTimeout(timer);
    if (h?.ok) {
      onProgress?.("ready", "Backend ready.");
      return h;
    }
    if (Date.now() - started > maxMs) {
      onProgress?.("offline", "Backend did not respond.");
      throw new Error(`Backend at ${url} did not respond within ${Math.round(maxMs / 1000)}s`);
    }
    if (!reportedWaking) {
      onProgress?.("waking", "Waking backend (HF Spaces cold-start can take 30–60 s)…");
      reportedWaking = true;
    } else {
      onProgress?.("waking", `Waking backend… ${Math.round((Date.now() - started) / 1000)} s`);
    }
    await sleep(delay);
    delay = Math.min(delay * 1.4, 3000);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function ensureTrailingSlash(s: string): string {
  return s.endsWith("/") ? s : `${s}/`;
}

// ---- WS tunnel -------------------------------------------------------------

/** Opens a WebSocket to the backend that the backend can use to request
 *  zarr chunks from the user's browser. Each inbound {req_id, path} is
 *  satisfied by fetching `new URL(path, window.location.href)` — which the
 *  existing service worker intercepts for /local-data/<id>/... URLs. */
export function openBrowserTunnel(
  backendUrl: string,
  sessionId: string,
  opts: { onError?: (err: Error) => void } = {},
): { close: () => void; ready: Promise<void> } {
  const wsUrl = new URL(`ws/bridge/${encodeURIComponent(sessionId)}`, ensureTrailingSlash(backendUrl))
    .toString()
    .replace(/^http/, "ws");
  const ws = new WebSocket(wsUrl);

  const ready = new Promise<void>((resolve, reject) => {
    ws.addEventListener("open", () => resolve(), { once: true });
    ws.addEventListener(
      "error",
      () => {
        reject(new Error(`Could not open tunnel to ${wsUrl}`));
      },
      { once: true },
    );
  });

  ws.addEventListener("message", async (ev) => {
    let msg: { type?: string; req_id?: number; path?: string };
    try {
      msg = JSON.parse(String(ev.data));
    } catch {
      return;
    }
    if (msg.type !== "request" || typeof msg.req_id !== "number" || typeof msg.path !== "string") return;

    try {
      // `path` comes from the backend as the relative URL seen by the service
      // worker, e.g. "local-data/<id>/<path>". Resolve against our own origin.
      const url = new URL(msg.path, window.location.href);
      const res = await fetch(url.toString(), { cache: "no-store" });
      if (!res.ok) {
        ws.send(JSON.stringify({ type: "response", req_id: msg.req_id, found: false }));
        return;
      }
      // Cloudflare Pages returns the SPA index.html (status 200, HTML) for
      // any unknown route. Treat that as not-found — otherwise the remote
      // side tries to parse HTML as zarr JSON and explodes.
      const ct = (res.headers.get("content-type") || "").toLowerCase();
      if (ct.includes("text/html")) {
        ws.send(JSON.stringify({ type: "response", req_id: msg.req_id, found: false }));
        return;
      }
      const buf = await res.arrayBuffer();
      const b64 = bytesToBase64(new Uint8Array(buf));
      ws.send(JSON.stringify({ type: "response", req_id: msg.req_id, found: true, bytes_b64: b64 }));
    } catch (err) {
      ws.send(
        JSON.stringify({
          type: "response",
          req_id: msg.req_id,
          found: false,
          error: (err as Error).message,
        }),
      );
      opts.onError?.(err as Error);
    }
  });

  return {
    ready,
    close: () => {
      try {
        ws.close();
      } catch {
        /* ignore */
      }
    },
  };
}

// Chunk-friendly base64 (btoa+String.fromCharCode chokes on large inputs).
function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + chunk)));
  }
  return btoa(binary);
}

// ---- /api/analysis/run -----------------------------------------------------

export interface RemoteRunBody {
  layers: {
    varName: string;
    url: string;
    scalePath: string;
    axesOrder: string[];
    voxelNm: [number, number, number];
    offsetNm: [number, number, number];
  }[];
  tables: { name: string; columns: string[]; rows: (number | string | null)[][] }[];
  code: string;
  timeoutMs: number;
  sessionId: string;
}

/** POST to the backend and return the parsed result, mapped into the same
 *  shape the Pyodide worker emits so the existing renderer can consume it. */
export async function postAnalysisRequest(
  backendUrl: string,
  body: RemoteRunBody,
): Promise<CustomAnalysisResult> {
  const res = await fetch(new URL("api/analysis/run", ensureTrailingSlash(backendUrl)).toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Remote analysis failed (${res.status}): ${text.slice(0, 400)}`);
  }
  const raw = (await res.json()) as RemoteResponse;
  if (raw.kind === "error") {
    throw new Error(raw.traceback || raw.message || "analysis failed");
  }
  return adaptResponse(raw, backendUrl);
}

interface RemoteResponse {
  kind: "customResult" | "error";
  message?: string;
  traceback?: string;
  table?: { name: string; columns: string[]; rows: (number | string | null)[][] };
  narration?: string;
  stdout?: string;
  plotPngDataUrl?: string;
  fly?: { pos: [number, number, number]; segmentId?: string; layer?: string };
  annotations?: CustomAnalysisResult["annotations"];
  highlight?: CustomAnalysisResult["highlight"];
  addSourceLayer?: CustomAnalysisResult["addSourceLayer"];
  newLayer?: {
    synthesizedId: string;
    name: string;
    type: "image" | "segmentation";
    shape: number[];
    dtype: string;
    serveUrl?: string;
  };
  newMeshLayer?: {
    synthesizedId: string;
    name: string;
    type: "segmentation";
    shape: number[];
    dtype: string;
    meshIds: string[];
    serveUrl?: string;
  };
}

function adaptResponse(raw: RemoteResponse, backendUrl: string): CustomAnalysisResult {
  const out: CustomAnalysisResult = {
    table: raw.table,
    narration: raw.narration,
    stdout: raw.stdout,
    plotPngDataUrl: raw.plotPngDataUrl,
    fly: raw.fly,
    annotations: raw.annotations,
    highlight: raw.highlight,
    addSourceLayer: raw.addSourceLayer,
  };
  // For remote newLayer, instead of reading from the browser's /synthesized/
  // (which doesn't exist on the backend's origin), add the layer pointing at
  // the server-served URL. The frontend has an `addSourceLayer` channel
  // already; we route newLayer through it.
  if (raw.newLayer) {
    const source = `zarr://${raw.newLayer.serveUrl || new URL(`api/data/${raw.newLayer.synthesizedId}/`, ensureTrailingSlash(backendUrl)).toString()}`;
    out.addSourceLayer = {
      source,
      name: raw.newLayer.name,
      type: raw.newLayer.type,
    };
  }
  // Mesh-bearing layers come back as a Neuroglancer precomputed source —
  // the seg voxels and the legacy mesh dir share one layer artifact, so
  // NG renders both the labels and the meshes from a single source.
  if (raw.newMeshLayer) {
    const url = raw.newMeshLayer.serveUrl
      || new URL(`api/data/${raw.newMeshLayer.synthesizedId}/`, ensureTrailingSlash(backendUrl)).toString();
    out.addSourceLayer = {
      source: `precomputed://${url}`,
      name: raw.newMeshLayer.name,
      type: raw.newMeshLayer.type,
    };
  }
  return out;
}
