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
  // Precomputed segmentation volumes. Backend uses tensorstore's
  // neuroglancer_precomputed driver to read these — no chunk size /
  // encoding / sharding metadata needed on the wire (tensorstore
  // re-reads the info file from baseUrl). Empty / omitted is fine.
  precomputedVolumeLayers?: {
    varName: string;
    baseUrl: string;
    scaleKey: string;
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
 *  shape the Pyodide worker emits so the existing renderer can consume it.
 *  Optional `signal` lets the caller abort the in-flight request — pair
 *  it with `cancelAnalysisRequest` below to also tell the server to
 *  terminate the subprocess (the abort alone just drops the response). */
export async function postAnalysisRequest(
  backendUrl: string,
  body: RemoteRunBody,
  signal?: AbortSignal,
): Promise<CustomAnalysisResult> {
  const res = await withColdStartRetry(backendUrl, signal, () =>
    fetch(new URL("api/analysis/run", ensureTrailingSlash(backendUrl)).toString(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    }),
  );
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    // Try to pull out the FastAPI 'detail' field — that's the
    // human-readable bit. Falls back to the raw body if the response
    // isn't JSON or doesn't have a detail. Cap at 2000 chars so a
    // truly enormous traceback doesn't blow up the alert / modal,
    // but leave enough room for tensorstore's per-driver fallback
    // chains (which run a few hundred chars each).
    let message = text;
    try {
      const parsed = JSON.parse(text) as { detail?: unknown };
      if (typeof parsed.detail === "string") message = parsed.detail;
    } catch {
      /* not JSON; use raw text */
    }
    throw new Error(`Remote analysis failed (${res.status}): ${message.slice(0, 2000)}`);
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

/** Push a long permalink suffix (everything after the page URL —
 *  `?d=...&q=...#!{...}`) to the backend's share store and return a
 *  short id. Throws on failure so the caller can fall back to the
 *  full URL. */
/** Ask the HF backend to probe a remote zarr/n5 source for its
 *  multiscale shape. Used when the browser worker can't read the
 *  source format (currently N5). Response matches the worker's
 *  LayerInspection shape so the existing scale-picker can consume
 *  it unchanged. */
/**
 * Wrap a backend fetch with HF-Space cold-start handling: a network-
 * level failure (fetch throws TypeError) triggers waitForBackendReady,
 * which polls /api/health until the Space comes back. We then retry
 * the original call once. After that retry, if it still fails, throw
 * a clear error instead of the raw 'Failed to fetch' so users see an
 * actionable message.
 *
 * Only catches network-level failures — HTTP 4xx/5xx responses are
 * passed through unchanged (the caller's own status-code handling
 * still runs).
 */
async function withColdStartRetry<T>(
  backendUrl: string,
  signal: AbortSignal | undefined,
  call: () => Promise<T>,
): Promise<T> {
  try {
    return await call();
  } catch (err) {
    if (signal?.aborted || (err as Error).name === "AbortError") throw err;
    const msg = (err as Error).message || "";
    const isNetwork = err instanceof TypeError || /Failed to fetch|NetworkError/i.test(msg);
    if (!isNetwork) throw err;
    try {
      await waitForBackendReady(backendUrl, { maxMs: 90_000 });
      return await call();
    } catch (retryErr) {
      if (signal?.aborted || (retryErr as Error).name === "AbortError") throw retryErr;
      throw new Error(
        `Backend unreachable after a wake-up retry. The HF Space may be sleeping or down — try again in a minute. ` +
          `(Original: ${msg}; retry: ${(retryErr as Error).message || "no further detail"})`,
      );
    }
  }
}

export async function inspectSourceRemote(
  backendUrl: string,
  url: string,
  defaultVoxelNm: [number, number, number],
  signal?: AbortSignal,
): Promise<{
  isMultiscale: boolean;
  axes: { name: string }[];
  scales: {
    path: string;
    shape: number[];
    voxelNm: [number, number, number];
    offsetNm: [number, number, number];
    downsample: [number, number, number];
    approxBytes: number;
  }[];
}> {
  const res = await withColdStartRetry(backendUrl, signal, () =>
    fetch(new URL("api/inspect-source", ensureTrailingSlash(backendUrl)).toString(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, defaultVoxelNm }),
      signal,
    }),
  );
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    let message = text;
    try {
      const parsed = JSON.parse(text) as { detail?: unknown };
      if (typeof parsed.detail === "string") message = parsed.detail;
    } catch {
      /* not JSON */
    }
    throw new Error(`Remote inspect failed (${res.status}): ${message.slice(0, 400)}`);
  }
  return await res.json();
}

export interface ShareCreateResult {
  id: string;
  /** True when the backend wrote to persistent storage (HF Datasets).
   *  False means /tmp fallback — link will die when the Space restarts. */
  persistent: boolean;
}

export async function createShareLink(
  backendUrl: string,
  suffix: string,
): Promise<ShareCreateResult> {
  const res = await fetch(new URL("api/share", ensureTrailingSlash(backendUrl)).toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ suffix }),
  });
  if (!res.ok) {
    throw new Error(`share create failed: HTTP ${res.status}`);
  }
  const body = (await res.json()) as { id?: string; persistent?: boolean };
  if (!body.id) throw new Error("share response missing id");
  return { id: body.id, persistent: body.persistent === true };
}

/** Fetch a previously-stored permalink suffix by id. Returns the raw
 *  string the frontend can splice back into window.location. */
export async function fetchShareLink(
  backendUrl: string,
  shareId: string,
): Promise<string> {
  const res = await fetch(
    new URL(`api/share/${encodeURIComponent(shareId)}`, ensureTrailingSlash(backendUrl)).toString(),
    { method: "GET" },
  );
  if (!res.ok) {
    throw new Error(`share fetch failed: HTTP ${res.status}`);
  }
  const body = (await res.json()) as { suffix?: string };
  if (typeof body.suffix !== "string") throw new Error("share response missing suffix");
  return body.suffix;
}

/** Tell the backend to terminate the subprocess running `sessionId`'s
 *  analysis. Best-effort — the backend may have already finished, or the
 *  Space may be sleeping again. We don't surface failures to the user;
 *  the caller has already aborted the local fetch. */
export async function cancelAnalysisRequest(
  backendUrl: string,
  sessionId: string,
): Promise<void> {
  try {
    await fetch(
      new URL(`api/analysis/cancel/${encodeURIComponent(sessionId)}`, ensureTrailingSlash(backendUrl)).toString(),
      { method: "POST" },
    );
  } catch {
    /* swallow — best-effort */
  }
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
  // Mesh-bearing layers come back as a Neuroglancer precomputed source.
  // Surfaced as `meshLayer` (not addSourceLayer) so the consumer can build
  // a layer spec that disables the default (volume) subsource — meshes
  // render alongside the user's existing seg without a duplicate slab.
  if (raw.newMeshLayer) {
    const url = raw.newMeshLayer.serveUrl
      || new URL(`api/data/${raw.newMeshLayer.synthesizedId}/`, ensureTrailingSlash(backendUrl)).toString();
    out.meshLayer = {
      source: `precomputed://${url}`,
      name: raw.newMeshLayer.name,
      meshIds: raw.newMeshLayer.meshIds,
    };
  }
  return out;
}
