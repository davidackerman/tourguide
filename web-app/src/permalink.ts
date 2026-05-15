import yaml from "js-yaml";
import type { DatasetDescriptor } from "./descriptor.js";
import { parseDescriptor } from "./descriptor.js";

// Analysis-derived tables that aren't backed by a layer.csv URL —
// e.g. the regionprops outputs of Σ Analyze or λ Custom. Embedding
// their rows in the permalink lets a recipient see the computed
// CSV without re-running the analysis.
export interface SharedTable {
  organelle_class: string;
  layer_name: string;
  columns: string[];
  rows: unknown[][];
}

// A single agent turn snapshot for replay on the recipient side. We
// strip large blobs at the share boundary (truncateForShare below);
// the FULL run-time turn cards keep their data in-memory and aren't
// affected.
export interface SerializedTraceItem {
  tool: string;
  args?: Record<string, unknown>;
  result?: unknown;
  error?: string;
  llmMs?: number;
  toolMs?: number;
}

export interface SerializedPlot {
  // PNG data URL. Dropped (set undefined) if oversize at the boundary;
  // title + explanation still ship so the recipient knows what was
  // shown even when the picture isn't embedded.
  png?: string;
  title?: string;
  explanation?: string;
}

export interface SerializedTurn {
  question: string;
  // The agent's final answer text (from onAnswer). Optional because
  // visual-only turns (fly_to / highlight) don't always emit one.
  answer?: string;
  // Status line as last shown — e.g. "✓ Done (4.2 s · think 1.8 s · tools 2.4 s)".
  status?: string;
  // Per-step trace, in order. Result fields trimmed at boundary.
  trace: SerializedTraceItem[];
  // make_plot outputs from this turn.
  plots?: SerializedPlot[];
}

interface PermalinkState {
  descriptor?: DatasetDescriptor;
  query?: string;
  catalogIndex?: number;
  // Full Neuroglancer viewer state (camera position, selected segments,
  // layout, per-layer visibility) captured via viewer.state.toJSON().
  // Restored after descriptor load so the recipient lands on the exact
  // same view the sharer had.
  viewerState?: Record<string, unknown>;
  // True if viewerState was decoded from the NG-format `#!{...}` hash
  // (which Neuroglancer's own UrlHashBinding already applied on viewer
  // mount). False/absent for the legacy `?v=` form, which NG can't read,
  // so we still need to apply it manually.
  viewerStateFromHash?: boolean;
  // Recent custom-analysis prompts (most recent first). Populated into
  // the Custom analysis history dropdown so the recipient can re-run the
  // sharer's queries with one click.
  analysisPrompts?: string[];
  // Computed (non-layer.csv) analysis tables to ship along with the
  // share link. Sender-side picks which to include; recipient re-
  // ingests them into the sql.js DB after the descriptor loads.
  analysisTables?: SharedTable[];
  // Full agent-conversation snapshot — questions, answers, trace
  // items, plots — so the recipient sees the same scrollback the
  // sender did. Tables aren't repeated here; they live in
  // `analysisTables` (and the trace items' table-bearing results
  // are stripped of row data at the boundary to avoid double-embed).
  analysisSession?: SerializedTurn[];
}

function base64UrlEncode(s: string): string {
  const utf8 = new TextEncoder().encode(s);
  let bin = "";
  utf8.forEach((b) => (bin += String.fromCharCode(b)));
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64UrlDecode(s: string): string {
  const padded = s.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (s.length % 4)) % 4);
  const bin = atob(padded);
  const utf8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) utf8[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(utf8);
}

function bytesToBase64Url(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64UrlToBytes(s: string): Uint8Array {
  const padded = s.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (s.length % 4)) % 4);
  const bin = atob(padded);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// gzip a string → bytes via CompressionStream. Available in Chromium /
// Firefox / Safari since 2023. Drops the URL payload size by ~4-5×
// for analysis tables (lots of repeated numeric column names).
async function gzipString(s: string): Promise<Uint8Array> {
  const stream = new Blob([s]).stream().pipeThrough(new CompressionStream("gzip"));
  const buf = await new Response(stream).arrayBuffer();
  return new Uint8Array(buf);
}

async function gunzipToString(bytes: Uint8Array): Promise<string> {
  // Cast around the SharedArrayBuffer | ArrayBuffer split — Blob's
  // BlobPart type narrowed in TS 5.5 and Uint8Array's `buffer` is
  // typed as ArrayBufferLike. At runtime browsers happily take a
  // Uint8Array; the cast is purely a TypeScript appeasement.
  const blob = new Blob([bytes as BlobPart]);
  const stream = blob.stream().pipeThrough(new DecompressionStream("gzip"));
  const buf = await new Response(stream).arrayBuffer();
  return new TextDecoder().decode(buf);
}

// Turn a SharedTable[] into a base64url-of-gzipped-JSON string for the
// 't' query param. Async because CompressionStream is. Returned string
// is URL-safe: no '+' or '/'.
export async function encodeSharedTables(tables: SharedTable[]): Promise<string> {
  if (tables.length === 0) return "";
  const json = JSON.stringify(tables);
  const gz = await gzipString(json);
  return bytesToBase64Url(gz);
}

export async function decodeSharedTables(encoded: string): Promise<SharedTable[]> {
  if (!encoded) return [];
  const bytes = base64UrlToBytes(encoded);
  const json = await gunzipToString(bytes);
  const parsed = JSON.parse(json) as unknown;
  if (!Array.isArray(parsed)) throw new Error("Expected array of tables");
  return parsed as SharedTable[];
}

// Trim a SerializedTurn[] for embedding in a share link.
//
// Design: the share is "replay-friendly" — we keep enough text for the
// recipient to read the agent's reasoning (questions, answers, tool
// args incl. Python code, short results), and we DROP plot PNGs and
// table rows entirely. Plot images regenerate when the recipient
// clicks Replay (which re-runs the saved Python on their backend);
// table rows already ship via `analysisTables`, so leaving them in
// the trace would double-embed. Keeps a 10-turn session well under
// 100 KB even with a full agent trace.
//
// Three caps:
//  - each trace item's `result` / `args` JSON capped at TRUNC_JSON_BYTES
//    chars (replaced with a "[truncated …]" marker if exceeded)
//  - python_on_layers result's `table.rows` stripped (rows live in
//    `analysisTables`)
//  - plot PNG dropped from every saved plot; title + explanation kept
//    as a placeholder so the recipient knows what plot existed
const TRUNC_JSON_BYTES = 10_000;

function truncatedJson(value: unknown): unknown {
  let s: string;
  try {
    s = typeof value === "string" ? value : JSON.stringify(value);
  } catch {
    return "[unserializable]";
  }
  if (s.length <= TRUNC_JSON_BYTES) return value;
  return `[truncated: ${s.length} chars → first ${TRUNC_JSON_BYTES}]\n${s.slice(0, TRUNC_JSON_BYTES)}`;
}

export function truncateSessionForShare(turns: SerializedTurn[]): SerializedTurn[] {
  return turns.map((t) => ({
    question: t.question,
    answer: t.answer,
    status: t.status,
    trace: t.trace.map((it) => {
      // Strip python_on_layers result tables — they're double-embedded
      // via analysisTables; keeping them here would bloat for nothing.
      let result = it.result;
      if (result && typeof result === "object") {
        const r = result as Record<string, unknown>;
        const table = r.table as { rows?: unknown[] } | undefined;
        if (table && Array.isArray(table.rows) && table.rows.length > 0) {
          result = { ...r, table: { ...table, rows: [], _note: "rows live in analysisTables" } };
        }
      }
      return {
        tool: it.tool,
        args: it.args ? (truncatedJson(it.args) as Record<string, unknown>) : undefined,
        result: result !== undefined ? truncatedJson(result) : undefined,
        error: it.error,
        llmMs: it.llmMs,
        toolMs: it.toolMs,
      };
    }),
    // Drop plot PNG bytes always. Keeping title + explanation gives the
    // recipient a placeholder card explaining what plot was here and
    // pointing them at the Replay button for the originating step.
    plots: t.plots?.map((p) => ({
      png: undefined,
      title: p.title,
      explanation: p.explanation,
    })),
  }));
}

export async function encodeSharedSession(turns: SerializedTurn[]): Promise<string> {
  if (turns.length === 0) return "";
  const json = JSON.stringify(truncateSessionForShare(turns));
  const gz = await gzipString(json);
  return bytesToBase64Url(gz);
}

export async function decodeSharedSession(encoded: string): Promise<SerializedTurn[]> {
  if (!encoded) return [];
  const bytes = base64UrlToBytes(encoded);
  const json = await gunzipToString(bytes);
  const parsed = JSON.parse(json) as unknown;
  if (!Array.isArray(parsed)) throw new Error("Expected array of turns");
  return parsed as SerializedTurn[];
}

// Tourguide state goes in the query string; Neuroglancer owns the hash
// (its native `#!{...}` format). Putting our params in the hash made NG
// log "URL hash is expected to be of the form '#!{...}'" and fall back
// to default state on every refresh — they fight over the same slot.
//
// Returned suffix has the form `?d=...&q=...#!<ng-state>` (either half
// optional). The base URL (everything before this suffix) is the bare
// page URL; the caller appends.
export async function encodeState(state: PermalinkState): Promise<string> {
  const params: string[] = [];
  if (state.catalogIndex !== undefined) {
    params.push(`c=${state.catalogIndex}`);
  } else if (state.descriptor) {
    const y = yaml.dump(state.descriptor, { lineWidth: 120 });
    params.push(`d=${base64UrlEncode(y)}`);
  }
  if (state.query) params.push(`q=${encodeURIComponent(state.query)}`);
  if (state.analysisPrompts && state.analysisPrompts.length > 0) {
    params.push(`p=${base64UrlEncode(JSON.stringify(state.analysisPrompts))}`);
  }
  if (state.analysisTables && state.analysisTables.length > 0) {
    const t = await encodeSharedTables(state.analysisTables);
    if (t) params.push(`t=${t}`);
  }
  if (state.analysisSession && state.analysisSession.length > 0) {
    const n = await encodeSharedSession(state.analysisSession);
    if (n) params.push(`n=${n}`);
  }
  const search = params.length > 0 ? "?" + params.join("&") : "";
  // NG accepts its state encoded as `#!<json>` — we just use the same
  // format so a recipient's NG sees what they expect.
  const hash =
    state.viewerState && Object.keys(state.viewerState).length > 0
      ? "#!" + encodeURIComponent(JSON.stringify(state.viewerState))
      : "";
  return search + hash;
}

// Reads tourguide state from the page's query string + Neuroglancer state
// from the hash. Falls back to legacy hash params (`#c=…&q=…&v=…&p=…`)
// for permalinks generated before the query-string move, so old links
// still work.
//
// `analysisTables` is left undefined here even when 't=...' is present
// in the URL — gunzip is async and the rest of decode is sync. The
// caller should also call `decodeSharedTablesFromUrl(search)` to pull
// them out separately.
export function decodeState(search: string, hash: string): PermalinkState {
  const state: PermalinkState = {};
  const queryStr = search.startsWith("?") ? search.slice(1) : search;
  const params = new URLSearchParams(queryStr);

  // Legacy fallback: if the query string is empty but the hash is in the
  // old `#c=…&q=…` form (no `#!` prefix), treat the hash as our params.
  const hashTrimmed = hash.startsWith("#") ? hash.slice(1) : hash;
  const isNgHash = hashTrimmed.startsWith("!") || hashTrimmed.startsWith("+!");
  if (queryStr.length === 0 && hashTrimmed.length > 0 && !isNgHash) {
    const legacy = new URLSearchParams(hashTrimmed);
    legacy.forEach((v, k) => params.set(k, v));
  }

  const c = params.get("c");
  if (c !== null) {
    const idx = Number(c);
    if (Number.isInteger(idx) && idx >= 0) state.catalogIndex = idx;
  }
  const d = params.get("d");
  if (d) {
    try {
      const yamlText = base64UrlDecode(d);
      state.descriptor = parseDescriptor(yamlText);
    } catch (err) {
      console.error("Failed to decode permalink descriptor:", err);
    }
  }
  const q = params.get("q");
  if (q) state.query = q;
  const p = params.get("p");
  if (p) {
    try {
      const parsed = JSON.parse(base64UrlDecode(p));
      if (Array.isArray(parsed)) state.analysisPrompts = parsed.map(String);
    } catch (err) {
      console.error("Failed to decode permalink analysis prompts:", err);
    }
  }

  // Neuroglancer state — the `#!{...}` form, or the legacy `v=` query
  // param if present (for share links built before this commit).
  if (isNgHash) {
    const ngEncoded = hashTrimmed.startsWith("!") ? hashTrimmed.slice(1) : hashTrimmed.slice(2);
    try {
      state.viewerState = JSON.parse(decodeURIComponent(ngEncoded));
      state.viewerStateFromHash = true;
    } catch (err) {
      console.error("Failed to decode NG hash state:", err);
    }
  } else {
    const v = params.get("v");
    if (v) {
      try {
        state.viewerState = JSON.parse(base64UrlDecode(v));
      } catch (err) {
        console.error("Failed to decode permalink viewer state:", err);
      }
    }
  }
  return state;
}

export async function buildPermalinkURL(state: PermalinkState, origin?: string): Promise<string> {
  const base = origin ?? `${window.location.origin}${window.location.pathname}`;
  return base + (await encodeState(state));
}

// Pull shared analysis tables out of a URL's query string, if any.
// Lives separate from decodeState because it's async (gunzip). Caller
// awaits this AFTER decodeState's synchronous output is in hand.
export async function decodeSharedTablesFromUrl(search: string): Promise<SharedTable[]> {
  const queryStr = search.startsWith("?") ? search.slice(1) : search;
  const params = new URLSearchParams(queryStr);
  const t = params.get("t");
  if (!t) return [];
  try {
    return await decodeSharedTables(t);
  } catch (err) {
    console.error("Failed to decode shared analysis tables:", err);
    return [];
  }
}

// Same shape as decodeSharedTablesFromUrl but for the session (`n=`).
// Recipient calls this after decodeState's sync output to get the
// captured turn-by-turn agent history.
export async function decodeSharedSessionFromUrl(search: string): Promise<SerializedTurn[]> {
  const queryStr = search.startsWith("?") ? search.slice(1) : search;
  const params = new URLSearchParams(queryStr);
  const n = params.get("n");
  if (!n) return [];
  try {
    return await decodeSharedSession(n);
  } catch (err) {
    console.error("Failed to decode shared session:", err);
    return [];
  }
}

// Strip layers whose source points at a browser-local FileSystemHandle.
// Those URLs (`zarr://<host>/local-data/<id>/...`) only resolve on the
// machine that picked the folder, so a recipient sees broken layers.
export function descriptorWithoutLocalLayers(d: DatasetDescriptor): {
  cleaned: DatasetDescriptor;
  removed: { name: string; source: string }[];
} {
  const removed: { name: string; source: string }[] = [];
  const cleaned: DatasetDescriptor = {
    ...d,
    layers: d.layers.filter((l) => {
      const sources = Array.isArray(l.source) ? l.source : [l.source];
      // Both `local-data/` (FileSystemHandle URLs) and `api/data/<sess>/`
      // (HF Space synthesized session artifacts) are bound to one
      // user / one container and don't survive a permalink. Strip both.
      const ephemeralHit = sources.find(
        (s) => /\/local-data\//.test(s) || /\.hf\.space\/api\/data\//.test(s),
      );
      if (ephemeralHit) removed.push({ name: l.name, source: ephemeralHit });
      return !ephemeralHit;
    }),
  };
  return { cleaned, removed };
}
