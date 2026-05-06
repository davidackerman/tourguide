import yaml from "js-yaml";
import type { DatasetDescriptor } from "./descriptor.js";
import { parseDescriptor } from "./descriptor.js";

interface PermalinkState {
  descriptor?: DatasetDescriptor;
  query?: string;
  catalogIndex?: number;
  // Full Neuroglancer viewer state (camera position, selected segments,
  // layout, per-layer visibility) captured via viewer.state.toJSON().
  // Restored after descriptor load so the recipient lands on the exact
  // same view the sharer had.
  viewerState?: Record<string, unknown>;
  // Recent custom-analysis prompts (most recent first). Populated into
  // the Custom analysis history dropdown so the recipient can re-run the
  // sharer's queries with one click.
  analysisPrompts?: string[];
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

// Tourguide state goes in the query string; Neuroglancer owns the hash
// (its native `#!{...}` format). Putting our params in the hash made NG
// log "URL hash is expected to be of the form '#!{...}'" and fall back
// to default state on every refresh — they fight over the same slot.
//
// Returned suffix has the form `?d=...&q=...#!<ng-state>` (either half
// optional). The base URL (everything before this suffix) is the bare
// page URL; the caller appends.
export function encodeState(state: PermalinkState): string {
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

export function buildPermalinkURL(state: PermalinkState, origin?: string): string {
  const base = origin ?? `${window.location.origin}${window.location.pathname}`;
  return base + encodeState(state);
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
      const isLocal = /\/local-data\//.test(l.source);
      if (isLocal) removed.push({ name: l.name, source: l.source });
      return !isLocal;
    }),
  };
  return { cleaned, removed };
}
