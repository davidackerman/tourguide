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

export function encodeState(state: PermalinkState): string {
  const parts: string[] = [];
  if (state.catalogIndex !== undefined) {
    parts.push(`c=${state.catalogIndex}`);
  } else if (state.descriptor) {
    const y = yaml.dump(state.descriptor, { lineWidth: 120 });
    parts.push(`d=${base64UrlEncode(y)}`);
  }
  if (state.query) parts.push(`q=${encodeURIComponent(state.query)}`);
  if (state.viewerState && Object.keys(state.viewerState).length > 0) {
    parts.push(`v=${base64UrlEncode(JSON.stringify(state.viewerState))}`);
  }
  if (state.analysisPrompts && state.analysisPrompts.length > 0) {
    parts.push(`p=${base64UrlEncode(JSON.stringify(state.analysisPrompts))}`);
  }
  return parts.length > 0 ? "#" + parts.join("&") : "";
}

export function decodeState(hash: string): PermalinkState {
  const state: PermalinkState = {};
  const h = hash.startsWith("#") ? hash.slice(1) : hash;
  if (!h) return state;
  const params = new URLSearchParams(h);
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
  const v = params.get("v");
  if (v) {
    try {
      state.viewerState = JSON.parse(base64UrlDecode(v));
    } catch (err) {
      console.error("Failed to decode permalink viewer state:", err);
    }
  }
  const p = params.get("p");
  if (p) {
    try {
      const parsed = JSON.parse(base64UrlDecode(p));
      if (Array.isArray(parsed)) state.analysisPrompts = parsed.map(String);
    } catch (err) {
      console.error("Failed to decode permalink analysis prompts:", err);
    }
  }
  return state;
}

export function buildPermalinkURL(state: PermalinkState, origin?: string): string {
  const base = origin ?? `${window.location.origin}${window.location.pathname}`;
  return base + encodeState(state);
}
