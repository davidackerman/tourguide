import yaml from "js-yaml";
import type { DatasetDescriptor } from "./descriptor.js";
import { parseDescriptor } from "./descriptor.js";

interface PermalinkState {
  descriptor?: DatasetDescriptor;
  query?: string;
  catalogIndex?: number;
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
  return state;
}

export function buildPermalinkURL(state: PermalinkState, origin?: string): string {
  const base = origin ?? `${window.location.origin}${window.location.pathname}`;
  return base + encodeState(state);
}
