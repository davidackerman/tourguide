// Disk-backed saved-state store for the bridge.
//
// The browser can't write to the user's disk (sandbox), so saved workspace
// states are persisted here, in the always-on local bridge process. Each save
// is one JSON file in a common, user-overridable directory; filenames carry a
// datetime so they sort and are findable by hand. The agent learns the
// directory from the `dir` field returned by save/list.
//
//   default dir:  ~/.tourguide/saved-states
//   override:     TG_STATE_DIR=/some/where
//
// File shape == the SavedTourguideState the browser serializes (id, name,
// createdAt, viewerState, descriptorState, tableIds, plotIds, annotations),
// written verbatim so restore can apply it directly.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export function stateDir() {
  const override = process.env.TG_STATE_DIR;
  return override && override.trim()
    ? path.resolve(override)
    : path.join(os.homedir(), ".tourguide", "saved-states");
}

function ensureDir() {
  const dir = stateDir();
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

// A filesystem-safe slug for the optional human name (keeps letters, digits,
// dash, underscore; collapses the rest to '-').
function slug(name) {
  const base = (name ?? "state").toString().trim().toLowerCase();
  const cleaned = base.replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  return cleaned.slice(0, 60) || "state";
}

// Compact, sortable, filename-safe timestamp: 2026-06-04T1305-22 (no colons).
function stamp(iso) {
  return iso.replace(/:/g, "").replace(/\.\d+Z$/, "").replace(/Z$/, "");
}

function fileFor(record) {
  const id8 = (record.id ?? "").toString().slice(0, 8) || "noid";
  return `${slug(record.name)}__${stamp(record.createdAt)}__${id8}.json`;
}

/** Write a full SavedTourguideState to disk. Returns {id, name, createdAt, path, dir}. */
export function saveState(record) {
  const dir = ensureDir();
  const file = path.join(dir, fileFor(record));
  fs.writeFileSync(file, JSON.stringify(record, null, 2));
  return { id: record.id, name: record.name, createdAt: record.createdAt, path: file, dir };
}

function readAll() {
  const dir = stateDir();
  let names = [];
  try {
    names = fs.readdirSync(dir).filter((n) => n.endsWith(".json"));
  } catch {
    return []; // dir not created yet → nothing saved
  }
  const out = [];
  for (const n of names) {
    try {
      const rec = JSON.parse(fs.readFileSync(path.join(dir, n), "utf8"));
      out.push({ record: rec, path: path.join(dir, n) });
    } catch {
      /* skip an unreadable/corrupt file rather than failing the whole list */
    }
  }
  // Newest first by createdAt, falling back to filename.
  out.sort((a, b) => (b.record.createdAt ?? "").localeCompare(a.record.createdAt ?? ""));
  return out;
}

/** Summaries for list_saved_states, plus the directory they live in. */
export function listStates() {
  const dir = stateDir();
  const savedStates = readAll().map(({ record, path: p }) => ({
    id: record.id,
    name: record.name,
    createdAt: record.createdAt,
    path: p,
  }));
  return { savedStates, dir };
}

// --- share states: small state blobs served by id for short LAN links -------
// A Tourguide short link (…?state=<id>) points here instead of carrying the
// whole encoded state in the URL. The recipient's browser fetches it from the
// bridge over the LAN (http→http, same network) — no public hosting needed.

function shareDir() {
  const override = process.env.TG_SHARE_STATE_DIR;
  return override && override.trim()
    ? path.resolve(override)
    : path.join(os.homedir(), ".tourguide", "shared-states");
}

/** Reject ids that aren't plain tokens (no path traversal in the GET handler). */
export function isValidShareId(id) {
  return typeof id === "string" && /^[A-Za-z0-9_-]{1,64}$/.test(id);
}

export function saveShareState(id, state) {
  if (!isValidShareId(id)) throw new Error("bad share id");
  const dir = shareDir();
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, `${id}.json`), JSON.stringify(state));
}

export function getShareState(id) {
  if (!isValidShareId(id)) return null;
  try {
    return JSON.parse(fs.readFileSync(path.join(shareDir(), `${id}.json`), "utf8"));
  } catch {
    return null;
  }
}

/** Full record for a saved id, or null if not on disk. */
export function getState(id) {
  for (const { record } of readAll()) {
    if (record.id === id) return record;
  }
  return null;
}
