// Service worker that intercepts requests under /local-data/<id>/<path...>
// and serves the bytes from a FileSystemDirectoryHandle the page picked
// (via window.showDirectoryPicker). This lets bundled Neuroglancer load
// zarr/n5/precomputed datasets straight from a folder on the user's
// laptop with no upload, no server, no install.
//
// Handles are stored in IndexedDB by the page so they survive page reloads
// and SW restarts.

const DB_NAME = "tourguide-handles";
const STORE = "handles";
// Second IndexedDB for analysis-generated (synthesized) zarrs. The analysis
// worker writes zarr v2 files here; this SW serves them under /synthesized/.
const SYNTH_DB = "tourguide-synthesized";
const SYNTH_STORE = "files";

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(self.clients.claim());
});

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function getHandle(id) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly").objectStore(STORE).get(id);
    tx.onsuccess = () => resolve(tx.result || null);
    tx.onerror = () => reject(tx.error);
  });
}

function openSynthDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(SYNTH_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(SYNTH_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function getSynth(key) {
  const db = await openSynthDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(SYNTH_STORE, "readonly").objectStore(SYNTH_STORE).get(key);
    tx.onsuccess = () => resolve(tx.result == null ? null : tx.result);
    tx.onerror = () => reject(tx.error);
  });
}

async function resolvePath(rootHandle, path) {
  const parts = path.split("/").filter(Boolean);
  if (parts.length === 0) {
    return { kind: "directory", handle: rootHandle };
  }
  let dir = rootHandle;
  for (let i = 0; i < parts.length - 1; i++) {
    try {
      dir = await dir.getDirectoryHandle(parts[i]);
    } catch (e) {
      // Sparse zarr volumes legitimately omit directories for all-fill
      // slabs. Surface those as 404s, not 500s — every consumer (zarrita,
      // our tunnel reader) already handles 404 == "use fill value".
      return { kind: "missing", error: e };
    }
  }
  const last = parts[parts.length - 1];
  // Try as a file first — this is the common case for chunk fetches.
  try {
    const fileHandle = await dir.getFileHandle(last);
    return { kind: "file", handle: fileHandle };
  } catch (e) {
    if (e && e.name === "NotFoundError") {
      // Maybe it's a directory.
      try {
        return { kind: "directory", handle: await dir.getDirectoryHandle(last) };
      } catch {
        return { kind: "missing", error: e };
      }
    }
    return { kind: "missing", error: e };
  }
}

function corsHeaders(extra = {}) {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, HEAD",
    "Access-Control-Allow-Headers": "*",
    "Cross-Origin-Resource-Policy": "cross-origin",
    ...extra,
  };
}

async function serveFile(fileHandle, req) {
  const file = await fileHandle.getFile();
  const range = req.headers.get("Range");
  if (range) {
    const m = /^bytes=(\d+)-(\d*)$/.exec(range);
    if (m) {
      const start = parseInt(m[1], 10);
      const end = m[2] ? parseInt(m[2], 10) : file.size - 1;
      const slice = file.slice(start, end + 1);
      return new Response(slice, {
        status: 206,
        headers: corsHeaders({
          "Content-Range": `bytes ${start}-${end}/${file.size}`,
          "Content-Length": String(end - start + 1),
          "Accept-Ranges": "bytes",
          "Content-Type": file.type || "application/octet-stream",
        }),
      });
    }
  }
  return new Response(file, {
    status: 200,
    headers: corsHeaders({
      "Content-Length": String(file.size),
      "Accept-Ranges": "bytes",
      "Content-Type": file.type || "application/octet-stream",
    }),
  });
}

async function serveSynthesized(id, path, request) {
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders() });
  }
  const key = `${id}/${path}`;
  const value = await getSynth(key);
  if (value == null) {
    return new Response(`Not found: ${path}`, { status: 404, headers: corsHeaders() });
  }
  const blob = value instanceof Uint8Array ? new Blob([value]) : new Blob([value]);
  const size = blob.size;
  const range = request.headers.get("Range");
  if (range) {
    const m = /^bytes=(\d+)-(\d*)$/.exec(range);
    if (m) {
      const start = parseInt(m[1], 10);
      const end = m[2] ? parseInt(m[2], 10) : size - 1;
      const slice = blob.slice(start, end + 1);
      return new Response(slice, {
        status: 206,
        headers: corsHeaders({
          "Content-Range": `bytes ${start}-${end}/${size}`,
          "Content-Length": String(end - start + 1),
          "Accept-Ranges": "bytes",
          "Content-Type": "application/octet-stream",
        }),
      });
    }
  }
  return new Response(blob, {
    status: 200,
    headers: corsHeaders({
      "Content-Length": String(size),
      "Accept-Ranges": "bytes",
      "Content-Type": "application/octet-stream",
    }),
  });
}

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;

  // /synthesized/<id>/<path...> — zarr files written by the analysis worker
  // into the tourguide-synthesized IndexedDB store.
  const mSynth = /^\/synthesized\/([^/]+)(?:\/(.*))?$/.exec(url.pathname);
  if (mSynth) {
    const id = decodeURIComponent(mSynth[1]);
    const path = mSynth[2] ? decodeURIComponent(mSynth[2]) : "";
    event.respondWith(serveSynthesized(id, path, event.request));
    return;
  }

  // /local-data/<id>/<path...> — files under a FileSystemDirectoryHandle.
  const m = /^\/local-data\/([^/]+)(?:\/(.*))?$/.exec(url.pathname);
  if (!m) return;
  const id = decodeURIComponent(m[1]);
  const path = m[2] ? decodeURIComponent(m[2]) : "";
  event.respondWith(
    (async () => {
      if (event.request.method === "OPTIONS") {
        return new Response(null, { status: 204, headers: corsHeaders() });
      }
      try {
        const root = await getHandle(id);
        if (!root) {
          return new Response(`Local folder handle '${id}' not registered`, {
            status: 404,
            headers: corsHeaders(),
          });
        }
        const resolved = await resolvePath(root, path);
        if (resolved.kind === "missing") {
          return new Response(`Not found: ${path}`, {
            status: 404,
            headers: corsHeaders(),
          });
        }
        if (resolved.kind === "file") {
          return serveFile(resolved.handle, event.request);
        }
        const entries = [];
        for await (const [name, h] of resolved.handle.entries()) {
          entries.push({ name, kind: h.kind });
        }
        return new Response(JSON.stringify(entries), {
          status: 200,
          headers: corsHeaders({ "Content-Type": "application/json" }),
        });
      } catch (err) {
        return new Response(`Service worker error: ${err && err.message}`, {
          status: 500,
          headers: corsHeaders(),
        });
      }
    })(),
  );
});
