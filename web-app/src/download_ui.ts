// Download zarr layers as .zarr.zip files. Lets the user pick which layers
// from the loaded dataset to bundle up.
//
// Sources we know how to walk:
//   - Local-folder layers (zarr://.../local-data/<id>/<subpath>/)
//     → walk the FileSystemDirectoryHandle, ZIP every file.
//   - Synthesized layers (zarr://.../synthesized/<id>/...)
//     → already implemented in custom_analysis_ui; we duplicate the SW path
//       walk via IndexedDB here so this module is self-contained.
//   - Remote zarrs (zarr://https://..., s3://...)
//     → fetch every file under the layer URL recursively. No directory
//       listing API on raw HTTP, so we follow the zarr v2 metadata
//       (.zgroup -> multiscales -> sN/.zarray -> chunks) and only fetch
//       what we know exists. Slow + bandwidth-heavy; we warn the user.

import type { DatasetDescriptor } from "./descriptor.js";
import { getStoredHandle } from "./local_folder.js";

interface FileEntry {
  path: string; // relative path inside the zip
  bytes: Uint8Array;
}

export function openDownloadDialog(getDescriptor: () => DatasetDescriptor | null): void {
  const d = getDescriptor();
  if (!d) {
    alert("Load a dataset first.");
    return;
  }
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  overlay.innerHTML = `
    <div class="modal modal-download">
      <div class="modal-header">
        <h2>Download layers</h2>
        <button class="modal-close" aria-label="Close">×</button>
      </div>
      <div class="modal-body">
        <p class="hint">Each selected layer is bundled as a single <code>&lt;name&gt;.zarr.zip</code>. Local-folder layers are read directly from your disk (fast). Synthesized layers come from your browser's IndexedDB (fast). Remote zarrs are downloaded from the network (slower).</p>
        <div class="download-layer-list" data-layer-list></div>
        <div class="analysis-progress" data-progress hidden>
          <div class="progress-line" data-progress-text></div>
          <div class="progress-bar"><div class="progress-bar-fill indeterminate"></div></div>
        </div>
        <pre class="custom-error" data-error hidden></pre>
      </div>
      <div class="modal-footer">
        <button class="btn-secondary" data-action="cancel">Close</button>
        <button class="btn-primary" data-action="download" disabled>Download</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);

  const $ = <T extends HTMLElement>(s: string): T => overlay.querySelector<T>(s)!;
  const layerList = $<HTMLDivElement>("[data-layer-list]");
  const errEl = $<HTMLPreElement>("[data-error]");
  const progressEl = $<HTMLDivElement>("[data-progress]");
  const progressText = $<HTMLDivElement>("[data-progress-text]");
  const downloadBtn = $<HTMLButtonElement>("[data-action='download']");
  const close = (): void => overlay.remove();
  $<HTMLButtonElement>("[data-action='cancel']").addEventListener("click", close);
  $<HTMLButtonElement>(".modal-close").addEventListener("click", close);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });

  // Render checkboxes per layer.
  const layers = d.layers.filter((l) => /^zarr/.test(l.source));
  if (layers.length === 0) {
    layerList.innerHTML = `<p class="hint">This dataset has no zarr layers to download.</p>`;
  } else {
    for (const l of layers) {
      const row = document.createElement("label");
      row.className = "download-layer-row";
      const kind = classifySource(l.source);
      row.innerHTML = `
        <input type="checkbox" data-source="${escapeAttr(l.source)}" data-name="${escapeAttr(l.name)}" ${kind === "remote" ? "" : "checked"} />
        <span class="download-layer-name">${escapeHtml(l.name)}</span>
        <span class="download-layer-meta">${l.type} · ${kind}</span>
      `;
      layerList.appendChild(row);
    }
    const refreshBtn = (): void => {
      const checked = layerList.querySelectorAll<HTMLInputElement>("input:checked").length;
      downloadBtn.disabled = checked === 0;
    };
    layerList.addEventListener("change", refreshBtn);
    refreshBtn();
  }

  downloadBtn.addEventListener("click", async () => {
    errEl.hidden = true;
    downloadBtn.disabled = true;
    progressEl.hidden = false;
    const checkedRows = Array.from(
      layerList.querySelectorAll<HTMLInputElement>("input:checked"),
    );
    try {
      for (const row of checkedRows) {
        const layerName = row.dataset.name || "layer";
        const source = row.dataset.source || "";
        progressText.textContent = `Downloading ${layerName}…`;
        const entries = await collectEntries(source, (msg) => {
          progressText.textContent = `${layerName}: ${msg}`;
        });
        if (!entries.length) {
          throw new Error(`Nothing fetched for layer ${layerName}`);
        }
        const zipBytes = buildZip(entries);
        triggerDownload(
          new Blob([zipBytes as BlobPart], { type: "application/zip" }),
          `${safeFilename(layerName)}.zarr.zip`,
        );
      }
      progressEl.hidden = true;
      progressText.textContent = "";
    } catch (err) {
      progressEl.hidden = true;
      errEl.hidden = false;
      errEl.textContent = (err as Error).message;
    } finally {
      downloadBtn.disabled = false;
    }
  });
}

// --- source classification + dispatch ---------------------------------------

function classifySource(source: string): "local" | "synth" | "remote" {
  // Strip zarr:// prefix.
  const url = source.replace(/^zarr(?:\d?):\/\//, "");
  if (url.includes("/local-data/")) return "local";
  if (url.includes("/synthesized/")) return "synth";
  return "remote";
}

async function collectEntries(
  source: string,
  onProgress: (msg: string) => void,
): Promise<FileEntry[]> {
  const url = source.replace(/^zarr(?:\d?):\/\//, "");
  const kind = classifySource(source);
  if (kind === "local") return collectLocal(url, onProgress);
  if (kind === "synth") return collectSynthesized(url, onProgress);
  return collectRemote(url, onProgress);
}

// --- local-folder walk ------------------------------------------------------

async function collectLocal(url: string, onProgress: (msg: string) => void): Promise<FileEntry[]> {
  // url like "https://tourguide-8j4.pages.dev/local-data/<id>/<subpath>/"
  const m = /\/local-data\/([^/]+)(?:\/(.*?))?\/?$/.exec(url);
  if (!m) throw new Error(`Cannot parse local-folder URL: ${url}`);
  const id = decodeURIComponent(m[1]);
  const subpath = m[2] ? decodeURIComponent(m[2]) : "";
  const root = await getStoredHandle(id);
  if (!root) throw new Error(`Local folder '${id}' is not registered (re-pick it with + Load your data).`);
  let dir: FileSystemDirectoryHandle = root;
  for (const part of subpath.split("/").filter(Boolean)) {
    dir = await dir.getDirectoryHandle(part);
  }
  const entries: FileEntry[] = [];
  await walkFS(dir, "", entries, onProgress);
  return entries;
}

async function walkFS(
  dir: FileSystemDirectoryHandle,
  prefix: string,
  out: FileEntry[],
  onProgress: (msg: string) => void,
): Promise<void> {
  // Iterate children. Some browsers expose .entries(); fall back to .keys/.values.
  const it = (dir as unknown as { entries: () => AsyncIterable<[string, FileSystemHandle]> }).entries();
  for await (const [name, handle] of it) {
    if (handle.kind === "file") {
      const file = await (handle as FileSystemFileHandle).getFile();
      out.push({ path: prefix + name, bytes: new Uint8Array(await file.arrayBuffer()) });
      if (out.length % 50 === 0) onProgress(`${out.length} files…`);
    } else if (handle.kind === "directory") {
      await walkFS(handle as FileSystemDirectoryHandle, prefix + name + "/", out, onProgress);
    }
  }
}

// --- synthesized walk (IndexedDB) -------------------------------------------

async function collectSynthesized(url: string, _onProgress: (msg: string) => void): Promise<FileEntry[]> {
  const m = /\/synthesized\/([^?#]+?)\/?(?:[?#]|$)/.exec(url);
  if (!m) throw new Error(`Cannot parse synthesized URL: ${url}`);
  const prefix = decodeURIComponent(m[1]) + "/";
  const db = await new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open("tourguide-synthesized", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("files");
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  const entries: FileEntry[] = [];
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction("files", "readonly");
    const store = tx.objectStore("files");
    const cursor = store.openCursor();
    cursor.onsuccess = () => {
      const c = cursor.result;
      if (!c) {
        resolve();
        return;
      }
      const key = String(c.key);
      if (key.startsWith(prefix)) {
        const v = c.value;
        const bytes =
          v instanceof Uint8Array
            ? v
            : new TextEncoder().encode(typeof v === "string" ? v : JSON.stringify(v));
        entries.push({ path: key.slice(prefix.length), bytes });
      }
      c.continue();
    };
    cursor.onerror = () => reject(cursor.error);
  });
  db.close();
  return entries;
}

// --- remote zarr v2 walk via metadata follow --------------------------------

async function collectRemote(url: string, onProgress: (msg: string) => void): Promise<FileEntry[]> {
  // We can't list HTTP directories. Walk by following zarr v2 metadata:
  //   <root>/.zgroup, <root>/.zattrs
  //   For each multiscale dataset path: .zarray, then enumerate chunks by
  //   shape/chunks/dimension_separator and fetch every existing one.
  const base = url.endsWith("/") ? url : url + "/";
  const entries: FileEntry[] = [];
  const fetchOne = async (rel: string, optional = false): Promise<Uint8Array | null> => {
    const res = await fetch(base + rel);
    if (!res.ok) {
      if (optional || res.status === 404) return null;
      throw new Error(`Failed ${rel}: HTTP ${res.status}`);
    }
    return new Uint8Array(await res.arrayBuffer());
  };

  const zgroup = await fetchOne(".zgroup", true);
  if (zgroup) entries.push({ path: ".zgroup", bytes: zgroup });
  const zattrs = await fetchOne(".zattrs", true);
  if (zattrs) entries.push({ path: ".zattrs", bytes: zattrs });

  const attrsObj: Record<string, unknown> = zattrs
    ? (JSON.parse(new TextDecoder().decode(zattrs)) as Record<string, unknown>)
    : {};
  const ms = (attrsObj.multiscales as Array<{ datasets?: Array<{ path: string }> }>) || [];
  const datasetPaths: string[] = [];
  if (ms.length > 0 && Array.isArray(ms[0].datasets)) {
    for (const d of ms[0].datasets) {
      if (d?.path) datasetPaths.push(d.path);
    }
  } else {
    // Single-array zarr at root.
    datasetPaths.push("");
  }

  for (const dp of datasetPaths) {
    const prefix = dp ? `${dp}/` : "";
    const zarrayBytes = await fetchOne(`${prefix}.zarray`, true);
    if (!zarrayBytes) continue; // not an array (probably a sub-group); skip
    entries.push({ path: `${prefix}.zarray`, bytes: zarrayBytes });
    const meta = JSON.parse(new TextDecoder().decode(zarrayBytes)) as {
      shape: number[];
      chunks: number[];
      dimension_separator?: string;
    };
    const dimSep = meta.dimension_separator || ".";
    const ndim = meta.shape.length;
    const nChunksPerDim = meta.shape.map((s, i) => Math.ceil(s / meta.chunks[i]));
    const totalChunks = nChunksPerDim.reduce((a, b) => a * b, 1);
    onProgress(`scale ${dp || "(root)"}: ${totalChunks.toLocaleString()} chunks`);
    let fetched = 0;
    let presentCount = 0;
    const idx = new Array(ndim).fill(0);
    while (true) {
      const key = idx.map(String).join(dimSep);
      const chunkBytes = await fetchOne(`${prefix}${key}`, true);
      fetched += 1;
      if (chunkBytes) {
        entries.push({ path: `${prefix}${key}`, bytes: chunkBytes });
        presentCount += 1;
      }
      if (fetched % 100 === 0) {
        onProgress(`scale ${dp || "(root)"}: ${fetched}/${totalChunks} fetched (${presentCount} present)`);
      }
      // Increment multidim index in row-major order.
      let d2 = ndim - 1;
      while (d2 >= 0) {
        idx[d2] += 1;
        if (idx[d2] < nChunksPerDim[d2]) break;
        idx[d2] = 0;
        d2 -= 1;
      }
      if (d2 < 0) break;
    }
  }
  return entries;
}

// --- ZIP builder + download trigger -----------------------------------------

function buildZip(entries: FileEntry[]): Uint8Array {
  const enc = new TextEncoder();
  const fileRecords: Uint8Array[] = [];
  const centralDir: Uint8Array[] = [];
  let offset = 0;

  const crcTable = (() => {
    const table = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      let c = i;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      table[i] = c >>> 0;
    }
    return table;
  })();
  const crc32 = (data: Uint8Array): number => {
    let c = 0xffffffff;
    for (let i = 0; i < data.length; i++) c = crcTable[(c ^ data[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };

  for (const e of entries) {
    const nameBytes = enc.encode(e.path);
    const crc = crc32(e.bytes);
    const size = e.bytes.length;
    const lfh = new Uint8Array(30 + nameBytes.length);
    const dv = new DataView(lfh.buffer);
    dv.setUint32(0, 0x04034b50, true);
    dv.setUint16(4, 20, true);
    dv.setUint16(6, 0, true);
    dv.setUint16(8, 0, true);
    dv.setUint16(10, 0, true);
    dv.setUint16(12, 0, true);
    dv.setUint32(14, crc, true);
    dv.setUint32(18, size, true);
    dv.setUint32(22, size, true);
    dv.setUint16(26, nameBytes.length, true);
    dv.setUint16(28, 0, true);
    lfh.set(nameBytes, 30);
    fileRecords.push(lfh, e.bytes);
    const cdh = new Uint8Array(46 + nameBytes.length);
    const cdv = new DataView(cdh.buffer);
    cdv.setUint32(0, 0x02014b50, true);
    cdv.setUint16(4, 20, true);
    cdv.setUint16(6, 20, true);
    cdv.setUint16(8, 0, true);
    cdv.setUint16(10, 0, true);
    cdv.setUint16(12, 0, true);
    cdv.setUint16(14, 0, true);
    cdv.setUint32(16, crc, true);
    cdv.setUint32(20, size, true);
    cdv.setUint32(24, size, true);
    cdv.setUint16(28, nameBytes.length, true);
    cdv.setUint16(30, 0, true);
    cdv.setUint16(32, 0, true);
    cdv.setUint16(34, 0, true);
    cdv.setUint16(36, 0, true);
    cdv.setUint32(38, 0, true);
    cdv.setUint32(42, offset, true);
    cdh.set(nameBytes, 46);
    centralDir.push(cdh);
    offset += lfh.length + e.bytes.length;
  }
  const cdSize = centralDir.reduce((n, b) => n + b.length, 0);
  const cdOffset = offset;
  const eocd = new Uint8Array(22);
  const edv = new DataView(eocd.buffer);
  edv.setUint32(0, 0x06054b50, true);
  edv.setUint16(4, 0, true);
  edv.setUint16(6, 0, true);
  edv.setUint16(8, entries.length, true);
  edv.setUint16(10, entries.length, true);
  edv.setUint32(12, cdSize, true);
  edv.setUint32(16, cdOffset, true);
  edv.setUint16(20, 0, true);
  const total = fileRecords.reduce((n, b) => n + b.length, 0) + cdSize + eocd.length;
  const out = new Uint8Array(total);
  let p = 0;
  for (const b of fileRecords) {
    out.set(b, p);
    p += b.length;
  }
  for (const b of centralDir) {
    out.set(b, p);
    p += b.length;
  }
  out.set(eocd, p);
  return out;
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

function safeFilename(s: string): string {
  return s.replace(/[^a-zA-Z0-9_.-]/g, "_") || "layer";
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function escapeAttr(s: string): string {
  return escapeHtml(s);
}
