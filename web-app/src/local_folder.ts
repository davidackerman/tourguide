// Local-folder loading via the File System Access API + a service worker.
// The user picks a directory; we persist its FileSystemDirectoryHandle in
// IndexedDB; the service worker (public/sw.js) then serves any request to
// /local-data/<id>/<path> from that folder. Bundled Neuroglancer fetches
// from those URLs as if they were a normal HTTP host.

const DB_NAME = "tourguide-handles";
const STORE = "handles";

declare global {
  interface Window {
    showDirectoryPicker?: (opts?: {
      mode?: "read" | "readwrite";
      id?: string;
      startIn?: "desktop" | "documents" | "downloads" | "music" | "pictures" | "videos";
    }) => Promise<FileSystemDirectoryHandle>;
  }
  interface FileSystemDirectoryHandle {
    queryPermission?: (opts: { mode: "read" | "readwrite" }) => Promise<"granted" | "denied" | "prompt">;
    requestPermission?: (opts: { mode: "read" | "readwrite" }) => Promise<"granted" | "denied" | "prompt">;
  }
}

export function isFsAccessSupported(): boolean {
  return typeof window !== "undefined" && typeof window.showDirectoryPicker === "function";
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function storeHandle(id: string, handle: FileSystemDirectoryHandle): Promise<void> {
  const db = await openDB();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).put(handle, id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

/** Read a previously-registered FileSystemDirectoryHandle by its id.
 *  Returns null if not found. Used by the download flow to walk the
 *  user's picked folder and ZIP a layer subtree. */
export async function getStoredHandle(id: string): Promise<FileSystemDirectoryHandle | null> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly").objectStore(STORE).get(id);
    tx.onsuccess = () => resolve((tx.result as FileSystemDirectoryHandle) || null);
    tx.onerror = () => reject(tx.error);
  });
}

async function ensureReadPermission(handle: FileSystemDirectoryHandle): Promise<void> {
  if (!handle.queryPermission || !handle.requestPermission) return;
  const opts = { mode: "read" as const };
  const have = await handle.queryPermission(opts);
  if (have === "granted") return;
  const got = await handle.requestPermission(opts);
  if (got !== "granted") throw new Error("Read permission denied for the picked folder");
}

export interface LocalFolderRegistration {
  id: string;
  name: string;
  baseUrl: string; // ready to drop into a zarr://, n5://, or precomputed:// prefix
}

export async function registerServiceWorker(): Promise<void> {
  if (!("serviceWorker" in navigator)) {
    throw new Error("Service Worker not supported in this browser — local-folder loading disabled.");
  }
  // Resolve sw.js relative to the document so it works at any base path
  // (Cloudflare project subpath, GH Pages, localhost).
  const swUrl = new URL("./sw.js", window.location.href);
  await navigator.serviceWorker.register(swUrl, { scope: "./" });
  await navigator.serviceWorker.ready;
}

export async function pickLocalFolder(): Promise<LocalFolderRegistration> {
  if (!isFsAccessSupported()) {
    throw new Error(
      "This browser doesn't support the File System Access API. Use Chrome/Edge, or run a local HTTP server (see the Local server tab).",
    );
  }
  // SW must be running for the folder URL to be servable.
  await registerServiceWorker();
  const handle = await window.showDirectoryPicker!({ mode: "read", id: "tourguide-data", startIn: "documents" });
  await ensureReadPermission(handle);
  // ID is folder-name + short random suffix so two picks of "data" don't collide.
  const suffix = Math.random().toString(36).slice(2, 8);
  const id = `${handle.name.replace(/[^a-zA-Z0-9_.-]/g, "_")}-${suffix}`;
  await storeHandle(id, handle);
  const baseUrl = new URL(
    `local-data/${encodeURIComponent(id)}/`,
    window.location.href,
  ).toString();
  return { id, name: handle.name, baseUrl };
}

/** Build a Neuroglancer source URL pointing at a subpath of a registered folder. */
export function buildLocalSourceUrl(
  reg: LocalFolderRegistration,
  kind: "zarr" | "n5" | "precomputed",
  subpath: string,
): string {
  const sub = subpath.replace(/^\/+/, "");
  return `${kind}://${reg.baseUrl}${sub}`;
}
