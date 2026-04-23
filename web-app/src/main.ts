import { BundledViewer } from "./bundled_viewer.js";
import { fetchCatalog, fetchDescriptor } from "./catalog.js";
import { openLoaderDialog } from "./loader_ui.js";
import { ingestDescriptor, type DatasetDB } from "./db.js";
import { renderStructuredBrowser } from "./browser.js";
import { renderQueryBox } from "./query_ui.js";
import { openSettingsDialog } from "./settings_ui.js";
import { openAnalysisDialog } from "./analysis_ui.js";
import { openCustomAnalysisDialog } from "./custom_analysis_ui.js";
import { loadSettings, backendFromSettings, type LLMBackend } from "./llm.js";
import { decodeState, buildPermalinkURL } from "./permalink.js";
import { registerServiceWorker, isFsAccessSupported } from "./local_folder.js";
import type { CatalogEntry, DatasetDescriptor } from "./descriptor.js";

const CATALOG_URL = "./catalog.json";

// Register the service worker early (best-effort) so /local-data/ URLs are
// servable as soon as the user picks a folder. Skipped silently in browsers
// that don't support either the SW or FS Access API.
if (isFsAccessSupported()) {
  void registerServiceWorker().catch((err) => {
    console.warn("Service worker registration failed:", err);
  });
}

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} not found`);
  return el as T;
};

const select = $<HTMLSelectElement>("dataset-select");
const loadBtn = $<HTMLButtonElement>("load-data-btn");
const analyzeBtn = $<HTMLButtonElement>("analyze-btn");
const customBtn = $<HTMLButtonElement>("custom-btn");
const settingsBtn = $<HTMLButtonElement>("settings-btn");
const shareBtn = $<HTMLButtonElement>("share-btn");
const backendIndicator = $<HTMLSpanElement>("backend-indicator");
const meta = $<HTMLDivElement>("dataset-meta");
const browserHost = $<HTMLDivElement>("browser-host");
const queryHost = $<HTMLDivElement>("query-host");
const ngHost = $<HTMLDivElement>("ng-host");
const viewer = new BundledViewer(ngHost);

let entries: CatalogEntry[] = [];
let currentDB: DatasetDB | null = null;
let currentDescriptor: DatasetDescriptor | null = null;
let currentCatalogIndex: number | null = null;
let currentIsCustom = false;
let backend: LLMBackend = backendFromSettings(loadSettings());

function updateBackendIndicator(): void {
  if (backend.isReady()) {
    backendIndicator.textContent = backend.name;
    backendIndicator.className = "backend-indicator ok";
  } else {
    backendIndicator.textContent = "No AI";
    backendIndicator.className = "backend-indicator";
  }
}
updateBackendIndicator();

renderQueryBox(queryHost, {
  getDB: () => currentDB,
  getDescriptor: () => currentDescriptor,
  getBackend: () => backend,
  viewer,
});

async function loadEntry(entry: CatalogEntry, index: number): Promise<void> {
  meta.textContent = `Loading ${entry.name}…`;
  meta.classList.add("placeholder");
  let descriptor: DatasetDescriptor;
  const baseUrl = new URL(CATALOG_URL, window.location.href).toString();
  try {
    descriptor = await fetchDescriptor(entry, baseUrl);
  } catch (err) {
    meta.textContent = `Failed to load: ${(err as Error).message}`;
    return;
  }
  currentIsCustom = false;
  currentCatalogIndex = index;
  await applyDescriptor(descriptor, baseUrl);
}

async function applyDescriptor(d: DatasetDescriptor, baseUrl: string | null): Promise<void> {
  currentDescriptor = d;
  viewer.loadDescriptor(d);
  renderMeta(d);
  await ingestAndRender(d, baseUrl);
}

async function ingestAndRender(d: DatasetDescriptor, baseUrl: string | null): Promise<void> {
  // Reset DB for every dataset switch, so a previous dataset's CSV tables
  // (or prior analysis tables) don't leak into the new one.
  currentDB = null;
  const csvLayers = d.layers.filter((l) => l.csv);
  if (csvLayers.length === 0) {
    browserHost.innerHTML = `<p class="placeholder">No organelle CSVs in this dataset — nothing to query yet. Add a <code>csv</code> field to layers in the descriptor to enable the structured browser, or click ∑ Analyze to compute stats on a zarr segmentation.</p>`;
    return;
  }
  browserHost.innerHTML = `<p class="placeholder">Loading ${csvLayers.length} CSV file(s)…</p>`;
  try {
    const db = await ingestDescriptor(d, baseUrl);
    currentDB = db;
    if (db.tables.length === 0) {
      browserHost.innerHTML = `<p class="placeholder">All CSVs failed to load — check the browser console.</p>`;
      return;
    }
    renderStructuredBrowser(browserHost, { db, viewer });
  } catch (err) {
    browserHost.innerHTML = `<p class="placeholder">Failed to ingest CSVs: ${(err as Error).message}</p>`;
    console.error(err);
  }
}

function renderMeta(d: DatasetDescriptor): void {
  meta.classList.remove("placeholder");
  meta.classList.add("dataset-meta");
  const layerLines = d.layers
    .map((l) => `<dt>${l.name}</dt><dd>${l.type}</dd>`)
    .join("");
  meta.innerHTML = `
    <h2>${escapeHtml(d.display_name)}</h2>
    <p>${escapeHtml(d.description ?? "")}</p>
    <dl>
      <dt>name</dt><dd>${escapeHtml(d.name)}</dd>
      <dt>voxel (nm)</dt><dd>${d.voxel_size_nm.join(" × ")}</dd>
      ${d.initial_position ? `<dt>position</dt><dd>${d.initial_position.join(", ")}</dd>` : ""}
      <dt>layers</dt><dd>${d.layers.length}</dd>
      ${layerLines}
    </dl>
  `;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function init(): Promise<void> {
  try {
    const catalog = await fetchCatalog(CATALOG_URL);
    entries = catalog.datasets;
  } catch (err) {
    meta.textContent = `Failed to load catalog: ${(err as Error).message}`;
    return;
  }
  select.innerHTML = "";
  if (entries.length === 0) {
    select.disabled = true;
    const opt = document.createElement("option");
    opt.textContent = "No datasets in catalog";
    select.appendChild(opt);
    return;
  }
  entries.forEach((entry, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = entry.name;
    select.appendChild(opt);
  });
  select.disabled = false;
  select.addEventListener("change", () => {
    const idx = Number(select.value);
    const entry = entries[idx];
    if (entry) void loadEntry(entry, idx);
  });

  const permalinkState = decodeState(window.location.hash);
  if (permalinkState.descriptor) {
    loadDescriptorDirect(permalinkState.descriptor);
  } else if (
    permalinkState.catalogIndex !== undefined &&
    permalinkState.catalogIndex < entries.length
  ) {
    select.value = String(permalinkState.catalogIndex);
    await loadEntry(entries[permalinkState.catalogIndex], permalinkState.catalogIndex);
  } else {
    await loadEntry(entries[0], 0);
  }
  if (permalinkState.query) {
    const input = document.querySelector<HTMLInputElement>(".query-input");
    if (input) input.value = permalinkState.query;
  }
}

function loadDescriptorDirect(d: DatasetDescriptor): void {
  currentIsCustom = true;
  currentCatalogIndex = null;
  void applyDescriptor(d, null);
  const customOptValue = "__custom__";
  let customOpt = select.querySelector<HTMLOptionElement>(`option[value="${customOptValue}"]`);
  if (!customOpt) {
    customOpt = document.createElement("option");
    customOpt.value = customOptValue;
    select.prepend(customOpt);
  }
  customOpt.textContent = `(custom) ${d.name}`;
  select.value = customOptValue;
}

loadBtn.addEventListener("click", () => {
  openLoaderDialog((d) => loadDescriptorDirect(d));
});

analyzeBtn.addEventListener("click", () => {
  openAnalysisDialog({
    getDescriptor: () => currentDescriptor,
    getDB: () => currentDB,
    setDB: (db) => {
      currentDB = db;
      console.log("[analysis] setDB", { tables: db.tables.map((t) => t.table_name) });
    },
    onTableAdded: () => {
      console.log("[analysis] onTableAdded", {
        hasDB: !!currentDB,
        tables: currentDB?.tables.map((t) => ({
          name: t.table_name,
          rows: t.row_count,
          cols: t.columns.length,
        })),
      });
      if (currentDB) renderStructuredBrowser(browserHost, { db: currentDB, viewer });
    },
  });
});

customBtn.addEventListener("click", () => {
  openCustomAnalysisDialog({
    getDescriptor: () => currentDescriptor,
    getDB: () => currentDB,
    setDB: (db) => {
      currentDB = db;
    },
    onTableAdded: () => {
      if (currentDB) renderStructuredBrowser(browserHost, { db: currentDB, viewer });
    },
    getBackend: () => backend,
    viewer,
  });
});

settingsBtn.addEventListener("click", () => {
  openSettingsDialog({
    onChange: (next) => {
      backend = next;
      updateBackendIndicator();
    },
  });
});

shareBtn.addEventListener("click", async () => {
  if (!currentDescriptor) {
    alert("Load a dataset first");
    return;
  }
  const queryInput = document.querySelector<HTMLInputElement>(".query-input");
  const query = queryInput?.value.trim() || undefined;
  const useCatalogIdx =
    !currentIsCustom && currentCatalogIndex !== null ? currentCatalogIndex : undefined;
  const url = buildPermalinkURL({
    catalogIndex: useCatalogIdx,
    descriptor: useCatalogIdx === undefined ? currentDescriptor : undefined,
    query,
  });
  try {
    await navigator.clipboard.writeText(url);
    shareBtn.textContent = "✓ Copied";
    setTimeout(() => (shareBtn.textContent = "🔗 Share"), 1500);
  } catch {
    prompt("Copy this URL:", url);
  }
});

void init();
