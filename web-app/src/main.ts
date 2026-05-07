import { BundledViewer } from "./bundled_viewer.js";
import { fetchCatalog, fetchDescriptor } from "./catalog.js";
import { openLoaderDialog } from "./loader_ui.js";
import { ingestDescriptor, type DatasetDB } from "./db.js";
import { renderStructuredBrowser } from "./browser.js";
import { renderQueryBox } from "./query_ui.js";
import { openSettingsDialog } from "./settings_ui.js";
import { openAnalysisDialog } from "./analysis_ui.js";
import { openCustomAnalysisDialog } from "./custom_analysis_ui.js";
import { openDownloadDialog } from "./download_ui.js";
import { loadSettings, backendFromSettings, type LLMBackend } from "./llm.js";
import {
  decodeState,
  buildPermalinkURL,
  descriptorWithoutLocalLayers,
  decodeSharedTablesFromUrl,
  type SharedTable,
} from "./permalink.js";
import { runQuery } from "./db.js";
import { loadPromptHistory, mergePrompts } from "./prompt_history.js";
import { registerServiceWorker, isFsAccessSupported } from "./local_folder.js";
import { createShareLink, fetchHealth, fetchShareLink } from "./remote_analysis.js";
import { openWelcomeDialog, hasSeenWelcome } from "./welcome_ui.js";
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
const downloadBtn = $<HTMLButtonElement>("download-btn");
const settingsBtn = $<HTMLButtonElement>("settings-btn");
const shareBtn = $<HTMLButtonElement>("share-btn");
const backendIndicator = $<HTMLButtonElement>("backend-indicator");
// Make the indicator a quick way into Settings when AI isn't configured —
// matches the title attribute on the button.
backendIndicator.addEventListener("click", () => settingsBtn.click());
const meta = $<HTMLDivElement>("dataset-meta");
const browserHost = $<HTMLDivElement>("browser-host");
const queryHost = $<HTMLDivElement>("query-host");
const ngHost = $<HTMLDivElement>("ng-host");
const viewer = new BundledViewer(ngHost);
// Expose for ad-hoc devtools debugging:
//   __tg.viewer.getNgState() / .getNgViewer().navigationState.coordinateSpace.value
(window as unknown as { __tg?: unknown }).__tg = { viewer };

// Neuroglancer's bindTitle helper rewrites document.title to
// 'neuroglancer' (or '<state> - neuroglancer') whenever the viewer's
// title signal fires. That clobbers our index.html <title>Tourguide</title>
// the moment the viewer mounts. Watch the <title> element and snap
// it back; the rewrite is idempotent so the observer's own writes
// don't re-trigger it.
(() => {
  const TITLE = "Tourguide";
  document.title = TITLE;
  const titleEl = document.querySelector("title");
  if (!titleEl) return;
  new MutationObserver(() => {
    if (document.title !== TITLE && !document.title.endsWith(`- ${TITLE}`)) {
      document.title = TITLE;
    }
  }).observe(titleEl, { childList: true, characterData: true, subtree: true });
})();

let entries: CatalogEntry[] = [];
let currentDB: DatasetDB | null = null;
let currentDescriptor: DatasetDescriptor | null = null;
let currentCatalogIndex: number | null = null;
let currentIsCustom = false;

// Empty-state placeholder shown in the viewer area when no dataset is
// loaded. Replaces what used to be an auto-loaded demo. Gives the user
// big obvious next-step buttons instead of waiting for a load they
// didn't ask for.
function renderEmptyViewerState(): void {
  ngHost.innerHTML = `
    <div class="empty-viewer">
      <div class="empty-viewer-card">
        <h2>No dataset loaded</h2>
        <p>Pick how you'd like to start:</p>
        <div class="empty-viewer-actions">
          <button class="btn-primary" data-empty-action="load">+ Load your data</button>
          <button class="btn-secondary" data-empty-action="catalog" ${entries.length === 0 ? "disabled" : ""}>Browse demo catalog</button>
          <button class="btn-secondary" data-empty-action="welcome">Show welcome again</button>
        </div>
        <p class="hint">
          Tourguide reads OME-Zarr / N5 / precomputed sources directly
          from S3, your lab server, or a local folder. Paste a
          Neuroglancer state JSON to import an existing view.
        </p>
      </div>
    </div>
  `;
  ngHost.querySelector("[data-empty-action='load']")?.addEventListener("click", () => loadBtn.click());
  ngHost.querySelector("[data-empty-action='catalog']")?.addEventListener("click", () => {
    if (entries.length > 0) {
      select.value = "0";
      void loadEntry(entries[0], 0);
    }
  });
  ngHost.querySelector("[data-empty-action='welcome']")?.addEventListener("click", () => {
    openWelcomeDialog(makeWelcomeOptions());
  });
}

// Centralized factory so every welcome-dialog opener uses the same
// callbacks (load-demo / open-loader / settings-changed) without
// having to duplicate the wiring everywhere it's invoked.
function makeWelcomeOptions(): Parameters<typeof openWelcomeDialog>[0] {
  return {
    onOpenLoader: () => loadBtn.click(),
    onLoadDemo: () => {
      // Prefer a non-synthetic catalog entry — the demo catalog has
      // 'demo_synthetic' first (placeholder) and a real OpenOrganelle
      // dataset second. Pick whichever real entry we can find.
      const realIdx = entries.findIndex((e) => !/synthetic|demo_synthetic/i.test(e.name));
      const idx = realIdx >= 0 ? realIdx : 0;
      if (entries[idx]) {
        select.value = String(idx);
        void loadEntry(entries[idx], idx);
      }
    },
    onSettingsChanged: () => {
      backend = backendFromSettings(loadSettings());
      updateBackendIndicator();
    },
  };
}

// On first run, surface the welcome modal automatically. Re-runs are
// silent — user can always reopen via the empty-state's 'Show welcome
// again' button or from Settings.
function maybeShowWelcome(): void {
  if (hasSeenWelcome()) return;
  openWelcomeDialog(makeWelcomeOptions());
}
let backend: LLMBackend = backendFromSettings(loadSettings());

function updateBackendIndicator(): void {
  const ready = backend.isReady();
  if (ready) {
    backendIndicator.textContent = backend.name;
    backendIndicator.className = "backend-indicator ok";
    backendIndicator.title = `${backend.name} ready. Click to change.`;
  } else {
    backendIndicator.textContent = "No AI";
    backendIndicator.className = "backend-indicator";
    backendIndicator.title = "Click to configure an AI backend (needed for Ask + Custom Python).";
  }
  // Custom Python and Ask both depend on AI — visually dim Custom in
  // the topbar when no backend is ready, and update its tooltip so
  // users know why. Σ Analyze and the structured browser don't need
  // AI, so we leave those alone.
  customBtn.disabled = !ready;
  customBtn.title = ready
    ? "Plain-English Python on one or more layers"
    : "Requires AI — set up in Settings";
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

// True while the very first applyDescriptor call (from permalink or default
// catalog entry on page boot) is in flight. Lets us preserve the URL hash
// so NG can restore its state from it; cleared after the first apply, so
// later user-initiated dataset switches strip the (now-stale) hash.
let initialApplyDone = false;

async function applyDescriptor(d: DatasetDescriptor, baseUrl: string | null): Promise<void> {
  // NG auto-syncs its viewer state to the URL hash on every interaction.
  // For *subsequent* dataset switches we strip the hash so a leftover
  // position/segments set from a previous dataset doesn't apply to the
  // wrong layers. But on initial load we MUST keep the hash — that's
  // where a pasted permalink's NG state lives, and stripping it here
  // is what was breaking permalinked centering: NG's setupDefaultViewer
  // reads the hash on mount; if we'd already wiped it, the recipient
  // got default-fit instead of the shared view.
  if (initialApplyDone && window.location.hash) {
    history.replaceState(null, "", window.location.pathname + window.location.search);
  }
  initialApplyDone = true;
  currentDescriptor = d;
  viewer.loadDescriptor(d);
  renderMeta(d);
  await ingestAndRender(d, baseUrl);
  // Intentionally no auto-frame: Neuroglancer fits its camera to the
  // first layer's bounds the moment the data source resolves, which is
  // exactly the framing the user expects. Forcing a position from our
  // side races NG's init and triggers 'Cannot set properties of
  // undefined (localPositionValid)' on some loads. The previous
  // over-zoom we were trying to fix came from descriptorToNgState
  // pre-setting crossSectionScale = max(voxel_nm); that default is
  // gone, so NG handles it correctly with no help from us.
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

async function maybeExpandShareId(): Promise<void> {
  const params = new URLSearchParams(window.location.search);
  const sid = params.get("s");
  if (!sid) return;
  const backendUrl = loadSettings().analysisBackendUrl.trim();
  if (!backendUrl) {
    console.warn("[share] ?s= present but no analysisBackendUrl configured — can't expand");
    return;
  }
  try {
    const suffix = await fetchShareLink(backendUrl, sid);
    // Replace the page URL with the long form and proceed. Suffix is
    // already the `?...#!...` part — strip our own ?s= bit cleanly
    // by rebuilding from origin + pathname.
    const base = window.location.origin + window.location.pathname;
    history.replaceState(null, "", base + suffix);
  } catch (err) {
    console.error("[share] failed to expand short link:", err);
    // Leave the URL as-is; init proceeds with no decoded state and the
    // user lands on the empty welcome screen rather than a broken page.
  }
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

  // Short-link expansion: `?s=<id>` means the real permalink suffix
  // lives in the backend's share store. Fetch it and replace the URL
  // before decode so the rest of the init logic sees a normal
  // permalink. We don't trigger a navigation — pushState keeps the
  // user on the page.
  await maybeExpandShareId();

  const permalinkState = decodeState(window.location.search, window.location.hash);
  if (permalinkState.descriptor) {
    loadDescriptorDirect(permalinkState.descriptor);
  } else if (
    permalinkState.catalogIndex !== undefined &&
    permalinkState.catalogIndex < entries.length
  ) {
    select.value = String(permalinkState.catalogIndex);
    await loadEntry(entries[permalinkState.catalogIndex], permalinkState.catalogIndex);
  } else {
    // No permalink → don't auto-load anything. The empty state in
    // #ng-host gives the user clear next-step buttons (Load / pick
    // from catalog / open Welcome). Auto-loading the first catalog
    // entry was confusing — users who wanted to start clean had to
    // wait for the demo to load before they could load their own.
    renderEmptyViewerState();
    // Show the first-run Welcome modal automatically. Returns
    // immediately if the user has dismissed it before.
    void maybeShowWelcome();
  }
  if (permalinkState.query) {
    const input = document.querySelector<HTMLInputElement>(".query-input");
    if (input) input.value = permalinkState.query;
  }
  // Merge any shared prompts into the local history dropdown — additive,
  // so opening someone else's permalink expands your suggestions instead
  // of replacing them.
  if (permalinkState.analysisPrompts) {
    mergePrompts(permalinkState.analysisPrompts);
  }
  // For NG-hash permalinks (`#!{...}`), NG's own UrlHashBinding applies
  // the state inside setupDefaultViewer the moment the viewer mounts —
  // and loadDescriptor knows to skip its own restoreState in that case,
  // so the hash-applied state survives. We only need to overlay manually
  // for the legacy `?v=` form (pre-hash share links), which NG can't see.
  if (permalinkState.viewerState && !permalinkState.viewerStateFromHash) {
    setTimeout(() => {
      try {
        viewer.applyNgState(permalinkState.viewerState as Record<string, unknown>);
      } catch (err) {
        console.error("Failed to apply permalinked viewer state:", err);
      }
    }, 100);
  }

  // Embedded analysis tables — recipient never ran Σ Analyze themselves
  // so we ingest the rows directly into their sql.js DB. Decoded
  // separately from decodeState because gunzip is async.
  try {
    const tables = await decodeSharedTablesFromUrl(window.location.search);
    if (tables.length > 0) await ingestSharedTables(tables);
  } catch (err) {
    console.error("Failed to ingest shared analysis tables:", err);
  }
}

// Add tables shipped in the permalink to the in-memory DB so the
// structured browser picks them up alongside the layer-CSV-derived
// ones. Mirrors the table-creation path in analysis_ui.ts (same
// safe-name + replace-on-duplicate behavior).
async function ingestSharedTables(tables: SharedTable[]): Promise<void> {
  const { loadSqlJs } = await import("./db.js");
  if (!currentDB) {
    const SQL = await loadSqlJs();
    currentDB = { db: new SQL.Database(), tables: [] };
  }
  const safeName = (s: string): string => s.replace(/[^a-zA-Z0-9_]/g, "_").toLowerCase();
  for (const t of tables) {
    const tableName = safeName(t.organelle_class);
    const types: Record<string, "INTEGER" | "REAL" | "TEXT"> = {};
    t.columns.forEach((col, i) => {
      let allInt = true;
      let allNum = true;
      for (const row of t.rows.slice(0, 200)) {
        const v = row[i];
        if (v === null || v === undefined) continue;
        if (typeof v !== "number") { allInt = false; allNum = false; }
        else if (!Number.isInteger(v)) allInt = false;
      }
      types[col] = allInt ? "INTEGER" : allNum ? "REAL" : "TEXT";
    });
    const cols = t.columns.map((c) => `"${c}" ${types[c]}`).join(", ");
    currentDB.db.run(`DROP TABLE IF EXISTS "${tableName}";`);
    currentDB.db.run(`CREATE TABLE "${tableName}" (${cols});`);
    const stmt = currentDB.db.prepare(
      `INSERT INTO "${tableName}" (${t.columns.map((c) => `"${c}"`).join(", ")}) VALUES (${t.columns.map(() => "?").join(", ")});`,
    );
    currentDB.db.run("BEGIN;");
    try {
      for (const row of t.rows) {
        stmt.run(row.map((v) => (v === undefined ? null : v as number | string)));
      }
      currentDB.db.run("COMMIT;");
    } finally {
      stmt.free();
    }
    const existingIdx = currentDB.tables.findIndex((x) => x.table_name === tableName);
    const ingested = {
      table_name: tableName,
      organelle_class: t.organelle_class,
      layer_name: t.layer_name,
      row_count: t.rows.length,
      columns: t.columns,
    };
    if (existingIdx >= 0) currentDB.tables[existingIdx] = ingested;
    else currentDB.tables.push(ingested);
  }
  // Refresh the structured browser so the imported tables show up
  // in the dropdown immediately.
  renderStructuredBrowser(browserHost, { db: currentDB, viewer });
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
  openLoaderDialog((d, ngState) => {
    loadDescriptorDirect(d);
    // The "Paste NG state" tab passes back the original NG state so we
    // can overlay the camera + selected segments on top of the freshly
    // loaded descriptor. loadDescriptorDirect → applyDescriptor is
    // fire-and-forget (awaits CSV ingest internally), so we defer past
    // its synchronous viewer.loadDescriptor before applying.
    if (ngState) {
      setTimeout(() => {
        try {
          viewer.applyNgState(ngState);
        } catch (err) {
          console.error("Failed to apply NG state from loader:", err);
        }
      }, 200);
    }
  });
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
    viewer,
  });
});

downloadBtn.addEventListener("click", () => {
  openDownloadDialog(() => currentDescriptor);
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
  // Detect layers whose source only resolves on the sharer's machine
  // (FileSystemDirectoryHandle URLs). The recipient won't be able to load
  // them, so we either drop them or bail entirely. Default: ask the user.
  const { cleaned, removed } = descriptorWithoutLocalLayers(currentDescriptor);
  let descriptorForShare: DatasetDescriptor = currentDescriptor;
  let useCatalogIdx =
    !currentIsCustom && currentCatalogIndex !== null ? currentCatalogIndex : undefined;
  if (removed.length > 0) {
    const reasons = removed.map((l) =>
      /\.hf\.space\/api\/data\//.test(l.source)
        ? `${l.name} (synthesized — expires when the analysis Space restarts)`
        : `${l.name} (local-folder pick — only works on your machine)`,
    ).join("\n  ");
    const ok = confirm(
      `${removed.length} layer(s) won't survive a share link:\n  ${reasons}\n\n` +
        `OK = strip those layers from the share link.\n` +
        `Cancel = include them anyway (recipient will see errors).`,
    );
    if (ok) {
      descriptorForShare = cleaned;
      // Catalog-index path implies the original descriptor is hosted; if
      // we cleaned, force inline the trimmed descriptor instead.
      useCatalogIdx = undefined;
    }
  }
  const queryInput = document.querySelector<HTMLInputElement>(".query-input");
  const query = queryInput?.value.trim() || undefined;
  // Snapshot live NG state (camera + selected segments + layout) so the
  // recipient lands on the same view, plus the recent Custom analysis
  // prompts so the dropdown carries over.
  const viewerState = viewer.getNgState() ?? undefined;
  const analysisPrompts = loadPromptHistory().slice(0, 10);
  // Computed analysis tables — anything in the local DB that the
  // recipient can't get just by re-loading the descriptor (i.e., not
  // backed by a layer.csv URL). Convention: organelle_class names
  // produced by Σ Analyze include '_computed_'; λ Custom uses the
  // table name. Either way, if the table doesn't match a layer.csv,
  // it's a candidate for the permalink. Also confirm via the
  // descriptor — a table whose name matches a layer.csv-derived one
  // can be skipped (recipient re-fetches).
  const sharedTables: SharedTable[] = [];
  if (currentDB) {
    const layerCsvOrganelles = new Set(
      descriptorForShare.layers
        .filter((l) => l.csv && l.organelle_class)
        .map((l) => l.organelle_class as string),
    );
    for (const t of currentDB.tables) {
      if (layerCsvOrganelles.has(t.organelle_class)) continue; // recipient gets this from layer.csv
      try {
        const colList = t.columns.map((c) => `"${c}"`).join(", ");
        const result = runQuery(currentDB.db, `SELECT ${colList} FROM "${t.table_name}";`);
        sharedTables.push({
          organelle_class: t.organelle_class,
          layer_name: t.layer_name,
          columns: result.columns,
          rows: result.rows,
        });
      } catch (err) {
        console.warn(`[share] couldn't dump table ${t.table_name}:`, err);
      }
    }
  }
  const url = await buildPermalinkURL({
    catalogIndex: useCatalogIdx,
    descriptor: useCatalogIdx === undefined ? descriptorForShare : undefined,
    query,
    viewerState,
    analysisPrompts: analysisPrompts.length > 0 ? analysisPrompts : undefined,
    analysisTables: sharedTables.length > 0 ? sharedTables : undefined,
  });
  // Soft size warning — most browsers handle URLs up to ~32k chars
  // fine, beyond that gets sketchy (Safari truncates at 80k, etc.).
  // For really big tables tell the user to use the CSV download
  // instead.
  if (url.length > 60_000) {
    const ok = confirm(
      `Share URL is ${url.length.toLocaleString()} characters long — some browsers / chat apps will truncate it. ` +
        `Embedded computed tables: ${sharedTables.length} (${sharedTables.reduce((s, t) => s + t.rows.length, 0).toLocaleString()} rows total).\n\n` +
        `OK = copy anyway. Cancel = abort and download tables as CSV from the structured browser instead.`,
    );
    if (!ok) return;
  }
  // If the backend is configured + reachable, swap the long URL for a
  // short `${origin}/?s=<id>` form by storing the suffix server-side.
  // Long URLs become unwieldy in chat apps and email; the short form
  // is ~30 chars regardless of payload size. Falls back to the long
  // URL if the backend is offline / errors / not configured.
  const backendUrl = loadSettings().analysisBackendUrl.trim();
  let finalUrl = url;
  let shortened = false;
  if (backendUrl) {
    try {
      const suffix = url.slice(url.indexOf("?") >= 0 ? url.indexOf("?") : url.indexOf("#"));
      if (suffix.length > 200) {
        // Check liveness quickly — don't make the user wait on a sleeping
        // Space; if it's not awake, just use the long URL.
        const ac = new AbortController();
        const timer = setTimeout(() => ac.abort(), 3000);
        const h = await fetchHealth(backendUrl, ac.signal);
        clearTimeout(timer);
        if (h?.ok) {
          const id = await createShareLink(backendUrl, suffix);
          const base = url.split("?")[0].split("#")[0];
          finalUrl = `${base}?s=${id}`;
          shortened = true;
        }
      }
    } catch (err) {
      console.warn("[share] short-link failed, using long URL:", err);
    }
  }
  try {
    await navigator.clipboard.writeText(finalUrl);
    const tablesHint = sharedTables.length > 0
      ? ` (${sharedTables.length} table${sharedTables.length === 1 ? "" : "s"} embedded)`
      : "";
    shareBtn.textContent = shortened
      ? `✓ Copied short link${tablesHint}`
      : `✓ Copied${tablesHint}`;
    setTimeout(() => (shareBtn.textContent = "🔗 Share"), 2200);
  } catch {
    prompt("Copy this URL:", finalUrl);
  }
});

void init();
