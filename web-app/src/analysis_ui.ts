// Modal UI for running browser-side segmentation analysis. Lets the user
// pick a segmentation layer + scale, watches progress, and on success injects
// the resulting per-object stats as a new table in the existing DatasetDB so
// the structured browser picks it up for free (sort / paginate / fly-to).

import type { Database } from "sql.js";
import type { DatasetDescriptor, DatasetLayer } from "./descriptor.js";
import type { DatasetDB, IngestedTable } from "./db.js";
import { loadSqlJs } from "./db.js";
import {
  AnalysisClient,
  SAFE_INPUT_BYTES,
  type AnalysisResult,
  type CustomAnalysisResult,
  type LayerInspection,
  type LayerScaleInfo,
  isZarrSource,
  normalizeZarrUrl,
} from "./analysis.js";
import { loadSettings } from "./llm.js";
import {
  fetchHealth,
  openBrowserTunnel,
  postAnalysisRequest,
  waitForBackendReady,
} from "./remote_analysis.js";

export interface AnalysisUICallbacks {
  getDescriptor: () => DatasetDescriptor | null;
  getDB: () => DatasetDB | null;
  setDB: (db: DatasetDB) => void;
  onTableAdded: () => void; // refresh browser UI
}

export function openAnalysisDialog(cb: AnalysisUICallbacks): void {
  const d = cb.getDescriptor();
  if (!d) {
    alert("Load a dataset first.");
    return;
  }
  // Any zarr layer is analyzable — the "Already labeled?" switch lets the
  // user decide whether to treat values as segment ids or threshold to a mask.
  // We don't filter by l.type === "segmentation" because the universal loader
  // often flags label volumes as "image" based on dtype/intensity heuristics.
  const segLayers = d.layers.filter((l) => isZarrSource(l.source));
  if (segLayers.length === 0) {
    alert("No zarr layers in this dataset. Browser analysis currently supports OME-Zarr only.");
    return;
  }

  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  overlay.innerHTML = `
    <div class="modal modal-analysis">
      <div class="modal-header">
        <h2>Analyze segmentation</h2>
        <button class="modal-close" aria-label="Close">×</button>
      </div>
      <div class="modal-body">
        <p class="hint">
          Runs connected-components + regionprops on a downsampled scale of a
          zarr segmentation, entirely in your browser. First run downloads the
          Pyodide Python runtime (~6 MB, cached after).
        </p>
        <div class="form-row">
          <label>
            <span>Layer</span>
            <select data-layer></select>
          </label>
          <label>
            <span>Already labeled?</span>
            <select data-labeled>
              <option value="true" selected>Yes — values are segment ids</option>
              <option value="false">No — run connected-components on mask</option>
            </select>
          </label>
        </div>
        <div data-scales-host>
          <p class="hint" data-inspect-status>Pick a layer to list available scales.</p>
        </div>
        <div class="analysis-progress" data-progress hidden>
          <div class="progress-line" data-progress-text></div>
          <div class="progress-bar"><div class="progress-bar-fill" data-progress-bar></div></div>
        </div>
        <p class="modal-error" data-error></p>
      </div>
      <div class="modal-footer">
        <button class="btn-secondary" data-cancel>Cancel</button>
        <label class="custom-remote-toggle" data-remote-row hidden>
          <input type="checkbox" data-remote-toggle />
          <span data-remote-label>Run on backend</span>
        </label>
        <button class="btn-primary" data-run disabled>Run analysis</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);

  const $ = <T extends HTMLElement>(sel: string): T => overlay.querySelector<T>(sel)!;
  const layerSel = $<HTMLSelectElement>("[data-layer]");
  const labeledSel = $<HTMLSelectElement>("[data-labeled]");
  const scalesHost = $<HTMLDivElement>("[data-scales-host]");
  const inspectStatus = $<HTMLParagraphElement>("[data-inspect-status]");
  const progressEl = $<HTMLDivElement>("[data-progress]");
  const progressText = $<HTMLDivElement>("[data-progress-text]");
  const progressBar = $<HTMLDivElement>("[data-progress-bar]");
  const errEl = $<HTMLParagraphElement>("[data-error]");
  const runBtn = $<HTMLButtonElement>("[data-run]");
  const cancelBtn = $<HTMLButtonElement>("[data-cancel]");
  const closeBtn = $<HTMLButtonElement>(".modal-close");
  const remoteRow = $<HTMLLabelElement>("[data-remote-row]");
  const remoteToggle = $<HTMLInputElement>("[data-remote-toggle]");
  const remoteLabel = $<HTMLSpanElement>("[data-remote-label]");

  const client = new AnalysisClient();

  // Wire the same Run-on-backend toggle the Custom modal has. When
  // checked, instead of using Pyodide we generate a regionprops Python
  // snippet and POST it to the HF Space — keeps one code path on the
  // backend (POST /api/analysis/run) and reuses the result-table plumbing.
  const analysisBackendUrl = loadSettings().analysisBackendUrl.trim();
  if (analysisBackendUrl) {
    remoteRow.hidden = false;
    const shortUrl = analysisBackendUrl.replace(/^https?:\/\//, "").replace(/\/$/, "");
    remoteLabel.innerHTML = `Run on <code>${escapeHtml(shortUrl)}</code> <span class="remote-muted">(checking …)</span>`;
    remoteToggle.disabled = true;
    void (async () => {
      const h = await fetchHealth(analysisBackendUrl);
      if (h?.ok) {
        remoteToggle.disabled = false;
        remoteLabel.innerHTML = `Run on <code>${escapeHtml(shortUrl)}</code> <span class="remote-ok">● ready</span>`;
      } else {
        remoteToggle.disabled = false;
        remoteLabel.innerHTML = `Run on <code>${escapeHtml(shortUrl)}</code> <span class="remote-warn">● unreachable (click Run to retry)</span>`;
      }
    })();
  }

  segLayers.forEach((l, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `${l.name} [${l.type}]${l.organelle_class ? ` — ${l.organelle_class}` : ""}`;
    layerSel.appendChild(opt);
  });

  let currentLayer: DatasetLayer = segLayers[0];
  let currentInspection: LayerInspection | null = null;
  let selectedScaleIdx: number | null = null;

  const close = (): void => {
    client.terminate();
    overlay.remove();
  };
  closeBtn.addEventListener("click", close);
  cancelBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });

  const showError = (msg: string): void => {
    errEl.textContent = msg;
  };
  const clearError = (): void => {
    errEl.textContent = "";
  };
  const showProgress = (msg: string, indeterminate = true): void => {
    progressEl.hidden = false;
    progressText.textContent = msg;
    progressBar.style.width = indeterminate ? "100%" : "0%";
    progressBar.classList.toggle("indeterminate", indeterminate);
  };
  const hideProgress = (): void => {
    progressEl.hidden = true;
  };

  const renderScales = (insp: LayerInspection): void => {
    scalesHost.innerHTML = "";
    if (insp.scales.length === 0) {
      scalesHost.innerHTML = `<p class="hint warn">No scales found.</p>`;
      return;
    }
    const hdr = document.createElement("p");
    hdr.className = "hint";
    hdr.textContent = insp.isMultiscale
      ? `OME-Zarr multiscale pyramid detected (${insp.scales.length} scales). Coarser scales are faster but lower resolution.`
      : "Single-scale zarr — only one resolution available.";
    scalesHost.appendChild(hdr);

    const tbl = document.createElement("table");
    tbl.className = "analysis-scales-table";
    tbl.innerHTML = `
      <thead>
        <tr>
          <th></th>
          <th>Path</th>
          <th>Shape</th>
          <th>Voxel (nm)</th>
          <th>Offset (nm, xyz)</th>
          <th>Size</th>
          <th></th>
        </tr>
      </thead>
      <tbody></tbody>
    `;
    const tb = tbl.querySelector("tbody")!;
    insp.scales.forEach((s, i) => {
      const tr = document.createElement("tr");
      // Warn but don't disable. Pyodide's WASM32 ceiling is ~4 GB, so above
      // roughly 1.5 GB the analysis *may* OOM — but a pure threshold op on
      // uint8 at 3 GB can still succeed, so let the user decide.
      const bytes = s.approxBytes;
      const risky = bytes > SAFE_INPUT_BYTES;
      const veryRisky = bytes > 3 * SAFE_INPUT_BYTES; // ~4.5 GB → guaranteed OOM
      const offStr = s.offsetNm.map((v) => format1(v)).join(", ");
      tr.innerHTML = `
        <td><input type="radio" name="scale" value="${i}" ${veryRisky ? "disabled" : ""}></td>
        <td><code>${escapeHtml(s.path || "(root)")}</code></td>
        <td><code>${s.shape.join("×")}</code></td>
        <td><code>${s.voxelNm.map((v) => format1(v)).join(" × ")}</code></td>
        <td><code>${offStr}</code></td>
        <td>${humanSize(bytes)}</td>
        <td>${
          veryRisky
            ? `<span class="chip chip-bad" title="Exceeds WASM memory limit — will OOM">beyond WASM cap</span>`
            : risky
              ? `<span class="chip chip-warn" title="May OOM: single-layer analysis typically needs ~6–10× the input size">may OOM</span>`
              : ""
        }</td>
      `;
      tb.appendChild(tr);
    });
    scalesHost.appendChild(tbl);

    // Auto-pick the coarsest scale that fits the safe budget.
    const coarsestFitIdx = [...insp.scales].reverse().findIndex((s) => s.approxBytes <= SAFE_INPUT_BYTES);
    if (coarsestFitIdx !== -1) {
      const realIdx = insp.scales.length - 1 - coarsestFitIdx;
      selectedScaleIdx = realIdx;
      const input = tbl.querySelector<HTMLInputElement>(`input[value="${realIdx}"]`);
      if (input) input.checked = true;
      runBtn.disabled = false;
    } else {
      // No scale fits the safe budget — pre-select the smallest that's not
      // beyond WASM cap, still usable if the user accepts the OOM risk.
      const idx = insp.scales.findIndex((s) => s.approxBytes <= 3 * SAFE_INPUT_BYTES);
      if (idx !== -1) {
        selectedScaleIdx = idx;
        const input = tbl.querySelector<HTMLInputElement>(`input[value="${idx}"]`);
        if (input) input.checked = true;
        runBtn.disabled = false;
      }
    }

    tbl.querySelectorAll<HTMLInputElement>('input[type="radio"]').forEach((inp) => {
      inp.addEventListener("change", () => {
        selectedScaleIdx = Number(inp.value);
        runBtn.disabled = false;
      });
    });
  };

  const inspectLayer = async (layer: DatasetLayer): Promise<void> => {
    clearError();
    scalesHost.innerHTML = `<p class="hint">Inspecting ${escapeHtml(layer.name)} …</p>`;
    selectedScaleIdx = null;
    runBtn.disabled = true;
    try {
      const url = normalizeZarrUrl(layer.source);
      const insp = await client.inspect(url, d.voxel_size_nm);
      currentInspection = insp;
      renderScales(insp);
    } catch (err) {
      showError((err as Error).message);
      scalesHost.innerHTML = "";
    }
  };

  layerSel.addEventListener("change", () => {
    currentLayer = segLayers[Number(layerSel.value)];
    void inspectLayer(currentLayer);
  });

  runBtn.addEventListener("click", async () => {
    if (!currentInspection || selectedScaleIdx == null) return;
    clearError();
    const scale: LayerScaleInfo = currentInspection.scales[selectedScaleIdx];
    const useRemote = analysisBackendUrl && remoteToggle.checked;

    // WASM-cap warning only matters for the local Pyodide path. Backend
    // has 16 GB real RAM; skip the prompt there.
    if (!useRemote && scale.approxBytes > SAFE_INPUT_BYTES) {
      const gb = (scale.approxBytes / 1024 ** 3).toFixed(2);
      const ok = confirm(
        `This scale is ${gb} GB. Pyodide has a ~4 GB WASM memory ceiling and ` +
          `scipy typically allocates 6–10× the input size. Analysis may OOM. Continue?`,
      );
      if (!ok) return;
    }
    runBtn.disabled = true;
    inspectStatus.textContent = "";
    const axesOrder = currentInspection.axes.map((a) => a.name);
    const url = normalizeZarrUrl(currentLayer.source);
    showProgress("Starting …");
    let tunnel: { close: () => void } | null = null;
    try {
      if (useRemote) {
        await waitForBackendReady(analysisBackendUrl, {
          onProgress: (_state, msg) => showProgress(msg),
        });
        const sessionId =
          typeof crypto?.randomUUID === "function"
            ? crypto.randomUUID()
            : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
        if (/\/local-data\//.test(url)) {
          showProgress("Opening tunnel to browser data …");
          const t = openBrowserTunnel(analysisBackendUrl, sessionId);
          await t.ready;
          tunnel = t;
        }
        showProgress("Running regionprops on backend …");
        const remote: CustomAnalysisResult = await postAnalysisRequest(analysisBackendUrl, {
          layers: [
            {
              varName: "image",
              url,
              scalePath: scale.path,
              axesOrder,
              voxelNm: scale.voxelNm,
              offsetNm: scale.offsetNm,
            },
          ],
          tables: [],
          code: buildRegionpropsCode(labeledSel.value === "true"),
          timeoutMs: 300_000,
          sessionId,
        });
        if (!remote.table) throw new Error("Backend returned no table.");
        // Adapt the {table: {columns, rows}} response into the AnalysisResult
        // shape that ingestResult already knows how to consume.
        const result: AnalysisResult = {
          columns: remote.table.columns,
          rows: remote.table.rows as (number | string)[][],
          shape: scale.shape,
          voxelNm: scale.voxelNm,
          labelCount: remote.table.rows.length,
        };
        showProgress(`Inserting ${result.rows.length.toLocaleString()} rows …`);
        await ingestResult(cb, currentLayer, scale, result);
        cb.onTableAdded();
        close();
      } else {
        // Local Pyodide path (unchanged).
        const maxVoxels = Math.floor((3 * SAFE_INPUT_BYTES) / Math.max(1, scale.approxBytes / productOf(scale.shape)));
        const result = await client.analyze(
          {
            url,
            scalePath: scale.path,
            axesOrder,
            voxelNm: scale.voxelNm,
            offsetNm: scale.offsetNm,
            maxVoxels,
            alreadyLabeled: labeledSel.value === "true",
          },
          (message) => showProgress(message),
        );
        showProgress(`Inserting ${result.rows.length.toLocaleString()} rows …`);
        await ingestResult(cb, currentLayer, scale, result);
        cb.onTableAdded();
        close();
      }
    } catch (err) {
      hideProgress();
      showError((err as Error).message);
      runBtn.disabled = false;
    } finally {
      tunnel?.close();
    }
  });

  void inspectLayer(currentLayer);
}

function productOf(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

function humanSize(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function format1(n: number): string {
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Backend-side regionprops Python. Mirrors the Pyodide kernel in
// analysis_worker.ts so the schema of the returned table matches exactly:
// object_id, volume, position_xyz, bbox_min/max_xyz, equivalent_diameter,
// n_voxels. Click-to-fly works out of the box because the columns line up.
function buildRegionpropsCode(alreadyLabeled: boolean): string {
  return `
import time as _t, numpy as np
spacing = layers["image"]["spacing"]
offsets = layers["image"]["offsets"]
axes    = layers["image"]["axes"]
arr = image
print(f"input shape={arr.shape} dtype={arr.dtype}", flush=True)

if arr.dtype == np.uint64:
    arr = arr.astype(np.int64, copy=False)

t0 = _t.time()
if ${alreadyLabeled ? "True" : "False"}:
    labels = arr.astype(np.int64, copy=False)
    n_labels = int(labels.max()) if labels.size else 0
else:
    structure = ndi.generate_binary_structure(arr.ndim, 1)
    labels, n_labels = ndi.label(arr > 0, structure=structure)
print(f"label step done in {_t.time()-t0:.2f}s; {n_labels} components", flush=True)

t1 = _t.time()
from skimage.measure import regionprops_table
props = ("label", "area", "bbox", "centroid", "equivalent_diameter_area")
tbl = regionprops_table(labels, spacing=tuple(spacing), properties=props)
print(f"regionprops in {_t.time()-t1:.2f}s; rows={len(tbl['label'])}", flush=True)

voxel_volume = float(np.prod(spacing))
def axis_col(prefix, world_axis):
    if world_axis not in axes: return [0.0] * len(tbl["label"])
    i = axes.index(world_axis); off = offsets[i]
    return [float(v) + off for v in tbl[f"{prefix}-{i}"].tolist()]
def bbox_pair(world_axis):
    if world_axis not in axes:
        n = len(tbl["label"]); return [0.0]*n, [0.0]*n
    i = axes.index(world_axis); s = spacing[i]; off = offsets[i]
    lo = [float(v)*s + off for v in tbl[f"bbox-{i}"].tolist()]
    hi = [float(v)*s + off for v in tbl[f"bbox-{i + len(axes)}"].tolist()]
    return lo, hi

cx = axis_col("centroid", "x"); cy = axis_col("centroid", "y"); cz = axis_col("centroid", "z")
bx0, bx1 = bbox_pair("x"); by0, by1 = bbox_pair("y"); bz0, bz1 = bbox_pair("z")

import pandas as pd
df = pd.DataFrame({
    "object_id":           [int(v) for v in tbl["label"].tolist()],
    "volume":              [float(v) for v in tbl["area"].tolist()],
    "position_x": cx, "position_y": cy, "position_z": cz,
    "bbox_min_x": bx0, "bbox_min_y": by0, "bbox_min_z": bz0,
    "bbox_max_x": bx1, "bbox_max_y": by1, "bbox_max_z": bz1,
    "equivalent_diameter": [float(v) for v in tbl["equivalent_diameter_area"].tolist()],
    "n_voxels":            [int(v / voxel_volume + 0.5) for v in tbl["area"].tolist()],
})
_TG_TABLE = df
_TG_TABLE_NAME = "regionprops"
_TG_NARRATION = f"Regionprops on backend: {len(df)} objects."
`;
}

// Inject the analysis result into the existing DatasetDB as a new table so
// the structured browser picks it up with zero extra wiring.
async function ingestResult(
  cb: AnalysisUICallbacks,
  layer: DatasetLayer,
  scale: LayerScaleInfo,
  result: AnalysisResult,
): Promise<void> {
  let db = cb.getDB();
  if (!db) {
    const SQL = await loadSqlJs();
    const database = new SQL.Database();
    db = { db: database, tables: [] };
    cb.setDB(db);
  }
  // organelle_class doubles as a Python identifier inside make_plot
  // (it becomes df_<organelle_class>). Keep it identifier-safe.
  const baseClass = safeTableName(layer.organelle_class ?? layer.name);
  const scaleSuffix = safeTableName(scale.path || "root");
  const organelleClass = `${baseClass}_computed_${scaleSuffix}`;
  const tableName = organelleClass;
  const types = inferTypes(result.columns, result.rows);
  createTable(db.db, tableName, types, result.columns);
  insertRows(db.db, tableName, result.columns, result.rows);

  const table: IngestedTable = {
    table_name: tableName,
    organelle_class: organelleClass,
    layer_name: layer.name,
    row_count: result.rows.length,
    columns: result.columns,
  };
  // Replace an existing computed table for the same layer+scale so re-runs don't duplicate.
  const existingIdx = db.tables.findIndex((t) => t.table_name === tableName);
  if (existingIdx >= 0) db.tables[existingIdx] = table;
  else db.tables.push(table);
}

function safeTableName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_]/g, "_").toLowerCase();
}

function inferTypes(
  columns: string[],
  rows: (number | string)[][],
): Record<string, "INTEGER" | "REAL" | "TEXT"> {
  const out: Record<string, "INTEGER" | "REAL" | "TEXT"> = {};
  columns.forEach((col, i) => {
    let allInt = true;
    let allNum = true;
    for (const row of rows.slice(0, 200)) {
      const v = row[i];
      if (v === null || v === undefined) continue;
      if (typeof v !== "number") {
        allInt = false;
        allNum = false;
      } else if (!Number.isInteger(v)) {
        allInt = false;
      }
    }
    out[col] = allInt ? "INTEGER" : allNum ? "REAL" : "TEXT";
  });
  return out;
}

function createTable(
  db: Database,
  tableName: string,
  types: Record<string, "INTEGER" | "REAL" | "TEXT">,
  columnOrder: string[],
): void {
  const cols = columnOrder.map((c) => `"${c}" ${types[c]}`).join(", ");
  db.run(`DROP TABLE IF EXISTS "${tableName}";`);
  db.run(`CREATE TABLE "${tableName}" (${cols});`);
}

function insertRows(
  db: Database,
  tableName: string,
  columns: string[],
  rows: (number | string)[][],
): void {
  const placeholders = columns.map(() => "?").join(", ");
  const colList = columns.map((c) => `"${c}"`).join(", ");
  const stmt = db.prepare(
    `INSERT INTO "${tableName}" (${colList}) VALUES (${placeholders});`,
  );
  db.run("BEGIN;");
  try {
    for (const row of rows) {
      const values = row.map((v) => (v === undefined ? null : v));
      stmt.run(values);
    }
    db.run("COMMIT;");
  } catch (e) {
    db.run("ROLLBACK;");
    throw e;
  } finally {
    stmt.free();
  }
}
