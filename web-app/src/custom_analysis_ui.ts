// Arbitrary-Python analysis UI. User picks 1+ zarr layers + their scales,
// writes (or has Gemini write) Python code that operates on the loaded numpy
// arrays plus any DataFrames, and gets back any combination of: a new table,
// a matplotlib figure, a viewer flyTo, or narration text.
//
// Kept separate from analysis_ui.ts so the two flows stay focused — the
// stats/regionprops flow is already a guided wizard, this one is more of an
// REPL.

import type { Database } from "sql.js";
import type { DatasetDescriptor, DatasetLayer } from "./descriptor.js";
import type { DatasetDB, IngestedTable } from "./db.js";
import { loadSqlJs, runQuery } from "./db.js";
import {
  AnalysisClient,
  type CustomAnalysisResult,
  type LayerInspection,
  isZarrSource,
  normalizeZarrUrl,
} from "./analysis.js";
import type { LLMBackend } from "./llm.js";
import type { BundledViewer } from "./bundled_viewer.js";

export interface CustomAnalysisUICallbacks {
  getDescriptor: () => DatasetDescriptor | null;
  getDB: () => DatasetDB | null;
  setDB: (db: DatasetDB) => void;
  onTableAdded: () => void;
  getBackend: () => LLMBackend;
  viewer: BundledViewer;
}

export function openCustomAnalysisDialog(cb: CustomAnalysisUICallbacks): void {
  const d = cb.getDescriptor();
  if (!d) {
    alert("Load a dataset first.");
    return;
  }
  const zarrLayers = d.layers.filter((l) => isZarrSource(l.source));
  if (zarrLayers.length === 0) {
    alert("No zarr layers in this dataset. Custom analysis currently supports OME-Zarr only.");
    return;
  }

  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  overlay.innerHTML = `
    <div class="modal modal-custom">
      <div class="modal-header">
        <h2>Custom analysis</h2>
        <button class="modal-close" aria-label="Close">×</button>
      </div>
      <div class="modal-body">
        <p class="hint">
          Runs arbitrary Python (numpy, scipy, scikit-image, pandas, matplotlib) in
          your browser. Each selected zarr layer is loaded as a numpy array
          bound to its variable name. Any existing CSV / analysis tables are
          available as <code>df_&lt;organelle_class&gt;</code>.
        </p>

        <h3>Layers</h3>
        <div data-layers-host></div>
        <button class="btn-secondary" data-action="add-layer" type="button">+ Add layer</button>

        <h3>Code</h3>
        <p class="hint">Set any of <code>_TG_TABLE</code> (a DataFrame), <code>_TG_FLY</code> (<code>{"pos": [x,y,z], "segment_id": "...", "layer": "..."}</code>), <code>_TG_NARRATION</code> (string). Any matplotlib figure is also auto-captured.</p>
        <textarea class="custom-code" data-code rows="14" placeholder="# Example: contact sites between two label volumes&#10;from scipy.ndimage import binary_dilation&#10;dilated = binary_dilation(mito > 0)&#10;contacts = dilated & (er > 0)&#10;print(f'{int(contacts.sum())} contact voxels')&#10;&#10;_TG_NARRATION = f'Mito-ER contact voxels: {int(contacts.sum())}'"></textarea>

        <div class="custom-ai-row">
          <input type="text" class="custom-ai-prompt" data-ai-prompt placeholder="Or describe what you want Gemini to compute (e.g. 'surface area of contact between mito and ER')" />
          <button class="btn-secondary" data-action="ask-ai" type="button">✨ Ask AI</button>
        </div>

        <div class="analysis-progress" data-progress hidden>
          <div class="progress-line" data-progress-text></div>
          <div class="progress-bar"><div class="progress-bar-fill indeterminate" data-progress-bar></div></div>
        </div>
        <pre class="custom-error" data-error hidden></pre>
        <div class="custom-output" data-output hidden></div>
      </div>
      <div class="modal-footer">
        <button class="btn-secondary" data-action="cancel">Close</button>
        <button class="btn-primary" data-action="run">Run</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);

  const $ = <T extends HTMLElement>(sel: string): T => overlay.querySelector<T>(sel)!;
  const layersHost = $<HTMLDivElement>("[data-layers-host]");
  const codeEl = $<HTMLTextAreaElement>("[data-code]");
  const promptEl = $<HTMLInputElement>("[data-ai-prompt]");
  const errEl = $<HTMLPreElement>("[data-error]");
  const outEl = $<HTMLDivElement>("[data-output]");
  const progressEl = $<HTMLDivElement>("[data-progress]");
  const progressText = $<HTMLDivElement>("[data-progress-text]");
  const runBtn = $<HTMLButtonElement>("[data-action='run']");
  const addLayerBtn = $<HTMLButtonElement>("[data-action='add-layer']");
  const askAiBtn = $<HTMLButtonElement>("[data-action='ask-ai']");
  const cancelBtn = $<HTMLButtonElement>("[data-action='cancel']");
  const closeBtn = $<HTMLButtonElement>(".modal-close");

  const client = new AnalysisClient();

  const close = (): void => {
    client.terminate();
    overlay.remove();
  };
  closeBtn.addEventListener("click", close);
  cancelBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });

  // --- Layer slots ---------------------------------------------------------

  interface LayerSlot {
    layer: DatasetLayer;
    varName: string;
    inspection?: LayerInspection;
    scaleIdx?: number;
    row: HTMLDivElement;
  }
  const slots: LayerSlot[] = [];

  const makeVarName = (layer: DatasetLayer): string => {
    const base = (layer.organelle_class ?? layer.name).replace(/[^a-zA-Z0-9_]/g, "_");
    // Ensure unique among existing slots.
    let n = base || "layer";
    let i = 1;
    while (slots.some((s) => s.varName === n)) {
      i += 1;
      n = `${base}_${i}`;
    }
    return n;
  };

  const renderSlot = async (slot: LayerSlot): Promise<void> => {
    slot.row.innerHTML = `
      <select data-slot-layer>
        ${zarrLayers
          .map(
            (l, i) =>
              `<option value="${i}" ${l.name === slot.layer.name ? "selected" : ""}>${escapeHtml(l.name)} [${l.type}]</option>`,
          )
          .join("")}
      </select>
      <input data-slot-var value="${escapeAttr(slot.varName)}" size="10" />
      <select data-slot-scale><option>Loading…</option></select>
      <button class="btn-remove" data-action="remove" type="button">×</button>
    `;
    slot.row
      .querySelector<HTMLSelectElement>("[data-slot-layer]")!
      .addEventListener("change", async (e) => {
        const idx = Number((e.target as HTMLSelectElement).value);
        slot.layer = zarrLayers[idx];
        slot.inspection = undefined;
        slot.scaleIdx = undefined;
        await loadScales(slot);
      });
    slot.row
      .querySelector<HTMLInputElement>("[data-slot-var]")!
      .addEventListener("change", (e) => {
        slot.varName = (e.target as HTMLInputElement).value.replace(/[^a-zA-Z0-9_]/g, "_") || "layer";
      });
    slot.row
      .querySelector<HTMLButtonElement>("[data-action='remove']")!
      .addEventListener("click", () => {
        const i = slots.indexOf(slot);
        if (i >= 0) slots.splice(i, 1);
        slot.row.remove();
      });
    await loadScales(slot);
  };

  const loadScales = async (slot: LayerSlot): Promise<void> => {
    const scaleSel = slot.row.querySelector<HTMLSelectElement>("[data-slot-scale]")!;
    scaleSel.innerHTML = `<option>Inspecting…</option>`;
    try {
      const url = normalizeZarrUrl(slot.layer.source);
      const insp = await client.inspect(url, d.voxel_size_nm);
      slot.inspection = insp;
      scaleSel.innerHTML = insp.scales
        .map(
          (s, i) =>
            `<option value="${i}">${escapeHtml(s.path || "(root)")} — ${s.shape.join("×")} @ ${s.voxelNm
              .map((v) => v.toFixed(1))
              .join("×")} nm</option>`,
        )
        .join("");
      // Default: coarsest scale under the safety cap.
      let defaultIdx = insp.scales.length - 1;
      for (let i = insp.scales.length - 1; i >= 0; i--) {
        if (insp.scales[i].shape.reduce((a, b) => a * b, 1) <= 32_000_000) {
          defaultIdx = i;
          break;
        }
      }
      scaleSel.value = String(defaultIdx);
      slot.scaleIdx = defaultIdx;
      scaleSel.addEventListener("change", () => {
        slot.scaleIdx = Number(scaleSel.value);
      });
    } catch (err) {
      scaleSel.innerHTML = `<option>inspect failed: ${(err as Error).message.slice(0, 80)}</option>`;
    }
  };

  const addLayer = async (layer?: DatasetLayer): Promise<void> => {
    const l = layer ?? zarrLayers[0];
    const row = document.createElement("div");
    row.className = "custom-layer-row";
    layersHost.appendChild(row);
    const slot: LayerSlot = { layer: l, varName: makeVarName(l), row };
    slots.push(slot);
    await renderSlot(slot);
  };

  addLayerBtn.addEventListener("click", () => void addLayer());
  // Start with one layer selected.
  void addLayer();

  // --- Run -----------------------------------------------------------------

  const showError = (msg: string): void => {
    errEl.hidden = false;
    errEl.textContent = msg;
  };
  const hideError = (): void => {
    errEl.hidden = true;
  };
  const showProgress = (msg: string): void => {
    progressEl.hidden = false;
    progressText.textContent = msg;
  };
  const hideProgress = (): void => {
    progressEl.hidden = true;
  };

  const collectTables = (): { name: string; columns: string[]; rows: (number | string | null)[][] }[] => {
    const db = cb.getDB();
    if (!db) return [];
    const out: { name: string; columns: string[]; rows: (number | string | null)[][] }[] = [];
    for (const t of db.tables) {
      const colList = t.columns.map((c) => `"${c}"`).join(", ");
      const res = runQuery(db.db, `SELECT ${colList} FROM "${t.table_name}";`);
      out.push({
        name: t.organelle_class,
        columns: res.columns,
        rows: res.rows as (number | string | null)[][],
      });
    }
    return out;
  };

  const renderOutput = (result: CustomAnalysisResult): void => {
    outEl.hidden = false;
    outEl.innerHTML = "";
    if (result.narration) {
      const p = document.createElement("p");
      p.className = "custom-narration";
      p.textContent = result.narration;
      outEl.appendChild(p);
    }
    if (result.stdout) {
      const pre = document.createElement("pre");
      pre.className = "custom-stdout";
      pre.textContent = result.stdout;
      outEl.appendChild(pre);
    }
    if (result.plotPngDataUrl) {
      const img = document.createElement("img");
      img.src = result.plotPngDataUrl;
      img.className = "custom-plot";
      outEl.appendChild(img);
    }
    if (result.table) {
      const info = document.createElement("p");
      info.className = "hint";
      info.textContent = `Inserted table '${result.table.name}' with ${result.table.rows.length} rows into the sidebar.`;
      outEl.appendChild(info);
      void ingestCustomTable(cb, result.table);
      cb.onTableAdded();
    }
    if (result.fly) {
      cb.viewer.flyTo(result.fly.pos, result.fly.segmentId, result.fly.layer);
      const info = document.createElement("p");
      info.className = "hint";
      info.textContent = `Flew viewer to (${result.fly.pos.map((n) => n.toFixed(0)).join(", ")}) nm.`;
      outEl.appendChild(info);
    }
  };

  runBtn.addEventListener("click", async () => {
    hideError();
    outEl.hidden = true;
    if (!slots.length) {
      showError("Add at least one layer first.");
      return;
    }
    for (const slot of slots) {
      if (!slot.inspection || slot.scaleIdx == null) {
        showError(`Layer '${slot.varName}' is still inspecting.`);
        return;
      }
    }
    runBtn.disabled = true;
    showProgress("Starting …");
    try {
      const result = await client.customAnalyze(
        {
          kind: "custom",
          layers: slots.map((s) => {
            const scale = s.inspection!.scales[s.scaleIdx!];
            return {
              varName: s.varName,
              url: normalizeZarrUrl(s.layer.source),
              scalePath: scale.path,
              axesOrder: s.inspection!.axes.map((a) => a.name),
              voxelNm: scale.voxelNm,
              offsetNm: scale.offsetNm,
            };
          }),
          tables: collectTables(),
          code: codeEl.value,
          timeoutMs: 60000,
        },
        (m) => showProgress(m),
      );
      hideProgress();
      renderOutput(result);
    } catch (err) {
      hideProgress();
      showError((err as Error).message);
    } finally {
      runBtn.disabled = false;
    }
  });

  askAiBtn.addEventListener("click", async () => {
    hideError();
    const question = promptEl.value.trim();
    if (!question) {
      showError("Type a prompt first (e.g. 'surface area of mito-ER contacts').");
      return;
    }
    const backend = cb.getBackend();
    if (!backend.isReady()) {
      showError("No AI backend configured. Open ⚙ AI to add one.");
      return;
    }
    askAiBtn.disabled = true;
    askAiBtn.textContent = "Thinking…";
    try {
      const systemPrompt = buildSystemPrompt(slots, collectTables());
      const raw = await backend.complete(
        [
          { role: "system", content: systemPrompt },
          { role: "user", content: question },
        ],
        { temperature: 0.1, maxTokens: 2000, jsonMode: true },
      );
      const code = extractPythonCode(raw);
      codeEl.value = code;
    } catch (err) {
      showError("AI request failed: " + (err as Error).message);
    } finally {
      askAiBtn.disabled = false;
      askAiBtn.textContent = "✨ Ask AI";
    }
  });
}

// ---------------------------------------------------------------------------

function buildSystemPrompt(
  slots: { varName: string; layer: DatasetLayer; inspection?: LayerInspection; scaleIdx?: number }[],
  tables: { name: string; columns: string[]; rows: unknown[][] }[],
): string {
  const layerDescs = slots.map((s) => {
    const scale = s.inspection?.scales[s.scaleIdx ?? 0];
    return `- ${s.varName}: numpy ndarray (dtype tbd), shape=${scale?.shape.join("×") ?? "?"}, spacing_nm=${
      scale?.voxelNm.map((v) => v.toFixed(2)).join(",") ?? "?"
    } (xyz), world offset_nm=${scale?.offsetNm.map((v) => v.toFixed(0)).join(",") ?? "?"} (xyz). Each value is a voxel intensity/label.`;
  });
  const tableDescs = tables.map(
    (t) => `- df_${t.name}: pandas DataFrame, columns=[${t.columns.join(", ")}], rows=${t.rows.length}`,
  );
  return `You write Python code that runs in a browser-side Pyodide sandbox.

AVAILABLE VARIABLES:
Layers (numpy ndarrays, already loaded):
${layerDescs.join("\n")}

Per-layer metadata is also available as \`layers["<varName>"]\` with keys: array, spacing (nm per voxel, array-axis order), offsets (world origin nm, array-axis order), axes (array-axis-order list of "x"/"y"/"z").

DataFrames:
${tableDescs.join("\n") || "(none)"}

Libraries (already imported):
- numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy.ndimage as ndi
- from skimage import measure as _sk_measure (available)

OUTPUT CONTRACT (set zero or more):
- \`_TG_TABLE\`: a pandas DataFrame — will be added to the sidebar as a new table. If it has object_id + position_x/y/z columns, rows become click-to-fly.
- \`_TG_TABLE_NAME\`: string — name for the table.
- \`_TG_FLY\`: dict \`{"pos": [x, y, z], "segment_id": "...", "layer": "..."}\` — world-nm position to fly the viewer to.
- \`_TG_NARRATION\`: string — short human-readable summary shown under the output.
- Any matplotlib figure you draw is auto-captured and shown as a PNG.

RULES:
- Return a JSON object exactly of shape: {"code": "...", "explanation": "one-line summary"}
- Output ONLY that JSON object, no markdown fences, no preamble.
- The \`code\` must be a single Python block. Do NOT reassign the provided ndarrays or DataFrames (they already exist as globals — use them by name). Do NOT add 'import' statements for the libs already imported above; you may import standard-lib modules freely.
- Convert positions with array_index * spacing + offset if you need world nm.
- Prefer operating on the loaded arrays directly; don't try to re-read from disk.
- Keep runtime under ~30s; prefer coarsest-scale arrays.`;
}

function extractPythonCode(raw: string): string {
  const trimmed = raw.trim();
  try {
    const obj = JSON.parse(
      trimmed.startsWith("{") ? trimmed : trimmed.slice(trimmed.indexOf("{"), trimmed.lastIndexOf("}") + 1),
    );
    if (typeof obj.code === "string") return obj.code;
  } catch {
    /* fallthrough */
  }
  // Fallback: extract fenced python if the model ignored JSON instructions.
  const fenced = trimmed.match(/```(?:python)?\s*([\s\S]*?)```/);
  if (fenced) return fenced[1].trim();
  return trimmed;
}

async function ingestCustomTable(
  cb: CustomAnalysisUICallbacks,
  tbl: { name: string; columns: string[]; rows: (number | string | null)[][] },
): Promise<void> {
  let db = cb.getDB();
  if (!db) {
    const SQL = await loadSqlJs();
    db = { db: new SQL.Database(), tables: [] };
    cb.setDB(db);
  }
  const tableName = tbl.name.replace(/[^a-zA-Z0-9_]/g, "_").toLowerCase();
  const types = inferTypes(tbl.columns, tbl.rows);
  createTable(db.db, tableName, types, tbl.columns);
  insertRows(db.db, tableName, tbl.columns, tbl.rows);
  const entry: IngestedTable = {
    table_name: tableName,
    organelle_class: tableName,
    layer_name: tableName,
    row_count: tbl.rows.length,
    columns: tbl.columns,
  };
  const existingIdx = db.tables.findIndex((t) => t.table_name === tableName);
  if (existingIdx >= 0) db.tables[existingIdx] = entry;
  else db.tables.push(entry);
}

function inferTypes(
  columns: string[],
  rows: (number | string | null)[][],
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
  rows: (number | string | null)[][],
): void {
  const placeholders = columns.map(() => "?").join(", ");
  const colList = columns.map((c) => `"${c}"`).join(", ");
  const stmt = db.prepare(
    `INSERT INTO "${tableName}" (${colList}) VALUES (${placeholders});`,
  );
  db.run("BEGIN;");
  try {
    for (const row of rows) {
      stmt.run(row.map((v) => (v === undefined ? null : v)));
    }
    db.run("COMMIT;");
  } catch (e) {
    db.run("ROLLBACK;");
    throw e;
  } finally {
    stmt.free();
  }
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
function escapeAttr(s: string): string {
  return escapeHtml(s);
}
