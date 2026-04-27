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
  SAFE_INPUT_BYTES,
  type CustomAnalysisResult,
  type LayerInspection,
  isZarrSource,
  normalizeZarrUrl,
} from "./analysis.js";
import type { LLMBackend } from "./llm.js";
import { loadSettings } from "./llm.js";
import type { BundledViewer } from "./bundled_viewer.js";
import {
  fetchHealth,
  openBrowserTunnel,
  postAnalysisRequest,
  waitForBackendReady,
} from "./remote_analysis.js";

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
        <label class="custom-remote-toggle" data-remote-row hidden>
          <input type="checkbox" data-remote-toggle />
          <span data-remote-label>Run on backend</span>
        </label>
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
  const remoteRow = $<HTMLLabelElement>("[data-remote-row]");
  const remoteToggle = $<HTMLInputElement>("[data-remote-toggle]");
  const remoteLabel = $<HTMLSpanElement>("[data-remote-label]");

  const client = new AnalysisClient();

  // Detect analysis backend availability. Show the configured URL so the
  // user knows exactly which Space the toggle would use. If the URL is
  // empty, hide the toggle and point to Settings with a small note.
  const analysisBackendUrl = loadSettings().analysisBackendUrl.trim();
  let backendReady = false;
  remoteRow.hidden = false;
  const shortUrl = analysisBackendUrl
    ? analysisBackendUrl.replace(/^https?:\/\//, "").replace(/\/$/, "")
    : "";
  if (!analysisBackendUrl) {
    // No backend configured: disable the toggle and nudge to Settings.
    remoteToggle.disabled = true;
    remoteToggle.checked = false;
    remoteLabel.innerHTML = `Run on backend <span class="remote-muted">(none — add one in ⚙ Settings)</span>`;
  } else {
    remoteToggle.disabled = true; // re-enabled after the health probe succeeds
    remoteLabel.innerHTML = `Run on <code>${escapeHtml(shortUrl)}</code> <span class="remote-muted">(checking …)</span>`;
    void (async () => {
      const h = await fetchHealth(analysisBackendUrl);
      backendReady = !!h?.ok;
      if (backendReady) {
        remoteToggle.disabled = false;
        remoteLabel.innerHTML = `Run on <code>${escapeHtml(shortUrl)}</code> <span class="remote-ok">● ready</span>`;
      } else {
        // Space might be asleep (cold start) OR gone. We can't distinguish
        // without trying harder; mark as "asleep" but still let the user try.
        remoteToggle.disabled = false;
        remoteLabel.innerHTML = `Run on <code>${escapeHtml(shortUrl)}</code> <span class="remote-warn">● unreachable (click Run to retry; may be asleep or deleted)</span>`;
      }
    })();
  }

  // Auto-enable the toggle when the selected-layer total is bigger than
  // Pyodide can comfortably hold. We re-evaluate whenever a slot changes.
  const refreshRemoteDefault = (): void => {
    if (!analysisBackendUrl || remoteToggle.checked) return;
    const total = slots.reduce(
      (n, s) => n + (s.inspection?.scales[s.scaleIdx ?? 0]?.approxBytes ?? 0),
      0,
    );
    if (total > SAFE_INPUT_BYTES) {
      remoteToggle.checked = true;
    }
  };

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
        .map((s, i) => {
          const risk =
            s.approxBytes > 3 * SAFE_INPUT_BYTES
              ? " [⚠ beyond WASM cap]"
              : s.approxBytes > SAFE_INPUT_BYTES
                ? " [⚠ may OOM]"
                : "";
          return `<option value="${i}">${escapeHtml(s.path || "(root)")} — ${s.shape.join("×")} @ ${s.voxelNm
            .map((v) => v.toFixed(1))
            .join("×")} nm (${humanBytes(s.approxBytes)})${risk}</option>`;
        })
        .join("");
      // Default: coarsest scale under the safe byte budget.
      let defaultIdx = insp.scales.length - 1;
      for (let i = insp.scales.length - 1; i >= 0; i--) {
        if (insp.scales[i].approxBytes <= SAFE_INPUT_BYTES) {
          defaultIdx = i;
          break;
        }
      }
      scaleSel.value = String(defaultIdx);
      slot.scaleIdx = defaultIdx;
      refreshRemoteDefault();
      scaleSel.addEventListener("change", () => {
        slot.scaleIdx = Number(scaleSel.value);
        refreshRemoteDefault();
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
    // Drop any previous Fix-with-AI button — caller decides when to add one.
  };
  // Show an error AND a "Fix with AI" button, which feeds the error back
  // to the LLM with the previous question + failed code as context, gets
  // a corrected version, and replaces the textarea so the user can Run.
  const showErrorWithFix = (msg: string, failedCode: string): void => {
    showError(msg);
    const backend = cb.getBackend();
    if (!backend.isReady() || !lastAiQuestion) return; // no AI configured / no prior Q
    const fixBtn = document.createElement("button");
    fixBtn.className = "btn-secondary btn-fix-ai";
    fixBtn.type = "button";
    fixBtn.textContent = "✨ Fix with AI";
    fixBtn.addEventListener("click", async () => {
      fixBtn.disabled = true;
      fixBtn.textContent = "Thinking…";
      try {
        const aiUseRemote = !!(analysisBackendUrl && remoteToggle.checked);
        const systemPrompt = buildSystemPrompt(slots, collectTables(), aiUseRemote);
        const raw = await backend.complete(
          [
            { role: "system", content: systemPrompt },
            { role: "user", content: lastAiQuestion },
            { role: "assistant", content: failedCode },
            {
              role: "user",
              content:
                "Your code raised this error:\n\n```\n" +
                msg.slice(0, 4000) +
                "\n```\n\nReturn the corrected Python ONLY — no JSON, no fences, no prose. Use the same _TG_* output channels as before.",
            },
          ],
          { temperature: 0.1, maxTokens: 2000 },
        );
        const fixed = extractPythonCode(raw);
        codeEl.value = fixed;
        hideError();
        // Auto-rerun the corrected code so the user doesn't have to click Run.
        runBtn.click();
      } catch (err) {
        showError("Fix-with-AI failed: " + (err as Error).message);
      } finally {
        fixBtn.disabled = false;
        fixBtn.textContent = "✨ Fix with AI";
      }
    });
    // Render below the message but inside the same container.
    errEl.appendChild(document.createElement("br"));
    errEl.appendChild(fixBtn);
  };
  const hideError = (): void => {
    errEl.hidden = true;
    errEl.textContent = "";
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

  // Keep a handle to the most recent code so download buttons work.
  let lastCode: string = "";
  // Track the last AI question so the "Fix with AI" button can include it
  // as conversational context when feeding back an error.
  let lastAiQuestion: string = "";

  const renderOutput = (result: CustomAnalysisResult): void => {
    outEl.hidden = false;
    outEl.innerHTML = "";

    const makeDownloadRow = (): HTMLDivElement => {
      const row = document.createElement("div");
      row.className = "custom-download-row";
      return row;
    };

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
      const row = makeDownloadRow();
      row.appendChild(makeDownloadButton("Download plot (PNG)", () => downloadDataUrl(result.plotPngDataUrl!, "plot.png")));
      outEl.appendChild(row);
    }
    if (result.table) {
      const info = document.createElement("p");
      info.className = "hint";
      info.textContent = `Inserted table '${result.table.name}' with ${result.table.rows.length} rows into the sidebar.`;
      outEl.appendChild(info);
      void ingestCustomTable(cb, result.table);
      cb.onTableAdded();
      const row = makeDownloadRow();
      const tbl = result.table;
      row.appendChild(
        makeDownloadButton(`Download ${tbl.name}.csv`, () => {
          downloadBlob(new Blob([tableToCsv(tbl)], { type: "text/csv" }), `${tbl.name}.csv`);
        }),
      );
      outEl.appendChild(row);
    }
    if (result.fly) {
      cb.viewer.flyTo(result.fly.pos, result.fly.segmentId, result.fly.layer);
      const info = document.createElement("p");
      info.className = "hint";
      info.textContent = `Flew viewer to (${result.fly.pos.map((n) => n.toFixed(0)).join(", ")}) nm.`;
      outEl.appendChild(info);
    }
    if (result.annotations) {
      cb.viewer.addAnnotationLayer(result.annotations.layerName, result.annotations.points);
      const info = document.createElement("p");
      info.className = "hint";
      info.textContent = `Added annotation layer '${result.annotations.layerName}' with ${result.annotations.points.length} points.`;
      outEl.appendChild(info);
    }
    if (result.highlight) {
      cb.viewer.highlightSegments(result.highlight.layer, result.highlight.ids);
      const info = document.createElement("p");
      info.className = "hint";
      info.textContent = `Highlighted ${result.highlight.ids.length} segments in '${result.highlight.layer}'.`;
      outEl.appendChild(info);
    }
    if (result.addSourceLayer) {
      cb.viewer.addLayerFromSpec({
        type: result.addSourceLayer.type,
        name: result.addSourceLayer.name,
        source: result.addSourceLayer.source,
      });
      const info = document.createElement("p");
      info.className = "hint";
      info.textContent = `Added layer '${result.addSourceLayer.name}' from ${result.addSourceLayer.source}.`;
      outEl.appendChild(info);
    }
    if (result.newLayer) {
      const { synthesizedId, name, type } = result.newLayer;
      // Build an origin-relative URL so NG + the SW can both resolve it.
      const url = new URL(`synthesized/${synthesizedId}/`, window.location.href).toString();
      cb.viewer.addLayerFromSpec({
        type,
        name,
        source: `zarr://${url}`,
      });
      const info = document.createElement("p");
      info.className = "hint";
      info.textContent = `Added synthesized ${type} layer '${name}' (${result.newLayer.shape.join("×")} ${result.newLayer.dtype}).`;
      outEl.appendChild(info);
      const row = makeDownloadRow();
      row.appendChild(
        makeDownloadButton(`Download ${name}.zarr.zip`, () => downloadSynthesizedZarr(synthesizedId, name)),
      );
      outEl.appendChild(row);
    }

    // Script download — always available once we've run something.
    const scriptRow = makeDownloadRow();
    scriptRow.appendChild(
      makeDownloadButton("Download .py script", () => {
        const header = buildScriptHeader(slots);
        downloadBlob(
          new Blob([header + "\n" + lastCode + "\n"], { type: "text/x-python" }),
          "analysis.py",
        );
      }),
    );
    outEl.appendChild(scriptRow);
  };

  runBtn.addEventListener("click", async () => {
    hideError();
    outEl.hidden = true;
    if (!slots.length) {
      showError("Add at least one layer first.");
      return;
    }
    lastCode = codeEl.value;
    for (const slot of slots) {
      if (!slot.inspection || slot.scaleIdx == null) {
        showError(`Layer '${slot.varName}' is still inspecting.`);
        return;
      }
    }
    const useRemote = analysisBackendUrl && remoteToggle.checked;

    // Skip the WASM OOM warning when running on the backend — it has 16 GB
    // real RAM and no Pyodide layer, so the budget isn't the same.
    if (!useRemote) {
      const totalBytes = slots.reduce(
        (n, s) => n + (s.inspection!.scales[s.scaleIdx!].approxBytes || 0),
        0,
      );
      if (totalBytes > SAFE_INPUT_BYTES) {
        const gb = (totalBytes / 1024 ** 3).toFixed(2);
        const ok = confirm(
          `Selected layers total ${gb} GB. Pyodide's WASM ceiling is ~4 GB and ` +
            `intermediates typically 2–4× the input size. Analysis may OOM. Continue?`,
        );
        if (!ok) return;
      }
    }

    const layersForRequest = slots.map((s) => {
      const scale = s.inspection!.scales[s.scaleIdx!];
      return {
        varName: s.varName,
        url: normalizeZarrUrl(s.layer.source),
        scalePath: scale.path,
        axesOrder: s.inspection!.axes.map((a) => a.name),
        voxelNm: scale.voxelNm,
        offsetNm: scale.offsetNm,
      };
    });

    runBtn.disabled = true;
    showProgress("Starting …");
    let tunnel: { close: () => void } | null = null;
    try {
      let result: CustomAnalysisResult;
      if (useRemote) {
        // Wait for backend to be ready (handles HF cold start).
        await waitForBackendReady(analysisBackendUrl, {
          onProgress: (_state, msg) => showProgress(msg),
        });
        backendReady = true;
        const needsTunnel = layersForRequest.some((l) => /\/local-data\//.test(l.url));
        const sessionId =
          typeof crypto?.randomUUID === "function"
            ? crypto.randomUUID()
            : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
        if (needsTunnel) {
          showProgress("Opening tunnel to browser data …");
          const t = openBrowserTunnel(analysisBackendUrl, sessionId);
          await t.ready;
          tunnel = t;
        }
        showProgress("Running on backend …");
        // Remote runs get a much longer budget than Pyodide because there
        // is no WASM memory cap and we often operate on whole multi-hundred-
        // million-voxel arrays. 5 min matches the backend sandbox default.
        result = await postAnalysisRequest(analysisBackendUrl, {
          layers: layersForRequest,
          tables: collectTables(),
          code: codeEl.value,
          timeoutMs: 300_000,
          sessionId,
        });
      } else {
        result = await client.customAnalyze(
          {
            kind: "custom",
            layers: layersForRequest,
            tables: collectTables(),
            code: codeEl.value,
            timeoutMs: 60000,
          },
          (m) => showProgress(m),
        );
      }
      hideProgress();
      renderOutput(result);
    } catch (err) {
      hideProgress();
      // Errors from analysis bubble up here (sandbox traceback, fetch failure,
      // or tunnel timeout). If we have an AI configured + a previous prompt,
      // surface the "Fix with AI" button so the user can try a one-click
      // self-correction loop instead of editing the code manually.
      showErrorWithFix((err as Error).message, codeEl.value);
    } finally {
      tunnel?.close();
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
      // Tell the LLM which runtime the code will run in so it picks the
      // right library set (Seung-lab extras are backend-only).
      const aiUseRemote = !!(analysisBackendUrl && remoteToggle.checked);
      const systemPrompt = buildSystemPrompt(slots, collectTables(), aiUseRemote);
      const raw = await backend.complete(
        [
          { role: "system", content: systemPrompt },
          { role: "user", content: question },
        ],
        // Don't use jsonMode — small models mangle embedded Python strings
        // badly (backslashes + quotes inside JSON break the parse). Plain
        // Python + a fence-stripper is much more robust.
        { temperature: 0.1, maxTokens: 2000 },
      );
      const code = extractPythonCode(raw);
      codeEl.value = code;
      lastAiQuestion = question; // remember for the Fix-with-AI loop on errors
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
  useRemote: boolean,
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
  // The Seung-lab stack only exists on the HF backend. In Pyodide the user
  // gets the stdlib scientific stack only. Branch the prompt so the LLM
  // doesn't hallucinate `cc3d.dust(...)` for code that will run locally.
  const runtimeHeader = useRemote
    ? `You write Python code that runs on a cloud HF Space container with ~16 GB RAM and 2 vCPU.`
    : `You write Python code that runs in a browser-side Pyodide sandbox (~2 GB usable memory, WASM).`;
  const librariesBlock = useRemote
    ? `Libraries (already imported; do NOT reimport):
- numpy as np, pandas as pd, matplotlib.pyplot as plt
- scipy.ndimage as ndi
- from skimage import measure as _sk_measure

Extra Seung-lab libraries (also already imported; 10-100× faster than scipy/skimage for label volumes — prefer these when applicable):
- cc3d — connected_components, statistics, dust (remove small components),
         largest_k, each_contiguous_region, each_neighboring_pair.
         Auto-parallelizes via OpenMP; don't bother threading it yourself.
- fastmorph — spherical_erode(labels, radius, anisotropy=...) /
              spherical_dilate / spherical_open / spherical_close. radius is a
              SCALAR in physical units. \`anisotropy\` follows the ARRAY axis
              order (NOT xyz) — for a (Z,Y,X)-shaped array, anisotropy is
              (sz, sy, sx). Easiest: just pass \`anisotropy=layers["<name>"]["spacing"]\`,
              which is already in array-axis order. Do NOT rebuild it into xyz
              order — that produces a wrong structuring element.
              Do NOT pass a per-axis list as radius — that raises a broadcast error.
              Operates DIRECTLY on label volumes: each label is eroded
              independently, fully-eroded labels disappear, labels are
              preserved (no relabeling needed). Do NOT loop over np.unique(labels)
              calling ndi.binary_erosion per-label — one fastmorph call replaces
              that entire pattern and is 100-1000× faster.
              **Default the radius to one voxel** when the user doesn't
              specify a size — i.e. \`radius = max(layers["<name>"]["spacing"])\`.
              Hard-coded "5 nm" or "10 nm" is meaningless at coarse scales
              (where one voxel may be 256+ nm) and disappears at fine scales
              (where one voxel is 2 nm). Scaling to the voxel size makes the
              operation visible at every level of a multiscale pyramid.
- fastremap — renumber, remap, mask, unique, refit; in-place relabeling at numpy speeds.
- edt — signed / unsigned Euclidean distance transform.
- kimimaro — TEASAR skeletonization for neuron/tubule volumes.
- zmesh — fast multi-resolution meshing from a labeled volume.`
    : `Libraries (already imported; do NOT reimport):
- numpy as np, pandas as pd, matplotlib.pyplot as plt
- scipy.ndimage as ndi
- from skimage import measure as _sk_measure

Note: cc3d / fastmorph / kimimaro / zmesh / fastremap / edt are NOT available in the browser Pyodide runtime. Use scipy.ndimage + skimage.measure / morphology instead.`;

  return `${runtimeHeader}

AVAILABLE VARIABLES:
Layers (numpy ndarrays, already loaded):
${layerDescs.join("\n")}

Per-layer metadata is a Python dict at \`layers["<varName>"]\`. Access via subscript syntax — \`layers["mito"]["spacing"]\`, NOT \`layers["mito"].spacing\` (attribute access raises AttributeError, dicts don't have attributes). Keys: \`array\` (numpy ndarray), \`spacing\` (nm per voxel, array-axis order), \`offsets\` (world origin nm, array-axis order), \`axes\` (array-axis-order list of "x"/"y"/"z").

DataFrames:
${tableDescs.join("\n") || "(none)"}

${librariesBlock}

OUTPUT CONTRACT (set zero or more):
- \`_TG_TABLE\`: a pandas DataFrame — will be added to the sidebar as a new table. If it has object_id + position_x/y/z columns, rows become click-to-fly.
- \`_TG_TABLE_NAME\`: string — name for the table.
- \`_TG_FLY\`: dict \`{"pos": [x, y, z], "segment_id": "...", "layer": "..."}\` — world-nm position to fly the viewer to.
- \`_TG_NARRATION\`: string — short human-readable summary shown under the output.
- \`_TG_ANNOTATIONS\`: \`{"layer_name": "<name>", "points": [{"pos": [x,y,z], "id": "...", "description": "..."}, ...]}\` — creates an annotation layer in the viewer with point markers. Positions in world nm.
- \`_TG_HIGHLIGHT\`: \`{"layer": "<existing NG layer name>", "ids": [1, 2, 3, ...]}\` — in a segmentation layer, show only these segment ids. Useful to focus attention on the objects your analysis selected.
- \`_TG_ADD_SOURCE_LAYER\`: \`{"source": "zarr://...", "name": "new_layer", "type": "segmentation"|"image"}\` — add a pre-existing remote zarr/n5/precomputed source as a new layer.
- \`_TG_NEW_LAYER\`: \`{"array": <numpy ndarray>, "name": "<name>", "type": "segmentation"|"image", "spacing": [sz,sy,sx]?, "offsets": [oz,oy,ox]?, "axes": ["z","y","x"]?}\` — the array is encoded as a zarr and added as a new layer in the viewer. spacing/offsets/axes default to the first input layer's values. Use this for derived masks (e.g. a contact-site mask).
- Any matplotlib figure you draw is auto-captured and shown as a PNG.

RULES:
- Reply with RAW PYTHON CODE ONLY. No JSON, no markdown fences, no prose, no preamble.
- Do NOT reassign the provided ndarrays or DataFrames (they already exist as globals — use them by name).
- Do NOT add 'import' statements for the libs already imported above; you may import standard-lib modules freely.
- Convert positions with array_index * spacing + offset if you need world nm.
- Prefer operating on the loaded arrays directly; don't try to re-read from disk.
- ALWAYS print BEFORE every potentially-slow operation and AFTER it with timing. This is mandatory, not optional. \`time\` is already imported, no need to re-import. Examples:
  \`print(f"input shape={mito.shape} dtype={mito.dtype} unique_labels={len(np.unique(mito))-1}", flush=True)\`
  \`t0=time.time(); eroded = fastmorph.spherical_erode(...); print(f"erode done in {time.time()-t0:.2f}s", flush=True)\`
  Without these prints the user sees a blank "Running on backend..." for minutes with no idea what's happening. Even a 200-ms call should be wrapped if it's the main work step.
- Pass \`flush=True\` on every print so HF Container Logs see it in real time (Python block-buffers stdout otherwise).
- Set \`_TG_NARRATION\` to a one- or two-sentence final human-readable summary (the answer, not the play-by-play).
- Keep runtime ${useRemote ? "under ~5 min on the HF backend (long-running ops are fine; report progress via print())" : "under ~30s; prefer coarsest-scale arrays"}.`;
}

function extractPythonCode(raw: string): string {
  let s = raw.trim();
  // Strip any fenced code block (```python ... ``` or ``` ... ```).
  const fenced = s.match(/```(?:python)?\s*([\s\S]*?)```/);
  if (fenced) s = fenced[1].trim();
  // If the model still wrapped everything in a JSON object (older prompts
  // or a stubborn model), pull the `code` value out by best-effort parse.
  if (s.startsWith("{")) {
    try {
      const obj = JSON.parse(s);
      if (typeof obj?.code === "string") s = obj.code;
    } catch {
      // Manual extraction: find the first "code": "..." field, un-escape it.
      const m = s.match(/"code"\s*:\s*"((?:[^"\\]|\\.)*)"/);
      if (m) {
        try {
          s = JSON.parse(`"${m[1]}"`);
        } catch {
          s = m[1].replace(/\\n/g, "\n").replace(/\\"/g, '"').replace(/\\\\/g, "\\");
        }
      }
    }
  }
  return s;
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

function humanBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
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

// --- Download helpers -------------------------------------------------------

function makeDownloadButton(label: string, onClick: () => void | Promise<void>): HTMLButtonElement {
  const btn = document.createElement("button");
  btn.className = "btn-secondary btn-download";
  btn.type = "button";
  btn.textContent = `⬇ ${label}`;
  btn.addEventListener("click", () => void onClick());
  return btn;
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

function downloadDataUrl(dataUrl: string, filename: string): void {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function tableToCsv(tbl: { columns: string[]; rows: (number | string | null)[][] }): string {
  const esc = (v: unknown): string => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [tbl.columns.map(esc).join(",")];
  for (const row of tbl.rows) {
    lines.push(row.map(esc).join(","));
  }
  return lines.join("\n");
}

function buildScriptHeader(
  slots: { varName: string; layer: DatasetLayer; inspection?: LayerInspection; scaleIdx?: number }[],
): string {
  const parts = [
    "# Tourguide custom analysis — extracted script",
    "# Generated " + new Date().toISOString(),
    "#",
    "# Input layers (already loaded as numpy ndarrays in the tourguide worker):",
  ];
  for (const s of slots) {
    const scale = s.inspection?.scales[s.scaleIdx ?? 0];
    parts.push(
      `#   ${s.varName}: ${s.layer.source} scale=${scale?.path ?? "?"} shape=${scale?.shape.join("×") ?? "?"} voxel_nm=${scale?.voxelNm.map((v) => v.toFixed(2)).join(",") ?? "?"} offset_nm=${scale?.offsetNm.map((v) => v.toFixed(0)).join(",") ?? "?"}`,
    );
  }
  parts.push(
    "#",
    "# To rerun outside tourguide, load each layer (e.g. via zarr/ome-zarr-py or",
    "# tensorstore), bind to the variable names above, then run the body below.",
    "",
  );
  return parts.join("\n");
}

// Download the synthesized zarr as a ZIP. We read each file from the same
// IndexedDB store the SW serves from, then assemble a minimal ZIP in-memory.
async function downloadSynthesizedZarr(id: string, name: string): Promise<void> {
  const prefix = `${id}/`;
  const entries: { path: string; bytes: Uint8Array }[] = [];
  const db = await new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open("tourguide-synthesized", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("files");
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction("files", "readonly");
    const store = tx.objectStore("files");
    const req = store.openCursor();
    req.onsuccess = () => {
      const cur = req.result;
      if (!cur) {
        resolve();
        return;
      }
      const key = String(cur.key);
      if (key.startsWith(prefix)) {
        const v = cur.value;
        const bytes = v instanceof Uint8Array ? v : new TextEncoder().encode(String(v));
        entries.push({ path: key.slice(prefix.length), bytes });
      }
      cur.continue();
    };
    req.onerror = () => reject(req.error);
  });
  db.close();
  if (entries.length === 0) {
    alert("Nothing to download — synthesized zarr is empty.");
    return;
  }
  const zipBytes = buildZip(entries);
  downloadBlob(
    new Blob([zipBytes as BlobPart], { type: "application/zip" }),
    `${name}.zarr.zip`,
  );
}

// Minimal STORE-only (no compression) ZIP writer. Good enough for the small
// synthesized zarrs we generate.
function buildZip(entries: { path: string; bytes: Uint8Array }[]): Uint8Array {
  const enc = new TextEncoder();
  const fileRecords: Uint8Array[] = [];
  const centralDir: Uint8Array[] = [];
  let offset = 0;

  const crcTable = (() => {
    const table = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      let c = i;
      for (let k = 0; k < 8; k++) {
        c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      }
      table[i] = c >>> 0;
    }
    return table;
  })();
  const crc32 = (data: Uint8Array): number => {
    let c = 0xffffffff;
    for (let i = 0; i < data.length; i++) {
      c = crcTable[(c ^ data[i]) & 0xff] ^ (c >>> 8);
    }
    return (c ^ 0xffffffff) >>> 0;
  };

  for (const e of entries) {
    const nameBytes = enc.encode(e.path);
    const crc = crc32(e.bytes);
    const size = e.bytes.length;
    // Local file header
    const lfh = new Uint8Array(30 + nameBytes.length);
    const dv = new DataView(lfh.buffer);
    dv.setUint32(0, 0x04034b50, true); // signature
    dv.setUint16(4, 20, true); // version
    dv.setUint16(6, 0, true); // flags
    dv.setUint16(8, 0, true); // method (0 = store)
    dv.setUint16(10, 0, true); // mtime
    dv.setUint16(12, 0, true); // mdate
    dv.setUint32(14, crc, true);
    dv.setUint32(18, size, true); // compressed
    dv.setUint32(22, size, true); // uncompressed
    dv.setUint16(26, nameBytes.length, true);
    dv.setUint16(28, 0, true); // extra len
    lfh.set(nameBytes, 30);
    fileRecords.push(lfh, e.bytes);
    // Central directory entry
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

  const total =
    fileRecords.reduce((n, b) => n + b.length, 0) + cdSize + eocd.length;
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
