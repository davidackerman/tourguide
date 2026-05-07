// Modal UI for running browser-side segmentation analysis. Lets the user
// pick a segmentation layer + scale, watches progress, and on success injects
// the resulting per-object stats as a new table in the existing DatasetDB so
// the structured browser picks it up for free (sort / paginate / fly-to).

import type { Database } from "sql.js";
import type { BundledViewer } from "./bundled_viewer.js";
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
  viewer: BundledViewer; // for adding the optional mesh layer to NG
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
          <div class="analysis-layers" data-layers-list>
            <div class="analysis-layers-header">
              <span>Layers <em class="hint" style="font-weight:normal;">(check one or more — multi-select runs them in series)</em></span>
              <span class="analysis-layer-actions">
                <button type="button" class="btn-tiny" data-action="select-all-layers">all</button>
                <button type="button" class="btn-tiny" data-action="select-none-layers">none</button>
              </span>
            </div>
            <div class="analysis-layers-host" data-layers-host></div>
          </div>
          <label>
            <span>Already labeled?</span>
            <select data-labeled>
              <option value="true" selected>Yes — values are segment ids</option>
              <option value="false">No — run connected-components on mask</option>
            </select>
          </label>
          <label class="mesh-toggle" data-mesh-row>
            <input type="checkbox" data-make-meshes />
            <span>Generate 3D meshes (zmesh, backend only) <em data-mesh-hint class="hint" style="font-style:italic;opacity:0.75;"></em></span>
          </label>
        </div>
        <div data-scales-host>
          <p class="hint" data-inspect-status>Pick a layer (single-layer mode shows scales). With multiple layers checked, each layer auto-picks its coarsest scale that fits the budget.</p>
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
  const layersHost = $<HTMLDivElement>("[data-layers-host]");
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
  const meshRow = $<HTMLLabelElement>("[data-mesh-row]");
  const meshCheckbox = $<HTMLInputElement>("[data-make-meshes]");
  const meshHint = $<HTMLElement>("[data-mesh-hint]");
  // zmesh ships only on the HF backend; gate the option behind the remote
  // toggle so the user can't pick it for a Pyodide run that would just
  // ignore it.
  const refreshMeshAvailability = (): void => {
    const remoteOn = !!(analysisBackendUrl && remoteToggle.checked && !remoteToggle.disabled);
    meshCheckbox.disabled = !remoteOn;
    meshRow.title = remoteOn ? "" : "Available only when 'Run on backend' is checked.";
    if (!remoteOn) meshCheckbox.checked = false;
  };
  // Auto-coarsen meshing for big inputs so zmesh stays under ~30 s.
  // Targets ≤32 M voxels of mesh input regardless of analyze scale.
  // Returned factor is 1/2/4/8 — applied as labels[::n,::n,::n] before
  // zmesh, with mesh spacing scaled by n. Meshes look smoother and
  // generate faster; regionprops is unaffected (analyze scale is
  // independent).
  const chooseMeshDownsample = (shape: number[]): number => {
    const nvox = shape.reduce((a, b) => a * b, 1);
    if (nvox <= 8e6) return 1;
    if (nvox <= 64e6) return 2;
    if (nvox <= 512e6) return 4;
    return 8;
  };
  const refreshMeshHint = (): void => {
    if (selectedScaleIdx == null || !currentInspection) {
      meshHint.textContent = "";
      return;
    }
    const sc = currentInspection.scales[selectedScaleIdx];
    const f = chooseMeshDownsample(sc.shape);
    meshHint.textContent = f === 1
      ? "— meshes at full analyze resolution"
      : `— meshes at ${f}× downsample (auto, for speed)`;
  };

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
      refreshMeshAvailability();
    })();
  }

  // Per-layer cache of inspection result + currently-selected scale,
  // keyed by the layer index in segLayers. Populated lazily as the
  // user checks layers in multi-layer mode (so we don't run an HTTP
  // probe for layers they're not going to analyze).
  type LayerSlot = {
    inspection: LayerInspection | null;
    selectedScaleIdx: number | null;
    inspecting: boolean;
    inspectError: string | null;
  };
  const layerSlots: LayerSlot[] = segLayers.map(() => ({
    inspection: null,
    selectedScaleIdx: null,
    inspecting: false,
    inspectError: null,
  }));

  // Render one row per zarr layer. Each row has a checkbox plus a
  // <select> for scale selection (hidden in single-layer mode where
  // the big radio table below is used instead). In multi-layer mode
  // each row's <select> drives that layer's scale choice — the radio
  // table is hidden.
  segLayers.forEach((l, i) => {
    const row = document.createElement("div");
    row.className = "analysis-layer-row";
    row.dataset.layerIdx = String(i);
    row.innerHTML = `
      <label class="analysis-layer-check">
        <input type="checkbox" data-layer-idx="${i}" ${i === 0 ? "checked" : ""} />
        <span>${escapeHtml(l.name)} <em class="hint" style="opacity:0.7;">[${l.type}]${l.organelle_class ? ` — ${escapeHtml(l.organelle_class)}` : ""}</em></span>
      </label>
      <select class="analysis-layer-scale" data-layer-scale="${i}" hidden>
        <option>(loading scales…)</option>
      </select>
      <span class="analysis-layer-status" data-layer-status="${i}"></span>
    `;
    layersHost.appendChild(row);
  });

  // Format an option label for one scale: 's0  1300×1500×220, 388 MB'
  // plus a 'beyond cap' badge if it'd OOM in Pyodide and we're not on
  // the backend.
  const scaleOptionLabel = (s: LayerScaleInfo, useRemote: boolean): string => {
    const path = s.path || "(root)";
    const shape = s.shape.join("×");
    const size = humanSize(s.approxBytes);
    const veryRisky = !useRemote && s.approxBytes > 3 * SAFE_INPUT_BYTES;
    const risky = !useRemote && s.approxBytes > SAFE_INPUT_BYTES;
    const tag = veryRisky ? " — beyond WASM cap" : risky ? " — may OOM" : "";
    return `${path}  ${shape}  ${size}${tag}`;
  };

  // Build the dropdown options for a layer's scale picker. Coarsest
  // fitting scale is preselected (matches the radio-table default).
  const populateLayerScaleSelect = (layerIdx: number): void => {
    const slot = layerSlots[layerIdx];
    const sel = layersHost.querySelector<HTMLSelectElement>(
      `[data-layer-scale="${layerIdx}"]`,
    );
    if (!sel) return;
    if (!slot.inspection) {
      sel.innerHTML = `<option>(loading scales…)</option>`;
      sel.disabled = true;
      return;
    }
    const useRemote = !!(analysisBackendUrl && remoteToggle.checked);
    sel.innerHTML = "";
    slot.inspection.scales.forEach((s, j) => {
      const opt = document.createElement("option");
      opt.value = String(j);
      const veryRisky = !useRemote && s.approxBytes > 3 * SAFE_INPUT_BYTES;
      opt.disabled = veryRisky;
      opt.textContent = scaleOptionLabel(s, useRemote);
      if (j === slot.selectedScaleIdx) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.disabled = false;
    sel.onchange = (): void => {
      slot.selectedScaleIdx = Number(sel.value);
    };
  };

  // Lazy inspection: probe a layer for its scales when first needed
  // in multi-layer mode. Already-inspected layers no-op.
  const ensureInspected = async (layerIdx: number): Promise<void> => {
    const slot = layerSlots[layerIdx];
    if (slot.inspection || slot.inspecting) return;
    slot.inspecting = true;
    const statusEl = layersHost.querySelector<HTMLSpanElement>(
      `[data-layer-status="${layerIdx}"]`,
    );
    if (statusEl) statusEl.textContent = "inspecting…";
    try {
      const url = normalizeZarrUrl(segLayers[layerIdx].source);
      const insp = await client.inspect(url, d.voxel_size_nm);
      slot.inspection = insp;
      const useRemote = !!(analysisBackendUrl && remoteToggle.checked);
      slot.selectedScaleIdx = autoPickScaleIdx(insp, useRemote);
      if (statusEl) statusEl.textContent = "";
      populateLayerScaleSelect(layerIdx);
    } catch (err) {
      slot.inspectError = (err as Error).message;
      if (statusEl) {
        statusEl.textContent = `✗ ${slot.inspectError.slice(0, 80)}`;
        statusEl.classList.add("err");
      }
    } finally {
      slot.inspecting = false;
    }
  };

  // Toggle visibility of per-row scale dropdowns based on how many
  // layers are checked. Single-layer mode keeps them hidden so the
  // big radio table is the single source of scale selection;
  // multi-layer mode shows them inline.
  const refreshPerRowScalePickers = (): void => {
    const idxs = getCheckedLayerIdxs();
    const useDropdowns = idxs.length >= 2;
    layersHost
      .querySelectorAll<HTMLSelectElement>("[data-layer-scale]")
      .forEach((sel) => {
        const idx = Number(sel.dataset.layerScale);
        const checked = idxs.includes(idx);
        sel.hidden = !(useDropdowns && checked);
      });
    if (useDropdowns) {
      // Kick off inspection for any newly-checked layer that we haven't
      // probed yet. Already-inspected ones no-op.
      idxs.forEach((idx) => void ensureInspected(idx));
    }
  };
  const getCheckedLayerIdxs = (): number[] => {
    const out: number[] = [];
    layersHost
      .querySelectorAll<HTMLInputElement>('input[type="checkbox"][data-layer-idx]')
      .forEach((cb) => {
        if (cb.checked) out.push(Number(cb.dataset.layerIdx));
      });
    return out;
  };
  const setAllLayers = (checked: boolean): void => {
    layersHost
      .querySelectorAll<HTMLInputElement>('input[type="checkbox"][data-layer-idx]')
      .forEach((cb) => (cb.checked = checked));
  };
  $<HTMLButtonElement>('[data-action="select-all-layers"]').addEventListener("click", () => {
    setAllLayers(true);
    onLayerSelectionChange();
  });
  $<HTMLButtonElement>('[data-action="select-none-layers"]').addEventListener("click", () => {
    setAllLayers(false);
    onLayerSelectionChange();
  });

  // currentLayer / currentInspection / selectedScaleIdx are only
  // used in single-layer mode (exactly one checkbox checked) — they
  // drive the per-scale radio table. In multi-layer mode the scale
  // selection is automatic per layer at run time.
  let currentLayer: DatasetLayer | null = segLayers[0];
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
    // Pyodide's WASM32 ceiling (~4 GB) only applies to local runs. On the
    // remote backend the input lives in 16 GB of real RAM, so don't disable
    // any scale and don't show the cap badges.
    const useRemote = !!(analysisBackendUrl && remoteToggle.checked);
    insp.scales.forEach((s, i) => {
      const tr = document.createElement("tr");
      const bytes = s.approxBytes;
      const risky = !useRemote && bytes > SAFE_INPUT_BYTES;
      const veryRisky = !useRemote && bytes > 3 * SAFE_INPUT_BYTES; // ~4.5 GB → guaranteed OOM locally
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

    // Auto-pick the coarsest scale that fits the safe budget. On remote,
    // every scale fits, so default to the *finest* (s0) — that's what the
    // user wants when they've explicitly opted into the backend.
    if (useRemote) {
      selectedScaleIdx = 0;
      const input = tbl.querySelector<HTMLInputElement>(`input[value="0"]`);
      if (input) input.checked = true;
      runBtn.disabled = false;
    } else {
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
    }

    tbl.querySelectorAll<HTMLInputElement>('input[type="radio"]').forEach((inp) => {
      inp.addEventListener("change", () => {
        selectedScaleIdx = Number(inp.value);
        runBtn.disabled = false;
        refreshMeshHint();
      });
    });
    refreshMeshHint();
  };

  // Re-render the scales table whenever the remote toggle flips, so the
  // 'beyond WASM cap' chips and disabled radios drop away when the backend
  // is selected (and reappear on switch back).
  remoteToggle.addEventListener("change", () => {
    if (currentInspection) renderScales(currentInspection);
    refreshMeshAvailability();
  });
  refreshMeshAvailability();

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

  // When the layer selection changes:
  //  - 0 checked → disable Run, show hint.
  //  - 1 checked → inspect the layer, show big radio-table scale picker.
  //  - 2+ checked → hide radio table; each row's inline <select> picks
  //                 the layer's scale (lazily inspected as checked).
  const onLayerSelectionChange = (): void => {
    refreshPerRowScalePickers();
    const idxs = getCheckedLayerIdxs();
    if (idxs.length === 0) {
      currentLayer = null;
      currentInspection = null;
      selectedScaleIdx = null;
      scalesHost.innerHTML = `<p class="hint warn">Check at least one layer.</p>`;
      runBtn.disabled = true;
      return;
    }
    if (idxs.length === 1) {
      currentLayer = segLayers[idxs[0]];
      void inspectLayer(currentLayer);
      return;
    }
    // Multi-layer mode: hide the per-scale radio table; scale lives in
    // each layer-row's <select>. Inspection runs in the background;
    // user can click Run as soon as they're ready (slots that are
    // still inspecting will block briefly at run time).
    currentLayer = null;
    currentInspection = null;
    selectedScaleIdx = null;
    scalesHost.innerHTML = `<p class="hint">Multi-layer mode — ${idxs.length} layers selected. Pick a scale per row above; default is the coarsest one that fits the budget${analysisBackendUrl && remoteToggle.checked ? " (backend lifts the WASM cap)" : ""}.</p>`;
    runBtn.disabled = false;
  };
  layersHost.addEventListener("change", (e) => {
    if ((e.target as HTMLElement).matches('input[type="checkbox"][data-layer-idx]')) {
      onLayerSelectionChange();
    }
  });
  // When the user toggles 'Run on backend', the per-row scale labels
  // need to refresh — 'beyond WASM cap' badges drop / reappear, and
  // the auto-picked scale flips between coarsest-fit (local) and s0
  // (backend). Re-run the dropdown population for any inspected slot.
  remoteToggle.addEventListener("change", () => {
    layerSlots.forEach((slot, idx) => {
      if (!slot.inspection) return;
      const useRemote = !!(analysisBackendUrl && remoteToggle.checked);
      slot.selectedScaleIdx = autoPickScaleIdx(slot.inspection, useRemote);
      populateLayerScaleSelect(idx);
    });
  });

  // Auto-pick a sensible scale for a layer when running in multi-layer
  // mode. Mirrors the radio-table default in renderScales: coarsest
  // scale that fits SAFE_INPUT_BYTES locally, finest (s0) when the
  // backend is selected.
  const autoPickScaleIdx = (insp: LayerInspection, useRemote: boolean): number => {
    if (useRemote) return 0;
    const reverseFitIdx = [...insp.scales].reverse().findIndex((s) => s.approxBytes <= SAFE_INPUT_BYTES);
    if (reverseFitIdx !== -1) return insp.scales.length - 1 - reverseFitIdx;
    const fallback = insp.scales.findIndex((s) => s.approxBytes <= 3 * SAFE_INPUT_BYTES);
    return fallback !== -1 ? fallback : insp.scales.length - 1;
  };

  // Run a single layer end-to-end. Returns nothing on success; throws
  // on failure (the caller decides whether to abort the rest of a
  // batch or continue). Captures the surrounding UI state (mesh
  // checkbox, labeled selector, remote toggle) once at call time —
  // we don't want them mutating mid-batch.
  const runOneLayer = async (
    layer: DatasetLayer,
    insp: LayerInspection,
    scaleIdx: number,
    useRemote: boolean,
    labelPrefix: string,
  ): Promise<void> => {
    const scale: LayerScaleInfo = insp.scales[scaleIdx];
    let progressTimer = 0;
    const axesOrder = insp.axes.map((a) => a.name);
    const url = normalizeZarrUrl(layer.source);
    showProgress(`${labelPrefix}Starting ${layer.name}…`);
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
        const makeMeshes = meshCheckbox.checked && !meshCheckbox.disabled;
        // Rotate the progress label based on elapsed time so the bar
        // at least narrates roughly which phase is running. Backend
        // doesn't stream real progress (would need SSE), but the
        // phase boundaries are predictable enough that calendar-
        // estimating them gives the user some signal.
        const t0 = performance.now();
        const phases: Array<[number, string]> = makeMeshes
          ? [
              [0, "Loading layer from zarr …"],
              [3, "Connected components + regionprops on backend …"],
              [12, "Computing region properties …"],
              [22, "Generating 3D meshes (zmesh) …"],
              [60, "Still meshing (large input) …"],
            ]
          : [
              [0, "Loading layer from zarr …"],
              [3, "Connected components + regionprops on backend …"],
              [12, "Computing region properties …"],
              [30, "Still running (large input) …"],
            ];
        const tickProgress = (): void => {
          const dt = (performance.now() - t0) / 1000;
          let label = phases[0][1];
          for (const [t, msg] of phases) if (dt >= t) label = msg;
          showProgress(label);
        };
        tickProgress();
        progressTimer = window.setInterval(tickProgress, 1000);
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
          code: buildRegionpropsCode(
            labeledSel.value === "true",
            makeMeshes,
            layer.name,
            chooseMeshDownsample(scale.shape),
          ),
          timeoutMs: 300_000,
          sessionId,
        });
        window.clearInterval(progressTimer);
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
        showProgress(`${labelPrefix}Inserting ${result.rows.length.toLocaleString()} rows for ${layer.name}…`);
        await ingestResult(cb, layer, scale, result);
        cb.onTableAdded();
        // Mesh layer (if the meshes checkbox was on). When the input was
        // already-labeled AND it's a segmentation layer in NG, the mesh
        // ids share the input layer's id space — attach the meshes to
        // that layer directly instead of creating a separate one. For
        // all other cases (cc3d-derived ids, image-typed inputs) fall
        // back to a standalone mesh-only layer to avoid mis-mapping ids.
        if (remote.meshLayer) {
          const isAlreadyLabeled = labeledSel.value === "true";
          const canAttach = isAlreadyLabeled && layer.type === "segmentation";
          let attached = false;
          if (canAttach) {
            attached = cb.viewer.attachMeshSourceToLayer({
              layerName: layer.name,
              meshSource: remote.meshLayer.source,
              segments: remote.meshLayer.meshIds,
            });
          }
          if (!attached) {
            cb.viewer.addMeshOnlyLayer({
              name: remote.meshLayer.name,
              source: remote.meshLayer.source,
              segments: remote.meshLayer.meshIds,
            });
            const desc = cb.getDescriptor();
            if (desc) {
              const i = desc.layers.findIndex((l) => l.name === remote.meshLayer!.name);
              const newLayer = {
                name: remote.meshLayer.name,
                type: "segmentation" as const,
                source: remote.meshLayer.source,
              };
              if (i >= 0) desc.layers[i] = newLayer;
              else desc.layers.push(newLayer);
            }
          }
        }
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
          (message) => showProgress(`${labelPrefix}${message}`),
        );
        showProgress(`${labelPrefix}Inserting ${result.rows.length.toLocaleString()} rows for ${layer.name}…`);
        await ingestResult(cb, layer, scale, result);
        cb.onTableAdded();
      }
    } finally {
      if (progressTimer) window.clearInterval(progressTimer);
      tunnel?.close();
    }
  };

  // Top-level Run handler: figure out which layers to process, in
  // single- or multi-layer mode, then call runOneLayer for each in
  // series. Errors per layer are captured and shown as a final
  // summary; we don't abort the batch on one bad layer.
  runBtn.addEventListener("click", async () => {
    clearError();
    const checkedIdxs = getCheckedLayerIdxs();
    if (checkedIdxs.length === 0) return;
    const useRemote = !!(analysisBackendUrl && remoteToggle.checked);
    runBtn.disabled = true;
    inspectStatus.textContent = "";

    // Pre-resolve the (layer, scale) pairs. In single-layer mode the
    // radio-table selection is authoritative. In multi-layer mode we
    // read each row's <select> via the cached layerSlots, kicking off
    // any not-yet-inspected layers and waiting on them so the run
    // doesn't fire half-loaded.
    type Job = { layer: DatasetLayer; insp: LayerInspection; scaleIdx: number };
    const jobs: Job[] = [];
    try {
      if (checkedIdxs.length === 1 && currentInspection && selectedScaleIdx != null) {
        jobs.push({
          layer: segLayers[checkedIdxs[0]],
          insp: currentInspection,
          scaleIdx: selectedScaleIdx,
        });
      } else {
        // Block on any pending inspections so each layer's <select>
        // has a real scale chosen before we read it.
        await Promise.all(checkedIdxs.map((i) => ensureInspected(i)));
        for (const i of checkedIdxs) {
          const slot = layerSlots[i];
          if (!slot.inspection) {
            throw new Error(
              `Couldn't inspect ${segLayers[i].name}${slot.inspectError ? `: ${slot.inspectError}` : ""}`,
            );
          }
          const scaleIdx = slot.selectedScaleIdx ?? autoPickScaleIdx(slot.inspection, useRemote);
          jobs.push({ layer: segLayers[i], insp: slot.inspection, scaleIdx });
        }
      }
    } catch (err) {
      hideProgress();
      showError(`Inspection failed: ${(err as Error).message}`);
      runBtn.disabled = false;
      return;
    }

    // WASM-cap warning per local-mode job — one combined confirm.
    if (!useRemote) {
      const oversize = jobs.filter((j) => j.insp.scales[j.scaleIdx].approxBytes > SAFE_INPUT_BYTES);
      if (oversize.length > 0) {
        const lines = oversize
          .map((j) => `  • ${j.layer.name}: ${(j.insp.scales[j.scaleIdx].approxBytes / 1024 ** 3).toFixed(2)} GB`)
          .join("\n");
        const ok = confirm(
          `${oversize.length} layer(s) exceed Pyodide's ~4 GB cap (scipy needs 6–10× input). May OOM:\n${lines}\n\nContinue?`,
        );
        if (!ok) {
          runBtn.disabled = false;
          return;
        }
      }
    }

    const errors: { layer: string; message: string }[] = [];
    for (let i = 0; i < jobs.length; i++) {
      const { layer, insp, scaleIdx } = jobs[i];
      const labelPrefix = jobs.length > 1 ? `[${i + 1}/${jobs.length}] ` : "";
      try {
        await runOneLayer(layer, insp, scaleIdx, useRemote, labelPrefix);
      } catch (err) {
        errors.push({ layer: layer.name, message: (err as Error).message });
      }
    }
    hideProgress();
    if (errors.length === 0) {
      close();
    } else if (errors.length === jobs.length) {
      // All failed — surface the messages and let the user fix.
      showError(
        `All ${jobs.length} layer(s) failed:\n` +
          errors.map((e) => `  • ${e.layer}: ${e.message}`).join("\n"),
      );
      runBtn.disabled = false;
    } else {
      // Partial success — succeeded layers are already in the DB.
      const failed = errors.length;
      const ok = jobs.length - failed;
      showError(
        `${ok}/${jobs.length} succeeded. Failed:\n` +
          errors.map((e) => `  • ${e.layer}: ${e.message}`).join("\n"),
      );
      runBtn.disabled = false;
    }
  });

  if (currentLayer) void inspectLayer(currentLayer);
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
function buildRegionpropsCode(
  alreadyLabeled: boolean,
  makeMeshes: boolean,
  layerName: string,
  meshDownsample: number,
): string {
  // The mesh block is appended (rather than nested) so it runs after the
  // labels variable is already populated by the regionprops step. We hand
  // the variable straight to _TG_NEW_MESH_LAYER and let the HF encoder
  // (_encode_new_mesh_layer_and_write) drive zmesh — that keeps the zmesh
  // API call in one place and avoids the analyze flow reimplementing it.
  const meshSafeName = layerName.replace(/[^a-zA-Z0-9_-]/g, "_");
  const meshBlock = makeMeshes
    ? `
# Generate 3D meshes for the labeled output. We auto-coarsen for speed:
# meshes are derived from a labels[::n, ::n, ::n] downsample, with mesh
# spacing scaled by n. zmesh runs faster and meshes look smoother;
# regionprops on the analyze scale stays accurate (analyze and meshing
# are independent). n is picked by tourguide based on the analyze
# scale's voxel count: 1× ≤8 M, 2× ≤64 M, 4× ≤512 M, 8× above.
_mesh_n = ${meshDownsample}
_mesh_labels = labels
if _mesh_labels.dtype == np.bool_:
    _mesh_labels = _mesh_labels.astype(np.uint32, copy=False)
elif str(_mesh_labels.dtype) not in ("uint8","uint16","uint32","uint64"):
    _mesh_labels = _mesh_labels.astype(np.uint32, copy=False)
if _mesh_n > 1:
    _mesh_labels = np.ascontiguousarray(_mesh_labels[::_mesh_n, ::_mesh_n, ::_mesh_n])
    _mesh_spacing = [s * _mesh_n for s in spacing]
else:
    _mesh_spacing = list(spacing)
_TG_NEW_MESH_LAYER = {
    "labels": _mesh_labels,
    "name": "${meshSafeName}_meshes",
    "spacing": _mesh_spacing,
    "offsets": offsets,
}
print(f"queued meshes for {len(np.unique(_mesh_labels)) - 1} segment(s) at {_mesh_n}x downsample", flush=True)
`
    : "";
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
_TG_NARRATION = f"Regionprops on backend: {len(df)} objects.${makeMeshes ? ` Meshes at ${meshDownsample}x downsample.` : ""}"
${meshBlock}`;
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
