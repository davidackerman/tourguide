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
  cancelAnalysisRequest,
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
          <strong>3D meshes</strong> require <em>Run on backend</em>
          (zmesh ships only on the HF Space).
        </p>
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
        <p class="hint" data-inspect-status>Each checked layer's scale + labeled dropdowns appear inline above. Default scale is the coarsest one that fits the budget; defaults to "Yes — values are segment ids".</p>
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
        <button class="btn-secondary" data-stop hidden>Stop analysis</button>
        <button class="btn-primary" data-run disabled>Run analysis</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);

  const $ = <T extends HTMLElement>(sel: string): T => overlay.querySelector<T>(sel)!;
  const layersHost = $<HTMLDivElement>("[data-layers-host]");
  const inspectStatus = $<HTMLParagraphElement>("[data-inspect-status]");
  const progressEl = $<HTMLDivElement>("[data-progress]");
  const progressText = $<HTMLDivElement>("[data-progress-text]");
  const progressBar = $<HTMLDivElement>("[data-progress-bar]");
  const errEl = $<HTMLParagraphElement>("[data-error]");
  const runBtn = $<HTMLButtonElement>("[data-run]");
  const stopBtn = $<HTMLButtonElement>("[data-stop]");
  const cancelBtn = $<HTMLButtonElement>("[data-cancel]");
  const closeBtn = $<HTMLButtonElement>(".modal-close");
  const remoteRow = $<HTMLLabelElement>("[data-remote-row]");
  const remoteToggle = $<HTMLInputElement>("[data-remote-toggle]");
  const remoteLabel = $<HTMLSpanElement>("[data-remote-label]");
  // zmesh ships only on the HF backend; gate the per-layer mesh
  // checkboxes behind the remote toggle so they can't be checked for
  // a Pyodide run that would just ignore them.
  const refreshMeshAvailability = (): void => {
    const remoteOn = !!(analysisBackendUrl && remoteToggle.checked && !remoteToggle.disabled);
    layersHost
      .querySelectorAll<HTMLInputElement>("[data-layer-mesh]")
      .forEach((cb) => {
        cb.disabled = !remoteOn;
        if (!remoteOn) cb.checked = false;
        const wrap = cb.closest<HTMLLabelElement>(".analysis-layer-mesh-toggle");
        if (wrap) wrap.title = remoteOn ? "" : "Available only when 'Run on backend' is checked.";
      });
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
  // No-op now that mesh selection is per-layer. The downsample hint
  // used to live next to the global toggle; with per-row checkboxes
  // we don't have a single place to put one without crowding rows.
  // The auto-downsample factor still applies at run time per layer.
  const refreshMeshHint = (): void => {
    /* intentionally empty */
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
    inspectingPromise?: Promise<void>;
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
  // Two-row layout per layer: line 1 has the checkbox + name + type
  // (always visible), line 2 has scale + already-labeled dropdowns
  // (only visible in multi-layer mode). Avoids the overlap that
  // happened when name and dropdown shared one row and text wrapped
  // under the <select>. Each row's labeled dropdown is independent so
  // mixed-input batches (one labeled volume + one mask volume) work.
  segLayers.forEach((l, i) => {
    const row = document.createElement("div");
    row.className = "analysis-layer-row";
    row.dataset.layerIdx = String(i);
    row.innerHTML = `
      <label class="analysis-layer-check">
        <input type="checkbox" data-layer-idx="${i}" ${i === 0 ? "checked" : ""} />
        <span class="analysis-layer-name">
          ${escapeHtml(l.name)}
          <em class="hint analysis-layer-type">[${l.type}]${l.organelle_class ? ` — ${escapeHtml(l.organelle_class)}` : ""}</em>
        </span>
      </label>
      <div class="analysis-layer-controls" data-layer-controls="${i}" hidden>
        <label class="analysis-layer-control">
          <span>Scale</span>
          <select class="analysis-layer-scale" data-layer-scale="${i}">
            <option>(loading scales…)</option>
          </select>
        </label>
        <label class="analysis-layer-control">
          <span>Labeled</span>
          <select class="analysis-layer-labeled" data-layer-labeled="${i}">
            <option value="true" selected>Yes — values are segment ids</option>
            <option value="false">No — connected-components on mask</option>
          </select>
        </label>
        <label class="analysis-layer-control analysis-layer-mesh-toggle" data-layer-mesh-row="${i}" title="Generate 3D meshes (zmesh, backend only)">
          <input type="checkbox" class="analysis-layer-mesh" data-layer-mesh="${i}" />
          <span>3D meshes</span>
        </label>
        <span class="analysis-layer-status" data-layer-status="${i}"></span>
      </div>
    `;
    layersHost.appendChild(row);
  });

  // Format an option label for one scale. Wide modal gives us room
  // for voxel-size info too — matches what the dropped radio table
  // showed: 's5 · 27×162×193 · 64×64×64 nm/vx · 12.9 MB'. Bullet-
  // separated so the columns scan even though a <select> can't be
  // a real grid. Trailing 'beyond cap' / 'may OOM' badge when the
  // scale would blow Pyodide's memory and we're not on the backend.
  const scaleOptionLabel = (s: LayerScaleInfo, useRemote: boolean): string => {
    const path = s.path || "(root)";
    const shape = s.shape.join("×");
    const vox = s.voxelNm.map((v) => format1(v)).join("×");
    const size = humanSize(s.approxBytes);
    const veryRisky = !useRemote && s.approxBytes > 3 * SAFE_INPUT_BYTES;
    const risky = !useRemote && s.approxBytes > SAFE_INPUT_BYTES;
    const tag = veryRisky ? " · beyond WASM cap" : risky ? " · may OOM" : "";
    return `${path} · ${shape} · ${vox} nm/vx · ${size}${tag}`;
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
      refreshMeshHint();
    };
    refreshMeshHint();
  };

  // AnalysisClient runs a single shared worker — only one inspect or
  // analyze can be in flight at a time. When multiple layer checkboxes
  // fire ensureInspected concurrently, all but the first would trip
  // 'Analysis already in progress'. Serialize via a chained promise:
  // each inspect waits for the previous to finish.
  let inspectQueue: Promise<unknown> = Promise.resolve();
  const ensureInspected = (layerIdx: number): Promise<void> => {
    const slot = layerSlots[layerIdx];
    if (slot.inspection) return Promise.resolve();
    if (slot.inspecting && slot.inspectingPromise) return slot.inspectingPromise;
    const statusEl = layersHost.querySelector<HTMLSpanElement>(
      `[data-layer-status="${layerIdx}"]`,
    );
    if (statusEl) statusEl.textContent = "queued…";
    slot.inspecting = true;
    const job = inspectQueue.then(async () => {
      // Re-check after we get the queue slot — the layer may have been
      // inspected by a parallel path while we were waiting.
      if (slot.inspection) {
        slot.inspecting = false;
        return;
      }
      if (statusEl) statusEl.textContent = "inspecting…";
      try {
        const url = normalizeZarrUrl(segLayers[layerIdx].source);
        const insp = await client.inspect(url, d.voxel_size_nm);
        slot.inspection = insp;
        const useRemote = !!(analysisBackendUrl && remoteToggle.checked);
        slot.selectedScaleIdx = autoPickScaleIdx(insp, useRemote);
        if (statusEl) {
          statusEl.textContent = "";
          statusEl.classList.remove("err");
        }
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
    });
    slot.inspectingPromise = job;
    // Chain so the next caller queues behind this one. Catch swallows
    // errors at the queue level — individual error handling already
    // happened inside the job above.
    inspectQueue = job.catch(() => {});
    return job;
  };

  // Show each row's controls (scale + labeled + mesh dropdowns)
  // whenever its checkbox is checked. Inspection runs lazily on first
  // check and populates the scale dropdown when ready; labeled and
  // mesh are immediately usable.
  const refreshPerRowScalePickers = (): void => {
    const idxs = getCheckedLayerIdxs();
    layersHost
      .querySelectorAll<HTMLDivElement>("[data-layer-controls]")
      .forEach((ctrls) => {
        const idx = Number(ctrls.dataset.layerControls);
        ctrls.hidden = !idxs.includes(idx);
      });
    // Mesh checkboxes need their disabled state synced after rows
    // toggle visible (the global mesh-availability check runs on the
    // remote toggle, but newly-shown rows haven't been processed yet).
    refreshMeshAvailability();
    // Kick off inspection for any newly-checked layer that we haven't
    // probed yet. Already-inspected ones no-op.
    idxs.forEach((idx) => void ensureInspected(idx));
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

  const close = (): void => {
    client.terminate();
    overlay.remove();
  };
  closeBtn.addEventListener("click", close);
  cancelBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });

  // Stop button: abort the in-flight fetch (so the frontend stops
  // waiting on the response) and POST a cancel to the backend (so
  // the sandbox subprocess gets terminated and frees the semaphore
  // slot). Both are best-effort — if the run finished a millisecond
  // before the user clicked, the abort is a no-op; if the backend
  // already returned, the cancel POST is a no-op. The userAborted
  // flag short-circuits the per-layer loop so a multi-layer batch
  // stops at the current layer instead of plowing through the rest.
  stopBtn.addEventListener("click", () => {
    userAborted = true;
    showProgress("Stopping…");
    const sid = activeRemoteSession;
    activeRemoteAbort?.abort();
    if (sid) void cancelAnalysisRequest(analysisBackendUrl, sid);
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

  // Re-render per-row scale labels whenever 'Run on backend' toggles,
  // so 'beyond WASM cap' tags drop / reappear and the auto-selected
  // scale flips between coarsest-fit (local) and s0 (backend).
  remoteToggle.addEventListener("change", () => {
    refreshMeshAvailability();
  });
  refreshMeshAvailability();

  // Layer selection drives whether each row's controls (scale +
  // labeled dropdowns) are visible. Always per-row now — no global
  // 'Already labeled?' dropdown, no separate radio-table scale picker.
  const onLayerSelectionChange = (): void => {
    refreshPerRowScalePickers();
    const idxs = getCheckedLayerIdxs();
    if (idxs.length === 0) {
      inspectStatus.textContent = "Check at least one layer to enable Run.";
      runBtn.disabled = true;
      return;
    }
    inspectStatus.textContent =
      idxs.length === 1
        ? `1 layer selected. Pick a scale + labeled mode above.`
        : `${idxs.length} layers selected — runs in series. Default scale per row is the coarsest one that fits the budget${analysisBackendUrl && remoteToggle.checked ? " (backend lifts the WASM cap)" : ""}.`;
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

  // Active remote-run state — set by the run handler when a remote
  // request is in flight, cleared in finally. The Stop button reads
  // these to (a) abort the local fetch and (b) POST a cancel to the
  // backend so the sandbox process actually stops, not just the
  // response wait. Only one remote call is in flight at a time
  // (multi-layer runs go in series), so a single slot is enough.
  let activeRemoteAbort: AbortController | null = null;
  let activeRemoteSession: string | null = null;
  let userAborted = false;

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
    alreadyLabeled: boolean,
    makeMeshes: boolean,
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
        // makeMeshes now comes in via the runOneLayer args (per-layer).
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
        const ac = new AbortController();
        activeRemoteAbort = ac;
        activeRemoteSession = sessionId;
        stopBtn.hidden = false;
        let remote: CustomAnalysisResult;
        try {
          remote = await postAnalysisRequest(
            analysisBackendUrl,
            {
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
                alreadyLabeled,
                makeMeshes,
                layer.name,
                chooseMeshDownsample(scale.shape),
              ),
              timeoutMs: 300_000,
              sessionId,
            },
            ac.signal,
          );
        } finally {
          activeRemoteAbort = null;
          activeRemoteSession = null;
          stopBtn.hidden = true;
        }
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
          const isAlreadyLabeled = alreadyLabeled;
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
            alreadyLabeled,
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
    type Job = {
      layer: DatasetLayer;
      insp: LayerInspection;
      scaleIdx: number;
      alreadyLabeled: boolean;
      makeMeshes: boolean;
    };
    const jobs: Job[] = [];
    try {
      // Always read per-row controls. Block on any pending inspections
      // so each layer's scale dropdown has a real value before we
      // read it.
      await Promise.all(checkedIdxs.map((i) => ensureInspected(i)));
      for (const i of checkedIdxs) {
        const slot = layerSlots[i];
        if (!slot.inspection) {
          throw new Error(
            `Couldn't inspect ${segLayers[i].name}${slot.inspectError ? `: ${slot.inspectError}` : ""}`,
          );
        }
        const scaleIdx = slot.selectedScaleIdx ?? autoPickScaleIdx(slot.inspection, useRemote);
        const perRowLabeled = layersHost.querySelector<HTMLSelectElement>(
          `[data-layer-labeled="${i}"]`,
        );
        const perRowMesh = layersHost.querySelector<HTMLInputElement>(
          `[data-layer-mesh="${i}"]`,
        );
        jobs.push({
          layer: segLayers[i],
          insp: slot.inspection,
          scaleIdx,
          alreadyLabeled: (perRowLabeled?.value ?? "true") === "true",
          makeMeshes: !!(perRowMesh && perRowMesh.checked && !perRowMesh.disabled),
        });
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

    userAborted = false;
    const errors: { layer: string; message: string }[] = [];
    for (let i = 0; i < jobs.length; i++) {
      if (userAborted) break;
      const { layer, insp, scaleIdx, alreadyLabeled, makeMeshes } = jobs[i];
      const labelPrefix = jobs.length > 1 ? `[${i + 1}/${jobs.length}] ` : "";
      try {
        await runOneLayer(layer, insp, scaleIdx, useRemote, labelPrefix, alreadyLabeled, makeMeshes);
      } catch (err) {
        // AbortError surfaces as a normal exception — translate to a
        // 'cancelled by user' message instead of a scary stack trace.
        const msg = (err as Error).name === "AbortError" || userAborted
          ? "cancelled"
          : (err as Error).message;
        errors.push({ layer: layer.name, message: msg });
      }
    }
    hideProgress();
    if (userAborted) {
      // Stop button was clicked — show a clean status, don't pop a
      // multi-line error. Successful layers (if any) are already in
      // the DB; the Run button stays enabled so the user can retry.
      const ok = jobs.length - errors.length;
      showError(
        ok > 0
          ? `Stopped. ${ok} layer(s) finished before stop; remaining cancelled.`
          : `Stopped — analysis cancelled.`,
      );
      runBtn.disabled = false;
      return;
    }
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

  // First run: the default-checked layer (idx 0) needs its controls
  // visible immediately. onLayerSelectionChange triggers inspection
  // for the checked rows; we then queue inspection for *all* other
  // layers up front too, so by the time the user toggles their
  // checkboxes the scale dropdowns are already populated. The
  // shared inspectQueue serializes them — no races, no extra HTTP
  // pressure beyond what'd happen if the user clicked them all
  // anyway.
  onLayerSelectionChange();
  segLayers.forEach((_, i) => void ensureInspected(i));
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

# Coordinate math (both centroid + bbox columns end in 'world-space nm'):
#   - regionprops_table with spacing=voxel_size_nm returns 'area' in
#     nm^3, 'centroid-i' in nm relative to voxel (0,0,0).
#   - 'bbox-i' is in voxel indices regardless of spacing — multiply by
#     spacing[i] then add offsets[i] to get nm.
#   - We add offsets[i] (the layer's world-space origin in nm) to land
#     in Neuroglancer's frame. axis_col / bbox_pair above do this.
import pandas as pd
df = pd.DataFrame({
    "object_id":             [int(v) for v in tbl["label"].tolist()],
    "volume_nm_3":           [float(v) for v in tbl["area"].tolist()],
    "com_x_nm": cx, "com_y_nm": cy, "com_z_nm": cz,
    "bbox_min_x_nm": bx0, "bbox_min_y_nm": by0, "bbox_min_z_nm": bz0,
    "bbox_max_x_nm": bx1, "bbox_max_y_nm": by1, "bbox_max_z_nm": bz1,
    "equivalent_diameter_nm": [float(v) for v in tbl["equivalent_diameter_area"].tolist()],
    "n_voxels":              [int(v / voxel_volume + 0.5) for v in tbl["area"].tolist()],
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
