import {
  loadSettings,
  saveSettings,
  backendFromSettings,
  GeminiBackend,
  WebLLMBackend,
  WEBLLM_MODELS,
  webllmModelLabel,
  hasWebGPU,
  diagnoseWebGPU,
  DEFAULT_ANALYSIS_BACKEND,
  listGeminiModels,
  type Settings,
  type LLMBackend,
  type WebLLMModelInfo,
  type GeminiModelInfo,
} from "./llm.js";

const GEMINI_MODELS_CACHE_KEY = "tourguide.geminiModels.v1";

interface CachedGeminiModels {
  fetchedAt: number;
  models: GeminiModelInfo[];
}

function loadCachedGeminiModels(): GeminiModelInfo[] | null {
  try {
    const raw = localStorage.getItem(GEMINI_MODELS_CACHE_KEY);
    if (!raw) return null;
    const cached = JSON.parse(raw) as CachedGeminiModels;
    // Auto-expire after 7 days so the dropdown picks up newly-released
    // models without the user having to click Refresh manually.
    if (Date.now() - cached.fetchedAt > 7 * 24 * 60 * 60 * 1000) return null;
    return cached.models;
  } catch {
    return null;
  }
}

function saveCachedGeminiModels(models: GeminiModelInfo[]): void {
  try {
    localStorage.setItem(
      GEMINI_MODELS_CACHE_KEY,
      JSON.stringify({ fetchedAt: Date.now(), models }),
    );
  } catch {
    /* localStorage full or disabled — caching is best-effort */
  }
}
import { waitForBackendReady, fetchHealth } from "./remote_analysis.js";

export interface SettingsUIOptions {
  onChange: (backend: LLMBackend) => void;
}

type GeminiSortMode = "default" | "rpd" | "rpm";

function geminiSortMode(): GeminiSortMode {
  return (localStorage.getItem("tourguide.geminiSort") as GeminiSortMode) || "default";
}

function formatGeminiLimits(m: GeminiModelInfo): string {
  // Compact "15rpm · 250K tpm · 500/day" — known limits only. Empty
  // string when nothing's known so the option label stays terse.
  if (m.rpm === undefined && m.rpd === undefined) return "";
  const parts: string[] = [];
  if (m.rpm !== undefined) parts.push(`${m.rpm} RPM`);
  if (m.tpm !== undefined) parts.push(`${(m.tpm / 1000).toFixed(0)}K TPM`);
  if (m.rpd !== undefined) parts.push(`${m.rpd}/day`);
  return ` (${parts.join(" · ")} free)`;
}

function sortGeminiModels(models: GeminiModelInfo[], mode: GeminiSortMode): GeminiModelInfo[] {
  const arr = [...models];
  if (mode === "rpd") {
    // Highest free RPD first. Models with unknown limits sink to the
    // bottom (they may or may not be generous; treat as 0 for sort).
    arr.sort((a, b) => (b.rpd ?? -1) - (a.rpd ?? -1) || a.id.localeCompare(b.id));
  } else if (mode === "rpm") {
    arr.sort((a, b) => (b.rpm ?? -1) - (a.rpm ?? -1) || a.id.localeCompare(b.id));
  }
  // 'default' keeps the order from listGeminiModels (newest gen first,
  // cheapest tier within a gen).
  return arr;
}

// Build the <option> list for the Gemini dropdown. Falls back to a small
// hardcoded set when no cache exists yet (first-run, before the user
// has clicked Refresh) so the dropdown isn't empty. Once they click
// Refresh, real model IDs from their key replace this stub.
function renderGeminiModelOptions(currentId: string): string {
  const cached = loadCachedGeminiModels();
  const sortMode = geminiSortMode();
  if (cached && cached.length > 0) {
    const sorted = sortGeminiModels(cached, sortMode);
    const list = sorted.map((m) => ({
      id: m.id,
      label: `${m.id}${formatGeminiLimits(m)}`,
    }));
    if (currentId && !list.some((m) => m.id === currentId)) {
      list.unshift({ id: currentId, label: `${currentId} (current)` });
    }
    return list
      .map((m) => `<option value="${m.id}" ${currentId === m.id ? "selected" : ""}>${m.label}</option>`)
      .join("");
  }
  // No cache — minimal stub so the dropdown isn't empty.
  const fallback: Array<{ id: string; label: string }> = [
    { id: "gemini-2.5-flash-lite", label: "gemini-2.5-flash-lite (15 RPM · 250K TPM · 500/day free) — click Refresh for current options" },
    { id: "gemini-2.5-flash", label: "gemini-2.5-flash (10 RPM · 250K TPM · 250/day free)" },
    { id: "gemini-2.5-pro", label: "gemini-2.5-pro (5 RPM · 250K TPM · 100/day free)" },
  ];
  if (currentId && !fallback.some((m) => m.id === currentId)) {
    fallback.unshift({ id: currentId, label: `${currentId} (current)` });
  }
  return fallback
    .map((m) => `<option value="${m.id}" ${currentId === m.id ? "selected" : ""}>${m.label}</option>`)
    .join("");
}

export function openSettingsDialog(opts: SettingsUIOptions): void {
  const current = loadSettings();
  const webgpu = hasWebGPU();
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  // Sort the model list by either agent-suitability ('recommended', the
  // default — strongest models on top, smaller / weaker beneath) or by
  // download size ascending (when the user is on a slow link or
  // memory-constrained machine and wants the smallest viable option
  // first). Ties broken by size asc / recommended desc respectively.
  type WebllmSort = "recommended" | "size";
  const sortKey: WebllmSort =
    (localStorage.getItem("tourguide.webllmSort") as WebllmSort) || "recommended";
  const sortModels = (sort: WebllmSort): WebLLMModelInfo[] => {
    const arr = [...WEBLLM_MODELS];
    if (sort === "recommended") {
      arr.sort((a, b) => b.recommended - a.recommended || a.sizeGB - b.sizeGB);
    } else {
      arr.sort((a, b) => a.sizeGB - b.sizeGB || b.recommended - a.recommended);
    }
    return arr;
  };
  const renderWebllmOptions = (sort: WebllmSort): string =>
    sortModels(sort)
      .map(
        (m) =>
          `<option value="${m.id}" ${current.webllmModel === m.id ? "selected" : ""}>${webllmModelLabel(m)}</option>`,
      )
      .join("");
  const webllmOptions = renderWebllmOptions(sortKey);
  overlay.innerHTML = `
    <div class="modal" role="dialog" aria-label="Settings">
      <header class="modal-header">
        <h2>Settings</h2>
        <button class="modal-close" aria-label="Close">×</button>
      </header>
      <div class="modal-body">
        <p class="hint">Two independent pieces of configuration, both optional. The app works without either one.</p>

        <h3>Analysis backend</h3>
        <div class="settings-section" data-section="analysis-backend">
          <p class="hint">Where to run heavy analyses that don't fit in Pyodide's ~4 GB ceiling. Leave empty to disable the remote path (everything still runs locally in your browser).</p>
          <label>
            Backend URL
            <input type="text" data-field="analysisBackendUrl" value="${escapeAttr(current.analysisBackendUrl)}" placeholder="${DEFAULT_ANALYSIS_BACKEND}" />
          </label>
          <div class="analysis-backend-row">
            <button class="btn-secondary" data-action="test-analysis-backend">Test backend</button>
            <span class="test-result" data-analysis-backend-result></span>
          </div>
          <p class="hint">
            Want isolated compute? <a href="https://huggingface.co/spaces/ackermand/tourguide-analysis?duplicate=true" target="_blank" rel="noopener">Duplicate this Space</a> into your own free HF account (~5 min, one-time), then paste the resulting URL above.
          </p>
        </div>

        <h3>AI backend</h3>
        <p class="hint">AI unlocks plain-English queries and auto-generated plot/analysis code. Pick none, a local in-browser model, or Gemini.</p>
        <div class="radio-group">
          <label class="radio-row">
            <input type="radio" name="backend" value="none" ${current.backend === "none" ? "checked" : ""}>
            <span><strong>None</strong> — structured browser only</span>
          </label>
          <label class="radio-row">
            <input type="radio" name="backend" value="webllm" ${current.backend === "webllm" ? "checked" : ""} ${webgpu ? "" : "disabled"}>
            <span><strong>WebLLM (in-browser)</strong> — runs locally, no key ${webgpu ? "" : "<em>(needs WebGPU — Chrome/Edge/Safari 18+)</em>"}</span>
          </label>
          <label class="radio-row">
            <input type="radio" name="backend" value="gemini" ${current.backend === "gemini" ? "checked" : ""}>
            <span><strong>Gemini (cloud)</strong> — paste your API key</span>
          </label>
        </div>

        <div class="settings-section" data-section="webllm">
          <div class="webllm-sort-row">
            <label class="webllm-sort-label">Sort by:
              <select data-field="webllmSort">
                <option value="recommended" ${sortKey === "recommended" ? "selected" : ""}>Recommended (best for agent first)</option>
                <option value="size" ${sortKey === "size" ? "selected" : ""}>Size (smallest first)</option>
              </select>
            </label>
          </div>
          <label>
            WebLLM model
            <select data-field="webllmModel">${webllmOptions}</select>
          </label>
          <p class="hint">★★★★★ = best for the agent loop (multi-step SQL + Python). Lower scores = smaller / less code-tuned, may give up after one tool error. Model downloads to your browser cache on first use; subsequent loads are instant and fully offline.</p>
          <div class="webgpu-diagnosis" data-webgpu-diagnosis>Checking WebGPU…</div>
          <button class="btn-secondary" data-action="test-webllm">Load now (optional)</button>
          <span class="test-result" data-webllm-result></span>
          <div class="webllm-progress" data-webllm-progress hidden>
            <div class="webllm-progress-text" data-webllm-progress-text></div>
            <div class="webllm-progress-bar"><div data-webllm-progress-fill></div></div>
          </div>
        </div>

        <div class="settings-section" data-section="gemini">
          <label>
            Gemini API key
            <input type="password" data-field="geminiApiKey" value="${escapeAttr(current.geminiApiKey)}" placeholder="AIza…" />
          </label>
          <div class="gemini-sort-row">
            <label class="gemini-sort-label">Sort by:
              <select data-field="geminiSort">
                <option value="default" ${geminiSortMode() === "default" ? "selected" : ""}>Newest generation first</option>
                <option value="rpd" ${geminiSortMode() === "rpd" ? "selected" : ""}>Free requests/day (most first)</option>
                <option value="rpm" ${geminiSortMode() === "rpm" ? "selected" : ""}>Free requests/min (most first)</option>
              </select>
            </label>
          </div>
          <label>
            Gemini model
            <select data-field="geminiModel">${renderGeminiModelOptions(current.geminiModel)}</select>
          </label>
          <div class="gemini-usage" data-gemini-usage></div>
          <div class="gemini-model-actions">
            <button class="btn-secondary btn-tiny" data-action="refresh-gemini-models" type="button">↻ Refresh available models</button>
            <span class="gemini-model-status" data-gemini-model-status></span>
          </div>
          <p class="hint">Model list is fetched from your key (cached 7 days). Rate limits are NOT returned by Google's API — they're hand-curated here from the <a href="https://aistudio.google.com/usage" target="_blank" rel="noopener">AI Studio dashboard</a> as of 2026-05, and tighten without notice. Models without listed limits don't have a known free-tier entry yet (likely still works, just unknown quota).</p>
          <p class="hint">Get a free key at <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener">aistudio.google.com</a>. Key is stored in your browser's localStorage only.</p>
          <button class="btn-secondary" data-action="test-gemini">Test key</button>
          <span class="test-result" data-test-result></span>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn-secondary" data-action="cancel">Cancel</button>
        <button class="btn-primary" data-action="save">Save</button>
      </div>
    </div>
  `;

  const close = (): void => overlay.remove();
  overlay.querySelector(".modal-close")!.addEventListener("click", close);
  overlay.querySelector("[data-action='cancel']")!.addEventListener("click", close);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });

  const get = (f: string): string => {
    const el = overlay.querySelector<HTMLInputElement | HTMLSelectElement>(`[data-field="${f}"]`);
    return el?.value ?? "";
  };
  const backendValue = (): Settings["backend"] => {
    const el = overlay.querySelector<HTMLInputElement>(`input[name="backend"]:checked`);
    return (el?.value as Settings["backend"]) ?? "none";
  };

  // Sort toggle: re-render the model dropdown without losing the
  // user's current selection. Persist the choice to localStorage so
  // it sticks across sessions independently of the saved Settings
  // (it's a UI preference, not a model choice).
  const webllmSortEl = overlay.querySelector<HTMLSelectElement>(`[data-field="webllmSort"]`);
  const webllmModelEl = overlay.querySelector<HTMLSelectElement>(`[data-field="webllmModel"]`);
  if (webllmSortEl && webllmModelEl) {
    webllmSortEl.addEventListener("change", () => {
      const sort = webllmSortEl.value as "recommended" | "size";
      localStorage.setItem("tourguide.webllmSort", sort);
      const previousSelection = webllmModelEl.value;
      webllmModelEl.innerHTML = renderWebllmOptions(sort);
      if (previousSelection) webllmModelEl.value = previousSelection;
    });
  }

  // Refresh-models button: list the user's available Gemini models via
  // the API, cache the result, and re-render the dropdown so they can
  // pick from real IDs instead of guessing. Useful when Google ships a
  // new family (gemini-3, gemini-3-flash-lite, etc.) — no app update
  // needed, just click Refresh.
  const refreshGeminiBtn = overlay.querySelector<HTMLButtonElement>(
    "[data-action='refresh-gemini-models']",
  )!;
  const refreshGeminiStatus = overlay.querySelector<HTMLSpanElement>(
    "[data-gemini-model-status]",
  )!;
  const geminiModelEl = overlay.querySelector<HTMLSelectElement>(
    `[data-field="geminiModel"]`,
  )!;
  const geminiSortEl = overlay.querySelector<HTMLSelectElement>(
    `[data-field="geminiSort"]`,
  );
  const geminiUsageEl = overlay.querySelector<HTMLDivElement>("[data-gemini-usage]")!;

  // Render the "X / Y today" line under the model dropdown for the
  // currently-selected model. Pulls counts from GeminiBackend's
  // localStorage tracker — accurate for this browser only; usage
  // from other apps using the same key isn't visible.
  const renderGeminiUsage = (): void => {
    const id = geminiModelEl.value;
    if (!id) {
      geminiUsageEl.innerHTML = "";
      return;
    }
    const { rpdUsed, rpmUsed } = GeminiBackend.getUsage(id);
    // Look up known free-tier limits via the cached models list
    // (which has them populated by listGeminiModels).
    const cached = loadCachedGeminiModels();
    const m = cached?.find((x) => x.id === id);
    const rpd = m?.rpd;
    const rpm = m?.rpm;
    const cells: string[] = [];
    if (rpm !== undefined) {
      const pct = Math.min(100, (rpmUsed / rpm) * 100);
      const cls = rpmUsed >= rpm ? "exceeded" : rpmUsed >= rpm * 0.8 ? "warn" : "ok";
      cells.push(
        `<div class="gemini-usage-cell ${cls}"><div class="gemini-usage-label">${rpmUsed}/${rpm} RPM</div><div class="gemini-usage-bar"><div style="width:${pct}%"></div></div></div>`,
      );
    } else {
      cells.push(`<div class="gemini-usage-cell"><div class="gemini-usage-label">${rpmUsed} RPM (no known limit)</div></div>`);
    }
    if (rpd !== undefined) {
      const pct = Math.min(100, (rpdUsed / rpd) * 100);
      const cls = rpdUsed >= rpd ? "exceeded" : rpdUsed >= rpd * 0.8 ? "warn" : "ok";
      cells.push(
        `<div class="gemini-usage-cell ${cls}"><div class="gemini-usage-label">${rpdUsed}/${rpd} today</div><div class="gemini-usage-bar"><div style="width:${pct}%"></div></div></div>`,
      );
    } else {
      cells.push(`<div class="gemini-usage-cell"><div class="gemini-usage-label">${rpdUsed} today (no known daily limit)</div></div>`);
    }
    geminiUsageEl.innerHTML = `
      <div class="gemini-usage-row">${cells.join("")}</div>
      <div class="gemini-usage-note">From this browser only. Resets at Pacific midnight (Google's RPD reset). Check <a href="https://aistudio.google.com/usage" target="_blank" rel="noopener">AI Studio</a> for cross-device totals.</div>
    `;
  };
  renderGeminiUsage();
  geminiModelEl.addEventListener("change", renderGeminiUsage);

  if (geminiSortEl) {
    geminiSortEl.addEventListener("change", () => {
      const mode = geminiSortEl.value as "default" | "rpd" | "rpm";
      localStorage.setItem("tourguide.geminiSort", mode);
      const previous = geminiModelEl.value;
      geminiModelEl.innerHTML = renderGeminiModelOptions(previous);
      if (previous) geminiModelEl.value = previous;
      renderGeminiUsage();
    });
  }
  refreshGeminiBtn.addEventListener("click", async () => {
    const key = get("geminiApiKey").trim();
    if (!key) {
      refreshGeminiStatus.textContent = "Paste a key first";
      refreshGeminiStatus.className = "gemini-model-status err";
      return;
    }
    refreshGeminiStatus.textContent = "Fetching…";
    refreshGeminiStatus.className = "gemini-model-status pending";
    refreshGeminiBtn.disabled = true;
    try {
      const models = await listGeminiModels(key);
      if (models.length === 0) {
        refreshGeminiStatus.textContent = "0 models returned (key may be unscoped)";
        refreshGeminiStatus.className = "gemini-model-status err";
        return;
      }
      saveCachedGeminiModels(models);
      const previous = geminiModelEl.value;
      geminiModelEl.innerHTML = renderGeminiModelOptions(previous);
      // If the previously-saved model wasn't in the new list, the
      // dropdown will show the first option as selected — flag that.
      if (!models.some((m) => m.id === previous)) {
        refreshGeminiStatus.textContent = `✓ ${models.length} models · previous "${previous}" not in list`;
      } else {
        refreshGeminiStatus.textContent = `✓ ${models.length} models loaded`;
      }
      refreshGeminiStatus.className = "gemini-model-status ok";
      renderGeminiUsage();
    } catch (err) {
      refreshGeminiStatus.textContent = (err as Error).message.slice(0, 200);
      refreshGeminiStatus.className = "gemini-model-status err";
    } finally {
      refreshGeminiBtn.disabled = false;
    }
  });

  const testBtn = overlay.querySelector<HTMLButtonElement>("[data-action='test-gemini']")!;
  const testResult = overlay.querySelector<HTMLSpanElement>("[data-test-result]")!;
  testBtn.addEventListener("click", async () => {
    const key = get("geminiApiKey").trim();
    if (!key) {
      testResult.textContent = "Paste a key first";
      testResult.className = "test-result err";
      return;
    }
    testResult.textContent = "Testing…";
    testResult.className = "test-result pending";
    testBtn.disabled = true;
    try {
      const backend = new GeminiBackend(key, get("geminiModel"));
      await backend.validate();
      testResult.textContent = "OK";
      testResult.className = "test-result ok";
    } catch (err) {
      testResult.textContent = (err as Error).message.slice(0, 120);
      testResult.className = "test-result err";
    } finally {
      testBtn.disabled = false;
    }
  });

  const testAnalysisBtn = overlay.querySelector<HTMLButtonElement>("[data-action='test-analysis-backend']")!;
  const analysisResult = overlay.querySelector<HTMLSpanElement>("[data-analysis-backend-result]")!;
  testAnalysisBtn.addEventListener("click", async () => {
    const url = get("analysisBackendUrl").trim();
    if (!url) {
      analysisResult.textContent = "URL empty — remote analysis disabled";
      analysisResult.className = "test-result";
      return;
    }
    analysisResult.textContent = "Pinging…";
    analysisResult.className = "test-result pending";
    testAnalysisBtn.disabled = true;
    try {
      const h = await waitForBackendReady(url, {
        maxMs: 90_000,
        onProgress: (state, msg) => {
          analysisResult.textContent = msg;
          analysisResult.className = state === "ready" ? "test-result ok" : "test-result pending";
        },
      });
      const memStr = h.mem_gb_total ? ` (${h.mem_gb_free?.toFixed(1) ?? "?"} / ${h.mem_gb_total.toFixed(1)} GB free)` : "";
      analysisResult.textContent = `● Ready ${h.version ?? ""}${memStr}`;
      analysisResult.className = "test-result ok";
    } catch (err) {
      analysisResult.textContent = (err as Error).message.slice(0, 140);
      analysisResult.className = "test-result err";
    } finally {
      testAnalysisBtn.disabled = false;
    }
  });

  // Quick, non-blocking health ping when the dialog opens so the status
  // reflects current reality without a click.
  void (async () => {
    const url = get("analysisBackendUrl").trim();
    if (!url) return;
    const h = await fetchHealth(url);
    if (h?.ok) {
      analysisResult.textContent = `● Ready`;
      analysisResult.className = "test-result ok";
    } else {
      analysisResult.textContent = `● Asleep — click Test to wake`;
      analysisResult.className = "test-result";
    }
  })();

  const testWebLLMBtn = overlay.querySelector<HTMLButtonElement>("[data-action='test-webllm']")!;
  const webllmResult = overlay.querySelector<HTMLSpanElement>("[data-webllm-result]")!;
  const webllmProgress = overlay.querySelector<HTMLDivElement>("[data-webllm-progress]")!;
  const webllmProgressText = overlay.querySelector<HTMLDivElement>("[data-webllm-progress-text]")!;
  const webllmProgressFill = overlay.querySelector<HTMLDivElement>("[data-webllm-progress-fill]")!;
  const webgpuDiagEl = overlay.querySelector<HTMLDivElement>("[data-webgpu-diagnosis]")!;

  const webllmRadio = overlay.querySelector<HTMLInputElement>(`input[name="backend"][value="webllm"]`)!;

  void (async () => {
    const d = await diagnoseWebGPU();
    if (d.available) {
      webgpuDiagEl.className = "webgpu-diagnosis ok";
      const f16Note = d.hasF16
        ? "shader-f16 supported — any model works."
        : "shader-f16 NOT supported on this GPU — pick a <code>f32</code> model variant from the dropdown (f16 ones will fail to load).";
      webgpuDiagEl.innerHTML = `<div>✓ WebGPU available</div><div>${f16Note}</div>`;
      webllmRadio.disabled = false;
      testWebLLMBtn.disabled = false;
    } else {
      webgpuDiagEl.className = "webgpu-diagnosis err";
      const lines: string[] = [`✗ ${d.reason}`];
      if (d.detail) lines.push(d.detail);
      if (d.fixHint) lines.push(`<strong>Fix:</strong> ${d.fixHint}`);
      webgpuDiagEl.innerHTML = lines.map((l) => `<div>${l}</div>`).join("");
      webllmRadio.disabled = true;
      testWebLLMBtn.disabled = true;
    }
  })();
  testWebLLMBtn.addEventListener("click", async () => {
    if (!hasWebGPU()) {
      webllmResult.textContent = "WebGPU unavailable";
      webllmResult.className = "test-result err";
      return;
    }
    const modelId = get("webllmModel") || WEBLLM_MODELS[0].id;
    webllmProgress.hidden = false;
    webllmResult.textContent = "";
    testWebLLMBtn.disabled = true;
    try {
      const backend = new WebLLMBackend(modelId, (p) => {
        webllmProgressText.textContent = p.text;
        webllmProgressFill.style.width = `${Math.round((p.progress ?? 0) * 100)}%`;
      });
      await backend.ensureInit();
      webllmResult.textContent = "Loaded";
      webllmResult.className = "test-result ok";
    } catch (err) {
      webllmResult.textContent = (err as Error).message.slice(0, 160);
      webllmResult.className = "test-result err";
    } finally {
      testWebLLMBtn.disabled = false;
    }
  });

  overlay.querySelector("[data-action='save']")!.addEventListener("click", () => {
    const next: Settings = {
      backend: backendValue(),
      geminiApiKey: get("geminiApiKey").trim(),
      geminiModel: get("geminiModel") || "gemini-2.5-flash",
      webllmModel: get("webllmModel") || WEBLLM_MODELS[0].id,
      analysisBackendUrl: get("analysisBackendUrl").trim(),
    };
    saveSettings(next);
    opts.onChange(backendFromSettings(next));
    close();
  });

  document.body.appendChild(overlay);
}

function escapeAttr(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/"/g, "&quot;");
}
