import {
  loadSettings,
  saveSettings,
  backendFromSettings,
  GeminiBackend,
  WebLLMBackend,
  WEBLLM_MODELS,
  hasWebGPU,
  diagnoseWebGPU,
  DEFAULT_ANALYSIS_BACKEND,
  type Settings,
  type LLMBackend,
} from "./llm.js";
import { waitForBackendReady, fetchHealth } from "./remote_analysis.js";

export interface SettingsUIOptions {
  onChange: (backend: LLMBackend) => void;
}

export function openSettingsDialog(opts: SettingsUIOptions): void {
  const current = loadSettings();
  const webgpu = hasWebGPU();
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  const webllmOptions = WEBLLM_MODELS.map(
    (m) => `<option value="${m.id}" ${current.webllmModel === m.id ? "selected" : ""}>${m.label}</option>`,
  ).join("");
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
          <label>
            WebLLM model
            <select data-field="webllmModel">${webllmOptions}</select>
          </label>
          <p class="hint">Model downloads to your browser cache on first use (~1–2 GB). Subsequent loads are instant and fully offline.</p>
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
          <label>
            Gemini model
            <select data-field="geminiModel">
              <option value="gemini-2.5-flash-lite" ${current.geminiModel === "gemini-2.5-flash-lite" ? "selected" : ""}>gemini-2.5-flash-lite (1000 req/day free, recommended)</option>
              <option value="gemini-2.5-flash" ${current.geminiModel === "gemini-2.5-flash" ? "selected" : ""}>gemini-2.5-flash (better quality, ~20–250 req/day free)</option>
              <option value="gemini-2.5-pro" ${current.geminiModel === "gemini-2.5-pro" ? "selected" : ""}>gemini-2.5-pro (highest quality, 100 req/day free)</option>
            </select>
          </label>
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
