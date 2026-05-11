import {
  loadSettings,
  saveSettings,
  backendFromSettings,
  GeminiBackend,
  AnthropicBackend,
  OpenAICompatibleBackend,
  OPENAI_COMPATIBLE_PRESETS,
  WebLLMBackend,
  WEBLLM_MODELS,
  webllmModelLabel,
  hasWebGPU,
  diagnoseWebGPU,
  DEFAULT_ANALYSIS_BACKEND,
  listGeminiModels,
  type Settings,
  type LLMBackend,
  type LLMProvider,
  type GeminiModelInfo,
} from "./llm.js";

// v2: bumped 2026-05-07 after correcting free-tier limits. Old caches
// had wrong RPD numbers (claimed 500 for 2.5-flash-lite which is now
// actually 20); invalidating forces a re-fetch with the right limits.
const GEMINI_MODELS_CACHE_KEY = "tourguide.geminiModels.v2";

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

// Build the <option> list for the Gemini dropdown. Falls back to a small
// hardcoded set when no cache exists yet (first-run, before the user
// has clicked Refresh) so the dropdown isn't empty. Once they click
// Refresh, real model IDs from their key replace this stub.
function renderGeminiModelOptions(currentId: string): string {
  const cached = loadCachedGeminiModels();
  const list = cached && cached.length > 0
    ? cached.map((m) => ({ id: m.id, label: m.id }))
    : [
        // Sensible defaults until Refresh runs. 3.1-flash-lite-preview
        // had the most generous free tier (500 RPD) as of 2026-05-07 —
        // the 2.5 series got crushed to 20 RPD around then.
        { id: "gemini-3.1-flash-lite-preview", label: "gemini-3.1-flash-lite-preview" },
        { id: "gemini-2.5-flash-lite", label: "gemini-2.5-flash-lite" },
        { id: "gemini-2.5-flash", label: "gemini-2.5-flash" },
        { id: "gemini-2.5-pro", label: "gemini-2.5-pro" },
      ];
  if (currentId && !list.some((m) => m.id === currentId)) {
    list.unshift({ id: currentId, label: `${currentId} (current)` });
  }
  return list
    .map((m) => `<option value="${m.id}" ${currentId === m.id ? "selected" : ""}>${m.label}</option>`)
    .join("");
}

export function openSettingsDialog(opts: SettingsUIOptions): void {
  const current = loadSettings();
  const webgpu = hasWebGPU();
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  // WebLLM model list, ordered by agent-suitability (recommended desc,
  // size asc as a tiebreaker). One static order keeps the dropdown
  // simple — users who want to see size-first can read the labels.
  const sortedWebllmModels = [...WEBLLM_MODELS].sort(
    (a, b) => b.recommended - a.recommended || a.sizeGB - b.sizeGB,
  );
  const webllmOptions = sortedWebllmModels
    .map(
      (m) =>
        `<option value="${m.id}" ${current.webllmModel === m.id ? "selected" : ""}>${webllmModelLabel(m)}</option>`,
    )
    .join("");
  overlay.innerHTML = `
    <div class="modal" role="dialog" aria-label="Settings">
      <header class="modal-header">
        <h2>Settings</h2>
        <button class="modal-close" aria-label="Close">×</button>
      </header>
      <div class="modal-body">
        <h3>Cloud LLM</h3>
        <p class="hint">AI unlocks plain-English queries and auto-generated plot / analysis code. Gemini's free tier is the easiest start, but any provider's API key works.</p>

        <label>
          Provider
          <select data-field="provider">
            <option value="gemini" ${current.backend === "gemini" ? "selected" : ""}>Gemini (recommended free tier)</option>
            <option value="anthropic" ${current.backend === "anthropic" ? "selected" : ""}>Anthropic Claude</option>
            <option value="openai" ${current.backend === "openai" ? "selected" : ""}>OpenAI</option>
            <option value="openrouter" ${current.backend === "openrouter" ? "selected" : ""}>OpenRouter (one key for Claude/Gemini/Llama/…)</option>
            <option value="xai" ${current.backend === "xai" ? "selected" : ""}>xAI Grok</option>
            <option value="openai_compatible" ${current.backend === "openai_compatible" ? "selected" : ""}>Local / custom (OpenAI-compatible)</option>
            <option value="none" ${current.backend === "none" ? "selected" : ""}>None — disables the agent</option>
          </select>
        </label>

        <!-- Per-provider sections. Visibility is driven by the
             dropdown above; values persist independently in
             localStorage so swapping providers doesn't lose what
             you typed. -->
        <div class="provider-section" data-provider-section="gemini" ${current.backend === "gemini" ? "" : "hidden"}>
          <label>
            Gemini API key
            <input type="text" class="api-key" data-field="geminiApiKey" value="${escapeAttr(current.geminiApiKey)}" placeholder="Get a free key at aistudio.google.com/app/apikey" autocomplete="off" />
          </label>
          <label>
            Gemini model
            <select data-field="geminiModel">${renderGeminiModelOptions(current.geminiModel)}</select>
          </label>
          <p class="hint">As of 2026-05-07, <code>gemini-3.1-flash-lite-preview</code> had the most generous free tier (15 RPM · 500 req/day) — quotas change, see <a href="https://aistudio.google.com/usage" target="_blank" rel="noopener">your dashboard</a>.</p>
          <button class="btn-secondary" data-action="test-gemini">Test key</button>
          <span class="test-result" data-test-result-gemini></span>
        </div>

        <div class="provider-section" data-provider-section="anthropic" ${current.backend === "anthropic" ? "" : "hidden"}>
          <label>
            Anthropic API key
            <input type="text" class="api-key" data-field="anthropicApiKey" value="${escapeAttr(current.anthropicApiKey)}" placeholder="Get a key at console.anthropic.com" autocomplete="off" />
          </label>
          <label>
            Claude model
            <input type="text" data-field="anthropicModel" value="${escapeAttr(current.anthropicModel || "claude-sonnet-4.6")}" placeholder="claude-sonnet-4.6" />
          </label>
          <p class="hint">Hits api.anthropic.com directly using <code>anthropic-dangerous-direct-browser-access</code>. Common model ids: <code>claude-sonnet-4.6</code>, <code>claude-opus-4.7</code>, <code>claude-haiku-4.5</code>.</p>
          <button class="btn-secondary" data-action="test-anthropic">Test key</button>
          <span class="test-result" data-test-result-anthropic></span>
        </div>

        <div class="provider-section" data-provider-section="openai" ${current.backend === "openai" ? "" : "hidden"}>
          <label>
            OpenAI API key
            <input type="text" class="api-key" data-field="openaiApiKey-openai" value="${escapeAttr(current.openaiApiKey)}" placeholder="sk-…" autocomplete="off" />
          </label>
          <label>
            Model
            <input type="text" data-field="openaiModel-openai" value="${escapeAttr(current.openaiModel)}" placeholder="${OPENAI_COMPATIBLE_PRESETS.openai.placeholderModel}" />
          </label>
          <p class="hint">Hits <code>${OPENAI_COMPATIBLE_PRESETS.openai.url}/chat/completions</code>. Get a key at <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener">platform.openai.com</a>.</p>
          <button class="btn-secondary" data-action="test-oai">Test key</button>
          <span class="test-result" data-test-result-oai></span>
        </div>

        <div class="provider-section" data-provider-section="openrouter" ${current.backend === "openrouter" ? "" : "hidden"}>
          <label>
            OpenRouter API key
            <input type="text" class="api-key" data-field="openaiApiKey-openrouter" value="${escapeAttr(current.openaiApiKey)}" placeholder="sk-or-…" autocomplete="off" />
          </label>
          <label>
            Model
            <input type="text" data-field="openaiModel-openrouter" value="${escapeAttr(current.openaiModel)}" placeholder="${OPENAI_COMPATIBLE_PRESETS.openrouter.placeholderModel}" />
          </label>
          <p class="hint">One key for Claude / Gemini / Llama / etc. Browse models at <a href="https://openrouter.ai/models" target="_blank" rel="noopener">openrouter.ai/models</a>. Hits <code>${OPENAI_COMPATIBLE_PRESETS.openrouter.url}/chat/completions</code>.</p>
          <button class="btn-secondary" data-action="test-oai">Test key</button>
          <span class="test-result" data-test-result-oai></span>
        </div>

        <div class="provider-section" data-provider-section="xai" ${current.backend === "xai" ? "" : "hidden"}>
          <label>
            xAI API key
            <input type="text" class="api-key" data-field="openaiApiKey-xai" value="${escapeAttr(current.openaiApiKey)}" placeholder="xai-…" autocomplete="off" />
          </label>
          <label>
            Model
            <input type="text" data-field="openaiModel-xai" value="${escapeAttr(current.openaiModel)}" placeholder="${OPENAI_COMPATIBLE_PRESETS.xai.placeholderModel}" />
          </label>
          <p class="hint">Hits <code>${OPENAI_COMPATIBLE_PRESETS.xai.url}/chat/completions</code>. Get a key at <a href="https://console.x.ai" target="_blank" rel="noopener">console.x.ai</a>.</p>
          <button class="btn-secondary" data-action="test-oai">Test key</button>
          <span class="test-result" data-test-result-oai></span>
        </div>

        <div class="provider-section" data-provider-section="openai_compatible" ${current.backend === "openai_compatible" ? "" : "hidden"}>
          <label>
            Base URL
            <input type="text" data-field="openaiBaseUrl" value="${escapeAttr(current.openaiBaseUrl)}" placeholder="http://localhost:11434/v1" />
          </label>
          <label>
            API key (optional for local servers)
            <input type="text" class="api-key" data-field="openaiApiKey-custom" value="${escapeAttr(current.openaiApiKey)}" autocomplete="off" />
          </label>
          <label>
            Model
            <input type="text" data-field="openaiModel-custom" value="${escapeAttr(current.openaiModel)}" placeholder="llama3.2" />
          </label>
          <p class="hint">For Ollama (<code>http://localhost:11434/v1</code>), vLLM, LM Studio, llama.cpp server, or any other endpoint that speaks OpenAI's <code>/chat/completions</code>. Local URLs skip the auth check.</p>
          <button class="btn-secondary" data-action="test-oai">Test connection</button>
          <span class="test-result" data-test-result-oai></span>
        </div>

        <div class="provider-section" data-provider-section="none" ${current.backend === "none" ? "" : "hidden"}>
          <p class="hint">Agent disabled. The structured browser still works for ingested CSVs.</p>
        </div>

        <p class="hint storage-note">🔒 Your API key is stored in your browser's localStorage only — Tourguide's frontend is static-hosted, it never touches a server we control. The key is sent directly to the provider you picked.</p>

        <details class="settings-advanced">
          <summary>Advanced — local WebLLM, analysis backend, model refresh</summary>

          <h4>WebLLM (in-browser, no API key)</h4>
          <div class="settings-section" data-section="webllm">
            <p class="hint">Use WebLLM to run a small LLM entirely in your browser via WebGPU. ${webgpu ? "" : "<em>(needs WebGPU — Chrome / Edge / Safari 18+.)</em>"} Set Provider to "None" above and toggle this on instead — picking it here overrides the cloud Provider selection on Save.</p>
            <label class="radio-row">
              <input type="checkbox" data-field="useWebllm" ${current.backend === "webllm" ? "checked" : ""} ${webgpu ? "" : "disabled"}>
              <span>Use WebLLM instead of a cloud provider</span>
            </label>
            <label>
              WebLLM model
              <select data-field="webllmModel">${webllmOptions}</select>
            </label>
            <p class="hint">Listed best-for-agent first. Top entries handle multi-step SQL + Python reliably; lower entries are smaller / less code-tuned. Model downloads to your browser cache on first use.</p>
            <div class="webgpu-diagnosis" data-webgpu-diagnosis>Checking WebGPU…</div>
            <button class="btn-secondary" data-action="test-webllm">Load now (optional)</button>
            <span class="test-result" data-webllm-result></span>
            <div class="webllm-progress" data-webllm-progress hidden>
              <div class="webllm-progress-text" data-webllm-progress-text></div>
              <div class="webllm-progress-bar"><div data-webllm-progress-fill></div></div>
            </div>
          </div>

          <h4>Refresh Gemini model list</h4>
          <div class="gemini-model-actions">
            <button class="btn-secondary btn-tiny" data-action="refresh-gemini-models" type="button">↻ Refresh available models</button>
            <span class="gemini-model-status" data-gemini-model-status></span>
          </div>

          <h4>Analysis backend (heavy compute)</h4>
          <div class="settings-section" data-section="analysis-backend">
            <p class="hint">Where to run heavy analyses that don't fit in Pyodide's ~4 GB ceiling. Leave empty to disable the remote path (everything still runs locally).</p>
            <div class="analysis-backend-fork-callout">
              <strong>For real use:</strong> fork the Space.
              The default URL points to a shared demo Space with limited
              CPU + memory and possible cold starts.
              <a class="btn-link" href="https://huggingface.co/spaces/ackermand/tourguide-analysis?duplicate=true" target="_blank" rel="noopener">Duplicate this Space →</a>
              (~5 min, one-time). Paste the resulting URL below.
              On your Space's <em>Files</em> tab, click <em>"Sync with
              upstream"</em> for updates.
            </div>
            <label>
              Backend URL
              <input type="text" data-field="analysisBackendUrl" value="${escapeAttr(current.analysisBackendUrl)}" placeholder="${DEFAULT_ANALYSIS_BACKEND}" />
            </label>
            <div class="analysis-backend-row">
              <button class="btn-secondary" data-action="test-analysis-backend">Test backend</button>
              <span class="test-result" data-analysis-backend-result></span>
            </div>
          </div>
        </details>
      </div>
      <div class="modal-footer">
        <button class="btn-link" data-action="show-welcome" type="button">Show welcome again</button>
        <span style="flex:1"></span>
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
  // Lazy-import the welcome dialog so this Settings module doesn't drag
  // in welcome_ui's dependencies on every Settings open. Imported on
  // click only.
  overlay.querySelector("[data-action='show-welcome']")?.addEventListener("click", async () => {
    close();
    const { openWelcomeDialog, clearWelcomeSeen } = await import("./welcome_ui.js");
    clearWelcomeSeen();
    openWelcomeDialog({
      onOpenLoader: () => document.getElementById("load-data-btn")?.click(),
      onLoadDemo: () => {
        // The demo loader lives in main.ts (it has access to the
        // catalog). Defer to the dataset dropdown — picking the first
        // non-synthetic entry there triggers the same load path. The
        // dropdown's change handler does the work.
        const sel = document.getElementById("dataset-select") as HTMLSelectElement | null;
        if (sel && sel.options.length > 0) {
          // Pick whichever option index isn't 'demo_synthetic'.
          let target = 0;
          for (let i = 0; i < sel.options.length; i++) {
            if (!/synthetic/i.test(sel.options[i].textContent ?? "")) {
              target = i;
              break;
            }
          }
          sel.value = String(target);
          sel.dispatchEvent(new Event("change"));
        }
      },
      onSettingsChanged: () => {
        opts.onChange(backendFromSettings(loadSettings()));
      },
    });
  });

  const get = (f: string): string => {
    const el = overlay.querySelector<HTMLInputElement | HTMLSelectElement>(`[data-field="${f}"]`);
    return el?.value ?? "";
  };
  // Provider dropdown drives which provider section is visible.
  // Each provider section's inputs persist independently via Save —
  // switching providers doesn't clear what you've typed in the
  // others. WebLLM has its own checkbox inside Advanced that
  // overrides the provider dropdown if checked.
  const providerSelect = overlay.querySelector<HTMLSelectElement>(`[data-field="provider"]`)!;
  const providerSections = overlay.querySelectorAll<HTMLDivElement>(`[data-provider-section]`);
  const showProviderSection = (provider: string): void => {
    providerSections.forEach((sec) => {
      sec.hidden = sec.getAttribute("data-provider-section") !== provider;
    });
  };
  providerSelect.addEventListener("change", () => showProviderSection(providerSelect.value));

  // The OpenAI-compatible group (openai / openrouter / xai / custom)
  // shares one stored API key + model under the hood — just shown
  // through different placeholders / labels per provider. We keep the
  // four input nodes in lockstep so the user typing the OpenRouter
  // key while on the OpenRouter section also updates what they'd see
  // if they switched to the OpenAI section. Same for model.
  const oaiKeyInputs = overlay.querySelectorAll<HTMLInputElement>(
    `input[data-field^="openaiApiKey-"]`,
  );
  const oaiModelInputs = overlay.querySelectorAll<HTMLInputElement>(
    `input[data-field^="openaiModel-"]`,
  );
  const syncSiblings = (src: HTMLInputElement, group: NodeListOf<HTMLInputElement>): void => {
    group.forEach((el) => {
      if (el !== src) el.value = src.value;
    });
  };
  oaiKeyInputs.forEach((el) =>
    el.addEventListener("input", () => syncSiblings(el, oaiKeyInputs)),
  );
  oaiModelInputs.forEach((el) =>
    el.addEventListener("input", () => syncSiblings(el, oaiModelInputs)),
  );
  // Helper: read whichever shared OpenAI-compatible field is currently
  // visible (they all hold the same value after sync, so any non-empty
  // one works — but pick the one matching the active provider for
  // cleanliness).
  const getOaiKey = (): string => {
    const live = overlay.querySelector<HTMLInputElement>(`input[data-field^="openaiApiKey-"]`);
    return live?.value ?? "";
  };
  const getOaiModel = (): string => {
    const live = overlay.querySelector<HTMLInputElement>(`input[data-field^="openaiModel-"]`);
    return live?.value ?? "";
  };

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
    } catch (err) {
      refreshGeminiStatus.textContent = (err as Error).message.slice(0, 200);
      refreshGeminiStatus.className = "gemini-model-status err";
    } finally {
      refreshGeminiBtn.disabled = false;
    }
  });

  // Test buttons — one per provider section. Each runs validate() on
  // a freshly-constructed backend with the key/model from its own
  // section, so the user gets feedback specific to where they're
  // typing.
  const wireTestButton = (
    selector: string,
    resultSelector: string,
    build: () => { backend: { validate(): Promise<void> } | null; emptyMsg: string },
  ): void => {
    const btn = overlay.querySelector<HTMLButtonElement>(selector);
    const out = overlay.querySelector<HTMLSpanElement>(resultSelector);
    if (!btn || !out) return;
    btn.addEventListener("click", async () => {
      const { backend, emptyMsg } = build();
      if (!backend) {
        out.textContent = emptyMsg;
        out.className = "test-result err";
        return;
      }
      out.textContent = "Testing…";
      out.className = "test-result pending";
      btn.disabled = true;
      try {
        await backend.validate();
        out.textContent = "OK";
        out.className = "test-result ok";
      } catch (err) {
        out.textContent = (err as Error).message.slice(0, 200);
        out.className = "test-result err";
      } finally {
        btn.disabled = false;
      }
    });
  };
  wireTestButton(
    "[data-action='test-gemini']",
    "[data-test-result-gemini]",
    () => {
      const key = get("geminiApiKey").trim();
      if (!key) return { backend: null, emptyMsg: "Paste a key first" };
      return { backend: new GeminiBackend(key, get("geminiModel")), emptyMsg: "" };
    },
  );
  wireTestButton(
    "[data-action='test-anthropic']",
    "[data-test-result-anthropic]",
    () => {
      const key = get("anthropicApiKey").trim();
      if (!key) return { backend: null, emptyMsg: "Paste a key first" };
      return { backend: new AnthropicBackend(key, get("anthropicModel") || "claude-sonnet-4.6"), emptyMsg: "" };
    },
  );
  // OAI-compatible group: one Test button per provider section, all
  // hooked to the same builder (they share state).
  overlay.querySelectorAll<HTMLButtonElement>("[data-action='test-oai']").forEach((btn) => {
    const out = btn.parentElement?.querySelector<HTMLSpanElement>("[data-test-result-oai]");
    if (!out) return;
    btn.addEventListener("click", async () => {
      const provider = providerSelect.value;
      const baseUrl =
        provider === "openai_compatible"
          ? get("openaiBaseUrl").trim()
          : OPENAI_COMPATIBLE_PRESETS[provider]?.url ?? "";
      const key = getOaiKey().trim();
      const model = getOaiModel().trim();
      if (!baseUrl) {
        out.textContent = "Base URL is empty";
        out.className = "test-result err";
        return;
      }
      if (!model) {
        out.textContent = "Model is empty";
        out.className = "test-result err";
        return;
      }
      out.textContent = "Testing…";
      out.className = "test-result pending";
      btn.disabled = true;
      try {
        const backend = new OpenAICompatibleBackend(baseUrl, key, model);
        await backend.validate();
        out.textContent = "OK";
        out.className = "test-result ok";
      } catch (err) {
        out.textContent = (err as Error).message.slice(0, 200);
        out.className = "test-result err";
      } finally {
        btn.disabled = false;
      }
    });
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

  const webllmCheckbox = overlay.querySelector<HTMLInputElement>(`input[data-field="useWebllm"]`)!;

  void (async () => {
    const d = await diagnoseWebGPU();
    if (d.available) {
      webgpuDiagEl.className = "webgpu-diagnosis ok";
      const f16Note = d.hasF16
        ? "shader-f16 supported — any model works."
        : "shader-f16 NOT supported on this GPU — pick a <code>f32</code> model variant from the dropdown (f16 ones will fail to load).";
      webgpuDiagEl.innerHTML = `<div>✓ WebGPU available</div><div>${f16Note}</div>`;
      webllmCheckbox.disabled = false;
      testWebLLMBtn.disabled = false;
    } else {
      webgpuDiagEl.className = "webgpu-diagnosis err";
      const lines: string[] = [`✗ ${d.reason}`];
      if (d.detail) lines.push(d.detail);
      if (d.fixHint) lines.push(`<strong>Fix:</strong> ${d.fixHint}`);
      webgpuDiagEl.innerHTML = lines.map((l) => `<div>${l}</div>`).join("");
      webllmCheckbox.disabled = true;
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
    // The visible Provider dropdown is the source of truth UNLESS
    // the WebLLM checkbox is on (in which case backend = "webllm"
    // overrides whatever cloud provider was picked).
    const cloudProvider = providerSelect.value as LLMProvider;
    const useWebllm = webllmCheckbox.checked;
    const backend: LLMProvider = useWebllm ? "webllm" : cloudProvider;
    // Preserve the openai_compatible base URL the user typed; for the
    // preset providers (openai/openrouter/xai), the base URL comes
    // from OPENAI_COMPATIBLE_PRESETS — but we still store any
    // user-typed URL so switching back to "openai_compatible" later
    // re-shows it.
    const next: Settings = {
      backend,
      geminiApiKey: get("geminiApiKey").trim(),
      geminiModel: get("geminiModel") || "gemini-2.5-flash",
      anthropicApiKey: get("anthropicApiKey").trim(),
      anthropicModel: get("anthropicModel").trim() || "claude-sonnet-4.6",
      openaiApiKey: getOaiKey().trim(),
      openaiModel: getOaiModel().trim(),
      openaiBaseUrl: get("openaiBaseUrl").trim(),
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
