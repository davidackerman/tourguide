// First-run welcome modal — explains the two optional backends
// (AI for plain-English queries, Analysis for heavy compute on zarr)
// and walks the user through configuring them. Skippable; remembers
// dismissal in localStorage so it never reappears unless the user
// asks to see it again from Settings.

import {
  loadSettings,
  saveSettings,
  GeminiBackend,
  AnthropicBackend,
  OpenAICompatibleBackend,
  OPENAI_COMPATIBLE_PRESETS,
  WEBLLM_MODELS,
  webllmModelLabel,
  hasWebGPU,
  DEFAULT_ANALYSIS_BACKEND,
  type LLMProvider,
} from "./llm.js";

const WELCOME_DISMISSED_KEY = "tourguide.welcomeDismissed.v1";

export function hasSeenWelcome(): boolean {
  try {
    return localStorage.getItem(WELCOME_DISMISSED_KEY) === "1";
  } catch {
    return false;
  }
}

function markWelcomeSeen(): void {
  try {
    localStorage.setItem(WELCOME_DISMISSED_KEY, "1");
  } catch {
    /* localStorage disabled; fine, modal will just keep appearing */
  }
}

export function clearWelcomeSeen(): void {
  try {
    localStorage.removeItem(WELCOME_DISMISSED_KEY);
  } catch {
    /* fine */
  }
}

export interface WelcomeOptions {
  // Called when the user clicks "Open loader" so the caller can
  // bring up the regular dataset loader dialog.
  onOpenLoader: () => void;
  // Called when the user clicks "Load demo dataset" — caller loads
  // a public OpenOrganelle / cellmap dataset from the catalog so a
  // first-time user can poke around without bringing their own data.
  onLoadDemo: () => void;
  // Called when settings change so the topbar AI indicator etc.
  // refresh without a page reload.
  onSettingsChanged: () => void;
}

export function openWelcomeDialog(opts: WelcomeOptions): void {
  const settings = loadSettings();
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  // Pre-populate AI choice from existing settings so a user who hits
  // 'Show welcome again' sees their current state, not blank.
  const initialAi = settings.backend;
  const webgpu = hasWebGPU();
  // Full list, ordered by agent suitability (recommended desc, size
  // asc tiebreak) — same order Settings uses. The welcome modal had a
  // top-4 cap which hid Llama / Qwen-3 / Gemma options the user might
  // actually want; no reason to hide them here.
  const webllmAll = [...WEBLLM_MODELS].sort(
    (a, b) => b.recommended - a.recommended || a.sizeGB - b.sizeGB,
  );

  overlay.innerHTML = `
    <div class="modal modal-welcome" role="dialog" aria-label="Welcome to tourguide">
      <header class="modal-header">
        <h2>Welcome to tourguide</h2>
        <button class="modal-close" aria-label="Skip">×</button>
      </header>
      <div class="modal-body">
        <p class="welcome-blurb">
          A 3D microscopy viewer with built-in structured-data browsing,
          plain-English queries, and Python analysis. Two optional
          backends to set up — both can be skipped and configured later.
        </p>

        <section class="welcome-step">
          <h3>1. AI backend <span class="welcome-step-tag">optional</span></h3>
          <p class="hint">
            Powers plain-English queries and auto-generated plot / analysis
            code. Gemini's free tier is the easiest start, but any provider's
            API key works. Without this, the structured browser and viewer
            still work — you just can't ask questions in English.
          </p>

          <label class="welcome-provider-label">
            Provider
            <select data-welcome-provider>
              <option value="gemini" ${initialAi === "gemini" ? "selected" : ""}>Gemini (recommended free tier)</option>
              <option value="anthropic" ${initialAi === "anthropic" ? "selected" : ""}>Anthropic Claude</option>
              <option value="openai" ${initialAi === "openai" ? "selected" : ""}>OpenAI</option>
              <option value="openrouter" ${initialAi === "openrouter" ? "selected" : ""}>OpenRouter (one key for Claude/Gemini/Llama/…)</option>
              <option value="xai" ${initialAi === "xai" ? "selected" : ""}>xAI Grok</option>
              <option value="openai_compatible" ${initialAi === "openai_compatible" ? "selected" : ""}>Local / custom (OpenAI-compatible)</option>
              <option value="webllm" ${initialAi === "webllm" ? "selected" : ""} ${webgpu ? "" : "disabled"}>WebLLM (local, in-browser) ${webgpu ? "" : "— needs WebGPU"}</option>
              <option value="none" ${initialAi === "none" ? "selected" : ""}>None — disables the agent</option>
            </select>
          </label>

          <div class="welcome-provider-section" data-welcome-provider-section="gemini" ${initialAi === "gemini" ? "" : "hidden"}>
            <label>
              Gemini API key
              <input type="password" data-welcome-gemini-key value="${escapeAttr(settings.geminiApiKey)}" placeholder="Get a free key at aistudio.google.com/apikey" autocomplete="off" />
            </label>
            <p class="hint">~500 req/day on Flash Lite. Stored in browser localStorage only.</p>
          </div>

          <div class="welcome-provider-section" data-welcome-provider-section="anthropic" ${initialAi === "anthropic" ? "" : "hidden"}>
            <label>
              Anthropic API key
              <input type="password" data-welcome-anthropic-key value="${escapeAttr(settings.anthropicApiKey)}" placeholder="Get a key at console.anthropic.com" autocomplete="off" />
            </label>
            <label>
              Model
              <input type="text" data-welcome-anthropic-model value="${escapeAttr(settings.anthropicModel || "claude-sonnet-4.6")}" placeholder="claude-sonnet-4.6" />
            </label>
          </div>

          <div class="welcome-provider-section" data-welcome-provider-section="openai" ${initialAi === "openai" ? "" : "hidden"}>
            <label>
              OpenAI API key
              <input type="password" data-welcome-oai-key value="${escapeAttr(settings.openaiApiKey)}" placeholder="sk-…" autocomplete="off" />
            </label>
            <label>
              Model
              <input type="text" data-welcome-oai-model value="${escapeAttr(settings.openaiModel)}" placeholder="${OPENAI_COMPATIBLE_PRESETS.openai.placeholderModel}" />
            </label>
          </div>

          <div class="welcome-provider-section" data-welcome-provider-section="openrouter" ${initialAi === "openrouter" ? "" : "hidden"}>
            <label>
              OpenRouter API key
              <input type="password" data-welcome-oai-key value="${escapeAttr(settings.openaiApiKey)}" placeholder="sk-or-…" autocomplete="off" />
            </label>
            <label>
              Model
              <input type="text" data-welcome-oai-model value="${escapeAttr(settings.openaiModel)}" placeholder="${OPENAI_COMPATIBLE_PRESETS.openrouter.placeholderModel}" />
            </label>
            <p class="hint">One key for Claude / Gemini / Llama / etc — browse at <a href="https://openrouter.ai/models" target="_blank" rel="noopener">openrouter.ai/models</a>.</p>
          </div>

          <div class="welcome-provider-section" data-welcome-provider-section="xai" ${initialAi === "xai" ? "" : "hidden"}>
            <label>
              xAI API key
              <input type="password" data-welcome-oai-key value="${escapeAttr(settings.openaiApiKey)}" placeholder="xai-…" autocomplete="off" />
            </label>
            <label>
              Model
              <input type="text" data-welcome-oai-model value="${escapeAttr(settings.openaiModel)}" placeholder="${OPENAI_COMPATIBLE_PRESETS.xai.placeholderModel}" />
            </label>
          </div>

          <div class="welcome-provider-section" data-welcome-provider-section="openai_compatible" ${initialAi === "openai_compatible" ? "" : "hidden"}>
            <label>
              Base URL
              <input type="text" data-welcome-oai-baseurl value="${escapeAttr(settings.openaiBaseUrl)}" placeholder="http://localhost:11434/v1" />
            </label>
            <label>
              API key (optional for local servers)
              <input type="password" data-welcome-oai-key value="${escapeAttr(settings.openaiApiKey)}" autocomplete="off" />
            </label>
            <label>
              Model
              <input type="text" data-welcome-oai-model value="${escapeAttr(settings.openaiModel)}" placeholder="llama3.2" />
            </label>
            <p class="hint">For Ollama / vLLM / LM Studio / llama.cpp server, or any other OpenAI-compatible endpoint. Local URLs skip auth.</p>
          </div>

          <div class="welcome-provider-section" data-welcome-provider-section="webllm" ${initialAi === "webllm" ? "" : "hidden"}>
            <label>
              WebLLM model
              <select data-welcome-webllm-model ${webgpu ? "" : "disabled"}>
                ${webllmAll
                  .map(
                    (m) =>
                      `<option value="${m.id}" ${settings.webllmModel === m.id ? "selected" : ""}>${webllmModelLabel(m)}</option>`,
                  )
                  .join("")}
              </select>
            </label>
            <p class="hint">Runs on your GPU via WebGPU. ~2-4 GB one-time download per model; fully offline after that, no quotas. Slower per token.${webgpu ? "" : " <em>(needs Chrome / Edge / Safari 18+)</em>"}</p>
          </div>

          <div class="welcome-provider-section" data-welcome-provider-section="none" ${initialAi === "none" ? "" : "hidden"}>
            <p class="hint">Agent disabled. The structured browser still works for ingested CSVs. You can flip this on later from Settings.</p>
          </div>

          <p class="hint storage-note">🔒 Your API key is stored in your browser's localStorage only — Tourguide's frontend is static-hosted, it never touches a server we control. The key is sent directly to the provider you picked.</p>
        </section>

        <section class="welcome-step">
          <h3>2. Analysis backend <span class="welcome-step-tag">optional</span></h3>
          <p class="hint">
            Used by <strong>Σ Analyze</strong> and <strong>🐍 Custom</strong> for
            heavy Python (regionprops, meshes, scikit-image) on multi-GB zarrs
            that don't fit in the browser's WASM memory. Without this, both
            tools fall back to a smaller in-browser Python (Pyodide) that's
            limited to ~4 GB inputs.
          </p>
          <div class="welcome-backend-choice">
            <label class="welcome-radio">
              <input type="radio" name="welcome-analysis" value="default"
                     ${settings.analysisBackendUrl.trim() === "" || settings.analysisBackendUrl.trim() === DEFAULT_ANALYSIS_BACKEND ? "checked" : ""} />
              <div>
                <strong>Use the shared demo Space</strong>
                <p class="hint">
                  <code>${DEFAULT_ANALYSIS_BACKEND}</code> — convenient for
                  trying things out. Rate-limited and shared with other users;
                  expect cold starts (~60 s) and contention. <strong>Don't
                  rely on this for real work.</strong>
                </p>
              </div>
            </label>
            <label class="welcome-radio">
              <input type="radio" name="welcome-analysis" value="custom"
                     ${settings.analysisBackendUrl.trim() !== "" && settings.analysisBackendUrl.trim() !== DEFAULT_ANALYSIS_BACKEND ? "checked" : ""} />
              <div>
                <strong>Use my own fork (recommended for real use)</strong>
                <p class="hint">
                  <a href="https://huggingface.co/spaces/ackermand/tourguide-analysis?duplicate=true"
                     target="_blank" rel="noopener">Click here to duplicate the
                  Space into your own free HF account</a> (~5 min, one-time).
                  Then paste the resulting <code>https://&lt;you&gt;-tourguide-analysis.hf.space</code>
                  URL below. Your fork has its own quota and won't contend
                  with anyone.
                </p>
                <input type="text" data-welcome-analysis-url
                       value="${escapeAttr(settings.analysisBackendUrl)}"
                       placeholder="https://&lt;you&gt;-tourguide-analysis.hf.space" />
                <p class="hint" style="margin-top:0.4rem;">
                  <strong>Keeping your fork up to date:</strong> on your
                  Space's <em>Files</em> tab on huggingface.co there's a
                  <em>"Sync with upstream"</em> button when an update is
                  available — one click to pull in the maintainer's latest
                  app.py.
                </p>
              </div>
            </label>
          </div>
        </section>

        <section class="welcome-step">
          <h3>3. Get started</h3>
          <p class="hint">
            New here? Load a small public OpenOrganelle dataset so you
            can poke around before bringing your own. Otherwise jump
            straight into the loader to paste a URL, drop a Neuroglancer
            state, or pick a YAML / local folder.
          </p>
          <div class="welcome-getstarted-row">
            <button class="btn-secondary" data-welcome-load-demo type="button">🧬 Load demo dataset (jrc_hela-2, ~public S3)</button>
            <button class="btn-primary" data-welcome-load-mine type="button">📂 Load my data</button>
          </div>
          <p class="hint">Both will save your settings first, then start loading.</p>
        </section>
      </div>
      <div class="modal-footer">
        <button class="btn-secondary" data-welcome-skip>Save &amp; close (no data load)</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);

  const close = (saved: boolean): void => {
    if (saved) markWelcomeSeen();
    overlay.remove();
  };
  // Closing the modal always tries to persist whatever's typed —
  // otherwise a user who enters an API key but doesn't click a "Load"
  // button loses it on dismiss. persistSettings's validate step may
  // surface an "API key didn't validate, save anyway?" confirm; if
  // the user cancels there, we leave the modal open.
  const saveAndClose = async (): Promise<void> => {
    if (!(await persistSettings())) return;
    markWelcomeSeen();
    close(true);
  };
  overlay.querySelector(".modal-close")!.addEventListener("click", () => {
    void saveAndClose();
  });
  overlay.querySelector("[data-welcome-skip]")!.addEventListener("click", () => {
    void saveAndClose();
  });
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) {
      void saveAndClose();
    }
  });

  // Provider dropdown drives which provider section is visible.
  // Same pattern Settings uses — each section's inputs persist on
  // Save into its own slot of the Settings type, so swapping
  // providers doesn't lose what you typed in the others.
  const providerSelect = overlay.querySelector<HTMLSelectElement>("[data-welcome-provider]")!;
  const providerSections = overlay.querySelectorAll<HTMLDivElement>("[data-welcome-provider-section]");
  providerSelect.addEventListener("change", () => {
    const v = providerSelect.value;
    providerSections.forEach((s) => {
      s.hidden = s.getAttribute("data-welcome-provider-section") !== v;
    });
  });

  // Save settings (extracted so both 'Load demo' and 'Load my data'
  // commit the user's AI / analysis-backend choices before navigating).
  // Returns false if the user cancelled an invalid-Gemini-key warning.
  const persistSettings = async (): Promise<boolean> => {
    const aiChoice = (providerSelect.value as LLMProvider);
    const geminiKey = overlay.querySelector<HTMLInputElement>("[data-welcome-gemini-key]")?.value.trim() ?? settings.geminiApiKey;
    const anthropicKey = overlay.querySelector<HTMLInputElement>("[data-welcome-anthropic-key]")?.value.trim() ?? settings.anthropicApiKey;
    const anthropicModel = overlay.querySelector<HTMLInputElement>("[data-welcome-anthropic-model]")?.value.trim() || settings.anthropicModel || "claude-sonnet-4.6";
    // OpenAI-compatible group: only the visible section's inputs are
    // populated in the DOM (others were never rendered for the
    // un-shown providers, since the welcome dialog renders a fresh
    // copy of each section). Read whichever pair is in the active
    // section, fall back to stored settings.
    const visibleOaiKey = overlay.querySelector<HTMLInputElement>(`[data-welcome-provider-section="${aiChoice}"] [data-welcome-oai-key]`);
    const visibleOaiModel = overlay.querySelector<HTMLInputElement>(`[data-welcome-provider-section="${aiChoice}"] [data-welcome-oai-model]`);
    const visibleOaiUrl = overlay.querySelector<HTMLInputElement>("[data-welcome-oai-baseurl]");
    const oaiKey = visibleOaiKey?.value.trim() ?? settings.openaiApiKey;
    const oaiModel = visibleOaiModel?.value.trim() ?? settings.openaiModel;
    const oaiUrl = visibleOaiUrl?.value.trim() ?? settings.openaiBaseUrl;
    const webllmModel = overlay.querySelector<HTMLSelectElement>("[data-welcome-webllm-model]")?.value ?? settings.webllmModel;
    const analysisChoice = overlay.querySelector<HTMLInputElement>('input[name="welcome-analysis"]:checked')?.value ?? "default";
    let analysisUrl: string;
    if (analysisChoice === "custom") {
      const typed = overlay.querySelector<HTMLInputElement>("[data-welcome-analysis-url]")?.value.trim() ?? "";
      analysisUrl = typed;
    } else {
      analysisUrl = DEFAULT_ANALYSIS_BACKEND;
    }
    const next = {
      ...settings,
      backend: aiChoice,
      geminiApiKey: geminiKey,
      anthropicApiKey: anthropicKey,
      anthropicModel: anthropicModel,
      openaiApiKey: oaiKey,
      openaiModel: oaiModel,
      openaiBaseUrl: oaiUrl,
      webllmModel,
      analysisBackendUrl: analysisUrl,
    };
    saveSettings(next);
    // Validate the picked provider's key (where we can — local
    // OAI-compatible endpoints might not even be running yet, so
    // skip those). A failed validate still lets the user save
    // anyway (typo'd key + click anyway is faster than retyping
    // when the user knows the key is right).
    try {
      if (aiChoice === "gemini" && geminiKey) {
        await new GeminiBackend(geminiKey, next.geminiModel).validate();
      } else if (aiChoice === "anthropic" && anthropicKey) {
        await new AnthropicBackend(anthropicKey, anthropicModel).validate();
      } else if (
        (aiChoice === "openai" || aiChoice === "openrouter" || aiChoice === "xai") &&
        oaiKey
      ) {
        const url = OPENAI_COMPATIBLE_PRESETS[aiChoice].url;
        await new OpenAICompatibleBackend(url, oaiKey, oaiModel).validate();
      }
    } catch (err) {
      const ok = confirm(
        `API key didn't validate: ${(err as Error).message.slice(0, 200)}\n\nSave settings anyway?`,
      );
      if (!ok) return false;
    }
    opts.onSettingsChanged();
    return true;
  };

  overlay.querySelector("[data-welcome-load-demo]")!.addEventListener("click", async () => {
    if (!(await persistSettings())) return;
    markWelcomeSeen();
    close(true);
    opts.onLoadDemo();
  });
  overlay.querySelector("[data-welcome-load-mine]")!.addEventListener("click", async () => {
    if (!(await persistSettings())) return;
    markWelcomeSeen();
    close(true);
    opts.onOpenLoader();
  });
}

function escapeAttr(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}
