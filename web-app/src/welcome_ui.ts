// First-run welcome modal — explains the two optional backends
// (AI for plain-English queries, Analysis for heavy compute on zarr)
// and walks the user through configuring them. Skippable; remembers
// dismissal in localStorage so it never reappears unless the user
// asks to see it again from Settings.

import {
  loadSettings,
  saveSettings,
  GeminiBackend,
  WEBLLM_MODELS,
  webllmModelLabel,
  hasWebGPU,
  DEFAULT_ANALYSIS_BACKEND,
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
            Powers the <strong>Ask</strong> input (plain-English questions
            over your data) and <strong>🐍 Custom</strong> (LLM-generated Python
            on your layers). Without this, the structured browser, Σ Analyze,
            and the viewer all still work — you just can't ask questions in
            English.
          </p>
          <div class="welcome-radio-group">
            <label class="welcome-radio">
              <input type="radio" name="welcome-ai" value="none" ${initialAi === "none" ? "checked" : ""} />
              <div>
                <strong>None</strong>
                <p class="hint">Skip AI. Configure later from Settings.</p>
              </div>
            </label>
            <label class="welcome-radio">
              <input type="radio" name="welcome-ai" value="gemini" ${initialAi === "gemini" ? "checked" : ""} />
              <div>
                <strong>Gemini (cloud)</strong>
                <p class="hint">
                  Free API key from
                  <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener">aistudio.google.com</a>.
                  Best quality; rate-limited (~500 req/day on Flash Lite).
                  Stored in browser localStorage only.
                </p>
                <input type="password" data-welcome-gemini-key
                       value="${escapeAttr(settings.geminiApiKey)}"
                       placeholder="(paste key)" autocomplete="off" />
              </div>
            </label>
            <label class="welcome-radio ${webgpu ? "" : "welcome-disabled"}">
              <input type="radio" name="welcome-ai" value="webllm" ${initialAi === "webllm" ? "checked" : ""} ${webgpu ? "" : "disabled"} />
              <div>
                <strong>WebLLM (local, in-browser)</strong>
                <p class="hint">
                  Runs on your GPU via WebGPU. ~2-4 GB one-time download per model;
                  fully offline after that, no quotas. Slower per token.
                  ${webgpu ? "" : "<em>(needs Chrome / Edge / Safari 18+)</em>"}
                </p>
                <select data-welcome-webllm-model ${webgpu ? "" : "disabled"}>
                  ${webllmAll
                    .map(
                      (m) =>
                        `<option value="${m.id}" ${settings.webllmModel === m.id ? "selected" : ""}>${webllmModelLabel(m)}</option>`,
                    )
                    .join("")}
                </select>
              </div>
            </label>
          </div>
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
        <button class="btn-secondary" data-welcome-skip>Skip — I'll set up later</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);

  const close = (saved: boolean): void => {
    if (saved) markWelcomeSeen();
    overlay.remove();
  };
  overlay.querySelector(".modal-close")!.addEventListener("click", () => {
    markWelcomeSeen();
    close(true);
  });
  overlay.querySelector("[data-welcome-skip]")!.addEventListener("click", () => {
    markWelcomeSeen();
    close(true);
  });
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) {
      markWelcomeSeen();
      close(true);
    }
  });

  // Save settings (extracted so both 'Load demo' and 'Load my data'
  // commit the user's AI / analysis-backend choices before navigating).
  // Returns false if the user cancelled an invalid-Gemini-key warning.
  const persistSettings = async (): Promise<boolean> => {
    const aiChoice = (overlay.querySelector<HTMLInputElement>('input[name="welcome-ai"]:checked')?.value as "none" | "gemini" | "webllm") ?? "none";
    const geminiKey = overlay.querySelector<HTMLInputElement>("[data-welcome-gemini-key]")?.value.trim() ?? settings.geminiApiKey;
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
      webllmModel,
      analysisBackendUrl: analysisUrl,
    };
    saveSettings(next);
    if (aiChoice === "gemini" && geminiKey) {
      try {
        const backend = new GeminiBackend(geminiKey, next.geminiModel);
        await backend.validate();
      } catch (err) {
        const ok = confirm(
          `Gemini API key didn't validate: ${(err as Error).message.slice(0, 200)}\n\nSave settings anyway?`,
        );
        if (!ok) return false;
      }
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
