export interface LLMMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface LLMCompleteOptions {
  temperature?: number;
  maxTokens?: number;
  jsonMode?: boolean;
  onToken?: (text: string, accumulated: string) => void;
}

export interface LLMBackend {
  id: string;
  name: string;
  complete(messages: LLMMessage[], options?: LLMCompleteOptions): Promise<string>;
  isReady(): boolean;
}

export class NullBackend implements LLMBackend {
  id = "none";
  name = "No AI configured";
  async complete(): Promise<string> {
    throw new Error("No AI backend configured. Configure one in settings to use plain-English queries.");
  }
  isReady(): boolean {
    return false;
  }
}

export function hasWebGPU(): boolean {
  return typeof navigator !== "undefined" && !!(navigator as unknown as { gpu?: unknown }).gpu;
}

export interface WebGPUDiagnosis {
  available: boolean;
  reason: string;
  detail?: string;
  fixHint?: string;
  features?: string[];
  hasF16?: boolean;
}

export async function diagnoseWebGPU(): Promise<WebGPUDiagnosis> {
  if (typeof navigator === "undefined") {
    return { available: false, reason: "No navigator object (not in a browser)." };
  }
  const gpu = (navigator as unknown as {
    gpu?: { requestAdapter: (opts?: unknown) => Promise<unknown> };
  }).gpu;
  if (!gpu) {
    const ua = navigator.userAgent;
    const isChrome = /Chrome\//.test(ua) && !/Edg\//.test(ua);
    const isMac = /Mac OS X/.test(ua);
    const isLinux = /Linux/.test(ua) && !/Android/.test(ua);
    const nav = navigator as unknown as {
      userAgentData?: unknown;
      brave?: unknown;
    };
    const win = (typeof window !== "undefined" ? window : {}) as unknown as {
      process?: { versions?: { electron?: string } };
    };
    const isElectron = !!win.process?.versions?.electron || /Electron/i.test(ua);
    const looksLikeWebView = isChrome && !nav.userAgentData;
    let hint: string;
    if (isElectron || looksLikeWebView) {
      hint = "You appear to be viewing this in an embedded webview (e.g. VS Code's Simple Browser, an Electron app, or an in-app browser) — these report a Chrome user agent but don't expose navigator.gpu. Open the URL in a real Chrome / Edge / Safari 18+ window instead.";
    } else if (isChrome && isMac) {
      hint = "On Chrome/Mac, WebGPU is on by default in Chrome 113+. If navigator.gpu is missing, check: (a) chrome://settings/system → 'Use hardware acceleration when available' is ON; (b) chrome://gpu shows WebGPU as 'Hardware accelerated'; (c) your Chrome version is recent (chrome://version). Relaunch Chrome after any toggle.";
    } else if (isChrome && isLinux) {
      hint = "On Chrome/Linux, WebGPU often needs chrome://flags/#enable-unsafe-webgpu set to Enabled, then Chrome relaunched. Also check chrome://gpu.";
    } else if (isChrome) {
      hint = "Your Chrome doesn't expose navigator.gpu. Check chrome://gpu and chrome://flags/#enable-unsafe-webgpu.";
    } else {
      hint = "WebGPU isn't available. Use Chrome, Edge, or Safari 18+.";
    }
    return {
      available: false,
      reason: "navigator.gpu is undefined",
      detail: `User agent: ${ua.slice(0, 140)}`,
      fixHint: hint,
    };
  }
  try {
    const adapter = (await gpu.requestAdapter()) as unknown as {
      features?: { values: () => IterableIterator<string>; has?: (f: string) => boolean };
    } | null;
    if (!adapter) {
      return {
        available: false,
        reason: "navigator.gpu.requestAdapter() returned null",
        fixHint: "Your GPU/driver reported no compatible adapter. Check chrome://gpu — look for 'WebGPU: Hardware accelerated'.",
      };
    }
    const features: string[] = [];
    try {
      if (adapter.features) {
        for (const f of adapter.features.values()) features.push(f);
      }
    } catch {
      /* ignore */
    }
    const hasF16 = features.includes("shader-f16");
    return { available: true, reason: "ok", features, hasF16 };
  } catch (err) {
    return {
      available: false,
      reason: "requestAdapter() threw",
      detail: (err as Error).message,
    };
  }
}

export interface WebLLMProgress {
  text: string;
  progress?: number;
}

export class WebLLMBackend implements LLMBackend {
  id = "webllm";
  name: string;
  private modelId: string;
  private engine: unknown = null;
  private initPromise: Promise<void> | null = null;
  private progressCallback: (p: WebLLMProgress) => void;

  constructor(modelId: string, onProgress: (p: WebLLMProgress) => void = () => {}) {
    this.modelId = modelId;
    this.name = `WebLLM · ${modelId.split("-").slice(0, 3).join("-")}`;
    this.progressCallback = onProgress;
  }

  isReady(): boolean {
    // WebLLM can run whenever WebGPU is present; it lazy-loads the model on first complete().
    return hasWebGPU();
  }

  isLoaded(): boolean {
    return this.engine !== null;
  }

  setProgressCallback(fn: (p: WebLLMProgress) => void): void {
    this.progressCallback = fn;
  }

  async ensureInit(): Promise<void> {
    if (this.engine) return;
    if (!hasWebGPU()) throw new Error("WebGPU is not available in this browser. Try Chrome/Edge.");
    if (!this.initPromise) {
      this.initPromise = (async () => {
        const mod = await import("@mlc-ai/web-llm");
        const engine = await mod.CreateMLCEngine(this.modelId, {
          initProgressCallback: (p: { text: string; progress?: number }) => {
            this.progressCallback({ text: p.text, progress: p.progress });
          },
        });
        this.engine = engine;
      })();
      this.initPromise.catch(() => {
        this.initPromise = null;
      });
    }
    await this.initPromise;
  }

  async complete(messages: LLMMessage[], options: LLMCompleteOptions = {}): Promise<string> {
    await this.ensureInit();
    type Chunk = { choices: Array<{ delta?: { content?: string | null } }> };
    type NonStreamResponse = { choices: Array<{ message: { content: string | null } }> };
    type MLCEngineLike = {
      chat: {
        completions: {
          create: (req: unknown) => Promise<NonStreamResponse | AsyncIterable<Chunk>>;
        };
      };
    };
    const engine = this.engine as MLCEngineLike;
    const useStream = !!options.onToken;
    // Note: WebLLM's response_format requires a full JSON schema string for
    // json_object mode and throws BindingError otherwise. Skip it; the agent
    // prompts already instruct "respond with one JSON object" and our parser
    // tolerates fenced/wrapped JSON. JSON mode is honored on Gemini only.
    const result = await engine.chat.completions.create({
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      temperature: options.temperature ?? 0.1,
      max_tokens: options.maxTokens ?? 1024,
      stream: useStream,
    });
    if (!useStream) {
      const r = result as NonStreamResponse;
      return r.choices[0].message.content ?? "";
    }
    const stream = result as AsyncIterable<Chunk>;
    let accumulated = "";
    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta?.content ?? "";
      if (delta) {
        accumulated += delta;
        options.onToken?.(delta, accumulated);
      }
    }
    return accumulated;
  }
}

// Listed best-for-the-agent first (Qwen2.5-Coder 3B / 7B) since the
// agent loop is mostly SQL + Python tool calls, which the Coder line
// is finetuned for. The 1.5B Coder used to be the default but it's
// too small to recover from tool errors on its own. 3B is the
// realistic floor for reliable agent use; 7B is best if the user has
// the VRAM headroom.
export const WEBLLM_MODELS: Array<{ id: string; label: string }> = [
  { id: "Qwen2.5-Coder-3B-Instruct-q4f16_1-MLC", label: "Qwen2.5-Coder 3B · f16 (best for agent / SQL / Python, ~2 GB) — recommended" },
  { id: "Qwen2.5-Coder-3B-Instruct-q4f32_1-MLC", label: "Qwen2.5-Coder 3B · f32 (same, Intel Mac / older GPU, ~2.6 GB)" },
  { id: "Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC", label: "Qwen2.5-Coder 7B · f16 (strongest, needs ~6 GB VRAM)" },
  { id: "Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC", label: "Qwen2.5-Coder 1.5B · f16 (fastest, ~1 GB — may fail on multi-step agent tasks)" },
  { id: "Qwen2.5-Coder-1.5B-Instruct-q4f32_1-MLC", label: "Qwen2.5-Coder 1.5B · f32 (same, Intel Mac compatible, ~1.3 GB)" },
  { id: "Llama-3.2-3B-Instruct-q4f16_1-MLC", label: "Llama-3.2 3B · f16 (general, less code-tuned, ~2 GB)" },
  { id: "Llama-3.2-3B-Instruct-q4f32_1-MLC", label: "Llama-3.2 3B · f32 (same, Intel Mac compatible, ~2.6 GB)" },
  { id: "Qwen2.5-3B-Instruct-q4f16_1-MLC", label: "Qwen2.5 3B · f16 (general, ~2 GB)" },
  { id: "Llama-3.2-1B-Instruct-q4f32_1-MLC", label: "Llama-3.2 1B · f32 (smallest, widest compat, ~700 MB — for testing only)" },
];

export class GeminiBackend implements LLMBackend {
  id = "gemini";
  name = "Gemini (cloud)";
  private apiKey: string;
  private model: string;

  constructor(apiKey: string, model = "gemini-2.5-flash") {
    this.apiKey = apiKey;
    this.model = model;
  }

  isReady(): boolean {
    return this.apiKey.length > 0;
  }

  // Process-wide counter so the agent can show "this turn made N API
  // calls" — the actual variable that matters when a free-tier quota
  // bites. Static so any caller can read it.
  static requestCount = 0;
  // Most recently observed retryDelay from a 429 (seconds). UI can
  // surface it as "wait Ns before retrying".
  static lastRetryDelaySeconds: number | undefined;

  async complete(messages: LLMMessage[], options: LLMCompleteOptions = {}): Promise<string> {
    const useStream = !!options.onToken;
    const action = useStream ? "streamGenerateContent?alt=sse&key=" : "generateContent?key=";
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${this.model}:${action}${encodeURIComponent(this.apiKey)}`;
    const systemParts: string[] = [];
    const contents: Array<{ role: string; parts: Array<{ text: string }> }> = [];
    for (const m of messages) {
      if (m.role === "system") {
        systemParts.push(m.content);
      } else {
        contents.push({
          role: m.role === "assistant" ? "model" : "user",
          parts: [{ text: m.content }],
        });
      }
    }
    const body: Record<string, unknown> = {
      contents,
      generationConfig: {
        temperature: options.temperature ?? 0.1,
        maxOutputTokens: options.maxTokens ?? 1024,
        // Gemini 2.5 enables "thinking" by default, which silently burns
        // maxOutputTokens before any visible text is emitted. Disable it
        // so small-budget requests (e.g. the 10-token validate ping)
        // actually return a response.
        thinkingConfig: { thinkingBudget: 0 },
        ...(options.jsonMode ? { responseMimeType: "application/json" } : {}),
      },
    };
    if (systemParts.length > 0) {
      body.systemInstruction = { parts: [{ text: systemParts.join("\n\n") }] };
    }
    GeminiBackend.requestCount += 1;
    let res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.status === 429) {
      // Parse Gemini's retryInfo block to see how long to wait. Free-tier
      // RPM bumps usually give a small delay (~10–30s); we honor it once
      // and retry, then surface the original error if still capped.
      // Daily-quota 429s come back with retryDelay ≈ 0s — the wait won't
      // help, but we still surface the message clearly.
      const text = await res.text();
      const delaySec = parseRetryDelaySeconds(text);
      GeminiBackend.lastRetryDelaySeconds = delaySec;
      if (delaySec !== undefined && delaySec > 0 && delaySec <= 60) {
        options.onToken?.("", `Rate-limited; waiting ${delaySec}s and retrying…`);
        await new Promise((r) => setTimeout(r, delaySec * 1000 + 250));
        GeminiBackend.requestCount += 1;
        res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          const retryText = await res.text();
          throw new Error(formatQuotaError(res.status, retryText, delaySec));
        }
      } else {
        throw new Error(formatQuotaError(429, text, delaySec));
      }
    } else if (!res.ok) {
      const text = await res.text();
      throw new Error(`Gemini API error ${res.status}: ${text.slice(0, 400)}`);
    }
    if (!useStream) {
      const data = (await res.json()) as {
        candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
      };
      const text = data.candidates?.[0]?.content?.parts?.map((p) => p.text ?? "").join("") ?? "";
      if (!text) throw new Error("Gemini returned empty response");
      return text;
    }
    const reader = res.body?.getReader();
    if (!reader) throw new Error("Gemini stream missing body");
    const decoder = new TextDecoder();
    let buffer = "";
    let accumulated = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const json = line.slice(6).trim();
        if (!json) continue;
        try {
          const chunk = JSON.parse(json) as {
            candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
          };
          const t = chunk.candidates?.[0]?.content?.parts?.map((p) => p.text ?? "").join("") ?? "";
          if (t) {
            accumulated += t;
            options.onToken?.(t, accumulated);
          }
        } catch {
          /* ignore partial-JSON SSE noise */
        }
      }
    }
    if (!accumulated) throw new Error("Gemini returned empty stream");
    return accumulated;
  }

  async validate(): Promise<void> {
    await this.complete(
      [{ role: "user", content: "Reply with only the word: ok" }],
      { maxTokens: 64 },
    );
  }
}

// Pull retryDelay out of Gemini's structured 429 error body. Shape:
//   { "error": { "details": [{ "@type": ".../RetryInfo", "retryDelay": "30s" }] } }
// Returns undefined if the field isn't present.
function parseRetryDelaySeconds(body: string): number | undefined {
  try {
    const parsed = JSON.parse(body) as {
      error?: { details?: Array<{ "@type"?: string; retryDelay?: string }> };
    };
    const details = parsed.error?.details ?? [];
    for (const d of details) {
      if (d["@type"]?.includes("RetryInfo") && typeof d.retryDelay === "string") {
        const m = d.retryDelay.match(/^(\d+(?:\.\d+)?)s$/);
        if (m) return Number(m[1]);
      }
    }
  } catch {
    /* not JSON or not the expected shape */
  }
  return undefined;
}

// Format a 429 in a way that's actually actionable for the user, not
// just the raw API JSON. Pulls out which quota was hit (RPM vs RPD) by
// reading the QuotaFailure block, and tells them what to do next.
function formatQuotaError(status: number, body: string, delaySec?: number): string {
  let quotaId = "";
  try {
    const parsed = JSON.parse(body) as {
      error?: { details?: Array<{ "@type"?: string; violations?: Array<{ quotaId?: string }> }> };
    };
    const details = parsed.error?.details ?? [];
    for (const d of details) {
      if (d["@type"]?.includes("QuotaFailure")) {
        quotaId = d.violations?.[0]?.quotaId ?? "";
        break;
      }
    }
  } catch {
    /* fall through */
  }
  const isPerDay = /PerDay|RequestsPerDay|RPD/i.test(quotaId);
  const isPerMin = /PerMinute|RPM/i.test(quotaId);
  let advice: string;
  if (isPerDay) {
    advice = "Daily free-tier limit hit. Resets at midnight Pacific. Switch model in Settings, enable billing, or wait.";
  } else if (isPerMin) {
    const wait = delaySec !== undefined ? `~${delaySec}s` : "~60s";
    advice = `Per-minute limit hit. Wait ${wait} and try again, or switch model in Settings.`;
  } else {
    advice = "Free-tier quota exceeded. Switch model in Settings, enable billing, or wait.";
  }
  const quotaNote = quotaId ? ` [${quotaId}]` : "";
  return `Gemini ${status}${quotaNote}: ${advice}`;
}

const LS_KEY = "tourguide.settings.v1";

export interface Settings {
  backend: "none" | "gemini" | "webllm";
  geminiApiKey: string;
  geminiModel: string;
  webllmModel: string;
  // Optional HF-Space analysis backend (see hf-space/app.py). Empty string
  // disables the remote path — everything still works via Pyodide.
  analysisBackendUrl: string;
}

// Shared tourguide analysis Space. Each user who wants isolated compute can
// duplicate this to their own HF account and paste the new URL into Settings
// — but the default just works with no setup.
export const DEFAULT_ANALYSIS_BACKEND = "https://ackermand-tourguide-analysis.hf.space";

const DEFAULT_SETTINGS: Settings = {
  backend: "none",
  geminiApiKey: "",
  geminiModel: "gemini-2.5-flash-lite",
  webllmModel: WEBLLM_MODELS[0].id,
  analysisBackendUrl: DEFAULT_ANALYSIS_BACKEND,
};

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    const parsed = JSON.parse(raw) as Partial<Settings>;
    return { ...DEFAULT_SETTINGS, ...parsed };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(s: Settings): void {
  localStorage.setItem(LS_KEY, JSON.stringify(s));
}

export function backendFromSettings(s: Settings): LLMBackend {
  if (s.backend === "gemini" && s.geminiApiKey) {
    return new GeminiBackend(s.geminiApiKey, s.geminiModel);
  }
  if (s.backend === "webllm") {
    return new WebLLMBackend(s.webllmModel);
  }
  return new NullBackend();
}
