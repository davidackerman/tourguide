export interface LLMMessage {
  role: "system" | "user" | "assistant";
  content: string;
}


export interface LLMCompleteOptions {
  temperature?: number;
  maxTokens?: number;
  jsonMode?: boolean;
  onToken?: (text: string, accumulated: string) => void;
  // When this fires, the backend should abort its in-flight request and
  // throw. Gemini wires the signal directly to fetch(); WebLLM has an
  // interruptGenerate() method we call. The throw is what unwinds the
  // agent loop's `await` so it can stop instead of running another step.
  signal?: AbortSignal;
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
      // MLC's interrupt API — stops the in-progress generation. The
      // current await on chat.completions.create resolves with whatever
      // tokens have been produced so far, then we throw to unwind.
      interruptGenerate?: () => void;
    };
    const engine = this.engine as MLCEngineLike;
    // Hook the abort signal into MLC's interruptGenerate. WebLLM doesn't
    // accept an AbortSignal directly, so we listen and call its imperative
    // API when the signal fires.
    const onAbort = (): void => {
      try {
        engine.interruptGenerate?.();
      } catch {
        /* engine may be mid-teardown; nothing more we can do */
      }
    };
    if (options.signal) {
      if (options.signal.aborted) {
        throw new DOMException("Aborted before WebLLM call", "AbortError");
      }
      options.signal.addEventListener("abort", onAbort, { once: true });
    }
    const useStream = !!options.onToken;
    // Note: WebLLM's response_format requires a full JSON schema string for
    // json_object mode and throws BindingError otherwise. Skip it; the agent
    // prompts already instruct "respond with one JSON object" and our parser
    // tolerates fenced/wrapped JSON. JSON mode is honored on Gemini only.
    let result;
    try {
      result = await engine.chat.completions.create({
        messages: messages.map((m) => ({ role: m.role, content: m.content })),
        temperature: options.temperature ?? 0.1,
        max_tokens: options.maxTokens ?? 1024,
        stream: useStream,
      });
    } finally {
      options.signal?.removeEventListener("abort", onAbort);
    }
    // After the await: if the user aborted while we were generating,
    // throw rather than returning a half-baked response. The agent loop
    // catches AbortError and stops cleanly.
    if (options.signal?.aborted) {
      throw new DOMException("WebLLM generation aborted", "AbortError");
    }
    if (!useStream) {
      const r = result as NonStreamResponse;
      return r.choices[0].message.content ?? "";
    }
    const stream = result as AsyncIterable<Chunk>;
    let accumulated = "";
    for await (const chunk of stream) {
      if (options.signal?.aborted) {
        throw new DOMException("WebLLM stream aborted", "AbortError");
      }
      const delta = chunk.choices?.[0]?.delta?.content ?? "";
      if (delta) {
        accumulated += delta;
        options.onToken?.(delta, accumulated);
      }
    }
    return accumulated;
  }
}

// recommended: 1-5, agent-loop suitability (SQL + Python + structured-JSON
// tool format). Code-tuned 3B+ models score highest because the loop is
// dominated by code/SQL generation and JSON-mode output. Pure reasoning
// models (R1 distills) score lower — they're verbose and can blow past
// the JSON-only constraint. sizeGB is the approximate download / VRAM
// footprint for q4f16 quantization.
export interface WebLLMModelInfo {
  id: string;
  // Short human label (model family + size). Sort UI builds the full
  // option text from family + size + score, so this stays terse.
  family: string;
  size: string;
  sizeGB: number;
  // 5 = best for agent, 1 = barely usable for agent. Doesn't reflect
  // raw model strength — Qwen3-0.6B is a fine model, just not for a
  // multi-tool loop with structured output.
  recommended: 1 | 2 | 3 | 4 | 5;
  // Free-form context note appended to the option label.
  note?: string;
}

export const WEBLLM_MODELS: WebLLMModelInfo[] = [
  // Tier 5 — strongest agent performance.
  { id: "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC", family: "Hermes-2-Pro Llama-3", size: "8B · f16", sizeGB: 4.5, recommended: 5, note: "Nous fine-tune trained for function-calling / structured tool use — strongest WebLLM agent" },
  { id: "Hermes-3-Llama-3.1-8B-q4f16_1-MLC", family: "Hermes-3 Llama-3.1", size: "8B · f16", sizeGB: 4.5, recommended: 5, note: "newer Hermes, also tool-tuned" },
  { id: "Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC", family: "Qwen2.5-Coder", size: "7B · f16", sizeGB: 4.0, recommended: 5, note: "code-tuned, needs ~6 GB VRAM" },
  // Tier 4 — strong, more accessible.
  { id: "Qwen2.5-Coder-3B-Instruct-q4f16_1-MLC", family: "Qwen2.5-Coder", size: "3B · f16", sizeGB: 2.0, recommended: 4, note: "recommended sweet spot for agent" },
  { id: "Qwen2.5-Coder-3B-Instruct-q4f32_1-MLC", family: "Qwen2.5-Coder", size: "3B · f32", sizeGB: 2.6, recommended: 4, note: "Intel Mac / older GPU compatible" },
  { id: "Hermes-3-Llama-3.2-3B-q4f16_1-MLC", family: "Hermes-3 Llama-3.2", size: "3B · f16", sizeGB: 2.0, recommended: 4, note: "tool-tuned, accessible size" },
  { id: "Qwen3-8B-q4f16_1-MLC", family: "Qwen3", size: "8B · f16", sizeGB: 4.5, recommended: 4, note: "newer general, very capable" },
  { id: "Llama-3.1-8B-Instruct-q4f16_1-MLC", family: "Llama-3.1", size: "8B · f16", sizeGB: 4.5, recommended: 4, note: "general, solid all-rounder" },
  { id: "gemma-2-9b-it-q4f16_1-MLC", family: "Gemma-2", size: "9B · f16", sizeGB: 5.0, recommended: 4, note: "Google open model, large" },
  // Tier 3 — capable but smaller / less agent-tuned.
  { id: "Qwen3-4B-q4f16_1-MLC", family: "Qwen3", size: "4B · f16", sizeGB: 2.5, recommended: 3, note: "newer mid-size general" },
  { id: "Qwen2.5-7B-Instruct-q4f16_1-MLC", family: "Qwen2.5", size: "7B · f16", sizeGB: 4.0, recommended: 3, note: "general (non-coder)" },
  { id: "Llama-3-8B-Instruct-q4f16_1-MLC", family: "Llama-3", size: "8B · f16", sizeGB: 4.5, recommended: 3, note: "older general" },
  // Tier 2 — small / marginal for multi-step agent.
  { id: "Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC", family: "Qwen2.5-Coder", size: "1.5B · f16", sizeGB: 1.0, recommended: 2, note: "fast, code-tuned, may fail on multi-step" },
  { id: "Qwen2.5-Coder-1.5B-Instruct-q4f32_1-MLC", family: "Qwen2.5-Coder", size: "1.5B · f32", sizeGB: 1.3, recommended: 2, note: "Intel Mac compatible" },
  { id: "Qwen3-1.7B-q4f16_1-MLC", family: "Qwen3", size: "1.7B · f16", sizeGB: 1.1, recommended: 2, note: "newer, but small for agent" },
  { id: "Llama-3.2-3B-Instruct-q4f16_1-MLC", family: "Llama-3.2", size: "3B · f16", sizeGB: 2.0, recommended: 2, note: "general" },
  { id: "Llama-3.2-3B-Instruct-q4f32_1-MLC", family: "Llama-3.2", size: "3B · f32", sizeGB: 2.6, recommended: 2, note: "Intel Mac compatible" },
  { id: "Qwen2.5-3B-Instruct-q4f16_1-MLC", family: "Qwen2.5", size: "3B · f16", sizeGB: 2.0, recommended: 2, note: "general (non-coder)" },
  { id: "Phi-3.5-mini-instruct-q4f16_1-MLC", family: "Phi-3.5-mini", size: "3.8B · f16", sizeGB: 2.4, recommended: 2, note: "Microsoft small model" },
  { id: "gemma-2-2b-it-q4f16_1-MLC", family: "Gemma-2", size: "2B · f16", sizeGB: 1.5, recommended: 2, note: "Google small open model" },
  // Tier 1 — too small for reliable agent use; kept for low-spec / testing.
  { id: "Qwen3-0.6B-q4f16_1-MLC", family: "Qwen3", size: "0.6B · f16", sizeGB: 0.5, recommended: 1, note: "tiny — for testing only" },
  { id: "Llama-3.2-1B-Instruct-q4f32_1-MLC", family: "Llama-3.2", size: "1B · f32", sizeGB: 0.7, recommended: 1, note: "smallest, widest compat" },
];

// Build the user-facing label. Centralized so the settings dropdown
// doesn't have to know about the field shape. Order in the dropdown
// already reflects agent suitability (recommended desc) — no need to
// repeat it visually.
export function webllmModelLabel(m: WebLLMModelInfo): string {
  const sizeStr = `~${m.sizeGB.toFixed(1)} GB`;
  const note = m.note ? ` — ${m.note}` : "";
  return `${m.family} ${m.size} (${sizeStr})${note}`;
}

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
      signal: options.signal,
    });
    // 503 UNAVAILABLE = "model is overloaded; try again later". Gemini
    // doesn't include a Retry-After, so use a short exponential backoff
    // (1s, 2s, 4s) before surfacing the error. Aborts via the signal
    // jump out of the wait.
    let retries503 = 0;
    while (res.status === 503 && retries503 < 3) {
      const waitMs = 1000 * 2 ** retries503;
      retries503 += 1;
      options.onToken?.("", `Gemini overloaded (503); retrying in ${waitMs / 1000}s (attempt ${retries503}/3)…`);
      await new Promise<void>((resolve, reject) => {
        const t = setTimeout(resolve, waitMs);
        if (options.signal) {
          options.signal.addEventListener("abort", () => {
            clearTimeout(t);
            reject(new DOMException("aborted", "AbortError"));
          }, { once: true });
        }
      });
      GeminiBackend.requestCount += 1;
      res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: options.signal,
      });
    }
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
          signal: options.signal,
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
      // For 503 specifically (model overloaded after our retries),
      // a short friendly message beats dumping the JSON envelope.
      if (res.status === 503) {
        throw new Error(
          `Gemini is overloaded right now (503 UNAVAILABLE). Tried ${retries503} retries with backoff. Try again in a moment, or switch model in Settings.`,
        );
      }
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

export interface GeminiModelInfo {
  // Model ID for the API path: 'models/<id>'. Strip the 'models/' prefix
  // before passing to GeminiBackend's constructor.
  id: string;
  displayName: string;
  description?: string;
  // True if this model supports the generateContent action we use.
  supportsGenerateContent: boolean;
}

// Fetch the list of Gemini models the given key can access. Filters
// out anything that doesn't support generateContent (embeddings,
// retired, image-only) so the dropdown doesn't show models the agent
// can't actually use. Sorted with newest / lite-tier first.
export async function listGeminiModels(apiKey: string): Promise<GeminiModelInfo[]> {
  if (!apiKey) throw new Error("API key required to list models");
  const url = `https://generativelanguage.googleapis.com/v1beta/models?pageSize=200&key=${encodeURIComponent(apiKey)}`;
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`models.list failed (${res.status}): ${text.slice(0, 300)}`);
  }
  const data = (await res.json()) as {
    models?: Array<{
      name?: string;
      displayName?: string;
      description?: string;
      supportedGenerationMethods?: string[];
    }>;
  };
  const models = data.models ?? [];
  const out: GeminiModelInfo[] = [];
  for (const m of models) {
    const fullName = m.name ?? "";
    if (!fullName.startsWith("models/")) continue;
    const id = fullName.slice("models/".length);
    // Skip non-generateContent (embeddings, gemini-pro-vision-only, etc.).
    const methods = m.supportedGenerationMethods ?? [];
    const supportsGenerateContent = methods.includes("generateContent");
    if (!supportsGenerateContent) continue;
    // Skip Gemini 1.x — long retired, just clutter.
    if (/^gemini-1\./i.test(id) || /^chat-bison/i.test(id) || /^text-bison/i.test(id)) continue;
    out.push({
      id,
      displayName: m.displayName ?? id,
      description: m.description,
      supportsGenerateContent: true,
    });
  }
  // Sort: newest major version first, then 'flash-lite' before 'flash'
  // before 'pro' (cheapest tier first within a generation).
  const tierRank = (id: string): number => {
    if (/flash-lite/i.test(id)) return 0;
    if (/flash/i.test(id)) return 1;
    if (/pro/i.test(id)) return 2;
    return 3;
  };
  const verRank = (id: string): number => {
    const m = id.match(/gemini-(\d+(?:\.\d+)?)/i);
    return m ? -parseFloat(m[1]) : 0; // descending
  };
  out.sort((a, b) => verRank(a.id) - verRank(b.id) || tierRank(a.id) - tierRank(b.id) || a.id.localeCompare(b.id));
  return out;
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
  // Gemini is the recommended path — works without setup beyond
  // pasting an API key. Selecting it as the default doesn't enable AI
  // on its own (backendFromSettings still returns NullBackend until
  // the key is set), but it pre-checks the radio so the Settings
  // dialog surfaces the key field directly instead of forcing a click
  // through "None".
  backend: "gemini",
  geminiApiKey: "",
  // 2026-05-07: 3.1-flash-lite-preview had the most generous free
  // tier (15 RPM, 500 RPD) — the 2.5 series got crushed to ~20 RPD.
  // Users can switch via Settings → Refresh once Google moves the
  // generous limits elsewhere.
  geminiModel: "gemini-3.1-flash-lite-preview",
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
