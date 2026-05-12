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
  // Mark the system prompt as cacheable for the call's lifetime (~5 min
  // on Anthropic). Only honored by AnthropicBackend; other backends
  // ignore it. Worth setting for callers that resend the same large
  // system prompt several times in quick succession — namely the agent
  // loop, which pays full input price on a ~20k-token system block
  // every iteration.
  cacheSystem?: boolean;
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

// ---------------------------------------------------------------------------
// OpenAI-compatible backend
// ---------------------------------------------------------------------------
// Speaks the OpenAI chat-completions API at any baseUrl. Covers OpenAI
// itself, xAI Grok, OpenRouter, Together, Groq (the chip company),
// Fireworks, DeepInfra, Mistral, and any local server (Ollama / vLLM /
// LM Studio / llama.cpp) that exposes the same endpoint.
//
// What's deliberately not abstracted away:
//   - jsonMode → response_format: { type: "json_object" } (OpenAI's
//     own param; OpenRouter passes through; older / smaller local
//     servers might ignore it, in which case the agent's prompt
//     "respond with a single JSON object" still works.)
//   - signal wired to fetch + the wait-for-retry sleep.
//   - 5xx retry: same exponential backoff (1/2/4s) GeminiBackend uses.
export class OpenAICompatibleBackend implements LLMBackend {
  id = "openai_compatible";
  name = "OpenAI-compatible (cloud / local)";
  private baseUrl: string;
  private apiKey: string;
  private model: string;

  constructor(baseUrl: string, apiKey: string, model: string) {
    this.baseUrl = baseUrl.replace(/\/$/, ""); // tolerate trailing slash
    this.apiKey = apiKey;
    this.model = model;
  }

  isReady(): boolean {
    // Local endpoints (localhost / 127.0.0.1 / file scheme) commonly
    // skip auth — accept those without a key. Cloud endpoints need
    // a key.
    if (!this.baseUrl) return false;
    if (this.apiKey) return true;
    return isLocalUrl(this.baseUrl);
  }

  static requestCount = 0;

  async complete(messages: LLMMessage[], options: LLMCompleteOptions = {}): Promise<string> {
    const useStream = !!options.onToken;
    const url = `${this.baseUrl}/chat/completions`;
    // OpenAI uses the standard {role, content} message shape directly —
    // system messages stay as role:"system" (unlike Anthropic / Gemini).
    const body: Record<string, unknown> = {
      model: this.model,
      messages,
      temperature: options.temperature ?? 0.1,
      max_tokens: options.maxTokens ?? 1024,
      stream: useStream,
    };
    if (options.jsonMode) body.response_format = { type: "json_object" };

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;

    OpenAICompatibleBackend.requestCount += 1;
    let res = await this.doFetch(url, headers, body, options.signal);

    // Retry on 5xx (model overloaded / transient gateway errors). The
    // backoff is the same shape Gemini uses for 503; we deliberately
    // don't retry 4xx (bad request, auth, quota) — those won't get
    // better with another try.
    let retries5xx = 0;
    while (res.status >= 500 && res.status < 600 && retries5xx < 3) {
      const waitMs = 1000 * 2 ** retries5xx;
      retries5xx += 1;
      options.onToken?.("", `${this.providerLabel()} returned ${res.status}; retrying in ${waitMs / 1000}s (attempt ${retries5xx}/3)…`);
      await waitWithSignal(waitMs, options.signal);
      OpenAICompatibleBackend.requestCount += 1;
      res = await this.doFetch(url, headers, body, options.signal);
    }

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${this.providerLabel()} API error ${res.status}: ${text.slice(0, 400)}`);
    }

    if (!useStream) {
      const data = (await res.json()) as {
        choices?: Array<{ message?: { content?: string } }>;
      };
      const text = data.choices?.[0]?.message?.content ?? "";
      if (!text) throw new Error(`${this.providerLabel()} returned empty response`);
      return text;
    }

    // SSE stream parse — same {data: {...}\n\n} envelope as Gemini.
    // OpenAI emits `data: [DONE]` to mark the end of the stream;
    // ignore it.
    const reader = res.body?.getReader();
    if (!reader) throw new Error(`${this.providerLabel()} stream missing body`);
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
        if (!json || json === "[DONE]") continue;
        try {
          const chunk = JSON.parse(json) as {
            choices?: Array<{ delta?: { content?: string } }>;
          };
          const t = chunk.choices?.[0]?.delta?.content ?? "";
          if (t) {
            accumulated += t;
            options.onToken?.(t, accumulated);
          }
        } catch {
          // partial / malformed line — keep streaming
        }
      }
    }
    if (!accumulated) throw new Error(`${this.providerLabel()} returned empty stream`);
    return accumulated;
  }

  // Reads the host portion of the configured base URL so error
  // messages name the actual provider ("OpenRouter API error 401"
  // instead of "OpenAI-compatible API error 401" for an OpenRouter
  // request — clearer when the user has multiple providers configured).
  private providerLabel(): string {
    try {
      const host = new URL(this.baseUrl).host;
      if (host.includes("openrouter")) return "OpenRouter";
      if (host.includes("openai")) return "OpenAI";
      if (host.includes("x.ai")) return "xAI";
      if (host.includes("groq.com")) return "Groq";
      if (host.includes("together.ai")) return "Together";
      if (host.includes("fireworks")) return "Fireworks";
      if (host.includes("mistral")) return "Mistral";
      if (host.includes("localhost") || host.startsWith("127.")) return "Local LLM server";
      return host;
    } catch {
      return "OpenAI-compatible";
    }
  }

  private async doFetch(
    url: string,
    headers: Record<string, string>,
    body: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<Response> {
    return await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal,
    });
  }

  async validate(): Promise<void> {
    await this.complete(
      [{ role: "user", content: "Reply with only the word: ok" }],
      { maxTokens: 64 },
    );
  }
}

interface AnthropicUsage {
  input_tokens?: number;
  output_tokens?: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}

// Default model — Anthropic's IDs use hyphens, not dots
// ('claude-sonnet-4-6', not 'claude-sonnet-4.6'). Centralized so
// settings_ui / welcome_ui / the AnthropicBackend default all stay
// in sync. Users can pick a different model via the dropdown after
// hitting Refresh (lists what /v1/models returns for their key).
export const DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6";

// ---------------------------------------------------------------------------
// Anthropic Claude (native API)
// ---------------------------------------------------------------------------
// Hits api.anthropic.com/v1/messages directly. Different from
// OpenAI-compatible:
//   - system is a top-level field, NOT a role:"system" message
//   - x-api-key header (not Authorization: Bearer)
//   - anthropic-dangerous-direct-browser-access: true (Anthropic's
//     opt-in for client-side requests; without this they reject CORS)
//   - SSE envelope is content_block_delta events with .delta.text
//   - no response_format param — agent's prompt already says
//     "respond with a single JSON object" so it works fine
export class AnthropicBackend implements LLMBackend {
  id = "anthropic";
  name = "Anthropic Claude";
  private apiKey: string;
  private model: string;

  constructor(apiKey: string, model = DEFAULT_ANTHROPIC_MODEL) {
    this.apiKey = apiKey;
    this.model = model;
  }

  isReady(): boolean {
    return this.apiKey.length > 0;
  }

  static requestCount = 0;

  // Running totals across this page session. Cache stats let us verify
  // that the cache_control tag is actually taking effect; without this
  // it's hard to tell from outside whether the second iteration of an
  // agent loop hit the cached prefix or paid full input price.
  static totals = {
    inputTokens: 0,
    outputTokens: 0,
    cacheCreationTokens: 0,
    cacheReadTokens: 0,
  };

  static recordUsage(u: AnthropicUsage): void {
    if (typeof u.input_tokens === "number") AnthropicBackend.totals.inputTokens += u.input_tokens;
    if (typeof u.output_tokens === "number") AnthropicBackend.totals.outputTokens += u.output_tokens;
    if (typeof u.cache_creation_input_tokens === "number")
      AnthropicBackend.totals.cacheCreationTokens += u.cache_creation_input_tokens;
    if (typeof u.cache_read_input_tokens === "number")
      AnthropicBackend.totals.cacheReadTokens += u.cache_read_input_tokens;
    // Per-call log so the user can verify cache hits in DevTools
    // without waiting for the Anthropic Console (which lags). Streaming
    // calls fire this twice — once at message_start (has input + cache
    // counts) and once at message_delta (has output). Non-stream calls
    // fire it once with everything.
    console.log(
      `[anthropic] usage: in=${u.input_tokens ?? "-"} out=${u.output_tokens ?? "-"} cache_create=${u.cache_creation_input_tokens ?? 0} cache_read=${u.cache_read_input_tokens ?? 0}  · session totals: in=${AnthropicBackend.totals.inputTokens} out=${AnthropicBackend.totals.outputTokens} cache_create=${AnthropicBackend.totals.cacheCreationTokens} cache_read=${AnthropicBackend.totals.cacheReadTokens}`,
    );
  }

  async complete(messages: LLMMessage[], options: LLMCompleteOptions = {}): Promise<string> {
    const useStream = !!options.onToken;
    const url = `https://api.anthropic.com/v1/messages`;
    // Pull system messages OUT into a top-level field; flatten user+
    // assistant into the messages array. Anthropic's API rejects
    // role:"system" inside messages, so this transform is required.
    const systemParts: string[] = [];
    const apiMessages: Array<{ role: "user" | "assistant"; content: string }> = [];
    for (const m of messages) {
      if (m.role === "system") {
        systemParts.push(m.content);
      } else {
        apiMessages.push({ role: m.role, content: m.content });
      }
    }
    const body: Record<string, unknown> = {
      model: this.model,
      max_tokens: options.maxTokens ?? 1024,
      messages: apiMessages,
      stream: useStream,
      temperature: options.temperature ?? 0.1,
    };
    if (systemParts.length > 0) {
      // When cacheSystem is set, send the system prompt as a single
      // content block tagged cache_control:ephemeral. Anthropic caches
      // the prefix for ~5 minutes; subsequent calls with the same
      // system text hit the cache at $0.30/M instead of $3/M. The
      // minimum cacheable size is 1024 tokens (Sonnet/Opus) — smaller
      // prompts get the tag ignored, no error. Last-message block also
      // gets tagged so the user turn extends the cached prefix on
      // repeated calls within the same agent loop.
      if (options.cacheSystem) {
        body.system = [
          { type: "text", text: systemParts.join("\n\n"), cache_control: { type: "ephemeral" } },
        ];
      } else {
        body.system = systemParts.join("\n\n");
      }
    }

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "x-api-key": this.apiKey,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    };

    AnthropicBackend.requestCount += 1;
    let res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: options.signal,
    });

    let retries5xx = 0;
    while (res.status >= 500 && res.status < 600 && retries5xx < 3) {
      const waitMs = 1000 * 2 ** retries5xx;
      retries5xx += 1;
      options.onToken?.("", `Anthropic returned ${res.status}; retrying in ${waitMs / 1000}s (attempt ${retries5xx}/3)…`);
      await waitWithSignal(waitMs, options.signal);
      AnthropicBackend.requestCount += 1;
      res = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: options.signal,
      });
    }

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Anthropic API error ${res.status}: ${text.slice(0, 400)}`);
    }

    if (!useStream) {
      const data = (await res.json()) as {
        content?: Array<{ type?: string; text?: string }>;
        usage?: AnthropicUsage;
      };
      if (data.usage) AnthropicBackend.recordUsage(data.usage);
      const text = (data.content ?? [])
        .filter((c) => c.type === "text")
        .map((c) => c.text ?? "")
        .join("");
      if (!text) throw new Error("Anthropic returned empty response");
      return text;
    }

    // Anthropic SSE: events of various types arrive as
    //   event: content_block_delta
    //   data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}
    // We only care about content_block_delta events with a text_delta.
    const reader = res.body?.getReader();
    if (!reader) throw new Error("Anthropic stream missing body");
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
            type?: string;
            delta?: { type?: string; text?: string };
            message?: { usage?: AnthropicUsage };
            usage?: AnthropicUsage;
          };
          if (chunk.type === "content_block_delta" && chunk.delta?.type === "text_delta") {
            const t = chunk.delta.text ?? "";
            if (t) {
              accumulated += t;
              options.onToken?.(t, accumulated);
            }
          } else if (chunk.type === "message_delta" && chunk.usage) {
            // Anthropic's SSE includes the same input/cache stats in
            // BOTH message_start and message_delta, with output_tokens
            // updated to the final cumulative count in message_delta.
            // Recording both events double-counted everything in the
            // running totals, so we only record once per call, at
            // message_delta (which has the complete picture).
            AnthropicBackend.recordUsage(chunk.usage);
          }
        } catch {
          // partial / malformed line — keep streaming
        }
      }
    }
    if (!accumulated) throw new Error("Anthropic returned empty stream");
    return accumulated;
  }

  async validate(): Promise<void> {
    await this.complete(
      [{ role: "user", content: "Reply with only the word: ok" }],
      { maxTokens: 64 },
    );
  }
}

// Shared sleep that respects the agent's abort signal — used by
// retry/backoff paths in all backends. Without this, an abort during
// the wait would silently complete and fire the next attempt anyway.
function waitWithSignal(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("aborted", "AbortError"));
      return;
    }
    const t = setTimeout(resolve, ms);
    if (signal) {
      signal.addEventListener("abort", () => {
        clearTimeout(t);
        reject(new DOMException("aborted", "AbortError"));
      }, { once: true });
    }
  });
}

// True for URLs the user is running on their own machine — local
// servers commonly skip auth, so isReady() can return true even with
// an empty API key.
function isLocalUrl(url: string): boolean {
  try {
    const u = new URL(url);
    return (
      u.hostname === "localhost" ||
      u.hostname === "127.0.0.1" ||
      u.hostname === "[::1]" ||
      u.hostname.endsWith(".local")
    );
  } catch {
    return false;
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

export interface AnthropicModelInfo {
  id: string;
  displayName: string;
  createdAt?: string;
}

// Fetch the list of Anthropic models the given key can access. Hits
// /v1/models, which is paginated; we follow has_more until exhausted.
// Sorted by created_at desc so the newest model lands at the top of
// the dropdown.
export async function listAnthropicModels(apiKey: string): Promise<AnthropicModelInfo[]> {
  if (!apiKey) throw new Error("API key required to list models");
  const out: AnthropicModelInfo[] = [];
  let afterId: string | undefined;
  for (let i = 0; i < 10; i++) {
    const params = new URLSearchParams({ limit: "100" });
    if (afterId) params.set("after_id", afterId);
    const res = await fetch(`https://api.anthropic.com/v1/models?${params}`, {
      headers: {
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
        "anthropic-dangerous-direct-browser-access": "true",
      },
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`models.list failed (${res.status}): ${text.slice(0, 300)}`);
    }
    const data = (await res.json()) as {
      data?: Array<{ id?: string; display_name?: string; created_at?: string }>;
      has_more?: boolean;
      last_id?: string;
    };
    for (const m of data.data ?? []) {
      if (!m.id) continue;
      out.push({
        id: m.id,
        displayName: m.display_name ?? m.id,
        createdAt: m.created_at,
      });
    }
    if (!data.has_more || !data.last_id) break;
    afterId = data.last_id;
  }
  // Newest first by created_at (string ISO sorts correctly); tie-break by id.
  out.sort((a, b) => (b.createdAt ?? "").localeCompare(a.createdAt ?? "") || a.id.localeCompare(b.id));
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

// LLM provider selection. The four "openai*" values all use
// OpenAICompatibleBackend under the hood — they differ only in which
// base URL is locked in (presets or user-supplied). Splitting them at
// the type level lets the Settings UI radio-style remember "I'm using
// OpenRouter" vs "I'm using OpenAI" even though they share one key
// field, one model field, and one URL field.
export type LLMProvider =
  | "none"
  | "gemini"
  | "anthropic"
  | "openai"
  | "openrouter"
  | "xai"
  | "openai_compatible"
  | "webllm";

export interface Settings {
  backend: LLMProvider;
  geminiApiKey: string;
  geminiModel: string;
  // Native Anthropic Claude — direct to api.anthropic.com.
  anthropicApiKey: string;
  anthropicModel: string;
  // OpenAI-compatible. Shared across openai / openrouter / xai /
  // openai_compatible — only one of those is the live backend at a
  // time, picked via `backend`. The base URL is set by the preset
  // (OpenAI / OpenRouter / xAI) and editable when backend is
  // openai_compatible (the user supplies their own URL for local
  // Ollama / vLLM / LM Studio / etc).
  openaiApiKey: string;
  openaiModel: string;
  openaiBaseUrl: string;
  webllmModel: string;
  // Optional HF-Space analysis backend (see hf-space/app.py). Empty string
  // disables the remote path — everything still works via Pyodide.
  analysisBackendUrl: string;
}

// Provider → preset URL. Used by the Settings UI to auto-fill
// openaiBaseUrl when the user picks one of the locked-URL providers.
// `openai_compatible` is intentionally absent — that's the "Custom"
// option where the user types their own URL.
export const OPENAI_COMPATIBLE_PRESETS: Record<string, { url: string; placeholderModel: string; label: string }> = {
  openai: {
    url: "https://api.openai.com/v1",
    placeholderModel: "gpt-5",
    label: "OpenAI",
  },
  openrouter: {
    url: "https://openrouter.ai/api/v1",
    placeholderModel: "anthropic/claude-sonnet-4-6",
    label: "OpenRouter (one key for Claude/Gemini/Llama/...)",
  },
  xai: {
    url: "https://api.x.ai/v1",
    placeholderModel: "grok-4",
    label: "xAI Grok",
  },
};

// Shared tourguide analysis Space. Each user who wants isolated compute can
// duplicate this to their own HF account and paste the new URL into Settings
// — but the default just works with no setup.
export const DEFAULT_ANALYSIS_BACKEND = "https://ackermand-tourguide-analysis.hf.space";

const DEFAULT_SETTINGS: Settings = {
  // Gemini is the recommended path — works without setup beyond
  // pasting an API key. Selecting it as the default doesn't enable AI
  // on its own (backendFromSettings still returns NullBackend until
  // the key is set), but it pre-checks the dropdown so the Settings
  // dialog surfaces the key field directly.
  backend: "gemini",
  geminiApiKey: "",
  // 2026-05-07: 3.1-flash-lite-preview had the most generous free
  // tier (15 RPM, 500 RPD) — the 2.5 series got crushed to ~20 RPD.
  // Users can switch via Settings → Refresh once Google moves the
  // generous limits elsewhere.
  geminiModel: "gemini-3.1-flash-lite-preview",
  anthropicApiKey: "",
  anthropicModel: DEFAULT_ANTHROPIC_MODEL,
  openaiApiKey: "",
  openaiModel: "",
  openaiBaseUrl: "",
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
  if (s.backend === "anthropic" && s.anthropicApiKey) {
    return new AnthropicBackend(s.anthropicApiKey, s.anthropicModel);
  }
  // OpenAI-compatible group — preset providers (openai / openrouter /
  // xai) all use the URL from OPENAI_COMPATIBLE_PRESETS rather than
  // whatever's stored in settings, so a stale URL from a previous
  // selection can't leak across providers. The "openai_compatible"
  // catch-all uses the user-supplied URL directly.
  if (
    s.backend === "openai" ||
    s.backend === "openrouter" ||
    s.backend === "xai" ||
    s.backend === "openai_compatible"
  ) {
    const url =
      s.backend === "openai_compatible"
        ? s.openaiBaseUrl
        : (OPENAI_COMPATIBLE_PRESETS[s.backend]?.url ?? s.openaiBaseUrl);
    if (url && (s.openaiApiKey || isLocalUrl(url))) {
      return new OpenAICompatibleBackend(url, s.openaiApiKey, s.openaiModel);
    }
  }
  if (s.backend === "webllm") {
    return new WebLLMBackend(s.webllmModel);
  }
  return new NullBackend();
}
