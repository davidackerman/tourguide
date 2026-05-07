import type { LLMBackend } from "./llm.js";
import type { DatasetDB } from "./db.js";
import type { DatasetDescriptor } from "./descriptor.js";
import type { BundledViewer } from "./bundled_viewer.js";
import { runAgent, type AgentTraceItem } from "./agent.js";

export interface QueryUIContext {
  getDB: () => DatasetDB | null;
  getDescriptor: () => DatasetDescriptor | null;
  getBackend: () => LLMBackend;
  viewer: BundledViewer;
}

export function renderQueryBox(container: HTMLElement, ctx: QueryUIContext): void {
  container.innerHTML = "";
  const box = document.createElement("div");
  box.className = "query-box";
  box.innerHTML = `
    <form class="query-form">
      <input
        type="text"
        class="query-input"
        placeholder="Ask: 'largest mito' or 'plot mito volumes'"
        autocomplete="off"
      />
      <button class="btn-primary" type="submit" data-action="ask">Ask</button>
      <button class="btn-secondary" type="button" data-action="stop" hidden>Stop</button>
    </form>
    <div class="query-status" data-status></div>
    <div class="query-answer" data-answer></div>
    <div class="plot-output" data-plot hidden></div>
    <details class="query-details" data-details hidden>
      <summary>Show agent trace</summary>
      <div class="agent-trace-toolbar">
        <button class="btn-secondary btn-tiny" data-action="copy-trace" type="button">📋 Copy trace</button>
        <span class="agent-trace-copy-status" data-copy-status></span>
      </div>
      <div class="agent-trace" data-trace></div>
    </details>
  `;
  container.appendChild(box);

  const form = box.querySelector<HTMLFormElement>(".query-form")!;
  const input = box.querySelector<HTMLInputElement>(".query-input")!;
  const button = box.querySelector<HTMLButtonElement>("button[type=submit]")!;
  const stopBtn = box.querySelector<HTMLButtonElement>("[data-action='stop']")!;
  // Active AbortController for the in-flight query, or null when idle.
  // Exposed at module scope (closure-captured) so the Stop click handler
  // and the form-submit handler can share it without prop-drilling.
  let currentAbortController: AbortController | null = null;
  stopBtn.addEventListener("click", () => {
    if (currentAbortController) {
      currentAbortController.abort();
      // Visual nudge — the actual UI state flip happens in the catch /
      // finally of the in-flight runAgent call.
      stopBtn.disabled = true;
      stopBtn.textContent = "Stopping…";
      setTimeout(() => {
        stopBtn.disabled = false;
        stopBtn.textContent = "Stop";
      }, 1500);
    }
  });
  const statusEl = box.querySelector<HTMLDivElement>("[data-status]")!;
  const answerEl = box.querySelector<HTMLDivElement>("[data-answer]")!;
  const plotEl = box.querySelector<HTMLDivElement>("[data-plot]")!;
  const detailsEl = box.querySelector<HTMLDetailsElement>("[data-details]")!;
  const traceEl = box.querySelector<HTMLDivElement>("[data-trace]")!;
  const copyBtn = box.querySelector<HTMLButtonElement>("[data-action='copy-trace']")!;
  const copyStatus = box.querySelector<HTMLSpanElement>("[data-copy-status]")!;

  // Keep the structured trace in memory so the copy button can dump it
  // as plain text — DOM-walking the rendered HTML is fragile and loses
  // collapsed sections.
  const traceItems: AgentTraceItem[] = [];

  const formatTraceForCopy = (): string => {
    const lines: string[] = [];
    if (statusEl.textContent) lines.push(`# Status\n${statusEl.textContent}\n`);
    if (answerEl.textContent) lines.push(`# Answer\n${answerEl.textContent}\n`);
    lines.push(`# Trace (${traceItems.length} steps)`);
    traceItems.forEach((item, i) => {
      lines.push(`\n## Step ${i + 1}: ${item.tool}`);
      if (item.args && Object.keys(item.args).length > 0) {
        lines.push("args:\n```json\n" + JSON.stringify(item.args, null, 2) + "\n```");
      }
      if (item.error) {
        lines.push("error:\n```\n" + item.error + "\n```");
      } else if (item.result !== undefined) {
        const r = typeof item.result === "string" ? item.result : JSON.stringify(item.result, null, 2);
        lines.push("result:\n```\n" + r + "\n```");
      }
    });
    return lines.join("\n");
  };

  copyBtn.addEventListener("click", async () => {
    const text = formatTraceForCopy();
    try {
      await navigator.clipboard.writeText(text);
      copyStatus.textContent = "✓ copied";
      copyStatus.className = "agent-trace-copy-status ok";
    } catch {
      // Clipboard API can be blocked in non-secure contexts; fall back
      // to a window prompt() which lets the user copy manually.
      window.prompt("Copy trace:", text);
      copyStatus.textContent = "";
    }
    setTimeout(() => (copyStatus.textContent = ""), 1500);
  });

  const setStatus = (msg: string, kind: "" | "err" | "ok" | "pending" = ""): void => {
    statusEl.textContent = msg;
    statusEl.className = `query-status ${kind}`;
  };

  const appendTrace = (item: AgentTraceItem): void => {
    traceItems.push(item);
    detailsEl.hidden = false;
    const row = document.createElement("div");
    row.className = "agent-trace-item";
    const argStr = Object.keys(item.args ?? {}).length ? JSON.stringify(item.args, null, 2) : "";
    const safeArgs = argStr
      ? `<details class="agent-trace-block"><summary>args</summary><pre class="agent-trace-args">${escapeHtml(argStr)}</pre></details>`
      : "";
    let resultLine = "";
    if (item.error) {
      resultLine = `<details class="agent-trace-block" open><summary>error</summary><pre class="agent-trace-result agent-trace-error">${escapeHtml(item.error)}</pre></details>`;
    } else if (item.result !== undefined) {
      const r = typeof item.result === "string" ? item.result : JSON.stringify(item.result, null, 2);
      resultLine = `<details class="agent-trace-block"><summary>result</summary><pre class="agent-trace-result">${escapeHtml(r)}</pre></details>`;
    }
    row.innerHTML = `<span class="agent-trace-tool">${escapeHtml(item.tool)}</span>${safeArgs}${resultLine}`;
    traceEl.appendChild(row);
    traceEl.scrollTop = traceEl.scrollHeight;
  };

  const renderPlot = (
    pngDataUrl: string,
    code: string,
    title?: string,
    explanation?: string,
  ): void => {
    plotEl.hidden = false;
    plotEl.innerHTML = "";
    if (title) {
      const h = document.createElement("h3");
      h.className = "plot-title";
      h.textContent = title;
      plotEl.appendChild(h);
    }
    const img = document.createElement("img");
    img.src = pngDataUrl;
    img.className = "plot-image";
    plotEl.appendChild(img);
    if (explanation) {
      const p = document.createElement("p");
      p.className = "plot-explanation";
      p.textContent = explanation;
      plotEl.appendChild(p);
    }
    void code;
  };

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    const db = ctx.getDB();
    if (!db) {
      setStatus("Load a dataset with organelle CSVs first.", "err");
      return;
    }
    const backend = ctx.getBackend();
    if (!backend.isReady()) {
      setStatus("No AI configured. Click Settings to add a key or enable WebLLM.", "err");
      return;
    }
    button.hidden = true;
    stopBtn.hidden = false;
    setStatus("Thinking…", "pending");
    answerEl.textContent = "";
    plotEl.hidden = true;
    traceEl.innerHTML = "";
    traceItems.length = 0;
    detailsEl.hidden = true;
    let answeredOrFlew = false;
    // Fresh AbortController per query — Stop button calls abort()
    // which cascades through to the LLM backend (fetch / WebLLM
    // interrupt) and unwinds the agent loop's awaits.
    const abortController = new AbortController();
    currentAbortController = abortController;
    try {
      await runAgent(question, {
        db,
        descriptor: ctx.getDescriptor(),
        viewer: ctx.viewer,
        backend,
        signal: abortController.signal,
        callbacks: {
          onTrace: (t) => appendTrace(t),
          onProgress: (m) => setStatus(m, "pending"),
          onAnswer: (text) => {
            answerEl.textContent = text;
            answeredOrFlew = true;
          },
          onPlot: (png, code, title, explanation) => {
            renderPlot(png, code, title, explanation);
            answeredOrFlew = true;
          },
          onFly: (_pos, layer, id) => {
            setStatus(`Flew to ${layer}${id ? ` ${id}` : ""}`, "ok");
            answeredOrFlew = true;
          },
          onHighlight: (layer, ids) => {
            setStatus(`Showing ${ids.length} segment${ids.length === 1 ? "" : "s"} in ${layer}`, "ok");
            answeredOrFlew = true;
          },
        },
      });
      if (!answeredOrFlew) {
        setStatus("Agent finished without delivering an answer.", "");
      }
    } catch (err) {
      // AbortError is the user clicking Stop — show a calmer message
      // than "Error: ..." since they did this on purpose.
      if (abortController.signal.aborted || (err as Error).name === "AbortError") {
        setStatus("Stopped.", "");
      } else {
        setStatus((err as Error).message, "err");
        console.error(err);
      }
    } finally {
      currentAbortController = null;
      button.hidden = false;
      stopBtn.hidden = true;
    }
  });
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

