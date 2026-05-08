import type { LLMBackend } from "./llm.js";
import type { DatasetDB } from "./db.js";
import type { DatasetDescriptor } from "./descriptor.js";
import type { BundledViewer } from "./bundled_viewer.js";
import { runAgent, type AgentTraceItem, type AgentTurnSummary } from "./agent.js";
import { loadPromptHistory, recordPrompt } from "./prompt_history.js";
import { setPendingSession } from "./python_session.js";

export interface QueryUIContext {
  getDB: () => DatasetDB | null;
  // Optional: when set, python_on_layers tables are ingested into the
  // SQL DB so the next agent turn can run_sql against them. Without
  // this the agent only sees the table summary in its trace.
  setDB?: (db: DatasetDB) => void;
  getDescriptor: () => DatasetDescriptor | null;
  getBackend: () => LLMBackend;
  viewer: BundledViewer;
}

export function renderQueryBox(container: HTMLElement, ctx: QueryUIContext): void {
  container.innerHTML = "";
  const box = document.createElement("div");
  box.className = "query-box";
  box.innerHTML = `
    <div class="query-ai-hint" data-ai-hint hidden>
      ⚠ AI not configured — plain-English questions, agent trace, and 🐍 Custom Python need an AI backend.
      <button class="btn-link" data-action="open-settings">Set up in Settings</button>
    </div>
    <div class="session-turns" data-session hidden>
      <div class="session-turns-header">
        <span class="session-turns-label">This session</span>
        <button class="btn-link session-clear" type="button" data-action="new-session">New session</button>
      </div>
      <ol class="session-turns-list" data-session-list></ol>
    </div>
    <form class="query-form">
      <input
        type="text"
        class="query-input"
        placeholder="Ask: 'largest mito' or 'plot mito volumes'"
        autocomplete="off"
        list="tg-agent-history"
      />
      <datalist id="tg-agent-history" data-history-list></datalist>
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
        <div class="download-menu">
          <button class="btn-secondary btn-tiny" data-action="download-toggle" type="button">⬇ Download…</button>
          <div class="download-popover" data-download-popover hidden>
            <label><input type="checkbox" data-dl-trace checked> Trace</label>
            <label><input type="checkbox" data-dl-plots checked> Plots</label>
            <label><input type="checkbox" data-dl-scripts checked> Python scripts</label>
            <label><input type="checkbox" data-dl-session checked> Session history</label>
            <button class="btn-primary btn-tiny" data-action="download-go" type="button">Download .md</button>
          </div>
        </div>
        <span class="agent-trace-copy-status" data-copy-status></span>
      </div>
      <div class="agent-trace-question" data-trace-question hidden></div>
      <div class="agent-trace" data-trace></div>
    </details>
  `;
  container.appendChild(box);

  const form = box.querySelector<HTMLFormElement>(".query-form")!;
  const input = box.querySelector<HTMLInputElement>(".query-input")!;
  const button = box.querySelector<HTMLButtonElement>("button[type=submit]")!;
  const stopBtn = box.querySelector<HTMLButtonElement>("[data-action='stop']")!;
  const aiHint = box.querySelector<HTMLDivElement>("[data-ai-hint]")!;
  const aiHintBtn = box.querySelector<HTMLButtonElement>("[data-action='open-settings']")!;
  // Reflect backend readiness in the persistent hint above the Ask
  // input. Show the hint and dim the Ask button when no backend is
  // ready; clear both once the user configures one. Polled because
  // backend reconfigures from the Settings dialog don't fire an event
  // here — keeps the hint truthful without a pubsub.
  const refreshAiHint = (): void => {
    const ready = ctx.getBackend().isReady();
    aiHint.hidden = ready;
    button.disabled = !ready;
    button.title = ready ? "" : "Set up an AI backend in Settings to use Ask.";
    input.placeholder = ready
      ? "Ask: 'largest mito' or 'plot mito volumes'"
      : "(set up AI in Settings to ask questions)";
  };
  refreshAiHint();
  setInterval(refreshAiHint, 1500);
  aiHintBtn.addEventListener("click", () => {
    document.getElementById("settings-btn")?.click();
  });
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
  // Session memory — last ~3 turns are passed back to the agent on each
  // follow-up so "now do the same for ER" / "redo with a smaller radius"
  // resolves. Cleared by the "New session" button.
  const SESSION_CONTEXT_TURNS = 3;
  const sessionTurns: AgentTurnSummary[] = [];
  const sessionEl = box.querySelector<HTMLDivElement>("[data-session]")!;
  const sessionListEl = box.querySelector<HTMLOListElement>("[data-session-list]")!;
  const newSessionBtn = box.querySelector<HTMLButtonElement>("[data-action='new-session']")!;
  const renderSessionTurns = (): void => {
    sessionEl.hidden = sessionTurns.length === 0;
    sessionListEl.innerHTML = sessionTurns
      .map(
        (t) =>
          `<li><div class="session-turn-q">${escapeHtml(t.question)}</div><div class="session-turn-a">${escapeHtml(t.summary)}</div></li>`,
      )
      .join("");
  };
  newSessionBtn.addEventListener("click", () => {
    sessionTurns.length = 0;
    renderSessionTurns();
  });

  const historyList = box.querySelector<HTMLDataListElement>("[data-history-list]")!;
  const refreshHistoryList = (): void => {
    const items = loadPromptHistory("agent");
    historyList.innerHTML = items
      .map((p) => `<option value="${p.replace(/"/g, "&quot;")}"></option>`)
      .join("");
  };
  refreshHistoryList();

  // Shell-style ArrowUp/Down recall over the agent's prompt history.
  // Preserves whatever the user was typing before they started recalling
  // ("draft") so ArrowDown past index 0 brings it back. Index -1 means
  // "showing draft"; 0..N-1 walks the persisted history (most-recent
  // first). We refuse to recall while a query is in flight — it would
  // visually overwrite the in-progress prompt mid-execution.
  let historyIndex = -1;
  let draft = "";
  input.addEventListener("keydown", (e) => {
    if (button.hidden) return; // running; don't intercept
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    const items = loadPromptHistory("agent");
    if (items.length === 0) return;
    e.preventDefault();
    if (historyIndex === -1) draft = input.value;
    if (e.key === "ArrowUp") {
      historyIndex = Math.min(items.length - 1, historyIndex + 1);
    } else {
      historyIndex = Math.max(-1, historyIndex - 1);
    }
    input.value = historyIndex === -1 ? draft : items[historyIndex];
    // Move caret to end so the user can keep typing immediately.
    input.setSelectionRange(input.value.length, input.value.length);
  });
  // Any direct edit cancels recall mode so the next ArrowUp starts fresh.
  input.addEventListener("input", () => {
    historyIndex = -1;
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
  // Captured at submit time so 'Copy trace' includes the user's prompt
  // alongside the tool calls. Cleared at the start of each new query.
  let currentQuestion = "";
  // Plots emitted this turn (captured for the Download bundle alongside
  // the trace). We accumulate within a turn so a single download can
  // bundle multiple plots (rare but possible if make_plot is followed
  // by another).
  const plotsThisTurn: { png: string; code: string; title?: string }[] = [];

  const formatTraceForCopy = (): string => {
    const lines: string[] = [];
    if (currentQuestion) lines.push(`# Query\n${currentQuestion}\n`);
    if (statusEl.textContent) lines.push(`# Status\n${statusEl.textContent}\n`);
    if (answerEl.textContent) lines.push(`# Answer\n${answerEl.textContent}\n`);
    lines.push(`# Trace (${traceItems.length} step${traceItems.length === 1 ? "" : "s"})`);
    traceItems.forEach((item, i) => {
      lines.push("\n" + formatStepForCopy(item, i));
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

  // Download popover — single button, all categories checked by default,
  // user can deselect. Generates one self-contained .md so the result
  // opens cleanly in any markdown viewer (plots embed as data URLs).
  const dlToggle = box.querySelector<HTMLButtonElement>("[data-action='download-toggle']")!;
  const dlPopover = box.querySelector<HTMLDivElement>("[data-download-popover]")!;
  const dlGoBtn = box.querySelector<HTMLButtonElement>("[data-action='download-go']")!;
  const dlTrace = box.querySelector<HTMLInputElement>("[data-dl-trace]")!;
  const dlPlots = box.querySelector<HTMLInputElement>("[data-dl-plots]")!;
  const dlScripts = box.querySelector<HTMLInputElement>("[data-dl-scripts]")!;
  const dlSession = box.querySelector<HTMLInputElement>("[data-dl-session]")!;
  dlToggle.addEventListener("click", (e) => {
    e.stopPropagation();
    dlPopover.hidden = !dlPopover.hidden;
  });
  // Click-outside dismisses the popover. Listen on document so a click
  // anywhere not on the popover or the toggle closes it.
  document.addEventListener("click", (e) => {
    if (dlPopover.hidden) return;
    const t = e.target as Node;
    if (!dlPopover.contains(t) && t !== dlToggle) dlPopover.hidden = true;
  });
  dlGoBtn.addEventListener("click", () => {
    const md = buildSessionMarkdown({
      includeTrace: dlTrace.checked,
      includePlots: dlPlots.checked,
      includeScripts: dlScripts.checked,
      includeSession: dlSession.checked,
    });
    downloadBlob(new Blob([md], { type: "text/markdown" }), `tourguide-session-${Date.now()}.md`);
    dlPopover.hidden = true;
  });

  const buildSessionMarkdown = (opts: {
    includeTrace: boolean;
    includePlots: boolean;
    includeScripts: boolean;
    includeSession: boolean;
  }): string => {
    const out: string[] = [];
    out.push(`# Tourguide session\n`);
    out.push(`Generated ${new Date().toISOString()}\n`);
    if (currentQuestion) out.push(`## Question\n${currentQuestion}\n`);
    if (answerEl.textContent) out.push(`## Answer\n${answerEl.textContent}\n`);
    if (opts.includePlots && plotsThisTurn.length > 0) {
      out.push(`## Plots`);
      plotsThisTurn.forEach((p, i) => {
        const label = p.title ?? `Plot ${i + 1}`;
        out.push(`\n### ${label}\n`);
        out.push(`![${label}](${p.png})\n`);
      });
    }
    if (opts.includeScripts) {
      const scripts = traceItems
        .filter((t) => ["run_python", "make_plot", "python_on_layers"].includes(t.tool))
        .map((t) => ({ tool: t.tool, code: String((t.args as Record<string, unknown>)?.python ?? "") }))
        .filter((s) => s.code.length > 0);
      if (scripts.length > 0) {
        out.push(`\n## Python scripts`);
        scripts.forEach((s, i) => {
          out.push(`\n### Script ${i + 1} (${s.tool})\n`);
          out.push("```python\n" + s.code + "\n```\n");
        });
      }
    }
    if (opts.includeTrace && traceItems.length > 0) {
      out.push(`\n## Trace (${traceItems.length} steps)`);
      traceItems.forEach((it, i) => {
        out.push("\n" + formatStepForCopy(it, i));
      });
    }
    if (opts.includeSession && sessionTurns.length > 0) {
      out.push(`\n## Session history (most recent last)`);
      sessionTurns.forEach((t, i) => {
        out.push(`\n### Turn ${i + 1}\n- **Q:** ${t.question}\n- **A:** ${t.summary}`);
      });
    }
    return out.join("\n");
  };

  const setStatus = (msg: string, kind: "" | "err" | "ok" | "pending" = ""): void => {
    statusEl.textContent = msg;
    statusEl.className = `query-status ${kind}`;
  };

  // Format a single step the same way formatTraceForCopy does so the
  // per-step copy button output matches what "Copy trace" produces.
  const formatStepForCopy = (item: AgentTraceItem, index: number): string => {
    const lines: string[] = [`## Step ${index + 1}: ${item.tool}`];
    if (item.args && Object.keys(item.args).length > 0) {
      lines.push("args:\n```json\n" + JSON.stringify(item.args, null, 2) + "\n```");
    }
    if (item.error) {
      lines.push("error:\n```\n" + item.error + "\n```");
    } else if (item.result !== undefined) {
      const r = typeof item.result === "string" ? item.result : JSON.stringify(item.result, null, 2);
      lines.push("result:\n```\n" + r + "\n```");
    }
    return lines.join("\n");
  };

  const appendTrace = (item: AgentTraceItem): void => {
    const index = traceItems.length;
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
    // Tool name + per-step copy button. Click handler bound below
    // because innerHTML can't carry function refs.
    // For Python-emitting tools, also expose an "Open in Custom Python"
    // button that drops the code + layer choices into the dialog so
    // the user can edit and re-run without burning more LLM calls.
    const isPythonTool = ["run_python", "make_plot", "python_on_layers"].includes(item.tool);
    const openBtn = isPythonTool
      ? `<button class="btn-tiny agent-trace-open-step" type="button" title="Open this code in the Custom Python dialog">🐍 Edit</button>`
      : "";
    row.innerHTML = `
      <div class="agent-trace-item-header">
        <span class="agent-trace-tool">${escapeHtml(item.tool)}</span>
        <button class="btn-tiny agent-trace-copy-step" type="button" title="Copy this step">📋</button>
        ${openBtn}
        <span class="agent-trace-step-status" data-step-status></span>
      </div>
      ${safeArgs}${resultLine}`;
    const stepCopyBtn = row.querySelector<HTMLButtonElement>(".agent-trace-copy-step")!;
    const stepStatus = row.querySelector<HTMLSpanElement>("[data-step-status]")!;
    stepCopyBtn.addEventListener("click", async () => {
      const text = formatStepForCopy(item, index);
      try {
        await navigator.clipboard.writeText(text);
        stepStatus.textContent = "✓";
        stepStatus.className = "agent-trace-step-status ok";
      } catch {
        window.prompt("Copy step:", text);
      }
      setTimeout(() => (stepStatus.textContent = ""), 1500);
    });
    if (isPythonTool) {
      const openStepBtn = row.querySelector<HTMLButtonElement>(".agent-trace-open-step")!;
      openStepBtn.addEventListener("click", () => {
        const a = item.args as Record<string, unknown>;
        const code = String(a.python ?? a.code ?? "");
        const layers = Array.isArray(a.layers) ? (a.layers as unknown[]).map(String) : [];
        const skeletons = Array.isArray(a.skeletons)
          ? (a.skeletons as unknown[])
              .filter((s): s is Record<string, unknown> => !!s && typeof s === "object")
              .map((s) => ({
                layer: String(s.layer ?? s.name ?? ""),
                segmentIds: Array.isArray(s.segment_ids)
                  ? (s.segment_ids as unknown[]).map(String)
                  : Array.isArray(s.ids)
                    ? (s.ids as unknown[]).map(String)
                    : [],
              }))
              .filter((s) => s.layer)
          : [];
        setPendingSession({ layers, skeletons, code });
        // Click the Custom button instead of opening the dialog
        // directly — avoids importing the dialog module here and
        // keeps a single open-path the rest of the app already
        // exercises (e.g. focus / disabled-state checks).
        document.getElementById("custom-btn")?.click();
      });
    }
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
    img.title = "Click to enlarge";
    img.addEventListener("click", () => openFullscreenImage(pngDataUrl));
    plotEl.appendChild(img);
    if (explanation) {
      const p = document.createElement("p");
      p.className = "plot-explanation";
      p.textContent = explanation;
      plotEl.appendChild(p);
    }
    const toolbar = document.createElement("div");
    toolbar.className = "plot-toolbar";
    toolbar.appendChild(makeDownloadButton("⬇ PNG", () => downloadDataUrl(pngDataUrl, slugify(title) + ".png")));
    if (code) {
      toolbar.appendChild(
        makeDownloadButton("⬇ .py", () =>
          downloadBlob(new Blob([code], { type: "text/x-python" }), slugify(title) + ".py"),
        ),
      );
    }
    plotEl.appendChild(toolbar);
    if (code) {
      const det = document.createElement("details");
      det.className = "plot-code";
      det.innerHTML = `<summary>Show Python source</summary>`;
      const pre = document.createElement("pre");
      pre.textContent = code;
      det.appendChild(pre);
      plotEl.appendChild(det);
    }
  };

  // Fullscreen overlay for plot images. Click the image (or background)
  // or press Esc to close. We don't reuse .modal-overlay because that
  // pattern centers a modal card with a header — for a single image we
  // want edge-to-edge and a one-click dismiss anywhere on the dimmer.
  const openFullscreenImage = (src: string): void => {
    const overlay = document.createElement("div");
    overlay.className = "plot-fullscreen";
    overlay.innerHTML = `<button class="plot-fullscreen-close" aria-label="Close">×</button>`;
    const big = document.createElement("img");
    big.src = src;
    overlay.appendChild(big);
    const close = (): void => {
      overlay.remove();
      document.removeEventListener("keydown", onKey);
    };
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") close();
    };
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay || (e.target as HTMLElement).classList.contains("plot-fullscreen-close")) {
        close();
      }
    });
    document.addEventListener("keydown", onKey);
    document.body.appendChild(overlay);
  };

  const makeDownloadButton = (label: string, onClick: () => void): HTMLButtonElement => {
    const b = document.createElement("button");
    b.className = "btn-secondary btn-tiny";
    b.type = "button";
    b.textContent = label;
    b.addEventListener("click", onClick);
    return b;
  };

  const downloadBlob = (blob: Blob, filename: string): void => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  };

  const downloadDataUrl = (dataUrl: string, filename: string): void => {
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  const slugify = (s: string | undefined): string => {
    const t = (s ?? "plot").trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
    return t || "plot";
  };

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    currentQuestion = question;
    // Persist before running so a crash / refresh mid-query still leaves
    // the prompt recoverable in the dropdown.
    recordPrompt(question, "agent");
    refreshHistoryList();
    historyIndex = -1;
    draft = "";
    // The agent only NEEDS a descriptor (loaded layers). Organelle CSVs
    // are required for run_sql / make_plot / run_python (DataFrame
    // tools), but describe_dataset, python_on_layers, fly_to, and
    // highlight_segments work without a DB. Block only when neither
    // is loaded — and let the per-tool executors error helpfully when
    // the model picks a DB tool against an empty DB.
    const db = ctx.getDB();
    const descriptor = ctx.getDescriptor();
    if (!db && !descriptor) {
      setStatus("Load a dataset first.", "err");
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
    plotsThisTurn.length = 0;
    // Show the trace panel up front (collapsed by default) so even if
    // the agent fails or is stopped before any tool fires, the Copy
    // trace button is still reachable. Previously we only revealed it
    // on the first appendTrace, which hid the copy path on early
    // errors (model parse failures, 429s on first call, etc.).
    detailsEl.hidden = false;
    const traceQuestionEl = box.querySelector<HTMLDivElement>("[data-trace-question]");
    if (traceQuestionEl) {
      traceQuestionEl.hidden = false;
      traceQuestionEl.textContent = `Query: ${question}`;
    }
    let answeredOrFlew = false;
    // Capture a one-line outcome from this turn so the next turn's
    // priorTurns has something concrete to reference. We prefer the
    // textual answer; failing that, the highest-signal viewer change.
    let turnSummary = "";
    // Fresh AbortController per query — Stop button calls abort()
    // which cascades through to the LLM backend (fetch / WebLLM
    // interrupt) and unwinds the agent loop's awaits.
    const abortController = new AbortController();
    currentAbortController = abortController;
    try {
      await runAgent(question, {
        db,
        setDB: ctx.setDB,
        descriptor,
        viewer: ctx.viewer,
        backend,
        signal: abortController.signal,
        priorTurns: sessionTurns.slice(-SESSION_CONTEXT_TURNS),
        callbacks: {
          onTrace: (t) => appendTrace(t),
          onProgress: (m) => setStatus(m, "pending"),
          onAnswer: (text) => {
            answerEl.textContent = text;
            answeredOrFlew = true;
            turnSummary = text;
          },
          onPlot: (_png, _code, title) => {
            renderPlot(_png, _code, title);
            plotsThisTurn.push({ png: _png, code: _code, title });
            answeredOrFlew = true;
            if (!turnSummary) turnSummary = title ? `Showed plot: ${title}` : "Showed a plot";
          },
          onFly: (_pos, layer, id) => {
            setStatus(`Flew to ${layer}${id ? ` ${id}` : ""}`, "ok");
            answeredOrFlew = true;
            if (!turnSummary) turnSummary = `Flew to ${layer}${id ? ` ${id}` : ""}`;
          },
          onHighlight: (layer, ids) => {
            setStatus(`Showing ${ids.length} segment${ids.length === 1 ? "" : "s"} in ${layer}`, "ok");
            answeredOrFlew = true;
            if (!turnSummary) turnSummary = `Highlighted ${ids.length} segments in ${layer}`;
          },
        },
      });
      if (!answeredOrFlew) {
        setStatus("Agent finished without delivering an answer.", "");
      }
      sessionTurns.push({ question, summary: turnSummary || "(no visible result)" });
      renderSessionTurns();
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

