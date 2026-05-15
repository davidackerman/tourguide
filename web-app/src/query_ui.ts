import type { LLMBackend } from "./llm.js";
import type { DatasetDB } from "./db.js";
import type { DatasetDescriptor } from "./descriptor.js";
import type { BundledViewer } from "./bundled_viewer.js";
import { runAgent, type AgentTraceItem, type AgentTurnSummary, type AskField } from "./agent.js";
import { loadPromptHistory, recordPrompt } from "./prompt_history.js";
import { setPendingSession, peekPendingSession } from "./python_session.js";
import type { SerializedTurn } from "./permalink.js";

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

// Returned from renderQueryBox so the caller (main.ts) can snapshot
// the session for share-link embedding and restore one from a URL.
export interface QueryUIHandle {
  /** Snapshot every visible turn as SerializedTurn data — what the
   *  share-link encoder ships to the recipient. Empty when no turns
   *  have run yet. */
  getSerializedSession: () => SerializedTurn[];
  /** Render saved turns into the query thread as if they had just
   *  run. Used on share-link load. Cards visually match live ones
   *  but no agent is invoked. */
  replaySerializedSession: (turns: SerializedTurn[]) => void;
}

// Per-turn UI state. Each submit creates a new card; subsequent
// callbacks (onTrace / onPlot / onAnswer / onTable / etc.) render INTO
// this card's DOM. Old cards persist in the thread, so a user can
// scroll back through previous Q's with their plots, code, and
// downloads still functional. Page reload still wipes the thread —
// chat-style persistence across reloads would mean serializing plot
// data URLs into localStorage and is out of scope here.
interface TurnCard {
  index: number;
  question: string;
  root: HTMLElement;
  metaEl: HTMLElement;
  statusEl: HTMLElement;
  answerEl: HTMLElement;
  plotsEl: HTMLElement;
  tablesEl: HTMLElement;
  traceEl: HTMLElement;
  detailsEl: HTMLDetailsElement;
  traceItems: AgentTraceItem[];
  plots: { png: string; code: string; title?: string; explanation?: string }[];
  tables: { name: string; columns: string[]; rows: (number | string | null)[][] }[];
  metaLines: string[];
}

const TRACE_OPEN_KEY = "tourguide.agentTraceOpen";
const loadTraceOpenPref = (): boolean => {
  // Default to closed — the agent's status line + meta strip are
  // enough to follow what's happening; the full trace is for when
  // something looks off. Toggleable via the session toolbar.
  try {
    const v = localStorage.getItem(TRACE_OPEN_KEY);
    if (v === null) return false;
    return v === "1";
  } catch {
    return false;
  }
};
const saveTraceOpenPref = (v: boolean): void => {
  try {
    localStorage.setItem(TRACE_OPEN_KEY, v ? "1" : "0");
  } catch {
    /* private mode / quota — silently drop */
  }
};

export function renderQueryBox(container: HTMLElement, ctx: QueryUIContext): QueryUIHandle {
  container.innerHTML = "";
  const box = document.createElement("div");
  box.className = "query-box";
  box.innerHTML = `
    <div class="query-ai-hint" data-ai-hint hidden>
      ⚠ AI not configured — the agent needs an AI backend.
      <button class="btn-link" data-action="open-settings">Set up in Settings</button>
    </div>
    <div class="replay-banner" data-replay-banner hidden>
      <div class="replay-banner-text">
        <strong>📊 Shared session loaded.</strong>
        <span data-replay-banner-detail></span>
      </div>
      <button class="btn-primary btn-tiny" data-action="replay-shared" type="button">🔁 Replay</button>
    </div>
    <div class="session-toolbar" data-session-toolbar hidden>
      <button class="btn-secondary btn-tiny" data-action="copy-session" type="button">📋 Copy session</button>
      <div class="download-menu">
        <button class="btn-secondary btn-tiny" data-action="download-toggle" type="button">⬇ Download…</button>
        <div class="download-popover" data-download-popover hidden>
          <label><input type="checkbox" data-dl-trace checked> Trace</label>
          <label><input type="checkbox" data-dl-plots checked> Plots</label>
          <label><input type="checkbox" data-dl-scripts checked> Python scripts</label>
          <label><input type="checkbox" data-dl-tables checked> Tables (CSV)</label>
          <label><input type="checkbox" data-dl-session checked> Session history</label>
          <button class="btn-primary btn-tiny" data-action="download-go" type="button">Download</button>
        </div>
      </div>
      <label class="trace-toggle" title="Show the agent trace expanded by default on each turn">
        <input type="checkbox" data-trace-default /> trace
      </label>
      <button class="btn-link" data-action="new-session" type="button">New session</button>
      <span class="agent-trace-copy-status" data-copy-status></span>
    </div>
    <div class="session-thread" data-thread></div>
    <form class="query-form">
      <input
        type="text"
        class="query-input"
        placeholder="Ask: 'measure properties of mito' or 'plot mito volumes'"
        autocomplete="off"
      />
      <button class="btn-primary" type="submit" data-action="ask">Ask</button>
      <button class="btn-secondary" type="button" data-action="stop" hidden>Stop</button>
    </form>
  `;
  container.appendChild(box);

  const form = box.querySelector<HTMLFormElement>(".query-form")!;
  const input = box.querySelector<HTMLInputElement>(".query-input")!;
  const button = box.querySelector<HTMLButtonElement>("button[type=submit]")!;
  const stopBtn = box.querySelector<HTMLButtonElement>("[data-action='stop']")!;
  const aiHint = box.querySelector<HTMLDivElement>("[data-ai-hint]")!;
  const aiHintBtn = box.querySelector<HTMLButtonElement>("[data-action='open-settings']")!;
  const threadEl = box.querySelector<HTMLDivElement>("[data-thread]")!;
  const sessionToolbar = box.querySelector<HTMLDivElement>("[data-session-toolbar]")!;
  const replayBanner = box.querySelector<HTMLDivElement>("[data-replay-banner]")!;
  const replayBannerDetail = box.querySelector<HTMLSpanElement>("[data-replay-banner-detail]")!;
  const replayBtn = box.querySelector<HTMLButtonElement>("[data-action='replay-shared']")!;
  const copyBtn = box.querySelector<HTMLButtonElement>("[data-action='copy-session']")!;
  const copyStatus = box.querySelector<HTMLSpanElement>("[data-copy-status]")!;
  const newSessionBtn = box.querySelector<HTMLButtonElement>("[data-action='new-session']")!;
  const dlToggle = box.querySelector<HTMLButtonElement>("[data-action='download-toggle']")!;
  const dlPopover = box.querySelector<HTMLDivElement>("[data-download-popover]")!;
  const dlGoBtn = box.querySelector<HTMLButtonElement>("[data-action='download-go']")!;
  const dlTrace = box.querySelector<HTMLInputElement>("[data-dl-trace]")!;
  const dlPlots = box.querySelector<HTMLInputElement>("[data-dl-plots]")!;
  const dlScripts = box.querySelector<HTMLInputElement>("[data-dl-scripts]")!;
  const dlTables = box.querySelector<HTMLInputElement>("[data-dl-tables]")!;
  const dlSession = box.querySelector<HTMLInputElement>("[data-dl-session]")!;
  const traceDefaultCheckbox = box.querySelector<HTMLInputElement>("[data-trace-default]")!;
  traceDefaultCheckbox.checked = loadTraceOpenPref();
  traceDefaultCheckbox.addEventListener("change", () => {
    const isOpen = traceDefaultCheckbox.checked;
    saveTraceOpenPref(isOpen);
    // Apply to every already-rendered card so the checkbox feels like
    // a real toggle, not just a default for future cards. Without this
    // users have to manually expand/collapse each turn's <details>.
    for (const card of allCards) {
      card.detailsEl.open = isOpen;
    }
  });

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
      ? "Ask: 'measure properties of mito' or 'plot mito volumes'"
      : "(set up AI in Settings to ask questions)";
  };
  refreshAiHint();
  setInterval(refreshAiHint, 1500);
  aiHintBtn.addEventListener("click", () => {
    document.getElementById("settings-btn")?.click();
  });

  // Active AbortController for the in-flight query, or null when idle.
  let currentAbortController: AbortController | null = null;
  stopBtn.addEventListener("click", () => {
    if (currentAbortController) {
      currentAbortController.abort();
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
  // All rendered turn cards, oldest-first. Used by the session-wide
  // Copy / Download buttons to aggregate every turn's outputs.
  const allCards: TurnCard[] = [];

  newSessionBtn.addEventListener("click", () => {
    sessionTurns.length = 0;
    allCards.length = 0;
    threadEl.innerHTML = "";
    sessionToolbar.hidden = true;
  });

  // ArrowUp/Down recall through the agent's persisted prompt history.
  // Triggers ONLY when the input is empty OR the user is already in
  // recall mode (showing a previous prompt unmodified) — Claude-style.
  let historyIndex = -1;
  let lastRecallValue = "";
  input.addEventListener("keydown", (e) => {
    if (button.hidden) return; // running; don't intercept
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    const empty = input.value.length === 0;
    const stillOnRecall = historyIndex >= 0 && input.value === lastRecallValue;
    if (!empty && !stillOnRecall) return;
    const items = loadPromptHistory("agent");
    if (items.length === 0) return;
    e.preventDefault();
    if (e.key === "ArrowUp") {
      historyIndex = Math.min(items.length - 1, historyIndex + 1);
    } else {
      historyIndex = Math.max(-1, historyIndex - 1);
    }
    input.value = historyIndex === -1 ? "" : items[historyIndex];
    lastRecallValue = input.value;
    input.setSelectionRange(input.value.length, input.value.length);
  });
  input.addEventListener("input", () => {
    historyIndex = -1;
    lastRecallValue = "";
  });

  // ---- per-turn card construction ----------------------------------------

  // Newest cards prepend so the user sees the current turn right under
  // the input without scrolling. Older turns sit further down. The
  // sidebar is the natural scroll container — no nested scroll needed.
  const createTurnCard = (question: string): TurnCard => {
    const idx = allCards.length;
    const root = document.createElement("div");
    root.className = "turn-card";
    // Order matches the chronological flow of a turn:
    //   header (the user's question)
    //   meta (runtime + scale chosen)
    //   status (what the agent is doing right now)
    //   trace (what the agent did, expandable; lives BEFORE the
    //          results because chronologically the steps came first)
    //   answer (the agent's textual conclusion)
    //   plots / tables (the artifacts produced)
    root.innerHTML = `
      <div class="turn-header">
        <span class="turn-num">${idx + 1}</span>
        <span class="turn-q"></span>
      </div>
      <div class="turn-meta" data-meta hidden></div>
      <div class="turn-status" data-status></div>
      <details class="turn-details" data-details>
        <summary>Show agent trace</summary>
        <div class="agent-trace" data-trace></div>
      </details>
      <div class="turn-answer" data-answer></div>
      <div class="turn-plots" data-plots></div>
      <div class="turn-tables" data-tables></div>
    `;
    root.querySelector<HTMLSpanElement>(".turn-q")!.textContent = question;
    // Append so the thread reads chronologically top-to-bottom — same
    // muscle memory as Claude / ChatGPT / iMessage. The form pins to
    // the bottom of the agent pane; the newest card sits right above
    // it. Older turns scroll up off-screen.
    threadEl.appendChild(root);
    const detailsEl = root.querySelector<HTMLDetailsElement>("[data-details]")!;
    detailsEl.open = traceDefaultCheckbox.checked;
    const card: TurnCard = {
      index: idx,
      question,
      root,
      metaEl: root.querySelector<HTMLElement>("[data-meta]")!,
      statusEl: root.querySelector<HTMLElement>("[data-status]")!,
      answerEl: root.querySelector<HTMLElement>("[data-answer]")!,
      plotsEl: root.querySelector<HTMLElement>("[data-plots]")!,
      tablesEl: root.querySelector<HTMLElement>("[data-tables]")!,
      traceEl: root.querySelector<HTMLElement>("[data-trace]")!,
      detailsEl,
      traceItems: [],
      plots: [],
      tables: [],
      metaLines: [],
    };
    allCards.push(card);
    sessionToolbar.hidden = false;
    // Scroll the agent pane to the bottom so the new card is visible
    // directly above the form. requestAnimationFrame waits for the
    // appended card to be laid out before we measure scroll heights.
    requestAnimationFrame(() => {
      const pane = document.getElementById("query-host");
      if (pane) pane.scrollTop = pane.scrollHeight;
    });
    return card;
  };

  // Render any sticky per-step meta info (runtime / scales / etc.) in
  // a small grayed-out strip under the question. Multiple lines stack.
  const appendMeta = (card: TurnCard, info: string): void => {
    card.metaLines.push(info);
    card.metaEl.hidden = false;
    const line = document.createElement("div");
    line.className = "turn-meta-line";
    line.textContent = info;
    card.metaEl.appendChild(line);
  };

  // Render a structured ask_user form inside the current turn card.
  // Returns a Promise that resolves on Submit (with {field_id: value})
  // or rejects on Cancel / Stop. The form stays in the DOM after
  // submit (disabled) so the user can scroll back and see what they
  // chose alongside the agent's continuation.
  const renderAskUser = (
    card: TurnCard,
    prompt: string,
    fields: AskField[],
    signal?: AbortSignal,
  ): Promise<Record<string, unknown>> => {
    return new Promise<Record<string, unknown>>((resolve, reject) => {
      const wrap = document.createElement("div");
      wrap.className = "ask-form";
      // Heading makes the form unmissable when scrolling — the previous
      // version sat between meta and answer with no visual signal that
      // the agent was *waiting* for the user. Easy to type into the
      // chat input by mistake instead of using the radio buttons.
      const header = document.createElement("div");
      header.className = "ask-form-header";
      header.textContent = "❓ The agent has a question for you";
      wrap.appendChild(header);
      const promptEl = document.createElement("div");
      promptEl.className = "ask-form-prompt";
      promptEl.textContent = prompt;
      wrap.appendChild(promptEl);

      // Build each field's DOM and a getter that pulls its current value.
      // Using closures + a getValue function per field keeps this small;
      // a more elaborate version could use form FormData but the
      // checkbox-array case (multi) needs hand work either way.
      const getters: { id: string; get: () => unknown }[] = [];
      for (const field of fields) {
        const wrapField = document.createElement("div");
        wrapField.className = "ask-form-field";
        const labelEl = document.createElement("label");
        labelEl.className = "ask-form-label";
        labelEl.textContent = field.label;
        wrapField.appendChild(labelEl);

        if (field.type === "select") {
          // Up to 4 options → radio buttons (one click + visible
          // alternatives); more → <select> (compact).
          if (field.options.length <= 4) {
            const group = document.createElement("div");
            group.className = "ask-form-radio-group";
            for (const opt of field.options) {
              const id = `tg-ask-${field.id}-${opt.value}`;
              const rwrap = document.createElement("label");
              rwrap.className = "ask-form-radio";
              const input = document.createElement("input");
              input.type = "radio";
              input.name = `tg-ask-${field.id}`;
              input.value = opt.value;
              input.id = id;
              if (field.default === opt.value) input.checked = true;
              rwrap.appendChild(input);
              const span = document.createElement("span");
              span.textContent = opt.label;
              rwrap.appendChild(span);
              group.appendChild(rwrap);
            }
            // If no default matched, pre-check the first.
            if (!field.default && group.querySelector("input[type=radio]")) {
              (group.querySelector("input[type=radio]") as HTMLInputElement).checked = true;
            }
            wrapField.appendChild(group);
            getters.push({
              id: field.id,
              get: () => (group.querySelector("input[type=radio]:checked") as HTMLInputElement | null)?.value ?? "",
            });
          } else {
            const sel = document.createElement("select");
            for (const opt of field.options) {
              const o = document.createElement("option");
              o.value = opt.value;
              o.textContent = opt.label;
              if (field.default === opt.value) o.selected = true;
              sel.appendChild(o);
            }
            wrapField.appendChild(sel);
            getters.push({ id: field.id, get: () => sel.value });
          }
        } else if (field.type === "multi") {
          const group = document.createElement("div");
          group.className = "ask-form-checkbox-group";
          const defaults = new Set(field.default ?? []);
          for (const opt of field.options) {
            const wrap2 = document.createElement("label");
            wrap2.className = "ask-form-checkbox";
            const input = document.createElement("input");
            input.type = "checkbox";
            input.value = opt.value;
            if (defaults.has(opt.value)) input.checked = true;
            wrap2.appendChild(input);
            const span = document.createElement("span");
            span.textContent = opt.label;
            wrap2.appendChild(span);
            group.appendChild(wrap2);
          }
          wrapField.appendChild(group);
          getters.push({
            id: field.id,
            get: () => Array.from(group.querySelectorAll<HTMLInputElement>("input[type=checkbox]:checked")).map((i) => i.value),
          });
        } else if (field.type === "yesno") {
          const group = document.createElement("div");
          group.className = "ask-form-radio-group";
          const yesId = `tg-ask-${field.id}-yes`;
          const noId = `tg-ask-${field.id}-no`;
          for (const opt of [
            { id: yesId, label: "Yes", val: "yes" },
            { id: noId, label: "No", val: "no" },
          ]) {
            const rwrap = document.createElement("label");
            rwrap.className = "ask-form-radio";
            const input = document.createElement("input");
            input.type = "radio";
            input.name = `tg-ask-${field.id}`;
            input.value = opt.val;
            input.id = opt.id;
            if (field.default === true && opt.val === "yes") input.checked = true;
            else if (field.default === false && opt.val === "no") input.checked = true;
            rwrap.appendChild(input);
            const span = document.createElement("span");
            span.textContent = opt.label;
            rwrap.appendChild(span);
            group.appendChild(rwrap);
          }
          if (!group.querySelector("input[type=radio]:checked")) {
            (group.querySelector("input[type=radio]") as HTMLInputElement).checked = true;
          }
          wrapField.appendChild(group);
          getters.push({
            id: field.id,
            get: () => (group.querySelector("input[type=radio]:checked") as HTMLInputElement | null)?.value === "yes",
          });
        } else if (field.type === "text") {
          const input = document.createElement("input");
          input.type = "text";
          input.className = "ask-form-text";
          if (field.default !== undefined) input.value = field.default;
          if (field.placeholder) input.placeholder = field.placeholder;
          wrapField.appendChild(input);
          getters.push({ id: field.id, get: () => input.value });
        }
        wrap.appendChild(wrapField);
      }

      const buttonRow = document.createElement("div");
      buttonRow.className = "ask-form-buttons";
      const submitBtn = document.createElement("button");
      submitBtn.type = "button";
      submitBtn.className = "btn-primary btn-tiny";
      submitBtn.textContent = "Submit";
      const cancelBtn = document.createElement("button");
      cancelBtn.type = "button";
      cancelBtn.className = "btn-secondary btn-tiny";
      cancelBtn.textContent = "Cancel";
      buttonRow.appendChild(submitBtn);
      buttonRow.appendChild(cancelBtn);
      wrap.appendChild(buttonRow);
      // Slot the form between meta and answer so it sits visually with
      // the agent's "I need to know X" prompt and stays out of the
      // way once submitted.
      card.metaEl.after(wrap);

      // Disable the chat input + Ask button while the form is open
      // so the user can't type a free-text answer that the agent
      // would never see. We restore the original placeholder + state
      // when the form settles.
      const prevPlaceholder = input.placeholder;
      const prevInputDisabled = input.disabled;
      const prevButtonDisabled = button.disabled;
      input.disabled = true;
      input.placeholder = "↑ Answer the agent's question above to continue…";
      button.disabled = true;

      let settled = false;
      const settle = (ok: boolean, value?: Record<string, unknown>): void => {
        if (settled) return;
        settled = true;
        // Disable form inputs so the user can scroll back and see
        // what they picked, but can't submit twice or change the
        // answer after the agent's already moved on.
        wrap.classList.add("ask-form-submitted");
        wrap.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>(
          "input, select, button",
        ).forEach((el) => {
          el.disabled = true;
        });
        // Re-enable the chat input now that the question is answered.
        input.disabled = prevInputDisabled;
        button.disabled = prevButtonDisabled;
        input.placeholder = prevPlaceholder;
        if (ok && value) resolve(value);
        else reject(new DOMException("ask_user cancelled", "AbortError"));
      };
      submitBtn.addEventListener("click", () => {
        const out: Record<string, unknown> = {};
        for (const g of getters) out[g.id] = g.get();
        settle(true, out);
      });
      cancelBtn.addEventListener("click", () => settle(false));
      // Stop button propagation: aborting the agent rejects any
      // pending form so the loop can unwind without a stray Promise
      // hanging on.
      if (signal) {
        if (signal.aborted) settle(false);
        else signal.addEventListener("abort", () => settle(false), { once: true });
      }
      // Scroll the form into view + focus the first input so keyboard
      // users can answer immediately without reaching for the mouse,
      // and the form is impossible to miss visually.
      requestAnimationFrame(() => {
        wrap.scrollIntoView({ block: "center", behavior: "smooth" });
        const first = wrap.querySelector<HTMLElement>("input, select, button.btn-primary");
        first?.focus();
      });
    });
  };

  const setStatus = (
    card: TurnCard,
    msg: string,
    kind: "" | "err" | "ok" | "pending" = "",
  ): void => {
    card.statusEl.textContent = msg;
    card.statusEl.className = `turn-status ${kind}`;
  };

  // Pretty-print ms — sub-second uses ms, second+ uses s with 1 decimal.
  const fmtMs = (ms: number): string =>
    ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;

  // One-line turn timing for the status: "(4.2 s · think 1.8 s · tools 2.4 s)".
  const formatTurnTiming = (totalMs: number, llmTotalMs: number, toolTotalMs: number): string => {
    const parts = [fmtMs(totalMs)];
    if (llmTotalMs > 0) parts.push(`think ${fmtMs(llmTotalMs)}`);
    if (toolTotalMs > 0) parts.push(`tools ${fmtMs(toolTotalMs)}`);
    return `(${parts.join(" · ")})`;
  };

  const renderPlot = (
    card: TurnCard,
    pngDataUrl: string,
    code: string,
    title?: string,
    explanation?: string,
  ): void => {
    const wrapper = document.createElement("div");
    wrapper.className = "plot-output";
    if (title) {
      const h = document.createElement("h3");
      h.className = "plot-title";
      h.textContent = title;
      wrapper.appendChild(h);
    }
    const img = document.createElement("img");
    img.src = pngDataUrl;
    img.className = "plot-image";
    img.title = "Click to enlarge";
    img.addEventListener("click", () => openFullscreenImage(pngDataUrl));
    wrapper.appendChild(img);
    if (explanation) {
      const p = document.createElement("p");
      p.className = "plot-explanation";
      p.textContent = explanation;
      wrapper.appendChild(p);
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
    wrapper.appendChild(toolbar);
    if (code) {
      const det = document.createElement("details");
      det.className = "plot-code";
      det.innerHTML = `<summary>Show Python source</summary>`;
      const pre = document.createElement("pre");
      pre.textContent = code;
      det.appendChild(pre);
      wrapper.appendChild(det);
    }
    card.plotsEl.appendChild(wrapper);
    card.plots.push({ png: pngDataUrl, code, title, explanation });
  };

  // Inline summary card for an ingested table. The data is already in
  // the SQL DB by the time we get here (applyCustomResult handled
  // ingestion), so this is purely for affordance — show the user that
  // a table was saved + give them a one-click CSV download. Browsing
  // the full data lives in the dataset's structured browser pane.
  const renderTable = (
    card: TurnCard,
    table: { name: string; columns: string[]; rows: (number | string | null)[][] },
  ): void => {
    const row = document.createElement("div");
    row.className = "turn-table-row";
    const label = document.createElement("span");
    label.className = "turn-table-label";
    label.innerHTML = `📊 Saved table: <code>${escapeHtml(table.name)}</code> (${table.rows.length} row${table.rows.length === 1 ? "" : "s"}, ${table.columns.length} cols)`;
    row.appendChild(label);
    row.appendChild(
      makeDownloadButton(`⬇ ${table.name}.csv`, () => {
        downloadBlob(new Blob([tableToCsv(table)], { type: "text/csv" }), `${table.name}.csv`);
      }),
    );
    card.tablesEl.appendChild(row);
    card.tables.push(table);
  };

  // Per-table row cap for Copy session. python_on_layers can return
  // a customResult whose .table.rows is millions of entries (hemibrain
  // segmentation has 8.7M unique ids). JSON.stringify-ing all of it
  // hangs the clipboard API; truncate to a representative slice instead.
  // Full table stays in the local DB; users wanting everything can use
  // the per-step Download buttons or the CSV export.
  const COPY_EMBED_ROW_CAP = 5000;
  const truncateResultForCopy = (result: unknown): unknown => {
    if (!result || typeof result !== "object") return result;
    const r = result as Record<string, unknown>;
    const table = r.table as { columns?: string[]; rows?: unknown[][]; name?: string } | undefined;
    if (!table || !Array.isArray(table.rows) || table.rows.length <= COPY_EMBED_ROW_CAP) {
      return result;
    }
    return {
      ...r,
      table: {
        ...table,
        rows: table.rows.slice(0, COPY_EMBED_ROW_CAP),
        _truncated_note: `truncated: ${table.rows.length.toLocaleString()} total rows → first ${COPY_EMBED_ROW_CAP.toLocaleString()} embedded`,
      },
    };
  };

  const formatStepForCopy = (item: AgentTraceItem, index: number): string => {
    const lines: string[] = [`## Step ${index + 1}: ${item.tool}`];
    if (item.args && Object.keys(item.args).length > 0) {
      lines.push("args:\n```json\n" + JSON.stringify(item.args, null, 2) + "\n```");
    }
    if (item.error) {
      lines.push("error:\n```\n" + item.error + "\n```");
    } else if (item.result !== undefined) {
      const trimmed = truncateResultForCopy(item.result);
      const r = typeof trimmed === "string" ? trimmed : JSON.stringify(trimmed, null, 2);
      lines.push("result:\n```\n" + r + "\n```");
    }
    return lines.join("\n");
  };

  const appendTrace = (card: TurnCard, item: AgentTraceItem): void => {
    const index = card.traceItems.length;
    card.traceItems.push(item);
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
    const isPythonTool = ["run_python", "make_plot", "python_on_layers"].includes(item.tool);
    const openBtn = isPythonTool
      ? `<button class="btn-tiny agent-trace-open-step" type="button" title="Open this code in the Custom Python dialog">🐍 Edit</button>`
      : "";
    // Per-step timing badge: "think 1.8s · tool 2.4s". Skipped when both
    // are undefined (e.g. unknown-tool error before any timing was set).
    const llmStr = typeof item.llmMs === "number" ? `think ${fmtMs(item.llmMs)}` : "";
    const toolStr = typeof item.toolMs === "number" ? `tool ${fmtMs(item.toolMs)}` : "";
    const timingStr = [llmStr, toolStr].filter(Boolean).join(" · ");
    const timingBadge = timingStr
      ? `<span class="agent-trace-step-timing" title="LLM stream time + tool execution time">${escapeHtml(timingStr)}</span>`
      : "";
    row.innerHTML = `
      <div class="agent-trace-item-header">
        <span class="agent-trace-tool">${escapeHtml(item.tool)}</span>
        ${timingBadge}
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
        document.getElementById("custom-btn")?.click();
      });
    }
    card.traceEl.appendChild(row);
    card.traceEl.scrollTop = card.traceEl.scrollHeight;
  };

  // ---- session-wide aggregation (Copy / Download) ------------------------

  const formatSessionForCopy = (): string => {
    const lines: string[] = [`# Tourguide session (${allCards.length} turn${allCards.length === 1 ? "" : "s"})`];
    // Walk allCards in submit order so the markdown reads
    // chronologically even though the on-screen layout is newest-first.
    for (const card of allCards) {
      lines.push(`\n## Q${card.index + 1}: ${card.question}\n`);
      if (card.statusEl.textContent) lines.push(`_status: ${card.statusEl.textContent}_\n`);
      if (card.answerEl.textContent) lines.push(`${card.answerEl.textContent}\n`);
      if (card.traceItems.length > 0) {
        lines.push(`### Trace (${card.traceItems.length} step${card.traceItems.length === 1 ? "" : "s"})`);
        card.traceItems.forEach((it, i) => lines.push("\n" + formatStepForCopy(it, i)));
      }
    }
    return lines.join("\n");
  };

  copyBtn.addEventListener("click", async () => {
    const text = formatSessionForCopy();
    try {
      await navigator.clipboard.writeText(text);
      copyStatus.textContent = "✓ copied";
      copyStatus.className = "agent-trace-copy-status ok";
    } catch {
      window.prompt("Copy session:", text);
      copyStatus.textContent = "";
    }
    setTimeout(() => (copyStatus.textContent = ""), 1500);
  });

  dlToggle.addEventListener("click", (e) => {
    e.stopPropagation();
    dlPopover.hidden = !dlPopover.hidden;
  });
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
      includeTables: dlTables.checked,
      includeSession: dlSession.checked,
    });
    downloadBlob(new Blob([md], { type: "text/markdown" }), `tourguide-session-${Date.now()}.md`);
    dlPopover.hidden = true;
  });

  const buildSessionMarkdown = (opts: {
    includeTrace: boolean;
    includePlots: boolean;
    includeScripts: boolean;
    includeTables: boolean;
    includeSession: boolean;
  }): string => {
    const out: string[] = [];
    out.push(`# Tourguide session\n`);
    out.push(`Generated ${new Date().toISOString()}\n`);
    for (const card of allCards) {
      out.push(`\n## Q${card.index + 1}: ${card.question}`);
      if (card.answerEl.textContent) out.push(`\n${card.answerEl.textContent}`);
      if (opts.includePlots && card.plots.length > 0) {
        card.plots.forEach((p, i) => {
          const label = p.title ?? `Plot ${i + 1}`;
          out.push(`\n### ${label}\n`);
          out.push(`![${label}](${p.png})`);
        });
      }
      if (opts.includeScripts) {
        const scripts = card.traceItems
          .filter((t) => ["run_python", "make_plot", "python_on_layers"].includes(t.tool))
          .map((t) => ({ tool: t.tool, code: String((t.args as Record<string, unknown>)?.python ?? "") }))
          .filter((s) => s.code.length > 0);
        scripts.forEach((s, i) => {
          out.push(`\n### Script ${i + 1} (${s.tool})\n`);
          out.push("```python\n" + s.code + "\n```");
        });
      }
      if (opts.includeTables && card.tables.length > 0) {
        card.tables.forEach((t) => {
          out.push(`\n### Table: ${t.name} (${t.rows.length} rows)\n`);
          out.push("```csv\n" + tableToCsv(t) + "```");
        });
      }
      if (opts.includeTrace && card.traceItems.length > 0) {
        out.push(`\n### Trace (${card.traceItems.length} steps)`);
        card.traceItems.forEach((it, i) => out.push("\n" + formatStepForCopy(it, i)));
      }
    }
    if (opts.includeSession && sessionTurns.length > 0) {
      out.push(`\n## Session summary (the agent's per-turn recap, used as priorTurns context)`);
      sessionTurns.forEach((t, i) => {
        out.push(`\n### Turn ${i + 1}\n- **Q:** ${t.question}\n- **A:** ${t.summary}`);
      });
    }
    return out.join("\n");
  };

  // ---- shared helpers ----------------------------------------------------

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

  const tableToCsv = (tbl: { columns: string[]; rows: (number | string | null)[][] }): string => {
    const esc = (v: unknown): string => {
      if (v === null || v === undefined) return "";
      const s = String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const lines = [tbl.columns.map(esc).join(",")];
    for (const row of tbl.rows) lines.push(row.map(esc).join(","));
    return lines.join("\n") + "\n";
  };

  // ---- submit ------------------------------------------------------------

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    recordPrompt(question, "agent");
    historyIndex = -1;
    lastRecallValue = "";

    const db = ctx.getDB();
    const descriptor = ctx.getDescriptor();
    const backend = ctx.getBackend();
    if (!db && !descriptor) {
      // No card to render into yet — show the gate inline as a transient
      // status by creating a minimal card that won't pollute history.
      const card = createTurnCard(question);
      setStatus(card, "Load a dataset first.", "err");
      return;
    }
    if (!backend.isReady()) {
      const card = createTurnCard(question);
      setStatus(card, "No AI configured. Click Settings to add a key or enable WebLLM.", "err");
      return;
    }

    const card = createTurnCard(question);
    input.value = "";
    button.hidden = true;
    stopBtn.hidden = false;
    setStatus(card, "Thinking…", "pending");
    // Trace open/closed state is set by createTurnCard from the
    // session toolbar's "show trace" checkbox — don't re-stomp it
    // here. (Used to hard-code `false` and silently break the
    // toggle.)

    let answeredOrFlew = false;
    let turnSummary = "";
    const abortController = new AbortController();
    currentAbortController = abortController;
    const turnStart = performance.now();
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
          onTrace: (t) => appendTrace(card, t),
          onProgress: (m) => setStatus(card, m, "pending"),
          onAnswer: (text) => {
            card.answerEl.textContent = text;
            answeredOrFlew = true;
            turnSummary = text;
          },
          onPlot: (png, code, title, explanation) => {
            renderPlot(card, png, code, title, explanation);
            answeredOrFlew = true;
            if (!turnSummary) turnSummary = title ? `Showed plot: ${title}` : "Showed a plot";
          },
          onFly: (_pos, layer, id) => {
            setStatus(card, `Flew to ${layer}${id ? ` ${id}` : ""}`, "ok");
            answeredOrFlew = true;
            if (!turnSummary) turnSummary = `Flew to ${layer}${id ? ` ${id}` : ""}`;
          },
          onHighlight: (layer, ids) => {
            setStatus(card, `Showing ${ids.length} segment${ids.length === 1 ? "" : "s"} in ${layer}`, "ok");
            answeredOrFlew = true;
            if (!turnSummary) turnSummary = `Highlighted ${ids.length} segments in ${layer}`;
          },
          onTable: (tbl) => {
            renderTable(card, tbl);
            answeredOrFlew = true;
            if (!turnSummary) turnSummary = `Saved table ${tbl.name} (${tbl.rows.length} rows)`;
          },
          onMeta: (info) => appendMeta(card, info),
          onAskUser: (prompt, fields) => renderAskUser(card, prompt, fields, abortController.signal),
        },
      });
      // Aggregate per-step timings into a turn-level breakdown so the
      // user can see where the time went. LLM = thinking; tools =
      // network + analysis backend + worker.
      const totalMs = performance.now() - turnStart;
      const llmTotal = card.traceItems.reduce((s, t) => s + (t.llmMs ?? 0), 0);
      const toolTotal = card.traceItems.reduce((s, t) => s + (t.toolMs ?? 0), 0);
      const timing = formatTurnTiming(totalMs, llmTotal, toolTotal);
      if (!answeredOrFlew) {
        setStatus(card, `Agent finished without delivering an answer. ${timing}`, "");
      } else {
        // Always end with a clear "done" — overwrites any leftover
        // "Step N: writing answer …" that was hanging around when the
        // executor returned. Green "ok" styling so it's visually
        // distinct from the in-flight pending state.
        setStatus(card, `✓ Done ${timing}`, "ok");
      }
      sessionTurns.push({ question, summary: turnSummary || "(no visible result)" });
    } catch (err) {
      if (abortController.signal.aborted || (err as Error).name === "AbortError") {
        setStatus(card, "Stopped.", "");
      } else {
        setStatus(card, (err as Error).message, "err");
        console.error(err);
      }
    } finally {
      currentAbortController = null;
      button.hidden = false;
      stopBtn.hidden = true;
    }
  });

  // Snapshot every rendered turn into the share-link's SerializedTurn
  // shape. Reads from allCards (the in-memory store of rendered cards)
  // rather than maintaining a parallel array — keeps the two in sync
  // automatically as the agent runs.
  const getSerializedSession = (): SerializedTurn[] =>
    allCards.map((card) => ({
      question: card.question,
      answer: card.answerEl.textContent?.trim() || undefined,
      status: card.statusEl.textContent?.trim() || undefined,
      trace: card.traceItems.map((it) => ({
        tool: it.tool,
        args: it.args,
        result: it.result,
        error: it.error,
        llmMs: it.llmMs,
        toolMs: it.toolMs,
      })),
      plots: card.plots.length > 0
        ? card.plots.map((p) => ({
            png: p.png,
            title: p.title,
            explanation: p.explanation,
          }))
        : undefined,
    }));

  // Inverse: take a saved session and render it as if the agent had
  // just run those turns. Used on share-link load. Tables are NOT
  // re-rendered here — they ship separately via `analysisTables` and
  // get ingested into the SQL DB before this fires, so any table that
  // was in the original session is already in the structured browser.
  // Plot PNGs are NOT shipped (would bloat the URL); a stub card is
  // rendered for each saved plot pointing the recipient at the
  // matching python_on_layers / make_plot step's Replay button.
  const replaySerializedSession = (turns: SerializedTurn[]): void => {
    for (const t of turns) {
      const card = createTurnCard(t.question);
      for (const it of t.trace) {
        // SerializedTraceItem.args is optional (`?`) but AgentTraceItem.args
        // is required — default to {} when missing to keep appendTrace happy.
        appendTrace(card, { ...it, args: it.args ?? {} });
      }
      if (t.answer) card.answerEl.textContent = t.answer;
      if (t.plots && t.plots.length > 0) {
        for (const p of t.plots) {
          if (p.png) {
            // Future-proof: a future version might preserve the PNG
            // (or shipped a small thumbnail). Render normally then.
            renderPlot(card, p.png, "", p.title, p.explanation);
          } else {
            // Placeholder so the recipient knows a plot existed here
            // and how to regenerate it. Replay is one click away
            // via the 🐍 Edit button on the matching python step.
            const stub = document.createElement("div");
            stub.className = "plot-output plot-stub";
            const titleText = p.title ? ` "${p.title}"` : "";
            stub.innerHTML = `
              <p class="hint">📊 Plot${escapeHtml(titleText)} not embedded in share link (PNGs are big and regenerate cheaply). Click 🐍 <strong>Edit</strong> on the matching step below, then Run to regenerate.</p>
              ${p.explanation ? `<p class="plot-explanation">${escapeHtml(p.explanation)}</p>` : ""}
            `;
            card.plotsEl.appendChild(stub);
          }
        }
      }
      if (t.status) {
        // Preserve the original status verbatim (incl. ✓ + timings).
        // Classify via prefix so "✓ Done" stays green; everything else
        // falls into the neutral bin.
        const kind = t.status.startsWith("✓") ? "ok" : "";
        setStatus(card, t.status, kind);
      }
      // Also record into sessionTurns so the agent's prior-turn
      // context picks up these as if the live session ran them.
      sessionTurns.push({ question: t.question, summary: t.answer || "(no visible result)" });
    }
    // Surface the banner if any of the replayed turns has python that
    // would regenerate something on click. Naive-user view: "look,
    // there's more stuff you can get; click Replay to get it."
    const replayableStepCount = turns.reduce(
      (n, t) =>
        n +
        t.trace.filter(
          (it) => it.tool === "python_on_layers" || it.tool === "make_plot" || it.tool === "run_python",
        ).length,
      0,
    );
    const plotCount = turns.reduce((n, t) => n + (t.plots?.length ?? 0), 0);
    if (replayableStepCount > 0) {
      const detailParts: string[] = [];
      if (plotCount > 0) detailParts.push(`${plotCount} plot${plotCount === 1 ? "" : "s"}`);
      detailParts.push(
        `${replayableStepCount} analysis step${replayableStepCount === 1 ? "" : "s"}`,
      );
      replayBannerDetail.textContent =
        ` Plot images and any layers the analysis created weren't embedded — click Replay to regenerate ${detailParts.join(" / ")} on your backend.`;
      replayBanner.hidden = false;
    }
  };

  // Replay button: jump to the FIRST python step's Edit handler so
  // the Custom Python dialog opens preloaded with that step's code,
  // then have the dialog auto-click Run as soon as inspect finishes.
  // The per-step 🐍 Edit buttons in the trace keep the review-first
  // flow for users who want to tweak code before running.
  replayBtn.addEventListener("click", () => {
    const firstPythonBtn = box.querySelector<HTMLButtonElement>(
      ".agent-trace-item .agent-trace-open-step",
    );
    if (firstPythonBtn) {
      const containingDetails = firstPythonBtn.closest("details");
      if (containingDetails) (containingDetails as HTMLDetailsElement).open = true;
      // The Edit click handler synchronously calls setPendingSession
      // and then opens the dialog. Click first, then flip the
      // pending state's autorun flag — the dialog hasn't drained it
      // yet (drain happens on dialog mount, which is microtask-async).
      firstPythonBtn.click();
      const pending = peekPendingSession();
      if (pending) {
        setPendingSession({ ...pending, autorun: true });
      }
    }
  });

  return { getSerializedSession, replaySerializedSession };
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
