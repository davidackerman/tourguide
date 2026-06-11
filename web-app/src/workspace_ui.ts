// Workspace-mode sidebar panel.
//
// In workspace mode the agent lives *outside* Tourguide (Claude Desktop,
// Claude Code, a script, …) and drives the viewer over the Workspace API.
// So instead of the chat composer we show two things here:
//   1. Agent connection status (is anything driving this session?).
//   2. The "Agent Actions" history — a log of operations that changed the
//      workspace (NOT the agent's full conversation).
//
// This module owns only the DOM + a small imperative handle. The bridge
// (Phase 2) feeds it connection-status and action events; in Phase 1 it
// renders as an empty, disconnected scaffold.

import type {
  ActionHistoryEntry,
  ConnectionStatus,
  PlotArtifact,
} from "./workspace_api/protocol.js";

export interface WorkspacePanelHandle {
  setConnectionStatus(status: ConnectionStatus, detail?: string): void;
  addActionEntry(entry: ActionHistoryEntry): void;
  /** Replace the whole list (e.g. on reconnect when the bridge replays). */
  setActionEntries(entries: ActionHistoryEntry[]): void;
  clearActions(): void;
  /** Wire the "restore" affordance on entries that captured a saved state. */
  onRestoreSavedState(cb: (savedStateId: string) => void): void;
  /** Dock a plot thumbnail into the panel (clicking it enlarges). */
  addPlot(artifact: PlotArtifact): void;
  /** Wire the enlarge action when a docked plot thumbnail is clicked. */
  onOpenPlot(cb: (artifact: PlotArtifact) => void): void;
  /** Show a clickable share link at the top of the panel (Open + Copy). */
  addShareLink(url: string, label?: string): void;
}

const STATUS_LABEL: Record<ConnectionStatus, string> = {
  connected: "Agent connected",
  disconnected: "No agent connected",
  reconnecting: "Reconnecting…",
};

const esc = (s: string): string =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

export function renderWorkspacePanel(container: HTMLElement): WorkspacePanelHandle {
  container.innerHTML = `
    <div class="workspace-panel">
      <div class="workspace-conn" data-conn>
        <span class="conn-dot" data-conn-dot></span>
        <span class="conn-label" data-conn-label>No agent connected</span>
        <span class="conn-detail hint" data-conn-detail></span>
      </div>
      <div class="workspace-share" data-share hidden></div>
      <div class="workspace-plots" data-plots hidden>
        <header class="workspace-actions-header">
          <h3>Plots</h3>
        </header>
        <div class="workspace-plots-list" data-plots-list></div>
      </div>
      <div class="workspace-actions">
        <header class="workspace-actions-header">
          <h3 class="workspace-actions-title" data-actions-toggle title="Show/hide the agent trace">▾ Agent Trace</h3>
          <span class="workspace-actions-btns">
            <button class="btn-secondary btn-xs" data-export-actions title="Export the trace as JSON — exactly what the agent ran (sources, scales, recipes), so it can be reproduced (e.g. at full resolution on a cluster)">⤓ Export</button>
            <button class="btn-secondary btn-xs" data-clear-actions title="Clear the action log">Clear</button>
          </span>
        </header>
        <div class="workspace-actions-list" data-actions-list>
          <p class="placeholder">No agent actions yet. Connect an agent (e.g. <code>tourguide-mcp</code>) to drive this workspace.</p>
        </div>
      </div>
    </div>
  `;

  const dot = container.querySelector<HTMLSpanElement>("[data-conn-dot]")!;
  const label = container.querySelector<HTMLSpanElement>("[data-conn-label]")!;
  const detailEl = container.querySelector<HTMLSpanElement>("[data-conn-detail]")!;
  const list = container.querySelector<HTMLDivElement>("[data-actions-list]")!;
  const clearBtn = container.querySelector<HTMLButtonElement>("[data-clear-actions]")!;
  const plotsSection = container.querySelector<HTMLDivElement>("[data-plots]")!;
  const plotsList = container.querySelector<HTMLDivElement>("[data-plots-list]")!;
  const shareSection = container.querySelector<HTMLDivElement>("[data-share]")!;

  let entries: ActionHistoryEntry[] = [];
  let restoreCb: ((id: string) => void) | null = null;
  let openPlotCb: ((artifact: PlotArtifact) => void) | null = null;

  const renderEmpty = (): void => {
    list.innerHTML = `<p class="placeholder">No agent actions yet. Connect an agent (e.g. <code>tourguide-mcp</code>) to drive this workspace.</p>`;
  };

  const entryEl = (e: ActionHistoryEntry): string => {
    const time = (() => {
      try {
        return new Date(e.timestamp).toLocaleTimeString();
      } catch {
        return e.timestamp;
      }
    })();
    const statusClass = e.error ? "err" : "ok";
    const restoreBtn = e.savedStateId
      ? `<button class="btn-secondary btn-xs" data-restore="${esc(e.savedStateId)}">↩ Restore</button>`
      : "";
    const args = e.argsSummary ? `<span class="action-args">${esc(e.argsSummary)}</span>` : "";
    const result = e.error
      ? `<span class="action-result err">${esc(e.error)}</span>`
      : e.resultSummary
        ? `<span class="action-result">${esc(e.resultSummary)}</span>`
        : "";
    return `
      <div class="action-entry ${statusClass}" data-entry-id="${esc(e.id)}">
        <div class="action-entry-head">
          <span class="action-name">${esc(e.action)}</span>
          <span class="action-source">${esc(e.source)}</span>
          <span class="action-time hint">${esc(time)}</span>
        </div>
        ${args ? `<div class="action-entry-body">${args}</div>` : ""}
        ${result ? `<div class="action-entry-body">${result}</div>` : ""}
        ${restoreBtn ? `<div class="action-entry-actions">${restoreBtn}</div>` : ""}
      </div>
    `;
  };

  const render = (): void => {
    if (entries.length === 0) {
      renderEmpty();
      return;
    }
    // Newest first.
    list.innerHTML = entries.map(entryEl).slice().reverse().join("");
    list.querySelectorAll<HTMLButtonElement>("[data-restore]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.getAttribute("data-restore");
        if (id && restoreCb) restoreCb(id);
      });
    });
  };

  clearBtn.addEventListener("click", () => {
    entries = [];
    render();
  });

  // The trace is provenance, not always-on chrome — let it collapse.
  const toggle = container.querySelector<HTMLElement>("[data-actions-toggle]")!;
  toggle.addEventListener("click", () => {
    const hidden = list.hasAttribute("hidden");
    list.toggleAttribute("hidden", !hidden);
    toggle.textContent = hidden ? "▾ Agent Trace" : "▸ Agent Trace";
  });

  // Export the trace as JSON — a reproducible record of exactly what the agent
  // ran (ops, sources, scales, recipes), so someone can reproduce it elsewhere
  // (e.g. rerun at full resolution on a cluster).
  const exportBtn = container.querySelector<HTMLButtonElement>("[data-export-actions]")!;
  exportBtn.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(entries, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "tourguide-trace.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  });

  setConnDom("disconnected", "");
  function setConnDom(status: ConnectionStatus, detail: string): void {
    dot.className = `conn-dot conn-${status}`;
    label.textContent = STATUS_LABEL[status];
    detailEl.textContent = detail;
  }

  return {
    setConnectionStatus(status, detail = "") {
      setConnDom(status, detail);
    },
    addActionEntry(entry) {
      entries.push(entry);
      render();
    },
    setActionEntries(next) {
      entries = next.slice();
      render();
    },
    clearActions() {
      entries = [];
      render();
    },
    onRestoreSavedState(cb) {
      restoreCb = cb;
    },
    addPlot(artifact) {
      if (!artifact.pngDataUrl) return;
      plotsSection.hidden = false;
      const fig = document.createElement("figure");
      fig.className = "workspace-plot";
      fig.title = "Click to enlarge";
      const title = artifact.title || "Plot";
      const img = document.createElement("img");
      img.src = artifact.pngDataUrl;
      img.alt = title;
      const cap = document.createElement("figcaption");
      cap.textContent = title;
      fig.append(img, cap);
      fig.addEventListener("click", () => openPlotCb?.(artifact));
      plotsList.prepend(fig); // newest first
    },
    onOpenPlot(cb) {
      openPlotCb = cb;
    },
    addShareLink(url, label) {
      const name = label || "Open share link";
      shareSection.hidden = false;
      shareSection.innerHTML = `
        <header class="workspace-actions-header"><h3>Share link</h3></header>
        <div class="workspace-share-row">
          <a class="workspace-share-open" target="_blank" rel="noopener"></a>
          <button class="btn-secondary btn-xs" data-share-copy>Copy</button>
        </div>`;
      const a = shareSection.querySelector<HTMLAnchorElement>(".workspace-share-open")!;
      a.href = url;
      a.textContent = name;
      const btn = shareSection.querySelector<HTMLButtonElement>("[data-share-copy]")!;
      btn.addEventListener("click", () => {
        void navigator.clipboard?.writeText(url).then(
          () => { btn.textContent = "Copied"; },
          () => { btn.textContent = "Copy failed"; },
        );
      });
    },
  };
}
