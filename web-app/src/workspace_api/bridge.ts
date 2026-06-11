// Workspace API bridge (browser side).
//
// Wires the WebSocket transport to the operation handlers, records an
// action-history entry for every workspace-affecting op, streams those
// entries to the bridge server (for agent subscribers) and to the local
// Agent Actions panel, and reflects connection status.
//
// Read-only inspection ops (get_session, get_viewer_state, get_selection,
// list_*, get_table_schema, run_sql, export_session_summary) do NOT create
// history entries — the panel logs operations that *change* the workspace,
// not every poll.

import type { WorkspacePanelHandle } from "../workspace_ui.js";
import { createHandlers, type WorkspaceContext } from "./handlers.js";
import { BrowserWsTransport } from "./transport_ws.js";
import { ingestTableIntoDB, runQuery, type DatasetDB } from "../db.js";
import type {
  ActionHistoryEntry,
  ConnectionStatus,
  WorkspaceRequest,
  WorkspaceResponse,
} from "./protocol.js";

const READ_ONLY_OPS = new Set<string>([
  "get_session",
  "get_viewer_state",
  "get_selection",
  "list_tables",
  "get_table_schema",
  "run_sql",
  "list_saved_states",
  "export_session_summary",
  "launch_or_attach",
  "show_share_link",
]);

export interface WorkspaceBridgeHandle {
  stop(): void;
}

const nowIso = (): string => new Date().toISOString();

const truncate = (s: string, n = 160): string => (s.length > n ? `${s.slice(0, n)}…` : s);

const summarizeArgs = (params: unknown): string | undefined => {
  if (params === undefined || params === null) return undefined;
  try {
    return truncate(JSON.stringify(params));
  } catch {
    return undefined;
  }
};

const summarizeResult = (result: unknown): string | undefined => {
  if (result === undefined || result === null) return undefined;
  try {
    return truncate(JSON.stringify(result));
  } catch {
    return undefined;
  }
};

// Export every ingested table's data (columns + rows) from the sql.js DB so
// the per-session snapshot can carry it — the structured browser is rebuilt by
// re-ingesting on restore (the snapshot's tableIds alone can't bring data back).
function exportTables(
  db: DatasetDB | null,
): Array<{ name: string; columns: string[]; rows: (number | string | null)[][] }> {
  if (!db) return [];
  const out: Array<{ name: string; columns: string[]; rows: (number | string | null)[][] }> = [];
  for (const t of db.tables) {
    try {
      const cols = t.columns.map((c) => `"${c}"`).join(", ");
      const res = runQuery(db.db, `SELECT ${cols} FROM "${t.table_name}";`);
      out.push({ name: t.table_name, columns: res.columns, rows: res.rows as (number | string | null)[][] });
    } catch {
      /* a table that won't export is skipped, not fatal */
    }
  }
  return out;
}

export function startWorkspaceBridge(
  ctx: WorkspaceContext,
  panel: WorkspacePanelHandle,
  opts: { bridgeWsUrl: string; viewOnly?: boolean; viewOf?: string },
): WorkspaceBridgeHandle {
  const handlers = createHandlers(ctx);

  const transport = new BrowserWsTransport({
    sessionId: ctx.sessionId,
    mode: ctx.mode,
    bridgeWsUrl: opts.bridgeWsUrl,
    viewOf: opts.viewOf,
    onStatus: (status: ConnectionStatus, detail) => panel.setConnectionStatus(status, detail),
    onRequest: (request) => void handle(request),
    // Put the bridge-assigned label in the browser tab title so multiple
    // open workspace tabs are distinguishable at a glance.
    onRegistered: (label) => {
      if (label && typeof document !== "undefined") {
        document.title = `Tourguide — ${label}`;
      }
    },
    // Reopening a ?session=<id> link: the bridge sends back the last persisted
    // snapshot for this id; apply it so the workspace (layers/camera) returns.
    onRestore: (state) => {
      void applyRestore(state);
    },
  });

  // Reopening a ?session=<id> link: replay the persisted snapshot — viewer
  // (layers/camera), then tables (re-ingest into the DB), then plots (re-add
  // + display). So the whole workspace returns, not just the layers.
  async function applyRestore(state: unknown): Promise<void> {
    if (!state || typeof state !== "object") return;
    const s = state as {
      plots?: Array<Parameters<typeof ctx.store.addPlot>[0]>;
      tablesData?: Array<{ name: string; columns: string[]; rows: (number | string | null)[][] }>;
    };
    try {
      ctx.store.applyState(state as unknown as Parameters<typeof ctx.store.applyState>[0]);
    } catch (err) {
      console.warn("[bridge] restore viewer failed:", err);
    }
    const layerNames = (ctx.getDescriptor()?.layers ?? []).map((l) => l.name);
    for (const t of s.tablesData ?? []) {
      try {
        await ingestTableIntoDB({ getDB: ctx.getDB, setDB: ctx.setDB }, t, layerNames);
      } catch (err) {
        console.warn("[bridge] restore table failed:", t.name, err);
      }
    }
    const havePlot = new Set(ctx.store.listPlots().map((x) => x.id));
    for (const p of s.plots ?? []) {
      try {
        const pid = (p as { id?: string }).id;
        if (pid && havePlot.has(pid)) continue; // already present — don't duplicate
        const added = ctx.store.addPlot(p);
        havePlot.add(added.id);
        ctx.displayPlot(added);
      } catch (err) {
        console.warn("[bridge] restore plot failed:", err);
      }
    }
    ctx.refreshBrowser();
  }

  // Rolling auto-save: after a change, push a snapshot up for the bridge to
  // persist under this session id (debounced so a burst of ops saves once).
  let persistTimer: ReturnType<typeof setTimeout> | null = null;
  function schedulePersist(): void {
    if (persistTimer) clearTimeout(persistTimer);
    persistTimer = setTimeout(() => {
      try {
        transport.persist({
          ...ctx.store.snapshot(),
          plots: ctx.store.listPlots(),
          tablesData: exportTables(ctx.getDB()),
        });
      } catch (err) {
        console.warn("[bridge] persist snapshot failed:", err);
      }
    }, 1500);
  }

  // Let the user restore a saved state straight from a history entry.
  panel.onRestoreSavedState((savedStateId) => {
    void handle({
      id: `restore-${Date.now()}`,
      op: "restore_session_state",
      params: { id: savedStateId },
      source: "internal",
    });
  });

  async function handle(request: WorkspaceRequest): Promise<void> {
    const handler = handlers[request.op];
    let response: WorkspaceResponse;
    let result: unknown;
    let error: string | undefined;
    if (!handler) {
      error = `unknown op: ${request.op}`;
      response = { id: request.id, ok: false, error: { message: error } };
    } else {
      try {
        result = await handler(request.params ?? {});
        response = { id: request.id, ok: true, result };
      } catch (err) {
        error = (err as Error).message;
        response = { id: request.id, ok: false, error: { message: error, stack: (err as Error).stack } };
      }
    }
    transport.sendResponse(response);

    // Record + stream an action entry for workspace-changing ops (and any
    // op that errored — a failed write is worth surfacing).
    if (!READ_ONLY_OPS.has(request.op) || error) {
      const entry = buildEntry(request, result, error);
      panel.addActionEntry(entry);
      transport.sendEvent({ type: "action", entry });
    }
    // Auto-save the workspace after any successful change so reopening the
    // ?session=<id> link restores it — but a read-only viewer never persists
    // back (so it can't modify the owner's saved session).
    if (!READ_ONLY_OPS.has(request.op) && !error && !opts.viewOnly) schedulePersist();
  }

  function buildEntry(
    request: WorkspaceRequest,
    result: unknown,
    error: string | undefined,
  ): ActionHistoryEntry {
    const r = (result ?? {}) as Record<string, unknown>;
    const artifactIds =
      request.op === "show_plot" && typeof r.id === "string" ? [r.id] : undefined;
    const savedStateId =
      (request.op === "save_session_state" || request.op === "restore_session_state") &&
      typeof r.id === "string"
        ? r.id
        : undefined;
    return {
      id: request.id,
      timestamp: nowIso(),
      source: request.source ?? "internal",
      action: request.op,
      argsSummary: summarizeArgs(request.params),
      resultSummary: error ? undefined : summarizeResult(result),
      artifactIds,
      error,
      savedStateId,
    };
  }

  transport.start();
  return { stop: () => transport.stop() };
}
