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

export function startWorkspaceBridge(
  ctx: WorkspaceContext,
  panel: WorkspacePanelHandle,
  opts: { bridgeWsUrl: string },
): WorkspaceBridgeHandle {
  const handlers = createHandlers(ctx);

  const transport = new BrowserWsTransport({
    sessionId: ctx.sessionId,
    mode: ctx.mode,
    bridgeWsUrl: opts.bridgeWsUrl,
    onStatus: (status: ConnectionStatus, detail) => panel.setConnectionStatus(status, detail),
    onRequest: (request) => void handle(request),
    // Put the bridge-assigned label in the browser tab title so multiple
    // open workspace tabs are distinguishable at a glance.
    onRegistered: (label) => {
      if (label && typeof document !== "undefined") {
        document.title = `Tourguide — ${label}`;
      }
    },
  });

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
