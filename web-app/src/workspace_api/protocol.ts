// Tourguide Workspace API — protocol contract.
//
// This is the DURABLE artifact. MCP is just the first adapter; the Python
// SDK and any future adapter speak the same operations defined here. Keep
// this file free of DOM / viewer / DB imports so it can be shared verbatim
// with non-browser code (it is mirrored by the MCP adapter's schemas.py and
// the Python SDK).
//
// Wire shape: JSON-RPC-ish request/response envelopes relayed between an
// external agent and the live browser session by the local bridge server.
// HTTP carries request/response operations; WebSocket carries live streams
// (connection status, action history, task progress).

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

/** The complete set of app-level operations the Workspace API exposes. These
 *  are intentionally *semantic* (select_segments, fly_to) rather than raw
 *  viewer-state mutations; set_viewer_state is the escape hatch. */
export type WorkspaceOp =
  | "launch_or_attach"
  | "get_session"
  | "load_descriptor"
  | "get_viewer_state"
  | "set_viewer_state"
  | "get_selection"
  | "select_segments"
  | "fly_to"
  | "add_layer"
  | "add_annotations"
  | "list_tables"
  | "get_table_schema"
  | "run_sql"
  | "show_table"
  | "show_plot"
  | "save_session_state"
  | "restore_session_state"
  | "list_saved_states"
  | "start_recording"
  | "stop_recording"
  | "add_narration_note"
  | "export_session_summary";

export const WORKSPACE_OPS: readonly WorkspaceOp[] = [
  "launch_or_attach",
  "get_session",
  "load_descriptor",
  "get_viewer_state",
  "set_viewer_state",
  "get_selection",
  "select_segments",
  "fly_to",
  "add_layer",
  "add_annotations",
  "list_tables",
  "get_table_schema",
  "run_sql",
  "show_table",
  "show_plot",
  "save_session_state",
  "restore_session_state",
  "list_saved_states",
  "start_recording",
  "stop_recording",
  "add_narration_note",
  "export_session_summary",
] as const;

// ---------------------------------------------------------------------------
// Request / response envelopes
// ---------------------------------------------------------------------------

export interface WorkspaceRequest<P = unknown> {
  /** Correlates a response to its request across the bridge relay. */
  id: string;
  op: WorkspaceOp;
  params?: P;
  /** Which adapter issued this — used for action-history attribution. */
  source?: ActionSource;
}

export interface WorkspaceResponse<R = unknown> {
  id: string;
  ok: boolean;
  result?: R;
  error?: { message: string; stack?: string };
}

export type ActionSource = "mcp" | "python_sdk" | "local_api" | "internal";

// ---------------------------------------------------------------------------
// Session summary (get_session) — summary + references, NEVER giant blobs.
// ---------------------------------------------------------------------------

export interface SessionSummary {
  sessionId: string;
  mode: "workspace" | "chat";
  descriptor?: {
    id?: string;
    name?: string;
    source?: string;
  };
  viewer: {
    layers: Array<{
      name: string;
      type?: string;
      visible?: boolean;
    }>;
    selectedSegmentsByLayer: Record<string, string[]>;
    position?: number[];
  };
  tables: Array<{
    id: string;
    name: string;
    rowCount?: number;
    columns?: string[];
  }>;
  plots: Array<{
    id: string;
    title?: string;
    kind: string;
    sourceTable?: string;
  }>;
  savedStates: Array<{
    id: string;
    name?: string;
    createdAt: string;
  }>;
  recording: {
    active: boolean;
  };
}

// ---------------------------------------------------------------------------
// Annotations — minimal first version; richer NG schemas added later if needed.
// ---------------------------------------------------------------------------

export type WorkspaceAnnotation =
  | { type: "point"; position: number[]; label?: string }
  | { type: "line"; points: number[][]; label?: string }
  | { type: "bbox"; min: number[]; max: number[]; label?: string };

// ---------------------------------------------------------------------------
// Saved workspace state — agents should prefer these over raw viewer blobs.
// ---------------------------------------------------------------------------

export interface SavedTourguideState {
  id: string;
  name?: string;
  createdAt: string;
  viewerState: unknown;
  descriptorState?: unknown;
  tableIds?: string[];
  plotIds?: string[];
  annotations?: WorkspaceAnnotation[];
  timelineEventId?: string;
}

// ---------------------------------------------------------------------------
// Plot artifact — plots stay in Tourguide; agents call show_plot.
// ---------------------------------------------------------------------------

export interface PlotArtifact {
  id: string;
  title?: string;
  kind: "scatter" | "histogram" | "bar" | "line" | "custom";
  sourceTable?: string;
  spec: unknown;
  linkedSelection?: boolean;
  /** Rendered image (data URL) once Tourguide has drawn it. */
  pngDataUrl?: string;
}

// ---------------------------------------------------------------------------
// Action history — records operations that affect the workspace, NOT the
// agent's conversation. Streamed to the Agent Actions panel over WebSocket.
// ---------------------------------------------------------------------------

export interface ActionHistoryEntry {
  id: string;
  timestamp: string;
  source: ActionSource;
  action: string;
  argsSummary?: string;
  resultSummary?: string;
  artifactIds?: string[];
  error?: string;
  savedStateId?: string;
}

// ---------------------------------------------------------------------------
// Live WebSocket events (bridge -> agent, bridge -> browser as relevant).
// ---------------------------------------------------------------------------

export type WorkspaceEvent =
  | { type: "connection_status"; status: ConnectionStatus; sessionId?: string }
  | { type: "action"; entry: ActionHistoryEntry }
  | { type: "task_progress"; taskId: string; message: string; fraction?: number }
  | { type: "heartbeat"; at: string };

export type ConnectionStatus = "connected" | "disconnected" | "reconnecting";

// ---------------------------------------------------------------------------
// Launch/attach session bookkeeping (mirrored by the MCP launcher).
// ---------------------------------------------------------------------------

export interface WorkspaceSessionRecord {
  sessionId: string;
  createdAt: string;
  lastSeenAt: string;
  url: string;
  mode: "workspace" | "chat";
  status: "running" | "disconnected" | "crashed";
}
