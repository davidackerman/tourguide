// Tourguide Workspace API — protocol contract.
//
// This is the DURABLE artifact. MCP is just the first adapter; the Python
// SDK and any future adapter speak the same operations defined here. Keep
// this file free of DOM / viewer / DB imports so it can be shared verbatim
// with non-browser code (it is mirrored by the Python SDK's schemas.py).
//
// Wire shape: JSON-RPC-ish request/response envelopes relayed between an
// external agent and the live browser session by the local bridge server.
// HTTP carries request/response operations; WebSocket (or GET /events
// polling) carries live streams: connection status, action history, and
// user events (selection / camera changes) so an agent can react to what
// the person at the screen is doing.

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
  | "wait_for_ready"
  | "screenshot"
  | "get_viewer_state"
  | "set_viewer_state"
  | "show_share_link"
  | "get_selection"
  | "select_segments"
  | "fly_to"
  | "fly_to_segment"
  | "add_layer"
  | "add_annotations"
  | "list_tables"
  | "get_table_schema"
  | "run_sql"
  | "ingest_table"
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
  "wait_for_ready",
  "screenshot",
  "get_viewer_state",
  "set_viewer_state",
  "show_share_link",
  "get_selection",
  "select_segments",
  "fly_to",
  "fly_to_segment",
  "add_layer",
  "add_annotations",
  "list_tables",
  "get_table_schema",
  "run_sql",
  "ingest_table",
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
  /** Pin the op to a specific workspace tab (bridge routing). */
  session?: string;
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
    /** Dataset voxel size in nm [x, y, z] — agents need this to read voxels. */
    voxelSizeNm?: number[];
  };
  viewer: {
    /** True once every layer has finished loading its current view. */
    ready: boolean;
    layers: Array<{
      name: string;
      type?: string;
      visible?: boolean;
      /** Data source URL (zarr/n5/precomputed) so the agent can read it
       *  directly in its own environment — the workspace is a sink/source,
       *  not a compute runtime. */
      source?: string;
      /** On-disk path for local-folder layers, when the descriptor declared
       *  one (`paths:` block or per-layer `local_path`). The browser-served
       *  `/local-data/` URL is not readable from outside the tab; this is. */
      localPath?: string;
      organelleClass?: string;
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
// Annotations — point / line / bbox map 1:1 onto Neuroglancer's native
// point / line / axis_aligned_bounding_box annotation types.
// ---------------------------------------------------------------------------

export type WorkspaceAnnotation =
  | { type: "point"; position: number[]; label?: string; id?: string }
  | { type: "line"; points: number[][]; label?: string; id?: string }
  | { type: "bbox"; min: number[]; max: number[]; label?: string; id?: string };

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
// Plot artifact — plots stay in Tourguide; agents call show_plot with a
// PNG they rendered themselves.
// ---------------------------------------------------------------------------

export interface PlotArtifact {
  id: string;
  title?: string;
  kind: "scatter" | "histogram" | "bar" | "line" | "custom";
  sourceTable?: string;
  spec: unknown;
  linkedSelection?: boolean;
  /** Rendered image (data URL). */
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
// Live events (browser -> bridge -> agents). The bridge stamps each with a
// monotonically increasing `seq` so HTTP pollers can resume with ?since=.
// ---------------------------------------------------------------------------

export type WorkspaceEvent =
  | { type: "connection_status"; status: ConnectionStatus; sessionId?: string }
  | { type: "action"; entry: ActionHistoryEntry }
  | { type: "task_progress"; taskId: string; message: string; fraction?: number }
  /** The person at the screen changed which segments are visible. */
  | { type: "selection_changed"; selectedSegmentsByLayer: Record<string, string[]> }
  /** The person at the screen moved the camera (debounced). */
  | { type: "position_changed"; position: number[] }
  /** A different dataset was loaded in the tab. */
  | { type: "dataset_changed"; name?: string }
  | { type: "heartbeat"; at?: string };

export type ConnectionStatus = "connected" | "disconnected" | "reconnecting";

// ---------------------------------------------------------------------------
// Launch/attach session bookkeeping (mirrored by the MCP launcher).
// ---------------------------------------------------------------------------

export interface WorkspaceSessionRecord {
  sessionId: string;
  label?: string;
  createdAt: string;
  lastSeenAt: string;
  url: string;
  mode: "workspace" | "chat";
  status: "running" | "disconnected" | "crashed";
  readOnly?: boolean;
}
