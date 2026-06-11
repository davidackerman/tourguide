// Workspace API handlers — the app-level operations an external agent can
// drive. These run IN THE BROWSER (that's where the live viewer, DB and
// plots are). The bridge relays a WorkspaceRequest here, we execute against
// the live context, and return a plain-JSON result.
//
// Operations are intentionally semantic (select_segments, fly_to, …);
// set_viewer_state is the escape hatch for raw Neuroglancer blobs.

import type { BundledViewer } from "../bundled_viewer.js";
import type { DatasetDB } from "../db.js";
import type { DatasetDescriptor } from "../descriptor.js";
import type { LLMBackend } from "../llm.js";
import { runQuery, ingestTableIntoDB } from "../db.js";
import { renderPlotFromCode, runPlotQuery } from "../plot.js";
import { SessionStore } from "./session_state.js";
import type {
  PlotArtifact,
  SavedTourguideState,
  SessionSummary,
  WorkspaceAnnotation,
  WorkspaceOp,
} from "./protocol.js";

/** Everything the handlers need from the host app. main.ts supplies these. */
export interface WorkspaceContext {
  sessionId: string;
  mode: "workspace" | "chat";
  viewer: BundledViewer;
  store: SessionStore;
  getDB: () => DatasetDB | null;
  setDB: (db: DatasetDB) => void;
  getDescriptor: () => DatasetDescriptor | null;
  loadDescriptor: (d: DatasetDescriptor) => void;
  /** Re-render the structured table browser after a DB change. */
  refreshBrowser: () => void;
  /** Display a rendered plot artifact in the workspace UI. */
  displayPlot: (artifact: PlotArtifact) => void;
  /** Show a clickable share link in the workspace UI (avoids pasting a long
   *  URL through the agent's chat). */
  displayShareLink: (url: string, label?: string) => void;
  /** AI backend — only needed for the question-based show_plot path. */
  getBackend: () => LLMBackend;
}

export type HandlerMap = Record<WorkspaceOp, (params: any) => Promise<unknown>>;

const MAX_SQL_ROWS = 1000;

const firstSource = (source: string | string[] | undefined): string | undefined =>
  Array.isArray(source) ? source[0] : source;

const segmentationLayerNames = (ctx: WorkspaceContext): string[] => {
  const state = ctx.viewer.getNgState() as { layers?: Array<Record<string, unknown>> } | null;
  const layers = state?.layers ?? [];
  return layers
    .filter((l) => (l.type ?? "") === "segmentation")
    .map((l) => String(l.name ?? ""))
    .filter(Boolean);
};

const buildSessionSummary = (ctx: WorkspaceContext): SessionSummary => {
  const state = ctx.viewer.getNgState() as
    | { layers?: Array<Record<string, unknown>>; position?: number[] }
    | null;
  const ngLayers = state?.layers ?? [];
  const selectedSegmentsByLayer: Record<string, string[]> = {};
  for (const name of segmentationLayerNames(ctx)) {
    const seg = ctx.viewer.getVisibleSegments(name);
    if (seg.length > 0) selectedSegmentsByLayer[name] = seg;
  }
  const db = ctx.getDB();
  const descriptor = ctx.getDescriptor();
  // Index descriptor layers by name so we can hand the agent each layer's
  // data source URL + organelle class — it reads/computes on the raw data
  // itself; the workspace doesn't run the compute.
  const descLayers = new Map(
    (descriptor?.layers ?? []).map((l) => [l.name, l]),
  );
  return {
    sessionId: ctx.sessionId,
    mode: ctx.mode,
    descriptor: descriptor
      ? {
          id: descriptor.name,
          name: descriptor.display_name ?? descriptor.name,
          source: firstSource(descriptor.layers?.[0]?.source),
          voxelSizeNm: descriptor.voxel_size_nm,
        }
      : undefined,
    viewer: {
      layers: ngLayers.map((l) => {
        const name = String(l.name ?? "");
        const dl = descLayers.get(name);
        return {
          name,
          type: l.type ? String(l.type) : undefined,
          visible: l.visible === undefined ? true : Boolean(l.visible),
          source: dl ? firstSource(dl.source) : undefined,
          organelleClass: dl?.organelle_class,
        };
      }),
      selectedSegmentsByLayer,
      position: state?.position,
    },
    tables: (db?.tables ?? []).map((t) => ({
      id: t.table_name,
      name: t.organelle_class || t.table_name,
      rowCount: t.row_count,
      columns: t.columns,
    })),
    plots: ctx.store.plotSummaries(),
    savedStates: ctx.store.savedStateSummaries(),
    recording: { active: ctx.store.recordingState().active },
  };
};

const annotationsToPoints = (
  anns: WorkspaceAnnotation[],
): { pos: [number, number, number]; description?: string }[] => {
  const pts: { pos: [number, number, number]; description?: string }[] = [];
  for (const a of anns) {
    if (a.type === "point") {
      pts.push({ pos: a.position as [number, number, number], description: a.label });
    } else if (a.type === "line") {
      a.points.forEach((p, i) =>
        pts.push({ pos: p as [number, number, number], description: a.label ? `${a.label} [${i}]` : undefined }),
      );
    } else if (a.type === "bbox") {
      pts.push({ pos: a.min as [number, number, number], description: a.label ? `${a.label} min` : "min" });
      pts.push({ pos: a.max as [number, number, number], description: a.label ? `${a.label} max` : "max" });
    }
  }
  return pts;
};

const requireDB = (ctx: WorkspaceContext): DatasetDB => {
  const db = ctx.getDB();
  if (!db) throw new Error("No dataset DB loaded — load a dataset with tables first.");
  return db;
};

export function createHandlers(ctx: WorkspaceContext): HandlerMap {
  return {
    // launch_or_attach is normally answered by the bridge/launcher; if it
    // reaches the browser the session is already live, so just summarize.
    launch_or_attach: async () => ({ sessionId: ctx.sessionId, mode: ctx.mode, attached: true }),

    get_session: async () => buildSessionSummary(ctx),

    load_descriptor: async (p: { descriptor: DatasetDescriptor }) => {
      if (!p?.descriptor) throw new Error("load_descriptor: missing 'descriptor'");
      ctx.loadDescriptor(p.descriptor);
      return { name: p.descriptor.name };
    },

    get_viewer_state: async () => {
      const s = ctx.viewer.getNgState();
      if (!s) throw new Error("Viewer not mounted yet.");
      return s;
    },

    set_viewer_state: async (p: { state: Record<string, unknown> }) => {
      if (!p?.state) throw new Error("set_viewer_state: missing 'state'");
      ctx.viewer.applyNgState(p.state);
      return { ok: true };
    },

    show_share_link: async (p: { url: string; label?: string }) => {
      if (!p?.url) throw new Error("show_share_link: missing 'url'");
      ctx.displayShareLink(p.url, p.label);
      return { ok: true };
    },

    get_selection: async () => {
      const out: Record<string, string[]> = {};
      for (const name of segmentationLayerNames(ctx)) {
        out[name] = ctx.viewer.getVisibleSegments(name);
      }
      return { selectedSegmentsByLayer: out };
    },

    select_segments: async (p: { layer: string; segmentIds: string[] }) => {
      if (!p?.layer) throw new Error("select_segments: missing 'layer'");
      const ids = (p.segmentIds ?? []).map(String);
      ctx.viewer.highlightSegments(p.layer, ids);
      return { layer: p.layer, count: ids.length };
    },

    fly_to: async (p: { position: [number, number, number]; segmentId?: string; layer?: string }) => {
      if (!Array.isArray(p?.position) || p.position.length < 3) {
        throw new Error("fly_to: 'position' must be [x, y, z] in nm");
      }
      ctx.viewer.flyTo(
        [p.position[0], p.position[1], p.position[2]],
        p.segmentId !== undefined ? String(p.segmentId) : undefined,
        p.layer,
      );
      return { position: p.position };
    },

    add_layer: async (p: { layer: Record<string, unknown> }) => {
      if (!p?.layer?.name) throw new Error("add_layer: layer spec must include 'name'");
      ctx.viewer.addLayerFromSpec(p.layer);
      return { name: String(p.layer.name) };
    },

    add_annotations: async (p: { layerName?: string; annotations: WorkspaceAnnotation[] }) => {
      const anns = p?.annotations ?? [];
      if (anns.length === 0) throw new Error("add_annotations: 'annotations' is empty");
      const pts = annotationsToPoints(anns);
      ctx.viewer.addAnnotationLayer(p.layerName || "agent-annotations", pts);
      return { layerName: p.layerName || "agent-annotations", count: pts.length };
    },

    list_tables: async () => {
      const db = ctx.getDB();
      return {
        tables: (db?.tables ?? []).map((t) => ({
          id: t.table_name,
          name: t.organelle_class || t.table_name,
          rowCount: t.row_count,
          columns: t.columns,
        })),
      };
    },

    get_table_schema: async (p: { table: string }) => {
      if (!p?.table) throw new Error("get_table_schema: missing 'table'");
      const db = requireDB(ctx);
      const info = runQuery(db.db, `PRAGMA table_info("${p.table.replace(/"/g, '""')}");`);
      const nameIdx = info.columns.indexOf("name");
      const typeIdx = info.columns.indexOf("type");
      const columns = info.rows.map((r) => ({
        name: String(r[nameIdx]),
        type: typeIdx >= 0 ? String(r[typeIdx]) : "",
      }));
      if (columns.length === 0) throw new Error(`table not found: ${p.table}`);
      const meta = db.tables.find((t) => t.table_name === p.table);
      return { table: p.table, columns, rowCount: meta?.row_count };
    },

    run_sql: async (p: { sql: string }) => {
      if (!p?.sql) throw new Error("run_sql: missing 'sql'");
      const db = requireDB(ctx);
      const res = runQuery(db.db, p.sql);
      const truncated = res.rows.length > MAX_SQL_ROWS;
      return {
        columns: res.columns,
        rows: truncated ? res.rows.slice(0, MAX_SQL_ROWS) : res.rows,
        rowCount: res.rows.length,
        truncated,
      };
    },

    // Push a table the AGENT computed (in its own environment) into the
    // workspace. This is the core of the model: the agent owns compute, the
    // workspace displays the result. Shows in the structured browser with
    // click-to-fly. Creates the DB if the dataset had no tables yet.
    ingest_table: async (p: { name: string; columns: string[]; rows: unknown[][] }) => {
      if (!p?.name) throw new Error("ingest_table: missing 'name'");
      if (!Array.isArray(p.columns) || p.columns.length === 0) {
        throw new Error("ingest_table: 'columns' must be a non-empty string array");
      }
      if (!Array.isArray(p.rows)) throw new Error("ingest_table: 'rows' must be an array of rows");
      const name = p.name.replace(/[^a-zA-Z0-9_]/g, "_").toLowerCase();
      const rows = p.rows.map((r) =>
        (r ?? []).map((v) => (v === undefined || v === null ? null : (v as number | string))),
      );
      await ingestTableIntoDB(
        { getDB: ctx.getDB, setDB: ctx.setDB },
        { name, columns: p.columns, rows },
        (ctx.getDescriptor()?.layers ?? []).map((l) => l.name),
      );
      ctx.refreshBrowser();
      return { tableId: name, name, rowCount: rows.length, columns: p.columns };
    },

    show_table: async (p: { sql: string; name?: string }) => {
      if (!p?.sql) throw new Error("show_table: missing 'sql'");
      const db = requireDB(ctx);
      const res = runQuery(db.db, p.sql);
      const name = (p.name || "agent_result").replace(/[^a-zA-Z0-9_]/g, "_").toLowerCase();
      const rows = res.rows.map((r) =>
        r.map((v) => (v === undefined || v === null ? null : (v as number | string))),
      );
      await ingestTableIntoDB(
        { getDB: ctx.getDB, setDB: ctx.setDB },
        { name, columns: res.columns, rows },
        (ctx.getDescriptor()?.layers ?? []).map((l) => l.name),
      );
      ctx.refreshBrowser();
      return { tableId: name, name, rowCount: res.rows.length, columns: res.columns };
    },

    show_plot: async (p: {
      png?: string;
      code?: string;
      question?: string;
      title?: string;
      kind?: PlotArtifact["kind"];
      sourceTable?: string;
      linkedSelection?: boolean;
    }) => {
      let pngDataUrl: string;
      let usedCode = p.code ?? "";
      // Preferred path: the agent rendered the figure in its OWN environment
      // and hands us the image. No Pyodide, no AI backend — just display it.
      if (p.png) {
        pngDataUrl = p.png.startsWith("data:") ? p.png : `data:image/png;base64,${p.png}`;
      } else if (p.code) {
        const r = await renderPlotFromCode(p.code, requireDB(ctx));
        pngDataUrl = r.png_data_url;
      } else if (p.question) {
        const db = requireDB(ctx);
        const backend = ctx.getBackend();
        if (!backend.isReady()) {
          throw new Error("show_plot: 'question' path needs an AI backend; pass 'code' instead.");
        }
        const r = await runPlotQuery(p.question, db, backend);
        pngDataUrl = r.png_data_url;
        usedCode = r.code;
      } else {
        throw new Error(
          "show_plot: provide 'png' (an image the agent rendered — preferred), " +
            "'code' (matplotlib, runs in-browser), or 'question' (needs AI backend).",
        );
      }
      const artifact = ctx.store.addPlot({
        title: p.title,
        kind: p.kind ?? "custom",
        sourceTable: p.sourceTable,
        spec: { code: usedCode, question: p.question },
        linkedSelection: p.linkedSelection,
        pngDataUrl,
      });
      ctx.displayPlot(artifact);
      return { id: artifact.id, title: artifact.title, kind: artifact.kind, hasImage: true };
    },

    save_session_state: async (p: { name?: string; annotations?: WorkspaceAnnotation[] }) => {
      // Return the FULL serialized state: the bridge persists it to disk (the
      // browser sandbox can't write files). localStorage stays a local cache.
      return ctx.store.saveState(p?.name, p?.annotations);
    },

    restore_session_state: async (p: { id?: string; state?: SavedTourguideState }) => {
      // The bridge passes the full `state` when restoring from disk (works in a
      // fresh tab); otherwise fall back to a local lookup by id.
      if (p?.state) {
        const s = ctx.store.applyState(p.state);
        return { id: s.id, name: s.name };
      }
      if (!p?.id) throw new Error("restore_session_state: missing 'id'");
      const s = ctx.store.restoreState(p.id);
      return { id: s.id, name: s.name };
    },

    list_saved_states: async () => ({ savedStates: ctx.store.savedStateSummaries() }),

    start_recording: async () => ctx.store.startRecording(),

    stop_recording: async () => ctx.store.stopRecording(),

    add_narration_note: async (p: { text: string; position?: number[]; segmentId?: string }) => {
      if (!p?.text) throw new Error("add_narration_note: missing 'text'");
      return ctx.store.addNarrationNote(p.text, { position: p.position, segmentId: p.segmentId });
    },

    export_session_summary: async () => ctx.store.exportSummary(),
  };
}
