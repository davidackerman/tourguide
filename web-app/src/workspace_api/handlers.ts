// Workspace API handlers — the app-level operations an external agent can
// drive. These run IN THE BROWSER (that's where the live viewer, DB and
// plots are). The bridge relays a WorkspaceRequest here, we execute against
// the live context, and return a plain-JSON result.
//
// Operations are intentionally semantic (select_segments, fly_to, …);
// set_viewer_state is the escape hatch for raw Neuroglancer blobs.
//
// Principle: the agent owns compute, Tourguide owns visual state. Nothing in
// here runs Python or calls an LLM — show_plot takes a PNG the agent
// rendered, ingest_table takes rows the agent computed.

import type { BundledViewer } from "../bundled_viewer.js";
import type { DatasetDB } from "../db.js";
import type { DatasetDescriptor } from "../descriptor.js";
import { runQuery, ingestTableIntoDB } from "../db.js";
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
}

export type HandlerMap = Record<WorkspaceOp, (params: any) => Promise<unknown>>;

const MAX_SQL_ROWS = 1000;

const firstSource = (source: string | string[] | undefined): string | undefined =>
  Array.isArray(source) ? source[0] : source;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Only SELECT-shaped statements go through the query ops. Anything that
 *  writes (INSERT/UPDATE/DROP/…) or chains several statements is refused,
 *  so the agent can't mutate tables through what the tool surface calls
 *  a read-only query. ingest_table is the sanctioned write path. */
const assertReadOnlySql = (sql: string): void => {
  const stripped = sql
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/--[^\n]*/g, " ")
    .trim()
    .replace(/;+\s*$/, "");
  if (stripped.includes(";")) {
    throw new Error("run_sql: one statement at a time");
  }
  const first = (/^\s*(\w+)/.exec(stripped)?.[1] ?? "").toUpperCase();
  if (!["SELECT", "WITH", "EXPLAIN", "PRAGMA", "VALUES"].includes(first)) {
    throw new Error(
      `run_sql is read-only (got '${first || "?"}'). Use ingest_table to add or replace data.`,
    );
  }
  if (first === "PRAGMA" && !/^\s*PRAGMA\s+table_info/i.test(stripped)) {
    throw new Error("run_sql: only PRAGMA table_info is permitted");
  }
};

const quoteIdent = (s: string): string => `"${s.replace(/"/g, '""')}"`;

const POSITION_COLUMNS: Array<[string, string, string]> = [
  ["com_x_nm", "com_y_nm", "com_z_nm"],
  ["position_x_nm", "position_y_nm", "position_z_nm"],
  ["position_x", "position_y", "position_z"],
  ["com_x", "com_y", "com_z"],
];

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
      ready: ctx.viewer.isReady(),
      layers: ngLayers.map((l) => {
        const name = String(l.name ?? "");
        const dl = descLayers.get(name);
        return {
          name,
          type: l.type ? String(l.type) : undefined,
          visible: l.visible === undefined ? true : Boolean(l.visible),
          source: dl ? firstSource(dl.source) : undefined,
          localPath: dl?.local_path,
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

const requireDB = (ctx: WorkspaceContext): DatasetDB => {
  const db = ctx.getDB();
  if (!db) throw new Error("No dataset DB loaded — ingest a table first.");
  return db;
};

/** Pick the table that describes a layer: explicit name, else the table
 *  whose layer_name / table_name / organelle_class matches. */
const tableForLayer = (ctx: WorkspaceContext, layer: string, table?: string) => {
  const db = requireDB(ctx);
  if (table) {
    const t = db.tables.find((x) => x.table_name === table);
    if (!t) throw new Error(`table not found: ${table}`);
    return { db, table: t };
  }
  const cls = ctx.getDescriptor()?.layers.find((l) => l.name === layer)?.organelle_class;
  const t =
    db.tables.find((x) => x.layer_name === layer) ??
    db.tables.find((x) => x.table_name === layer) ??
    (cls ? db.tables.find((x) => x.organelle_class === cls) : undefined);
  if (!t) {
    throw new Error(
      `no table for layer '${layer}' (have: ${db.tables.map((x) => x.table_name).join(", ") || "none"}); pass 'table'`,
    );
  }
  return { db, table: t };
};

export function createHandlers(ctx: WorkspaceContext): HandlerMap {
  return {
    // launch_or_attach is normally answered by the bridge/launcher; if it
    // reaches the browser the session is already live, so just summarize.
    launch_or_attach: async () => ({ sessionId: ctx.sessionId, mode: ctx.mode, attached: true }),

    get_session: async () => buildSessionSummary(ctx),

    load_descriptor: async (p: { descriptor: DatasetDescriptor; wait?: boolean; timeoutMs?: number }) => {
      if (!p?.descriptor) throw new Error("load_descriptor: missing 'descriptor'");
      ctx.loadDescriptor(p.descriptor);
      let ready = false;
      if (p.wait !== false) {
        const deadline = Date.now() + (p.timeoutMs ?? 30_000);
        await sleep(250); // let NG mount the layers before polling readiness
        while (Date.now() < deadline) {
          if (ctx.viewer.isReady()) {
            ready = true;
            break;
          }
          await sleep(250);
        }
      }
      return { name: p.descriptor.name, ready };
    },

    // Block until every layer has loaded the chunks for the current view,
    // or the timeout passes. Lets an agent take a screenshot / read state
    // right after fly_to or load_descriptor without racing the loader.
    wait_for_ready: async (p: { timeoutMs?: number }) => {
      const deadline = Date.now() + (p?.timeoutMs ?? 30_000);
      while (Date.now() < deadline) {
        if (ctx.viewer.isReady()) return { ready: true };
        await sleep(200);
      }
      return { ready: false, timedOut: true };
    },

    // The agent's eyes: a PNG of the current view. Waits for the view to
    // finish loading first (bounded) so the image isn't half-drawn.
    screenshot: async (p: { waitForReadyMs?: number; maxWidth?: number }) => {
      const wait = p?.waitForReadyMs ?? 10_000;
      const deadline = Date.now() + wait;
      while (wait > 0 && Date.now() < deadline && !ctx.viewer.isReady()) await sleep(150);
      const shot = await ctx.viewer.screenshot(p?.maxWidth);
      const position = (ctx.viewer.getNgState() as { position?: number[] } | null)?.position;
      return { ...shot, ready: ctx.viewer.isReady(), position };
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

    // Fly to an object by id, looking its centroid up in the layer's table
    // (com_x_nm/com_y_nm/com_z_nm or the legacy position_* variants).
    fly_to_segment: async (p: { layer: string; segmentId: string | number; table?: string; select?: boolean }) => {
      if (!p?.layer) throw new Error("fly_to_segment: missing 'layer'");
      if (p.segmentId === undefined || p.segmentId === null) throw new Error("fly_to_segment: missing 'segmentId'");
      const { db, table } = tableForLayer(ctx, p.layer, p.table);
      const cols = POSITION_COLUMNS.find((c) => c.every((k) => table.columns.includes(k)));
      if (!cols) {
        throw new Error(`table '${table.table_name}' has no position columns (need com_x_nm/com_y_nm/com_z_nm)`);
      }
      const res = runQuery(
        db.db,
        `SELECT ${cols.map(quoteIdent).join(", ")} FROM ${quoteIdent(table.table_name)} WHERE object_id = ${Number(p.segmentId)} LIMIT 1`,
      );
      const row = res.rows[0];
      if (!row) throw new Error(`object_id ${p.segmentId} not found in '${table.table_name}'`);
      const position = [Number(row[0]), Number(row[1]), Number(row[2])] as [number, number, number];
      if (position.some((v) => !Number.isFinite(v))) {
        throw new Error(`object_id ${p.segmentId} has a non-numeric position in '${table.table_name}'`);
      }
      const id = String(p.segmentId);
      ctx.viewer.flyTo(position, id, p.layer);
      if (p.select !== false) ctx.viewer.highlightSegments(p.layer, [id]);
      return { layer: p.layer, segmentId: id, position, table: table.table_name };
    },

    add_layer: async (p: { layer: Record<string, unknown> }) => {
      if (!p?.layer?.name) throw new Error("add_layer: layer spec must include 'name'");
      ctx.viewer.addLayerFromSpec(p.layer);
      return { name: String(p.layer.name) };
    },

    add_annotations: async (p: { layerName?: string; annotations: WorkspaceAnnotation[]; replace?: boolean }) => {
      const anns = p?.annotations ?? [];
      if (anns.length === 0) throw new Error("add_annotations: 'annotations' is empty");
      for (const a of anns) {
        if (a.type === "point" && (!Array.isArray(a.position) || a.position.length < 3)) {
          throw new Error("add_annotations: point needs 'position' [x,y,z]");
        }
        if (a.type === "line" && (!Array.isArray(a.points) || a.points.length !== 2)) {
          throw new Error("add_annotations: line needs exactly two 'points' (use several lines for a polyline)");
        }
        if (a.type === "bbox" && (!Array.isArray(a.min) || !Array.isArray(a.max))) {
          throw new Error("add_annotations: bbox needs 'min' and 'max' [x,y,z]");
        }
      }
      const layerName = p.layerName || "agent-annotations";
      const count = ctx.viewer.addWorkspaceAnnotations(layerName, anns, p.replace === true);
      return { layerName, count };
    },

    list_tables: async () => {
      const db = ctx.getDB();
      return {
        tables: (db?.tables ?? []).map((t) => ({
          id: t.table_name,
          name: t.organelle_class || t.table_name,
          layer: t.layer_name,
          rowCount: t.row_count,
          columns: t.columns,
        })),
      };
    },

    get_table_schema: async (p: { table: string }) => {
      if (!p?.table) throw new Error("get_table_schema: missing 'table'");
      const db = requireDB(ctx);
      const info = runQuery(db.db, `PRAGMA table_info(${quoteIdent(p.table)});`);
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
      assertReadOnlySql(p.sql);
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
      assertReadOnlySql(p.sql);
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

    // Display a figure the agent rendered in ITS environment. No Pyodide,
    // no LLM — just show the image and keep it as a session artifact.
    show_plot: async (p: {
      png: string;
      title?: string;
      kind?: PlotArtifact["kind"];
      sourceTable?: string;
      linkedSelection?: boolean;
    }) => {
      if (!p?.png || typeof p.png !== "string") {
        throw new Error("show_plot: 'png' (base64 PNG or data URL, rendered by the agent) is required");
      }
      const pngDataUrl = p.png.startsWith("data:") ? p.png : `data:image/png;base64,${p.png}`;
      const artifact = ctx.store.addPlot({
        title: p.title,
        kind: p.kind ?? "custom",
        sourceTable: p.sourceTable,
        spec: {},
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
