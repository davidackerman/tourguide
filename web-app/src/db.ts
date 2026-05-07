import initSqlJs, { type Database, type SqlJsStatic } from "sql.js";
import sqlWasmUrl from "sql.js/dist/sql-wasm.wasm?url";
import Papa from "papaparse";
import type { DatasetDescriptor, DatasetLayer } from "./descriptor.js";

let sqlPromise: Promise<SqlJsStatic> | null = null;

export function loadSqlJs(): Promise<SqlJsStatic> {
  if (!sqlPromise) {
    sqlPromise = initSqlJs({ locateFile: () => sqlWasmUrl });
  }
  return sqlPromise;
}

export interface OrganelleRow {
  object_id: number | string;
  volume_nm_3?: number;
  surface_area_nm_2?: number;
  position_x_nm?: number;
  position_y_nm?: number;
  position_z_nm?: number;
  [extra: string]: unknown;
}

export interface IngestedTable {
  table_name: string;
  organelle_class: string;
  layer_name: string;
  row_count: number;
  columns: string[];
}

export interface DatasetDB {
  db: Database;
  tables: IngestedTable[];
}

// Canonical column names. Units explicit in the name so an agent
// reading the schema doesn't have to guess (cellmap-analyze CSVs use
// these conventions; tourguide-produced regionprops outputs match).
const STANDARD_NUMERIC_COLUMNS = [
  "volume_nm_3",
  "surface_area_nm_2",
  "position_x_nm",
  "position_y_nm",
  "position_z_nm",
];

// Maps every common variant of a column name to one of the canonical
// names above. Applied at CSV-ingest time so all tables — whether
// produced by cellmap-analyze, tourguide's own Σ Analyze, or hand-
// authored — converge on a single schema.
const COLUMN_ALIASES: Record<string, string> = {
  // Object id
  id: "object_id",
  obj_id: "object_id",
  object: "object_id",
  segment_id: "object_id",
  label: "object_id",
  // Position / centroid / center-of-mass — every flavor → position_*_nm
  position_x: "position_x_nm",
  position_y: "position_y_nm",
  position_z: "position_z_nm",
  com_x: "position_x_nm",
  com_y: "position_y_nm",
  com_z: "position_z_nm",
  com_x_nm: "position_x_nm",
  com_y_nm: "position_y_nm",
  com_z_nm: "position_z_nm",
  centroid_x: "position_x_nm",
  centroid_y: "position_y_nm",
  centroid_z: "position_z_nm",
  centroid_x_nm: "position_x_nm",
  centroid_y_nm: "position_y_nm",
  centroid_z_nm: "position_z_nm",
  x: "position_x_nm",
  y: "position_y_nm",
  z: "position_z_nm",
  // Volume → volume_nm_3 (cellmap-analyze writes 'volume_(nm^3)' which
  // the special-char sanitizer collapses to 'volume_nm_3' first; this
  // pass catches anything else)
  volume: "volume_nm_3",
  size: "volume_nm_3",
  vol: "volume_nm_3",
  // Surface area → surface_area_nm_2
  surface_area: "surface_area_nm_2",
  surface_area_nm_2_: "surface_area_nm_2",
  surface: "surface_area_nm_2",
  sa: "surface_area_nm_2",
};

function safeTableName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_]/g, "_").toLowerCase();
}

// Sanitize CSV headers to SQL-safe identifiers. cellmap-analyze emits
// columns like 'volume_(nm^3)', 'surface_area_(nm^2)', 'com_x_(nm)' —
// the parens and '^' are tokens to SQLite, so any unquoted reference
// errors with 'unrecognized token: "^"'. Gemini reliably quotes
// identifiers; WebLLM small models don't, and the resulting failures
// are unrecoverable. Easier to make the columns not need quoting at
// all: collapse anything non-alphanumeric to '_', squash repeats,
// trim edges. 'volume_(nm^3)' -> 'volume_nm_3'.
function normalizeHeader(h: string): string {
  const k = h
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "");
  return COLUMN_ALIASES[k] ?? k;
}

async function fetchCsv(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch CSV ${url}: HTTP ${res.status}`);
  return res.text();
}

function parseCsvText(text: string, normalizeColumns: boolean): {
  columns: string[];
  rows: Record<string, unknown>[];
} {
  const result = Papa.parse<Record<string, unknown>>(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    transformHeader: normalizeColumns ? normalizeHeader : (h) => h,
  });
  if (result.errors.length > 0) {
    console.warn("CSV parse warnings:", result.errors.slice(0, 3));
  }
  const columns = result.meta.fields ?? [];
  return { columns, rows: result.data };
}

function inferColumnTypes(
  columns: string[],
  rows: Record<string, unknown>[],
): Record<string, "INTEGER" | "REAL" | "TEXT"> {
  const out: Record<string, "INTEGER" | "REAL" | "TEXT"> = {};
  for (const col of columns) {
    // object_id can land as either 'object_id' (preferred) or 'label'
    // (skimage regionprops convention before COLUMN_ALIASES rewrites
    // it). Treat both as integer up-front.
    if (col === "object_id") {
      out[col] = "INTEGER";
      continue;
    }
    if (STANDARD_NUMERIC_COLUMNS.includes(col)) {
      out[col] = "REAL";
      continue;
    }
    let allInt = true;
    let allNum = true;
    let nonNullSeen = false;
    for (const row of rows.slice(0, 200)) {
      const v = row[col];
      if (v === null || v === undefined || v === "") continue;
      nonNullSeen = true;
      if (typeof v !== "number") {
        allInt = false;
        allNum = false;
      } else if (!Number.isInteger(v)) {
        allInt = false;
      }
    }
    if (!nonNullSeen) out[col] = "TEXT";
    else if (allInt) out[col] = "INTEGER";
    else if (allNum) out[col] = "REAL";
    else out[col] = "TEXT";
  }
  return out;
}

function createTable(
  db: Database,
  tableName: string,
  types: Record<string, "INTEGER" | "REAL" | "TEXT">,
): void {
  const cols = Object.entries(types)
    .map(([name, t]) => `"${name}" ${t}`)
    .join(", ");
  db.run(`DROP TABLE IF EXISTS "${tableName}";`);
  db.run(`CREATE TABLE "${tableName}" (${cols});`);
}

function insertRows(
  db: Database,
  tableName: string,
  columns: string[],
  rows: Record<string, unknown>[],
): void {
  const placeholders = columns.map(() => "?").join(", ");
  const colList = columns.map((c) => `"${c}"`).join(", ");
  const stmt = db.prepare(
    `INSERT INTO "${tableName}" (${colList}) VALUES (${placeholders});`,
  );
  db.run("BEGIN;");
  try {
    for (const row of rows) {
      const values = columns.map((c) => {
        const v = row[c];
        if (v === undefined || v === null || v === "") return null;
        if (typeof v === "boolean") return v ? 1 : 0;
        return v as number | string;
      });
      stmt.run(values);
    }
    db.run("COMMIT;");
  } catch (e) {
    db.run("ROLLBACK;");
    throw e;
  } finally {
    stmt.free();
  }
}

async function ingestLayer(
  db: Database,
  layer: DatasetLayer,
  baseUrl: string | null,
): Promise<IngestedTable | null> {
  if (!layer.csv) return null;
  const csvUrl = baseUrl ? new URL(layer.csv, baseUrl).toString() : layer.csv;
  const text = await fetchCsv(csvUrl);
  const { columns, rows } = parseCsvText(text, true);
  if (rows.length === 0) {
    console.warn(`CSV for layer ${layer.name} had no rows`);
    return null;
  }
  const types = inferColumnTypes(columns, rows);
  const organelleClass = layer.organelle_class ?? layer.name;
  const tableName = safeTableName(organelleClass);
  createTable(db, tableName, types);
  insertRows(db, tableName, columns, rows);
  return {
    table_name: tableName,
    organelle_class: organelleClass,
    layer_name: layer.name,
    row_count: rows.length,
    columns,
  };
}

export async function ingestDescriptor(
  d: DatasetDescriptor,
  catalogBaseUrl: string | null,
): Promise<DatasetDB> {
  const SQL = await loadSqlJs();
  const db = new SQL.Database();
  const tables: IngestedTable[] = [];
  for (const layer of d.layers) {
    if (!layer.csv) continue;
    try {
      const t = await ingestLayer(db, layer, catalogBaseUrl);
      if (t) tables.push(t);
    } catch (err) {
      console.error(`Failed to ingest CSV for layer ${layer.name}:`, err);
    }
  }
  return { db, tables };
}

export interface QueryResult {
  columns: string[];
  rows: unknown[][];
}

export function runQuery(db: Database, sql: string): QueryResult {
  const stmt = db.prepare(sql);
  try {
    const rows: unknown[][] = [];
    while (stmt.step()) {
      rows.push(stmt.get());
    }
    return { columns: stmt.getColumnNames(), rows };
  } finally {
    stmt.free();
  }
}
