import type { LLMBackend, LLMMessage } from "./llm.js";
import { WebLLMBackend, GeminiBackend } from "./llm.js";
import type { DatasetDB } from "./db.js";
import { runQuery } from "./db.js";
import type { BundledViewer } from "./bundled_viewer.js";
import { loadPyodide } from "./plot.js";
import type { DatasetDescriptor } from "./descriptor.js";

export interface AgentTraceItem {
  tool: string;
  args: Record<string, unknown>;
  result?: unknown;
  error?: string;
}

export interface AgentCallbacks {
  onTrace?: (item: AgentTraceItem) => void;
  onProgress?: (msg: string) => void;
  onAnswer?: (text: string) => void;
  onPlot?: (pngDataUrl: string, code: string, title?: string, explanation?: string) => void;
  onFly?: (position: [number, number, number], layer: string, objectId?: string) => void;
  onHighlight?: (layer: string, ids: string[]) => void;
}

export interface AgentContext {
  db: DatasetDB | null;
  descriptor: DatasetDescriptor | null;
  viewer: BundledViewer;
  backend: LLMBackend;
  callbacks: AgentCallbacks;
}

interface ToolCall {
  tool: string;
  args?: Record<string, unknown>;
}

// Cap the agent loop at 5 steps. Most legitimate flows finish in 2-4
// (run_sql -> fly_to -> done; run_python -> answer -> done). A higher
// cap mostly lets a confused model burn through your Gemini free-tier
// quota chasing its tail. If a real query needs more, the user can ask
// a follow-up.
const MAX_ITERATIONS = 5;

const SCHEMA_GUIDE = (db: DatasetDB): string => {
  const tableLines = db.tables
    .map((t) => {
      const cols = t.columns.map((c) => `"${c}"`).join(", ");
      return `  TABLE "${t.table_name}"  (class: ${t.organelle_class}, layer: ${t.layer_name}, rows: ${t.row_count})\n    columns: ${cols}`;
    })
    .join("\n");
  const dataframes = db.tables
    .map((t) => `  df_${t.organelle_class}  columns: ${t.columns.join(", ")}  rows: ${t.row_count}`)
    .join("\n");
  return `SQL tables:\n${tableLines}\n\nPython DataFrames (available to make_plot):\n${dataframes}`;
};

const SYSTEM_PROMPT = (db: DatasetDB | null, d: DatasetDescriptor | null): string => `
You are the agent for a 3D microscopy viewer. You respond to user questions by calling one tool at a time.

${d ? `DATASET: ${d.display_name}  (voxel size: ${d.voxel_size_nm.join(" × ")} nm)` : ""}
COORDINATES: ALL positions everywhere — CSV position_x/y/z columns, fly_to inputs, the viewer's display — are in NANOMETERS. Pass nm values directly; do NOT convert to voxels.

${db ? SCHEMA_GUIDE(db) : "No organelle database is loaded — only answer() and done() are useful."}

ON EACH TURN, respond with a single JSON object describing one tool call:

  {"tool": "<name>", "args": { ... }}

NO prose, NO markdown fences — only the JSON object.

TOOLS:

  run_sql(sql: string)
    Run a SELECT query against the organelle DB. Returns up to 50 rows.
    SQL dialect: SQLite. Quote identifiers with double quotes. SELECT only.

  fly_to(position: [x, y, z], layer: string, object_id?: string)
    Move the viewer camera to these NANOMETER coordinates, switch on this
    layer, and highlight object_id if given. Use position_x/position_y/
    position_z values from run_sql results AS-IS — no scaling.

  highlight_segments(layer: string, ids: number[] | string[])
    In a SEGMENTATION layer, show only these segment ids — everything
    else fades to background. Use this for "show me only the …" or
    "select these segments" requests. ids can be the object_id values
    from a run_sql result, or specific numeric ids the user named.

  make_plot(python: string, title?: string, explanation?: string)
    Run Python via Pyodide (numpy, pandas, matplotlib already imported).
    You get df_<class> DataFrames as globals. Produce exactly ONE matplotlib
    figure; do NOT call plt.show() or plt.savefig() — the harness handles it.
    Volumes in nm^3, surface area in nm^2, positions in nm; convert to um^3
    (divide by 1e9) when plotting for readability.

  run_python(python: string)
    Run arbitrary Python via Pyodide for computation that doesn't need a
    plot — e.g. statistics, transformations, intermediate results to feed
    a later tool. Same environment as make_plot: df_<class> DataFrames as
    globals, plus np / pd already imported. NOT terminal; the harness
    feeds the captured stdout (and any value you assign to a global named
    _out) back to you so you can call answer/fly_to next.
    Conventions:
      - print(...) anything you want returned; it lands in stdout.
      - Set _out = <python value> for a structured return — DataFrames /
        Series get to_dict()'d, otherwise json/repr.
      - Globals persist across run_python calls within one user turn.
    Use this for "what's the median volume?" / "compute X then fly to
    the result" flows. For visual answers, prefer make_plot.

  answer(text: string)
    Deliver a final text answer to the user. Use for counts, means,
    informational questions. Keep it concise.

  done()
    End this turn. You MUST call this after delivering any answer / plot / fly_to.

CRITICAL — fly_to expects POSITION coordinates (com_x_nm, com_y_nm, com_z_nm
or position_x, position_y, position_z), NOT a volume / surface_area / id.
ALWAYS include position columns in your SQL when you plan to fly_to a
result. Pass [com_x, com_y, com_z] from the row as position. If the
schema's position columns are named differently, use whatever the schema
shows — never invent values.

TYPICAL FLOWS:
  "show me the largest mito"       -> run_sql (SELECT object_id, com_x_nm, com_y_nm, com_z_nm, volume_nm_3 FROM mito ORDER BY volume_nm_3 DESC LIMIT 1)
                                      -> fly_to (position = [com_x, com_y, com_z], object_id from row) -> done
  "how many nuclei are there?"     -> run_sql (COUNT) -> answer -> done
  "plot mitochondrion volumes"     -> make_plot -> done
  "fly through the 3 biggest mitos" -> run_sql (SELECT object_id, com_x_nm, com_y_nm, com_z_nm, volume_nm_3 FROM mito ORDER BY volume_nm_3 DESC LIMIT 3)
                                      -> fly_to (row 1) -> fly_to (row 2) -> fly_to (row 3) -> done
  "show only the largest 10 mitos" -> run_sql (SELECT object_id FROM mito ORDER BY volume_nm_3 DESC LIMIT 10) -> highlight_segments -> done
  "densest region of mitos"        -> run_python (grid COMs, find max-count bin, set _out = (com, ids))
                                      -> fly_to (bin center) -> highlight_segments (ids in bin) -> done
  "median volume of nuclei"        -> run_python (df_nucleus["volume_(nm^3)"].median(); print) -> answer -> done
  "two closest mito pairs"         -> run_python (pdist over com_*, argmin; print pairs)       -> answer -> done

WHEN TO USE WHICH TOOL — IMPORTANT:
  - Filter / sort / count / "the N biggest/smallest"           -> run_sql
  - Distribution shape, scatter, histogram, "show on a chart"  -> make_plot
  - Anything spatial (density, clusters, neighbors, distance,
    "where are most of the X", "region", "near", bounding box,
    convex hull) or anything statistical beyond simple aggregates
    (median, percentile, std, IQR, correlation, regression)    -> run_python
  - Never reach for run_sql when the question implies geometric
    reasoning over com_x/com_y/com_z — SQL can't compute density
    or pairwise distances. ORDER BY volume DESC LIMIT 1 is NOT
    "the densest mito".

BUDGET: You have AT MOST 5 tool calls per question. Plan accordingly — most flows above finish in 2-4. If you can't reach an answer in 5, end with answer() explaining what's missing rather than running out silently.

Pick the minimum tool calls needed. Never repeat the same query. If a tool errors, read the error and adjust.
`.trim();

function extractJson(text: string): string {
  const trimmed = text.trim();
  if (trimmed.startsWith("{")) return trimmed;
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced) return fenced[1].trim();
  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  if (start >= 0 && end > start) return trimmed.slice(start, end + 1);
  throw new Error(`No JSON object found in: ${trimmed.slice(0, 200)}`);
}

const DONE_PATTERNS = [
  /^done\s*$/i,
  /^done\s*\(\s*\)\s*$/i,
  /^"?done"?\s*$/i,
  /^\s*\{?\s*"?tool"?\s*:\s*"?done"?\s*\}?\s*$/i,
];

function parseToolCall(raw: string): ToolCall {
  const trimmed = raw.trim();
  if (DONE_PATTERNS.some((p) => p.test(trimmed))) {
    return { tool: "done", args: {} };
  }
  const json = extractJson(raw);
  const parsed = JSON.parse(json) as ToolCall;
  if (!parsed.tool) throw new Error("Tool call missing 'tool' field");
  if (typeof parsed.tool === "string" && parsed.tool.trim().toLowerCase() === "done") {
    return { tool: "done", args: {} };
  }
  parsed.args = parsed.args ?? {};
  return parsed;
}

const TERMINAL_TOOLS = new Set(["answer", "make_plot"]);
// Tools that produce something the user actually sees — viewer
// changes count, since the camera move IS the answer to "show me X".
// Used to decide whether `done` is acceptable (vs. an empty-trace bail).
const DELIVERED_BY_TOOL = new Set(["answer", "make_plot", "fly_to", "highlight_segments"]);

function guardSqlReadOnly(sql: string): void {
  const banned = /\b(create|drop|alter|insert|update|delete|attach|pragma)\b/i;
  if (banned.test(sql)) throw new Error(`SQL must be SELECT only`);
}

async function execRunSql(
  args: Record<string, unknown>,
  ctx: AgentContext,
): Promise<{ columns: string[]; rows: unknown[][] }> {
  if (!ctx.db) throw new Error("No database loaded");
  const sql = String(args.sql ?? "").trim();
  if (!sql) throw new Error("run_sql requires 'sql'");
  guardSqlReadOnly(sql);
  const limitedSql = /\blimit\b/i.test(sql) ? sql : `${sql.replace(/;$/, "")} LIMIT 50`;
  const result = runQuery(ctx.db.db, limitedSql);
  return { columns: result.columns, rows: result.rows };
}

// Sanity bound for fly_to positions. EM volumes top out at ~1 mm
// (1e6 nm) per axis — anything above this is the model confusing a
// volume / surface_area / object_id value for a coordinate. Reject
// loudly so the model can self-correct (typical mistake: SELECT'd
// volume but not com_x/y/z, then put volume into position[0]).
const FLY_TO_MAX_NM = 1e7; // 10 mm — generous upper bound

async function execFlyTo(
  args: Record<string, unknown>,
  ctx: AgentContext,
): Promise<{ ok: boolean }> {
  const pos = args.position as unknown;
  if (!Array.isArray(pos) || pos.length !== 3 || pos.some((n) => typeof n !== "number")) {
    throw new Error("fly_to requires position: [x, y, z] numbers");
  }
  // Reject absurd magnitudes — these come from the model substituting
  // a volume / surface_area / object_id into a position slot. A 12-
  // billion-nm 'x' is 12 meters; obviously not a dataset coord.
  for (let i = 0; i < 3; i++) {
    const v = (pos as number[])[i];
    if (!Number.isFinite(v) || Math.abs(v) > FLY_TO_MAX_NM) {
      throw new Error(
        `fly_to position[${i}] = ${v} nm is out of dataset range. Positions must come from com_x_nm / com_y_nm / com_z_nm columns of the SQL result, in nanometers. SELECT those columns alongside object_id (e.g. 'SELECT object_id, com_x_nm, com_y_nm, com_z_nm FROM mito ORDER BY volume_nm_3 DESC LIMIT 1'), then pass the COM values as position. Do NOT pass volume / surface_area / object_id as position.`,
      );
    }
  }
  const layer = String(args.layer ?? "").trim();
  if (!layer) throw new Error("fly_to requires 'layer'");
  const objectId = args.object_id !== undefined ? String(args.object_id) : undefined;
  ctx.viewer.flyTo(pos as [number, number, number], objectId, layer);
  ctx.callbacks.onFly?.(pos as [number, number, number], layer, objectId);
  return { ok: true };
}

async function execHighlight(
  args: Record<string, unknown>,
  ctx: AgentContext,
): Promise<{ ok: boolean; count: number }> {
  const layer = String(args.layer ?? "").trim();
  if (!layer) throw new Error("highlight_segments requires 'layer'");
  const idsRaw = args.ids;
  if (!Array.isArray(idsRaw) || idsRaw.length === 0) {
    throw new Error("highlight_segments requires non-empty 'ids' array");
  }
  const ids = idsRaw.map((v) => String(v));
  ctx.viewer.highlightSegments(layer, ids);
  ctx.callbacks.onHighlight?.(layer, ids);
  return { ok: true, count: ids.length };
}

async function execMakePlot(
  args: Record<string, unknown>,
  ctx: AgentContext,
): Promise<{ rendered: boolean }> {
  if (!ctx.db) throw new Error("No database loaded");
  const code = String(args.python ?? args.code ?? "").trim();
  if (!code) throw new Error("make_plot requires 'python'");
  const title = args.title !== undefined ? String(args.title) : undefined;
  const explanation = args.explanation !== undefined ? String(args.explanation) : undefined;

  ctx.callbacks.onProgress?.("Loading Pyodide…");
  const py = await loadPyodide((m) => ctx.callbacks.onProgress?.(m));
  py.globals.set("_tables_json", tablesToJson(ctx.db));
  await py.runPythonAsync(SETUP_PY);
  await py.runPythonAsync(`
_dfs = _materialize_tables(_tables_json)
for _k, _v in _dfs.items():
    globals()[_k] = _v
import numpy as np
`);
  ctx.callbacks.onProgress?.("Rendering plot…");
  await py.runPythonAsync(`
plt.close("all")
${code}

_buf = io.BytesIO()
plt.savefig(_buf, format="png", bbox_inches="tight", dpi=120)
plt.close("all")
_PLOT_PNG = base64.b64encode(_buf.getvalue()).decode("ascii")
`);
  const b64 = py.globals.get("_PLOT_PNG") as string;
  if (!b64) throw new Error("Plot produced no figure");
  const pngDataUrl = `data:image/png;base64,${b64}`;
  ctx.callbacks.onPlot?.(pngDataUrl, code, title, explanation);
  return { rendered: true };
}

async function execAnswer(
  args: Record<string, unknown>,
  ctx: AgentContext,
): Promise<{ delivered: boolean }> {
  const text = String(args.text ?? "").trim();
  if (!text) throw new Error("answer requires 'text'");
  ctx.callbacks.onAnswer?.(text);
  return { delivered: true };
}

async function execRunPython(
  args: Record<string, unknown>,
  ctx: AgentContext,
): Promise<{ stdout: string; result?: string }> {
  if (!ctx.db) throw new Error("No database loaded");
  const code = String(args.python ?? args.code ?? "").trim();
  if (!code) throw new Error("run_python requires 'python'");

  ctx.callbacks.onProgress?.("Loading Pyodide…");
  const py = await loadPyodide((m) => ctx.callbacks.onProgress?.(m));
  // First use of Pyodide on this page: prime DataFrames + helpers. We
  // reuse the same setup make_plot does so globals (df_*, np, pd, plt)
  // are consistent regardless of which Python tool is called first.
  py.globals.set("_tables_json", tablesToJson(ctx.db));
  await py.runPythonAsync(SETUP_PY);
  await py.runPythonAsync(`
_dfs = _materialize_tables(_tables_json)
for _k, _v in _dfs.items():
    globals()[_k] = _v
import numpy as np
`);
  ctx.callbacks.onProgress?.("Running Python…");
  // Run the user code inside a try/finally so we always restore stdout,
  // and capture (a) printed output, (b) whatever the model assigned to
  // `_out`. Indenting under `try:` keeps multi-line code intact;
  // imports / assignments inside still bind globally because
  // runPythonAsync runs at module scope.
  const indented = code.replace(/^/gm, "    ");
  await py.runPythonAsync(`
import sys, io, json as _json
_buf = io.StringIO()
_old_stdout = sys.stdout
sys.stdout = _buf
_out = None
try:
${indented}
finally:
    sys.stdout = _old_stdout
_PY_STDOUT = _buf.getvalue()
def _serialize_out(v):
    if v is None:
        return None
    try:
        if hasattr(v, "to_dict"):
            return _json.dumps(v.to_dict(), default=str)
    except Exception:
        pass
    try:
        return _json.dumps(v, default=str)
    except Exception:
        return repr(v)
_PY_OUT = _serialize_out(_out)
`);
  const stdoutRaw = String(py.globals.get("_PY_STDOUT") ?? "");
  // Cap stdout so a chatty `print(df)` doesn't blow the LLM context.
  const stdout =
    stdoutRaw.length > 4000 ? stdoutRaw.slice(0, 4000) + "\n…(truncated)" : stdoutRaw;
  const outVal = py.globals.get("_PY_OUT");
  return {
    stdout,
    result: outVal === null || outVal === undefined ? undefined : String(outVal),
  };
}

const SETUP_PY = `
import json
import io
import base64
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _materialize_tables(_tables_json):
    tables = json.loads(_tables_json)
    out = {}
    for name, payload in tables.items():
        df = pd.DataFrame(payload["rows"], columns=payload["columns"])
        out[f"df_{name}"] = df
    return out
`;

function tablesToJson(db: DatasetDB): string {
  const obj: Record<string, { columns: string[]; rows: unknown[][] }> = {};
  for (const t of db.tables) {
    const colList = t.columns.map((c) => `"${c}"`).join(", ");
    const result = runQuery(db.db, `SELECT ${colList} FROM "${t.table_name}";`);
    obj[t.organelle_class] = { columns: result.columns, rows: result.rows };
  }
  return JSON.stringify(obj);
}

const TOOL_EXECUTORS: Record<
  string,
  (args: Record<string, unknown>, ctx: AgentContext) => Promise<unknown>
> = {
  run_sql: execRunSql,
  run_python: execRunPython,
  fly_to: execFlyTo,
  highlight_segments: execHighlight,
  make_plot: execMakePlot,
  answer: execAnswer,
};

// Translate raw SQLite / Pyodide errors into a short, directive hint
// the model can act on. WebLLM small models in particular don't
// recover from "unrecognized token" or "no such column" without one
// — and crucially, they can't re-read the schema in the system prompt
// once it's pushed back in context. So for column / table errors we
// reinject the actual schema as part of the hint.
// Returns "" when no useful hint is available.
function synthesizeErrorHint(
  tool: string,
  args: Record<string, unknown>,
  errMsg: string,
  ctx: AgentContext,
): string {
  const lower = errMsg.toLowerCase();
  if (tool === "run_sql") {
    const sql = String(args.sql ?? "");
    if (lower.includes("unrecognized token")) {
      // SQLite gives this when a column name has a special char (^, /,
      // parens) and isn't quoted. The schema lists the real column
      // names — re-emphasize that they go between double quotes.
      return `Column names with special characters like '(', ')', '^' must be wrapped in double quotes: "volume_(nm^3)" not volume_(nm^3). Or use only the alphanumeric prefix if it's unique.`;
    }
    if (lower.includes("no such column") && ctx.db) {
      const colMatch = errMsg.match(/no such column[:\s]+([^\s,;]+)/i);
      const bad = colMatch ? colMatch[1] : "that column";
      // Try to figure out which table the SQL targeted so we can list
      // ITS columns specifically (more focused than dumping every
      // table). Falls back to listing all tables' columns if FROM
      // can't be parsed.
      const fromMatch = sql.match(/from\s+"?([a-zA-Z0-9_]+)"?/i);
      const targetTable = fromMatch?.[1]?.toLowerCase();
      const targets = targetTable
        ? ctx.db.tables.filter((t) => t.table_name === targetTable)
        : ctx.db.tables;
      const schema = (targets.length > 0 ? targets : ctx.db.tables)
        .map((t) => `"${t.table_name}": ${t.columns.join(", ")}`)
        .join("\n");
      return `Column ${bad} doesn't exist. The actual columns are:\n${schema}\nUse one of these names verbatim.`;
    }
    if (lower.includes("no such table") && ctx.db) {
      const names = ctx.db.tables.map((t) => `"${t.table_name}"`).join(", ");
      return `That table doesn't exist. Available tables: ${names}.`;
    }
    if (lower.includes("syntax error")) {
      return `SQL syntax error. Re-check the SELECT clause, table name, and any quoted identifiers. Note: SQLite dialect.`;
    }
    if (sql.length === 0) {
      return `run_sql expects {"sql": "SELECT ..."}.`;
    }
  }
  if (tool === "run_python") {
    if (lower.includes("nameerror") || lower.includes("name '") && lower.includes("' is not defined")) {
      return `That variable doesn't exist. The DataFrames are named df_<organelle_class> (see the system prompt schema). np and pd are imported.`;
    }
    if (lower.includes("keyerror") || lower.includes("not in index")) {
      return `That column isn't on the DataFrame. Use df.columns to discover names — they include the suffix (e.g. 'volume_(nm^3)' is a literal column key in pandas, write df["volume_(nm^3)"]).`;
    }
    if (lower.includes("attributeerror")) {
      return `Wrong type / method. If you're calling a Series method on a DataFrame (or vice versa), index into the column first: df["col"].mean(), not df.mean()["col"].`;
    }
    if (lower.includes("indentationerror") || lower.includes("syntaxerror")) {
      return `Python syntax error. Watch indentation; the harness wraps your code under a try block, so it must be valid as-is.`;
    }
  }
  if (tool === "make_plot") {
    if (lower.includes("did not produce a figure")) {
      return `Your code ran but no matplotlib figure was open. Make sure you call plt.figure() / plt.plot() / plt.bar() etc. — at least one plot command before the script ends.`;
    }
  }
  return "";
}

function summarizeResult(result: unknown): string {
  if (result === undefined || result === null) return "null";
  if (typeof result === "object") {
    const asRecord = result as { columns?: string[]; rows?: unknown[][] };
    if (Array.isArray(asRecord.rows)) {
      const rows = asRecord.rows.slice(0, 10);
      const truncated = (asRecord.rows?.length ?? 0) > rows.length;
      return JSON.stringify(
        { columns: asRecord.columns, rows, truncated },
        null,
        0,
      );
    }
  }
  return JSON.stringify(result).slice(0, 800);
}

export async function runAgent(question: string, ctx: AgentContext): Promise<void> {
  if (!ctx.backend.isReady()) {
    throw new Error("No AI backend ready. Configure one in settings.");
  }
  if (ctx.backend instanceof WebLLMBackend && !ctx.backend.isLoaded()) {
    ctx.backend.setProgressCallback((p) => {
      const pct = p.progress !== undefined ? ` (${Math.round(p.progress * 100)}%)` : "";
      ctx.callbacks.onProgress?.(`Loading WebLLM model${pct}: ${p.text}`);
    });
    ctx.callbacks.onProgress?.("Loading WebLLM model (first use, ~1 GB one-time download)…");
    await ctx.backend.ensureInit();
  }
  const messages: LLMMessage[] = [
    { role: "system", content: SYSTEM_PROMPT(ctx.db, ctx.descriptor) },
    { role: "user", content: question },
  ];

  // Snapshot the cloud-API call counter so we can show "this turn used
  // N calls" — the actual number that matters when you're hitting a
  // free-tier quota. Local backends (WebLLM) leave this at 0.
  const startReqCount = GeminiBackend.requestCount;

  // Track whether the model has delivered any user-visible output
  // (answer text or a plot). If it tries to `done` without one — which
  // small models do after a single tool error — we push back once,
  // asking for a written answer summarizing what happened. Avoids the
  // 'Agent finished without delivering an answer' dead end.
  let deliveredOutput = false;
  let nudgedForAnswer = false;

  for (let i = 0; i < MAX_ITERATIONS; i++) {
    const stepStart = performance.now();
    let tokenCount = 0;
    let lastUpdate = 0;
    ctx.callbacks.onProgress?.(`Agent step ${i + 1}: thinking…`);
    const raw = await ctx.backend.complete(messages, {
      temperature: 0.1,
      jsonMode: true,
      maxTokens: 1500,
      onToken: (_t, accumulated) => {
        tokenCount += 1;
        const now = performance.now();
        if (now - lastUpdate > 120) {
          lastUpdate = now;
          const elapsed = ((now - stepStart) / 1000).toFixed(1);
          const tps = tokenCount / Math.max(0.001, (now - stepStart) / 1000);
          const preview = accumulated.replace(/\s+/g, " ").slice(-90);
          ctx.callbacks.onProgress?.(
            `Agent step ${i + 1}: ${tokenCount} tokens · ${tps.toFixed(0)} tok/s · ${elapsed}s — ${preview}`,
          );
        }
      },
    });
    const stepElapsed = ((performance.now() - stepStart) / 1000).toFixed(1);
    const apiCalls = GeminiBackend.requestCount - startReqCount;
    const apiNote = apiCalls > 0 ? ` · ${apiCalls} API call${apiCalls === 1 ? "" : "s"} this turn` : "";
    ctx.callbacks.onProgress?.(`Agent step ${i + 1}: done in ${stepElapsed}s (${tokenCount} tokens)${apiNote}`);

    let call: ToolCall;
    try {
      call = parseToolCall(raw);
    } catch (err) {
      messages.push({ role: "assistant", content: raw });
      messages.push({
        role: "user",
        content: `Parser error: ${(err as Error).message}. Respond with a single JSON object like {"tool": "...", "args": {...}}. No prose, no markdown.`,
      });
      continue;
    }

    if (call.tool === "done") {
      // If the model bails without ever calling answer / make_plot,
      // push back once and ask it to summarize. Some flows are valid
      // visual-only (fly_to → done) but most "I errored, give up"
      // flows silently leave the user with nothing on screen.
      if (!deliveredOutput && !nudgedForAnswer) {
        nudgedForAnswer = true;
        messages.push({ role: "assistant", content: JSON.stringify(call) });
        messages.push({
          role: "user",
          content: `You called done without delivering an answer or plot. Call answer(text) summarizing what you found, or — if a tool kept failing — explain in one sentence what blocked you. Don't call done again until you've called answer.`,
        });
        continue;
      }
      ctx.callbacks.onTrace?.({ tool: "done", args: {} });
      return;
    }

    const executor = TOOL_EXECUTORS[call.tool];
    if (!executor) {
      const trace: AgentTraceItem = {
        tool: call.tool,
        args: call.args ?? {},
        error: `Unknown tool: ${call.tool}`,
      };
      ctx.callbacks.onTrace?.(trace);
      messages.push({ role: "assistant", content: JSON.stringify(call) });
      messages.push({
        role: "user",
        content: `Error: tool "${call.tool}" does not exist. Pick from: run_sql, run_python, fly_to, highlight_segments, make_plot, answer, done.`,
      });
      continue;
    }

    messages.push({ role: "assistant", content: JSON.stringify(call) });
    const trace: AgentTraceItem = { tool: call.tool, args: call.args ?? {} };
    try {
      const result = await executor(call.args ?? {}, ctx);
      trace.result = result;
      ctx.callbacks.onTrace?.(trace);
      // answer / make_plot are clearly user-visible. fly_to and
      // highlight_segments also produce a visible viewer change, so
      // count them as "delivered" — the user sees the camera move
      // even if no text answer follows. This means a single fly_to
      // without an answer() is still considered a successful turn.
      if (DELIVERED_BY_TOOL.has(call.tool)) {
        deliveredOutput = true;
      }
      if (TERMINAL_TOOLS.has(call.tool)) {
        // answer / make_plot are inherently terminal — no need for a separate done call.
        return;
      }
      messages.push({
        role: "user",
        content: `tool_result: ${summarizeResult(result)}\n\nCall the next tool, or {"tool":"done"} when finished.`,
      });
    } catch (err) {
      const errMsg = (err as Error).message;
      trace.error = errMsg;
      ctx.callbacks.onTrace?.(trace);
      // Synthesize a hint specific to common error shapes so the model
      // can self-correct. Small WebLLM models can't translate raw
      // SQLite / Python errors into the right fix on their own — they
      // just trip into 'done' if the message ends with "give up".
      // Don't offer 'done' as an option here at all; the model can
      // still emit it, but biasing toward retry-with-hint catches the
      // common 80%.
      const hint = synthesizeErrorHint(call.tool, call.args ?? {}, errMsg, ctx);
      messages.push({
        role: "user",
        content: `tool_error: ${errMsg}${hint ? `\n\nHint: ${hint}` : ""}\n\nFix the call and try again. If you've tried twice with no progress, call answer() to explain what's blocking you.`,
      });
    }
  }
  throw new Error(`Agent exceeded ${MAX_ITERATIONS} iterations — giving up.`);
}
