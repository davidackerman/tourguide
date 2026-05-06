import type { LLMBackend, LLMMessage } from "./llm.js";
import { WebLLMBackend } from "./llm.js";
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

const MAX_ITERATIONS = 10;

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

  answer(text: string)
    Deliver a final text answer to the user. Use for counts, means,
    informational questions. Keep it concise.

  done()
    End this turn. You MUST call this after delivering any answer / plot / fly_to.

TYPICAL FLOWS:
  "show me the largest mito"       -> run_sql (ORDER BY volume DESC LIMIT 1) -> fly_to -> done
  "how many nuclei are there?"     -> run_sql (COUNT) -> answer -> done
  "plot mitochondrion volumes"     -> make_plot -> done
  "fly through the 3 biggest mitos" -> run_sql (LIMIT 3) -> fly_to (first) -> fly_to (second) -> fly_to (third) -> done
  "show only the largest 10 mitos" -> run_sql (ORDER BY volume DESC LIMIT 10) -> highlight_segments -> done

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

async function execFlyTo(
  args: Record<string, unknown>,
  ctx: AgentContext,
): Promise<{ ok: boolean }> {
  const pos = args.position as unknown;
  if (!Array.isArray(pos) || pos.length !== 3 || pos.some((n) => typeof n !== "number")) {
    throw new Error("fly_to requires position: [x, y, z] numbers");
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
  fly_to: execFlyTo,
  highlight_segments: execHighlight,
  make_plot: execMakePlot,
  answer: execAnswer,
};

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
    ctx.callbacks.onProgress?.(`Agent step ${i + 1}: done in ${stepElapsed}s (${tokenCount} tokens)`);

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
        content: `Error: tool "${call.tool}" does not exist. Pick from: run_sql, fly_to, highlight_segments, make_plot, answer, done.`,
      });
      continue;
    }

    messages.push({ role: "assistant", content: JSON.stringify(call) });
    const trace: AgentTraceItem = { tool: call.tool, args: call.args ?? {} };
    try {
      const result = await executor(call.args ?? {}, ctx);
      trace.result = result;
      ctx.callbacks.onTrace?.(trace);
      if (TERMINAL_TOOLS.has(call.tool)) {
        // answer / make_plot are inherently terminal — no need for a separate done call.
        return;
      }
      messages.push({
        role: "user",
        content: `tool_result: ${summarizeResult(result)}\n\nCall the next tool, or {"tool":"done"} when finished.`,
      });
    } catch (err) {
      trace.error = (err as Error).message;
      ctx.callbacks.onTrace?.(trace);
      messages.push({
        role: "user",
        content: `tool_error: ${(err as Error).message}\n\nAdjust and try again, or {"tool":"done"} to give up.`,
      });
    }
  }
  throw new Error(`Agent exceeded ${MAX_ITERATIONS} iterations — giving up.`);
}
