import type { LLMBackend, LLMMessage } from "./llm.js";
import { runQuery, type DatasetDB } from "./db.js";

const PYODIDE_VERSION = "0.27.0";
const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const PYODIDE_MJS = `${PYODIDE_INDEX_URL}pyodide.mjs`;

interface PyodideAPI {
  globals: { set(name: string, value: unknown): void; get(name: string): unknown };
  loadPackage(packages: string[]): Promise<void>;
  // Scans Python source for import statements and loads any matching
  // packages from Pyodide's prebuilt list (scipy, scikit-learn, sympy,
  // statsmodels, networkx, etc.). Lets the model write `import scipy`
  // without needing an explicit install step.
  loadPackagesFromImports(code: string): Promise<void>;
  runPythonAsync(code: string): Promise<unknown>;
  toPy(obj: unknown): unknown;
}

let pyodidePromise: Promise<PyodideAPI> | null = null;
type ProgressFn = (msg: string) => void;

export function loadPyodide(progress?: ProgressFn): Promise<PyodideAPI> {
  if (!pyodidePromise) {
    pyodidePromise = (async () => {
      progress?.("Downloading Pyodide runtime…");
      const mod = (await import(/* @vite-ignore */ PYODIDE_MJS)) as {
        loadPyodide: (opts: { indexURL: string }) => Promise<PyodideAPI>;
      };
      const py = await mod.loadPyodide({ indexURL: PYODIDE_INDEX_URL });
      progress?.("Loading numpy + pandas + matplotlib…");
      await py.loadPackage(["numpy", "pandas", "matplotlib"]);
      progress?.("Ready.");
      return py;
    })();
    pyodidePromise.catch((err) => {
      console.error("Pyodide load failed:", err);
      pyodidePromise = null;
    });
  }
  return pyodidePromise;
}

export interface PlotResult {
  png_data_url: string;
  stdout?: string;
  code: string;
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

const RUN_TEMPLATE = (userCode: string): string => `
plt.close("all")
${userCode}

_buf = io.BytesIO()
plt.savefig(_buf, format="png", bbox_inches="tight", dpi=120)
plt.close("all")
_PLOT_PNG = base64.b64encode(_buf.getvalue()).decode("ascii")
`;

function tablesToJson(db: DatasetDB): string {
  const obj: Record<string, { columns: string[]; rows: unknown[][] }> = {};
  for (const t of db.tables) {
    const colList = t.columns.map((c) => `"${c}"`).join(", ");
    const sql = `SELECT ${colList} FROM "${t.table_name}";`;
    const result = runQuery(db.db, sql);
    obj[t.organelle_class] = { columns: result.columns, rows: result.rows };
  }
  return JSON.stringify(obj);
}

function tablesSchemaForPrompt(db: DatasetDB): string {
  return db.tables
    .map((t) => {
      const cols = t.columns.join(", ");
      return `df_${t.organelle_class}  (columns: ${cols}, rows: ${t.row_count})`;
    })
    .join("\n");
}

const PLOT_SYSTEM_PROMPT = `You write small Python scripts that produce a single matplotlib figure.

You are given pandas DataFrames already loaded as global variables, named df_<organelle_class>
(e.g. df_mitochondria, df_nucleus). You may also use pd, plt, np.

Reply with a strict JSON object:
{
  "code": "... Python code ...",
  "title": "short title for the plot",
  "explanation": "one short sentence describing what the plot shows"
}

Rules:
- Output ONLY the JSON object. No markdown fences.
- The DataFrames listed above are ALREADY in globals(). Use them by name directly: \`df_mitochondria["volume"]\`. DO NOT reassign them. DO NOT write \`df_x = pd.DataFrame(globals()["df_x"])\` — that is a syntax error and unnecessary.
- DO NOT import anything — numpy (np), pandas (pd), and matplotlib.pyplot (plt) are already imported. Do not write 'import' statements.
- The code must produce exactly one matplotlib figure (use plt.figure() if needed) and end after plotting; do NOT call plt.show() or plt.savefig() — the harness handles that.
- Use seaborn-friendly clean styling (set fig size, axis labels, title).
- Volume is in nm^3, surface area in nm^2, positions in nm. Convert to micrometers when readable (e.g. divide volume by 1e9 for um^3).
- Never read files, network, or env. Never DDL. Use only the provided DataFrames and standard libs.`;

function userPlotPrompt(question: string, db: DatasetDB): string {
  return `AVAILABLE DATAFRAMES:

${tablesSchemaForPrompt(db)}

QUESTION: ${question}`;
}

function extractJson(text: string): string {
  const trimmed = text.trim();
  if (trimmed.startsWith("{")) return trimmed;
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced) return fenced[1].trim();
  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  if (start >= 0 && end > start) return trimmed.slice(start, end + 1);
  throw new Error("No JSON object found in plot model response");
}

interface PlotPlan {
  code: string;
  title?: string;
  explanation?: string;
}

export async function runPlotQuery(
  question: string,
  db: DatasetDB,
  backend: LLMBackend,
  progress?: ProgressFn,
): Promise<PlotResult & { plan: PlotPlan }> {
  if (!backend.isReady()) throw new Error("No AI backend ready.");
  progress?.("Asking AI for plot code…");
  const messages: LLMMessage[] = [
    { role: "system", content: PLOT_SYSTEM_PROMPT },
    { role: "user", content: userPlotPrompt(question, db) },
  ];
  const raw = await backend.complete(messages, { temperature: 0.1, jsonMode: true, maxTokens: 1500 });
  const plan = JSON.parse(extractJson(raw)) as PlotPlan;
  if (!plan.code) throw new Error("Plot model response missing 'code'");

  const py = await loadPyodide(progress);

  progress?.("Loading data into Python…");
  py.globals.set("_tables_json", tablesToJson(db));
  await py.runPythonAsync(SETUP_PY);
  await py.runPythonAsync(`
_dfs = _materialize_tables(_tables_json)
for _k, _v in _dfs.items():
    globals()[_k] = _v
import numpy as np
`);

  progress?.("Running plot code…");
  await py.runPythonAsync(RUN_TEMPLATE(plan.code));

  const b64 = py.globals.get("_PLOT_PNG") as string;
  if (!b64) throw new Error("Plot script did not produce a figure");
  return {
    plan,
    code: plan.code,
    png_data_url: `data:image/png;base64,${b64}`,
  };
}
