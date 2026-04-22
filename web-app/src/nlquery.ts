import type { LLMBackend, LLMMessage } from "./llm.js";
import { runQuery, type DatasetDB } from "./db.js";

export type Intent = "navigate" | "informational" | "plot";

export interface AgentPlan {
  intent: Intent;
  sql?: string;
  python?: string;
  answer_template?: string;
  reasoning?: string;
}

export interface AgentResult {
  plan: AgentPlan;
  sql_rows?: { columns: string[]; rows: unknown[][] };
  answer?: string;
  navigate_to?: {
    position: [number, number, number];
    object_id?: string;
    layer: string;
  };
}

function schemaSummary(db: DatasetDB): string {
  return db.tables
    .map((t) => {
      const cols = t.columns.map((c) => `"${c}"`).join(", ");
      return `TABLE "${t.table_name}"  (organelle_class: ${t.organelle_class}, layer: ${t.layer_name}, row_count: ${t.row_count})\n  columns: ${cols}`;
    })
    .join("\n\n");
}

const SYSTEM_PROMPT = `You are a data navigation assistant for a 3D microscopy viewer.
Given a user question about an organelle dataset and the database schema,
decide what to do and reply with a strict JSON object:

{
  "intent": "navigate" | "informational" | "plot",
  "sql": "... SQL query if intent is navigate or informational ...",
  "python": "... only if intent is plot (leave unset for now) ...",
  "answer_template": "short template with {placeholders} for values from the top SQL row",
  "reasoning": "one-sentence explanation"
}

Rules:
- "navigate" when the user wants to be taken somewhere or see a specific object/example
  (e.g. "show me the largest mito", "fly to nucleus 5", "take me to a lysosome near X").
  The SQL MUST select object_id, position_x, position_y, position_z, and any relevant
  metric columns, ORDER BY the relevant column, LIMIT 1 (or a small N if user says "top 5").
- "informational" when the user wants a number, count, or summary, not a location
  (e.g. "how many nuclei?", "average volume of mitochondria").
  The SQL returns aggregate rows.
- "plot" when the user wants a chart, histogram, scatter, distribution, etc.
  Leave sql unset; we'll handle plot separately.
- Use SQLite syntax. Quote all identifiers with double quotes.
- Never use DDL (CREATE/DROP/ALTER). SELECT only.
- Position columns are in nanometers. Volume in nm^3. Surface area in nm^2.
- answer_template example: "The largest mitochondrion is object {object_id} with volume {volume} nm^3."
- Output ONLY the JSON object. No markdown fences, no prose before or after.`;

function userPrompt(question: string, db: DatasetDB): string {
  return `SCHEMA:

${schemaSummary(db)}

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
  throw new Error("No JSON object found in model response");
}

function parsePlan(text: string): AgentPlan {
  const json = extractJson(text);
  const parsed = JSON.parse(json) as AgentPlan;
  if (!parsed.intent) throw new Error("Plan missing 'intent'");
  if (!["navigate", "informational", "plot"].includes(parsed.intent)) {
    throw new Error(`Plan intent must be navigate|informational|plot, got ${parsed.intent}`);
  }
  return parsed;
}

function guardSqlReadOnly(sql: string): void {
  const banned = /\b(create|drop|alter|insert|update|delete|attach|pragma)\b/i;
  if (banned.test(sql)) throw new Error(`SQL must be SELECT only; got: ${sql}`);
}

function fillTemplate(tpl: string, row: Record<string, unknown>): string {
  return tpl.replace(/\{(\w+)\}/g, (_, k) => {
    const v = row[k];
    if (v === undefined || v === null) return "?";
    if (typeof v === "number") {
      if (Math.abs(v) >= 1e5) return v.toExponential(3);
      return String(v);
    }
    return String(v);
  });
}

function layerForTable(db: DatasetDB, sql: string): string | undefined {
  const m = sql.match(/from\s+"([^"]+)"/i);
  if (!m) return undefined;
  const tableName = m[1];
  return db.tables.find((t) => t.table_name === tableName)?.layer_name;
}

export async function runNLQuery(
  question: string,
  db: DatasetDB,
  backend: LLMBackend,
): Promise<AgentResult> {
  if (!backend.isReady()) {
    throw new Error("No AI backend ready. Configure one in settings.");
  }
  const messages: LLMMessage[] = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: userPrompt(question, db) },
  ];
  const raw = await backend.complete(messages, { temperature: 0.1, jsonMode: true });
  const plan = parsePlan(raw);

  const result: AgentResult = { plan };

  if (plan.intent === "plot") {
    result.answer = "(Plot mode isn't wired up yet — coming soon.)";
    return result;
  }

  if (!plan.sql) {
    throw new Error("Agent plan missing 'sql'");
  }
  guardSqlReadOnly(plan.sql);

  const queryResult = runQuery(db.db, plan.sql);
  result.sql_rows = queryResult;

  if (queryResult.rows.length === 0) {
    result.answer = "No rows matched.";
    return result;
  }

  const rowObj: Record<string, unknown> = {};
  queryResult.columns.forEach((c, i) => (rowObj[c] = queryResult.rows[0][i]));

  if (plan.answer_template) {
    result.answer = fillTemplate(plan.answer_template, rowObj);
  }

  if (plan.intent === "navigate") {
    const px = rowObj.position_x;
    const py = rowObj.position_y;
    const pz = rowObj.position_z;
    if (typeof px === "number" && typeof py === "number" && typeof pz === "number") {
      const layer = layerForTable(db, plan.sql);
      if (layer) {
        result.navigate_to = {
          position: [px, py, pz],
          object_id: rowObj.object_id !== undefined ? String(rowObj.object_id) : undefined,
          layer,
        };
      }
    }
  }

  return result;
}
