// Most-recent-first list of prompts the user has run, persisted in
// localStorage so the dropdown survives reloads. Two scopes share this
// helper: Custom Analysis ("analysis") and the AI agent ("agent"). The
// permalink encoder also reads the analysis scope so a "Share" link
// carries the sharer's recent queries forward to the recipient.

export type HistoryScope = "analysis" | "agent";

const KEYS: Record<HistoryScope, string> = {
  analysis: "tourguide.analysisPromptHistory",
  agent: "tourguide.agentPromptHistory",
};
const MAX_PROMPTS = 20;

export function loadPromptHistory(scope: HistoryScope = "analysis"): string[] {
  try {
    const raw = localStorage.getItem(KEYS[scope]);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.map(String).slice(0, MAX_PROMPTS) : [];
  } catch {
    return [];
  }
}

export function savePromptHistory(prompts: string[], scope: HistoryScope = "analysis"): void {
  try {
    localStorage.setItem(KEYS[scope], JSON.stringify(prompts.slice(0, MAX_PROMPTS)));
  } catch {
    /* private mode / quota — silently drop */
  }
}

export function recordPrompt(prompt: string, scope: HistoryScope = "analysis"): void {
  const trimmed = prompt.trim();
  if (!trimmed) return;
  const existing = loadPromptHistory(scope);
  const deduped = [trimmed, ...existing.filter((p) => p !== trimmed)];
  savePromptHistory(deduped, scope);
}

export function mergePrompts(incoming: string[], scope: HistoryScope = "analysis"): void {
  if (!incoming || incoming.length === 0) return;
  const existing = loadPromptHistory(scope);
  const seen = new Set(existing);
  const merged = [...existing];
  for (const p of incoming) {
    const t = p.trim();
    if (!t || seen.has(t)) continue;
    merged.push(t);
    seen.add(t);
  }
  savePromptHistory(merged, scope);
}
