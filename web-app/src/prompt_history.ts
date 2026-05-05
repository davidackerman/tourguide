// Most-recent-first list of Custom Analysis prompts the user has run.
// Persisted in localStorage so the dropdown survives reloads, and read by
// the permalink encoder so a "Share" link carries the sharer's recent
// queries forward to the recipient.

const KEY = "tourguide.analysisPromptHistory";
const MAX_PROMPTS = 20;

export function loadPromptHistory(): string[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.map(String).slice(0, MAX_PROMPTS) : [];
  } catch {
    return [];
  }
}

export function savePromptHistory(prompts: string[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(prompts.slice(0, MAX_PROMPTS)));
  } catch {
    /* private mode / quota — silently drop */
  }
}

export function recordPrompt(prompt: string): void {
  const trimmed = prompt.trim();
  if (!trimmed) return;
  const existing = loadPromptHistory();
  const deduped = [trimmed, ...existing.filter((p) => p !== trimmed)];
  savePromptHistory(deduped);
}

export function mergePrompts(incoming: string[]): void {
  if (!incoming || incoming.length === 0) return;
  const existing = loadPromptHistory();
  const seen = new Set(existing);
  const merged = [...existing];
  for (const p of incoming) {
    const t = p.trim();
    if (!t || seen.has(t)) continue;
    merged.push(t);
    seen.add(t);
  }
  savePromptHistory(merged);
}
