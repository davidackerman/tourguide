// Display/workspace mode.
//
// Tourguide is moving from a chat-centered app to a persistent visual
// *workspace* controlled by external agents (Claude Desktop, Claude Code,
// Cursor, a local script, …) over the Workspace API. The mode is selected
// with `?mode=workspace` / `?mode=chat`.
//
// Migration policy (intentional, see the agent-workspace plan):
//   - During early phases the DEFAULT stays `chat` so existing links and
//     muscle memory keep working.
//   - Once the MCP adapter is usable, flip DEFAULT_MODE to "workspace" and
//     legacy chat lives on at `?mode=chat`.
//   - Eventually chat is removed after a deprecation window.
//
// Keep this module dependency-free so anything (UI, bridge, handlers) can
// import it without pulling in the viewer or DB.

export type TourguideMode = "workspace" | "chat";

// Flip to "workspace" when Phase 3 (MCP adapter) is usable. Until then the
// default remains chat so nothing the user relies on changes silently.
export const DEFAULT_MODE: TourguideMode = "chat";

/** Resolve the active mode from the current URL (`?mode=…`), falling back to
 *  DEFAULT_MODE. Unknown values are treated as the default. */
export function resolveMode(search: string = window.location.search): TourguideMode {
  try {
    const raw = new URLSearchParams(search).get("mode")?.toLowerCase();
    if (raw === "workspace" || raw === "chat") return raw;
  } catch {
    /* malformed query string — fall through to default */
  }
  return DEFAULT_MODE;
}

export function isWorkspaceMode(search?: string): boolean {
  return resolveMode(search) === "workspace";
}

/** Stamp the mode onto <body data-mode> so CSS can gate chat-only chrome
 *  (composer, AI provider indicator, …) purely with attribute selectors. */
export function applyModeToDocument(mode: TourguideMode): void {
  document.body.setAttribute("data-mode", mode);
}
