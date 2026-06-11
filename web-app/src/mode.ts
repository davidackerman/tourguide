// Display/workspace mode.
//
// Tourguide is moving from a chat-centered app to a persistent visual
// *workspace* controlled by external agents (Claude Desktop, Claude Code,
// Cursor, a local script, …) over the Workspace API. The mode is selected
// with `?mode=workspace` / `?mode=chat`.
//
// Migration status:
//   - `workspace` is now the DEFAULT (the MCP adapter is in daily use).
//   - legacy `?mode=chat` still works but is DEPRECATED — it logs a warning and
//     will be removed after the deprecation window. Don't build new features on
//     chat mode; drive the workspace from an external agent instead.
//
// Keep this module dependency-free so anything (UI, bridge, handlers) can
// import it without pulling in the viewer or DB.

export type TourguideMode = "workspace" | "chat";

// The agent-driven workspace is the product; chat mode is deprecated legacy.
export const DEFAULT_MODE: TourguideMode = "workspace";

let chatDeprecationWarned = false;

/** Resolve the active mode from the current URL (`?mode=…`), falling back to
 *  DEFAULT_MODE. Unknown values are treated as the default. */
export function resolveMode(search: string = window.location.search): TourguideMode {
  try {
    const raw = new URLSearchParams(search).get("mode")?.toLowerCase();
    if (raw === "workspace") return "workspace";
    if (raw === "chat") {
      if (!chatDeprecationWarned) {
        chatDeprecationWarned = true;
        console.warn(
          "[tourguide] ?mode=chat is DEPRECATED — the agent workspace is now the default. " +
            "Chat mode will be removed in a future release; drive the workspace from an external agent instead.",
        );
      }
      return "chat";
    }
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
