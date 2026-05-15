// Cross-surface handoff for Custom Python state. Lets the agent's
// per-step "Open in Custom Python" button push code + layer choices
// into a buffer that the Custom Analysis dialog drains on its next
// open. Module-level singleton because both surfaces are mounted in
// the same window, never multiplexed.
//
// State shape is intentionally minimal: only what the dialog needs to
// pre-populate slots — layer NAMES (not full descriptors), skeleton
// segment IDs, and the code. The dialog re-picks scale defaults and
// re-resolves layer descriptors from the live DatasetDescriptor, so a
// session generated from one dataset doesn't break if the user
// switches datasets in between (the layer lookup just fails cleanly).

export interface PendingPythonSession {
  layers: string[]; // layer names (zarr volumes)
  skeletons: { layer: string; segmentIds: string[] }[];
  code: string;
  // When true, the Custom Analysis dialog auto-clicks Run as soon as
  // its inspect finishes — used by the share-link Replay button so
  // the recipient doesn't have to interact with the dialog. Default
  // false; the per-step 🐍 Edit button keeps the review-first flow.
  autorun?: boolean;
}

let pending: PendingPythonSession | null = null;

export function setPendingSession(s: PendingPythonSession): void {
  pending = s;
}

export function consumePendingSession(): PendingPythonSession | null {
  const out = pending;
  pending = null;
  return out;
}

export function peekPendingSession(): PendingPythonSession | null {
  return pending;
}
