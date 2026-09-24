// Workspace session artifacts: saved states, plot artifacts, recording
// status, and narration notes. These are workspace-side objects (the agent
// references them by id; it does not hold them). Saved states persist to
// localStorage so they survive a reload; plots / recording / narration are
// per-session and in-memory.

import type {
  PlotArtifact,
  SavedTourguideState,
  WorkspaceAnnotation,
} from "./protocol.js";

const SAVED_STATES_KEY = "tourguide.workspace.savedStates";

export interface ViewerStateHooks {
  /** Live Neuroglancer state (camera + layers + selection). */
  getViewerState: () => unknown;
  applyViewerState: (state: unknown) => void;
  /** Current dataset descriptor (for descriptorState capture). */
  getDescriptorState: () => unknown;
  /** Reload a descriptor captured in a saved state when it differs from the
   *  one currently loaded (restoring after a dataset switch). */
  applyDescriptorState?: (descriptor: unknown, viewerState: unknown) => void;
  /** Ids of tables currently present, for the saved-state manifest. */
  getTableIds: () => string[];
}

export interface NarrationNote {
  id: string;
  at: string;
  text: string;
  /** Optional position/segment context the agent attached to the note. */
  position?: number[];
  segmentId?: string;
}

export interface RecordingState {
  active: boolean;
  startedAt?: string;
  stoppedAt?: string;
  /** Narration notes captured during the active recording window. */
  noteIds: string[];
}

export interface SessionExport {
  sessionId: string;
  exportedAt: string;
  savedStates: SavedTourguideState[];
  plots: Array<Omit<PlotArtifact, "pngDataUrl"> & { hasImage: boolean }>;
  tables: string[];
  recording: RecordingState;
  narrationNotes: NarrationNote[];
}

const uuid = (): string =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `id-${Math.floor(performance.now())}-${Math.floor(Math.random() * 1e9)}`;

const descriptorName = (d: unknown): string | undefined =>
  d && typeof d === "object" ? (d as { name?: string }).name : undefined;

export class SessionStore {
  private savedStates: SavedTourguideState[] = [];
  private plots: PlotArtifact[] = [];
  private narrationNotes: NarrationNote[] = [];
  private recording: RecordingState = { active: false, noteIds: [] };

  constructor(
    private readonly sessionId: string,
    private readonly hooks: ViewerStateHooks,
    private readonly nowIso: () => string,
  ) {
    this.savedStates = loadSavedStates();
  }

  // --- saved states --------------------------------------------------------

  saveState(name?: string, annotations?: WorkspaceAnnotation[]): SavedTourguideState {
    const state: SavedTourguideState = {
      id: uuid(),
      name,
      createdAt: this.nowIso(),
      viewerState: this.hooks.getViewerState(),
      descriptorState: this.hooks.getDescriptorState(),
      tableIds: this.hooks.getTableIds(),
      plotIds: this.plots.map((p) => p.id),
      annotations,
    };
    this.savedStates.push(state);
    persistSavedStates(this.savedStates);
    return state;
  }

  restoreState(id: string): SavedTourguideState {
    const found = this.savedStates.find((s) => s.id === id);
    if (!found) throw new Error(`saved state not found: ${id}`);
    this.applyViewer(found);
    return found;
  }

  // If the saved state belongs to a different dataset than the one loaded
  // now, reload that dataset first (with the saved camera/selection
  // overlaid); otherwise just reapply the viewer state.
  private applyViewer(state: SavedTourguideState): void {
    const current = descriptorName(this.hooks.getDescriptorState());
    const saved = descriptorName(state.descriptorState);
    if (state.descriptorState && saved && saved !== current && this.hooks.applyDescriptorState) {
      this.hooks.applyDescriptorState(state.descriptorState, state.viewerState);
    } else {
      this.hooks.applyViewerState(state.viewerState);
    }
  }

  /** Apply a full state object (e.g. one the bridge loaded from disk, which
   *  may not be in this tab's localStorage). Caches it locally too so the
   *  panel reflects it. */
  applyState(state: SavedTourguideState): SavedTourguideState {
    this.applyViewer(state);
    if (!this.savedStates.some((s) => s.id === state.id)) {
      this.savedStates.push(state);
      persistSavedStates(this.savedStates);
    }
    return state;
  }

  /** Current workspace state WITHOUT persisting it as a named saved-state —
   *  used for the rolling per-session auto-save (keyed by session id on the
   *  bridge), so reopening a ?session=<id> link restores the viewer. */
  snapshot(): SavedTourguideState {
    return {
      id: this.sessionId,
      createdAt: this.nowIso(),
      viewerState: this.hooks.getViewerState(),
      descriptorState: this.hooks.getDescriptorState(),
      tableIds: this.hooks.getTableIds(),
      plotIds: this.plots.map((p) => p.id),
    };
  }

  listSavedStates(): SavedTourguideState[] {
    return this.savedStates.slice();
  }

  savedStateSummaries(): Array<{ id: string; name?: string; createdAt: string }> {
    return this.savedStates.map((s) => ({ id: s.id, name: s.name, createdAt: s.createdAt }));
  }

  // --- plots ---------------------------------------------------------------

  addPlot(artifact: Omit<PlotArtifact, "id"> & { id?: string }): PlotArtifact {
    const plot: PlotArtifact = { ...artifact, id: artifact.id ?? uuid() };
    this.plots.push(plot);
    return plot;
  }

  listPlots(): PlotArtifact[] {
    return this.plots.slice();
  }

  plotSummaries(): Array<{ id: string; title?: string; kind: string; sourceTable?: string }> {
    return this.plots.map((p) => ({
      id: p.id,
      title: p.title,
      kind: p.kind,
      sourceTable: p.sourceTable,
    }));
  }

  // --- recording + narration ----------------------------------------------

  startRecording(): RecordingState {
    this.recording = { active: true, startedAt: this.nowIso(), noteIds: [] };
    return this.recording;
  }

  stopRecording(): RecordingState {
    this.recording = {
      ...this.recording,
      active: false,
      stoppedAt: this.nowIso(),
    };
    return this.recording;
  }

  recordingState(): RecordingState {
    return { ...this.recording, noteIds: this.recording.noteIds.slice() };
  }

  addNarrationNote(text: string, extra?: { position?: number[]; segmentId?: string }): NarrationNote {
    const note: NarrationNote = {
      id: uuid(),
      at: this.nowIso(),
      text,
      position: extra?.position,
      segmentId: extra?.segmentId,
    };
    this.narrationNotes.push(note);
    if (this.recording.active) this.recording.noteIds.push(note.id);
    return note;
  }

  listNarrationNotes(): NarrationNote[] {
    return this.narrationNotes.slice();
  }

  // --- export --------------------------------------------------------------

  exportSummary(): SessionExport {
    return {
      sessionId: this.sessionId,
      exportedAt: this.nowIso(),
      savedStates: this.listSavedStates(),
      // Strip heavy PNG payloads from the manifest; reference by hasImage.
      plots: this.plots.map(({ pngDataUrl, ...rest }) => ({
        ...rest,
        hasImage: !!pngDataUrl,
      })),
      tables: this.hooks.getTableIds(),
      recording: this.recordingState(),
      narrationNotes: this.listNarrationNotes(),
    };
  }
}

function loadSavedStates(): SavedTourguideState[] {
  try {
    const raw = localStorage.getItem(SAVED_STATES_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function persistSavedStates(states: SavedTourguideState[]): void {
  try {
    localStorage.setItem(SAVED_STATES_KEY, JSON.stringify(states));
  } catch {
    /* private mode / quota — saved states stay in-memory for this session */
  }
}
