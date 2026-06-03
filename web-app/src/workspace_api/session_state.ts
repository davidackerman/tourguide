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
  /** Ids of tables/plots currently present, for the saved-state manifest. */
  getTableIds: () => string[];
  getPlotIds: () => string[];
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
      plotIds: this.hooks.getPlotIds(),
      annotations,
    };
    this.savedStates.push(state);
    persistSavedStates(this.savedStates);
    return state;
  }

  restoreState(id: string): SavedTourguideState {
    const found = this.savedStates.find((s) => s.id === id);
    if (!found) throw new Error(`saved state not found: ${id}`);
    this.hooks.applyViewerState(found.viewerState);
    return found;
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
