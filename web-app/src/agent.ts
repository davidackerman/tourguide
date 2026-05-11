import type { LLMBackend, LLMMessage } from "./llm.js";
import { WebLLMBackend, GeminiBackend } from "./llm.js";
import type { DatasetDB } from "./db.js";
import { ingestTableIntoDB, runQuery } from "./db.js";
import type { BundledViewer } from "./bundled_viewer.js";
import { loadPyodide } from "./plot.js";
import type { DatasetDescriptor, DatasetLayer } from "./descriptor.js";
import {
  AnalysisClient,
  isZarrSource,
  normalizeZarrUrl,
  SAFE_INPUT_BYTES,
  type CustomAnalysisResult,
} from "./analysis.js";
import {
  probePrecomputedInfo,
  resolveBundledSubpath,
  probeMeshInfo,
} from "./precomputed_info.js";
import { loadSettings } from "./llm.js";
import { postAnalysisRequest, waitForBackendReady } from "./remote_analysis.js";

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
  // Fired when python_on_layers emits _TG_TABLE — UI can render a CSV
  // download / "table saved" hint inline. The data is already being
  // ingested into the SQL DB by applyCustomResult; this callback is
  // purely for surface-level affordance.
  onTable?: (table: { name: string; columns: string[]; rows: (number | string | null)[][] }) => void;
  // "Sticky" per-step metadata that should be visible after the step
  // completes, separate from the transient onProgress messages. Used
  // by python_on_layers to surface which runtime + scales were chosen
  // ("Ran on backend; scales: mito@s2 (3.1 GB)") so the user sees it
  // without expanding the agent trace.
  onMeta?: (info: string) => void;
  // Render a structured ask_user form inline in the current turn card
  // and resolve once the user submits. Reject with AbortError if the
  // user cancels or hits Stop. The agent loop awaits this naturally —
  // a no-op if the UI doesn't implement it (auto-resolves with
  // defaults so the agent doesn't hang).
  onAskUser?: (
    prompt: string,
    fields: AskField[],
  ) => Promise<Record<string, unknown>>;
}

// Structured input the agent can request from the user via ask_user.
// Each field has a recommended default; the user can submit unchanged
// to take the default — preserving the auto-pick experience while
// giving them an override at the same friction cost.
export type AskField =
  | { id: string; label: string; type: "select"; options: { label: string; value: string }[]; default?: string }
  | { id: string; label: string; type: "multi"; options: { label: string; value: string }[]; default?: string[] }
  | { id: string; label: string; type: "yesno"; default?: boolean }
  | { id: string; label: string; type: "text"; default?: string; placeholder?: string };

export interface AgentTurnSummary {
  // The user's question.
  question: string;
  // A short recap of what the agent did/answered. We don't replay full
  // tool traces — at ~3 prior turns × multi-step flow that would
  // dominate the context window. A single sentence per turn is enough
  // for "now do the same for ER" follow-ups to make sense.
  summary: string;
}

export interface AgentContext {
  db: DatasetDB | null;
  // Optional setter so python_on_layers can ingest a result table into
  // the SQL DB (and have the next turn run_sql against it). When
  // omitted the table is recorded in the trace but not persisted.
  setDB?: (db: DatasetDB) => void;
  descriptor: DatasetDescriptor | null;
  viewer: BundledViewer;
  backend: LLMBackend;
  callbacks: AgentCallbacks;
  // Last few turns in this session, oldest-first. The agent gets these
  // as a compressed context so follow-up prompts ("now do the same for
  // X", "redo with a different threshold") can refer back. Capped by
  // the caller — typically last 3.
  priorTurns?: AgentTurnSummary[];
  // When this aborts, the agent stops between iterations (and the
  // current LLM call gets aborted via fetch/interruptGenerate). Lets
  // the user hit a Stop button mid-question instead of waiting out
  // 5 iterations of a slow WebLLM model.
  signal?: AbortSignal;
}

interface ToolCall {
  tool: string;
  args?: Record<string, unknown>;
}

// Cap the agent loop at 5 steps. Most legitimate flows finish in 2-4
// (run_sql -> fly_to -> done; run_python -> answer -> done). A higher
// cap mostly lets a confused model burn through your Gemini free-tier
// quota chasing its tail. If a real query needs more, the user can ask
// a follow-up.
const MAX_ITERATIONS = 5;

// User-facing labels for each tool, shown in the status line as soon
// as the streaming model commits to a tool name (regex'd from the
// in-flight JSON). Replaces the raw tool name + token-rate noise
// that previously lived in the visible status.
const TOOL_LABELS: Record<string, string> = {
  describe_dataset: "checking dataset",
  run_sql: "querying tables",
  run_python: "computing",
  python_on_layers: "running on layers",
  make_plot: "rendering plot",
  fly_to: "flying to position",
  highlight_segments: "highlighting segments",
  ask_user: "asking you",
  answer: "writing answer",
};

// Tool docs that need organelle CSVs in the SQL DB to function (run_sql
// reads from the DB; run_python / make_plot bind df_<class> globals
// from it). Suppressed entirely from the prompt when the DB is empty
// so the model doesn't burn a turn calling a tool that errors with
// "No database loaded".
const DB_TOOL_DOCS = `  run_sql(sql: string)
    Run a SELECT query against the organelle DB. Returns up to 50 rows.
    SQL dialect: SQLite. Quote identifiers with double quotes. SELECT only.

  make_plot(python: string, title?: string, explanation?: string)
    Run Python via Pyodide (numpy, pandas, matplotlib already imported).
    You get df_<class> DataFrames as globals. Produce exactly ONE matplotlib
    figure; do NOT call plt.show() or plt.savefig() — the harness handles it.
    Volumes in nm^3, surface area in nm^2, positions in nm; convert to um^3
    (divide by 1e9) when plotting for readability.

  run_python(python: string)
    Run arbitrary Python via Pyodide for computation that doesn't need a
    plot — e.g. statistics, transformations, intermediate results to feed
    a later tool. Same environment as make_plot: df_<class> DataFrames as
    globals, plus np / pd already imported. NOT terminal; the harness
    feeds the captured stdout (and any value you assign to a global named
    _out) back to you so you can call answer/fly_to next.
    Available packages: numpy, pandas, matplotlib are pre-imported. Other
    Pyodide-prebuilt packages (scipy, scikit-learn, sympy, networkx,
    statsmodels, etc.) auto-load when you import them — just write
    'import scipy.stats' or 'from sklearn.cluster import DBSCAN' directly.
    DO NOT call micropip.install or pyodide.loadPackage; the harness
    handles it. If a package isn't available, you'll get
    ModuleNotFoundError — pick a different approach.

    DATA SHAPES — read carefully:
      df_<class>   pandas DataFrame. Index columns by name: df_mito["volume_nm_3"].
                   DO NOT use bracket-name on a numpy array.
      com_<class>  numpy (N, 3) float64 array of COMs in nanometers, ALREADY
                   extracted. Index by integer/slice: com_mito[:, 0] for x,
                   com_mito[i] for the i-th point. Pass com_<class>.T to
                   scipy.stats.gaussian_kde (it expects (D, N), not (N, D)).
                   USE com_<class> directly — do NOT rebuild it via
                   np.array(df_x[["com_x_nm",...]]). DO NOT call pd.read_csv
                   or any network fetch — the data is already loaded.

    JSON STRING ESCAPING — read carefully:
      Your code is a JSON string, so \\n inside a Python literal needs to
      survive JSON unescaping. Two safe patterns:
        1. Single-line strings, no newlines:
             print('Densest bin COM:', max_count_bin)   # no \\n needed
        2. If you really need a newline in a string, use a triple-quoted
           string OR escape twice:  "\\\\n" in the JSON becomes "\\n" in
           Python which is an actual newline at runtime.
      DO NOT write f'foo: {x}\\nbar: {y}' as a single non-triple-quoted
      Python literal — \\n in the JSON becomes a real line break and
      triggers SyntaxError: unterminated string literal.
    Conventions:
      - print(...) anything you want returned; it lands in stdout.
      - Set _out = <python value> for a structured return — DataFrames /
        Series get to_dict()'d, otherwise json/repr.
      - Globals persist across run_python calls within one user turn.
    Use this for "what's the median volume?" / "compute X then fly to
    the result" flows. For visual answers, prefer make_plot.`;

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
  const numpy = db.tables
    .map((t) => `  com_${t.organelle_class}  numpy (N, 3) array of COMs in nm — already extracted, use directly`)
    .join("\n");
  return `SQL tables (THIS is the truth — table and column names below are what actually exist; never invent or copy from examples):\n${tableLines}\n\nPython DataFrames:\n${dataframes}\n\nPre-extracted numpy arrays (USE THESE for spatial / scipy work — no need to convert from DataFrame):\n${numpy}`;
};

// Classify a source URL into a coarse kind. Includes mesh / skeleton
// detection so multi-source layers (a segmentation that has all three:
// volume + precomputed mesh + precomputed skeleton) surface them all.
function sourceKind(url: string): string {
  const m = url.match(/^(zarr2?|zarr3|n5|precomputed|graphene)/);
  const proto = m?.[1] ?? "url";
  // Path-based signals — common Janelia / cellmap convention is
  // .../neuroglancer/mesh/... and .../neuroglancer/skeleton/... for
  // the auxiliary precomputed sources attached to a segmentation.
  if (/\/skeleton\b|\/skeletons?\//i.test(url)) return "precomputed-skeleton";
  if (/\/mesh\b|\/meshes?\//i.test(url)) return "precomputed-mesh";
  return proto;
}

const LAYER_GUIDE = (d: DatasetDescriptor): string => {
  if (d.layers.length === 0) return "No layers loaded.";
  const lines = d.layers.map((l) => {
    const sourceList = Array.isArray(l.source) ? l.source : [l.source];
    const kinds = sourceList.map(sourceKind).filter((k, i, a) => a.indexOf(k) === i);
    const cls = l.organelle_class ? ` class=${l.organelle_class}` : "";
    // Surface the NG-state translation if any was applied — the
    // analysis pipeline silently folds it into the layer offset, but
    // it's worth flagging so the model doesn't claim "positions are
    // in raw zarr coords" when they're not.
    const tx = l.transform_offset_nm;
    const xform =
      tx && (tx[0] !== 0 || tx[1] !== 0 || tx[2] !== 0)
        ? `  ng_offset_nm=${tx.map((n) => n.toFixed(0)).join(",")}`
        : "";
    return `  "${l.name}"  type=${l.type}  sources=${kinds.join("+")}${cls}${xform}`;
  });
  return `Loaded Neuroglancer layers (THIS is what's actually loaded; never claim a layer exists if it isn't here, and never invent its resolution — call describe_dataset for per-layer scale info).

  sources= conventions:
    zarr / zarr2 / zarr3 / n5 — multiscale array on a per-axis voxel grid.
        python_on_layers loads the voxels as a numpy array.
    precomputed — Neuroglancer precomputed segmentation. The volume
        VOXELS are NOT readable by python_on_layers (no in-browser
        reader for that format). Skeletons + meshes bundled inside
        the precomputed dir often ARE accessible; check describe_dataset.
    precomputed-mesh — 3D mesh per segment, not loadable as an array.
    precomputed-skeleton — pass via 'skeletons' to python_on_layers
        for length / branching / geodesic metrics.

  For precomputed-volume layers, volume / regionprops on raw voxels
  is unavailable. Skeletons cover length-style queries. Mesh-based
  volume estimation isn't supported yet — tell the user.

  'ng_offset_nm' is the user-applied translation from the pasted NG
  state; tourguide adds it to per-object positions automatically so
  fly_to and click-to-fly land where the user sees the layer, not
  where the raw zarr would put it.

${lines.join("\n")}`;
};

const SYSTEM_PROMPT = (db: DatasetDB | null, d: DatasetDescriptor | null): string => `
You are the agent for a 3D microscopy viewer. You respond to user questions by calling one tool at a time.

${d ? `DATASET: ${d.display_name}  (default voxel size: ${d.voxel_size_nm.join(" × ")} nm — individual layers may differ; use describe_dataset to confirm)` : ""}
COORDINATES: ALL positions everywhere — CSV com_x_nm / com_y_nm / com_z_nm columns, fly_to inputs, the viewer's display — are in NANOMETERS in Neuroglancer's reference frame. Pass nm values directly; do NOT convert to voxels and do NOT add the layer's offset (already baked in).

${d ? LAYER_GUIDE(d) : ""}

${db ? SCHEMA_GUIDE(db) : "No organelle database is loaded — run_sql / run_python / make_plot are unavailable, but describe_dataset, python_on_layers, fly_to, highlight_segments, answer, done all work."}

NEVER GUESS DATASET PROPERTIES. If the user asks about a layer's resolution / scale / shape / dtype / available downsamplings, call describe_dataset(layer_name) and answer from its return value. Never invent numbers like "1 × 1 × 1 nm" — when uncertain, say so or call describe_dataset.

ON EACH TURN, respond with a single JSON object describing one tool call:

  {"tool": "<name>", "args": { ... }}

NO prose, NO markdown fences — only the JSON object.

TOOLS:

  describe_dataset(layer_name?: string)
    Returns information about loaded Neuroglancer layers. Without a name,
    returns a summary of every layer (name, type, source). With a name,
    returns scale-level detail for that layer: shape, voxel_size_nm
    (per-axis, in physical units), offset_nm, available scales, and
    approximate byte size at each scale. Use this whenever the user
    asks about resolution, dimensions, or what's loaded — never guess.

${db ? DB_TOOL_DOCS : "  (run_sql / make_plot / run_python omitted — they need organelle CSVs which aren't loaded. Use python_on_layers below to compute any per-object metrics + plots directly from the segmentation voxels.)"}

  fly_to(position: [x, y, z], layer: string, object_id?: string)
    Move the viewer camera to these NANOMETER coordinates, switch on this
    layer, and highlight object_id if given. Use com_x_nm / com_y_nm /
    com_z_nm values from run_sql results AS-IS — no scaling, no offset
    addition (already in nm in NG's frame).

  highlight_segments(layer: string, ids: number[] | string[])
    In a SEGMENTATION layer, show only these segment ids — everything
    else fades to background. Use this for "show me only the …" or
    "select these segments" requests. ids can be the object_id values
    from a run_sql result, or specific numeric ids the user named.

  ask_user(prompt: string, fields: AskField[])
    Pause for a structured user answer. Renders an inline form with
    your recommended defaults pre-selected; the user can submit
    unchanged to take them, or override. Returns {field_id: value, ...}.
    Use ONLY when the answer materially changes the result and your
    default would be a coin flip. Each field gets a clear default —
    set one even if it's just your best guess.

    Field types:
      {"id": "...", "label": "...", "type": "select", "options": [{"label":"...", "value":"..."}, ...], "default": "..."}
      {"id": "...", "label": "...", "type": "multi",  "options": [...], "default": ["..."]}
      {"id": "...", "label": "...", "type": "yesno",  "default": true}
      {"id": "...", "label": "...", "type": "text",   "default": "...", "placeholder": "..."}

    GOOD asks:
      - "erode mito" with no size → ask radius_nm with options
        [1 voxel, 25 nm, 50 nm (default), 100 nm].
      - "skeletonize ER" / instance vs binary mask matters → ask
        is_already_labeled (yesno, default false).
      - "measure mito at full res" with HF backend off → ask whether
        to enable backend or downsample to local-safe scale.
    BAD asks:
      - histogram bin count, plot title, color
      - things the user already spelled out ("erode by 50 nm" — don't
        ask the radius)
      - things describe_dataset already answered (resolution,
        layer types, scale paths)
      - questions where one answer is obviously correct given the
        prompt (axis-order conventions, dtype clamping for NG)

    After ask_user resolves, do NOT ask the same question again — its
    answer is now in the conversation. Proceed to the operation you
    were going to perform.

    Example flow for an ambiguous "erode mito":
      1) ask_user(
           prompt="What erosion radius?",
           fields=[{"id":"radius_nm","label":"Radius (nm)","type":"select",
                    "options":[
                      {"label":"1 voxel","value":"1vox"},
                      {"label":"25 nm","value":"25"},
                      {"label":"50 nm","value":"50"},
                      {"label":"100 nm","value":"100"}],
                    "default":"50"}])
      2) python_on_layers (using the resolved radius)
      3) done

  python_on_layers(python: string, layers?: string[], skeletons?: [{layer: string, segment_ids: string[]}], meshes?: [{layer: string, segment_ids: string[]}], runtime?: "auto"|"local"|"backend")
    HEAVY-LIFT path. Runs Python with the actual layer voxels (and/or
    precomputed skeletons) loaded as numpy arrays — use this when the
    user asks you to OPERATE ON LAYERS (erode, dilate, threshold,
    skeletonize, mesh, find contacts, distance transforms), PRODUCE A
    NEW LAYER, or analyze SKELETONS (length, branching, geodesic
    distances, tortuosity), or compute regionprops / connected
    components ("measure properties of mito", "list mito by volume").

    REQUIRED args every call: { "python": "<your code>", "layers": [...] }
    The "python" key is ALWAYS required — emitting just
    {"layers": ["mito"]} fails immediately with "python_on_layers
    requires 'python'". Even a trivial call needs the code body.
    Don't use "code" / "script" / "source" — only "python".

    Pass at least one of:
      'layers'    — array of zarr layer names from describe_dataset.
                    The harness picks the FINEST scale that fits the
                    chosen runtime's byte budget (≤1.5 GB for local
                    Pyodide, ≤5 GB for HF backend) and binds each as a
                    numpy array under its 'organelle_class' (or
                    sanitized layer name). Per-layer metadata is at
                    layers["<varName>"]: "array" (ndarray), "spacing"
                    / "voxel_size_nm" (voxel size nm, array-axis
                    order), "offsets" / "offset_nm" (world origin nm,
                    array-axis order), "axes" (e.g. ["z","y","x"]).
                    Both spacing/voxel_size_nm and offsets/offset_nm
                    are aliases — use whichever you remember.
      'skeletons' — array of {"layer": str, "segment_ids": [...]} for
                    layers with a precomputed-skeleton source
                    (describe_dataset's has_skeleton). Binds
                    '<layer>_skel = {seg_id: {"vertices": (N,3) f32 nm,
                    "edges": (M,2) u32}}'. A sibling
                    '<layer>_skel_missing_ids' lists IDs without files.
                    Always provide segment_ids explicitly — usually
                    from a prior run_sql ORDER BY <metric> LIMIT N.
      'meshes'    — array of {"layer": str, "segment_ids": [...]} for
                    layers with a precomputed-mesh source. Use when
                    the layer has no zarr volume (so regionprops on
                    voxels isn't available) but does have meshes —
                    THIS is how you compute volume / surface area
                    on precomputed-only datasets. Binds
                    '<layer>_mesh = {seg_id: {"vertices": (N,3) f32 nm,
                    "faces": (M,3) u32}}'. CURRENTLY supports only
                    the 'neuroglancer_legacy_mesh' on-disk format,
                    non-sharded — describe_dataset reports
                    mesh_format + mesh_analysis_supported so check
                    before requesting. Multi-LOD draco + sharded
                    variants aren't supported yet; the resolver
                    errors clearly if you try them.
      'runtime'   — "auto" (default; harness picks). Use "backend" only
                    when the user explicitly asks for full resolution
                    or you know the dataset is huge. Use "local" only
                    if you want to force Pyodide.

    RUNTIME AUTO-SELECTION (you don't usually need to think about this,
    but it shapes what code to write):
      - 'skeletons' or 'meshes' present           → forced LOCAL
      - code imports cc3d / fastmorph / fastremap
        / edt / kimimaro / zmesh, OR sets
        _TG_NEW_MESH_LAYER                        → forced BACKEND
      - else if total finest-scale ≤1.5 GB        → LOCAL
      - else if backend URL configured            → BACKEND
      - else                                       → LOCAL (downsampled)
    Result narration always tells the user which runtime + scale ran,
    so they can ask for full resolution as a follow-up.

    PACKAGES BY RUNTIME:
      LOCAL  (Pyodide): numpy, scipy.ndimage, skimage.measure /
              morphology, pandas, matplotlib. NO Seung-lab pkgs.
      BACKEND (HF Space): the LOCAL set + cc3d, fastmorph (incl.
              spherical_erode / spherical_dilate), fastremap, edt,
              kimimaro, zmesh. Use these for instance-label
              morphology, fast CC, and meshing. Per-layer
              'spacing'/'offset' come in array-axis order (typically
              z,y,x); pass anisotropy=layers["x"]["spacing"] to
              fastmorph.spherical_* directly without flipping.
    Output channels — set any to deliver results:
      _TG_NEW_LAYER       persist a derived volume as a new NG layer.
                          Set to {"array": ndarray, "name": str,
                          "type": "image"|"segmentation",
                          "spacing": [sz, sy, sx],   # array-axis order
                          "offsets": [oz, oy, ox],   # array-axis order
                          "axes": ["z","y","x"]}     # optional; defaults to first input layer's axes
                          Aliases accepted: "data" for "array",
                          "voxel_size_nm" / "spacing_nm" for "spacing",
                          "offset_nm" / "offset" for "offsets". If you
                          omit spacing/offsets, they default to the
                          first input layer's values — fine when the
                          output shape matches the input.
      _TG_NEW_MESH_LAYER  per-id meshing → precomputed mesh layer.
      _TG_PLOT            matplotlib figure (auto-captured).
      _TG_FLY             {"pos": [x,y,z], "segment_id": str?, "layer": str?}.
      _TG_HIGHLIGHT       {"layer": str, "ids": list}.
      _TG_NARRATION       short text shown to the user.
      _TG_ANNOTATIONS     {"layer_name": str, "points": [{"pos": [x,y,z], "id"?, "description"?}]}.
      _TG_ADD_SOURCE_LAYER add a remote zarr/n5/precomputed as a layer.
    Available libraries: numpy / scipy.ndimage / skimage / pandas / matplotlib.
    On the HF backend (when configured): cc3d, fastmorph (incl.
    spherical_erode / spherical_dilate), fastremap, edt, kimimaro, zmesh.
    Layer arrays come in array-axis order (typically z,y,x); 'spacing_nm'
    on each layer is in the SAME axis order — pass it directly to
    fastmorph.spherical_erode(anisotropy=...) without flipping.
    Examples:
      "measure properties of mito" -> python_on_layers
        layers=["mito"]
        python="""
# mito is type=segmentation in describe_dataset → ALREADY instance-
# labeled. Do NOT call cc3d.connected_components — that re-IDs from 1
# and breaks the link to existing meshes/skeletons + click-to-fly.
# Pass the array directly to regionprops_table; the resulting 'label'
# column IS the existing segment id.
import numpy as np, pandas as pd
from skimage import measure as _sk
spacing = layers["mito"]["spacing"]   # array-axis order (z,y,x typically)
offsets = layers["mito"]["offsets"]   # SAME axis order as spacing
props = _sk.regionprops_table(
    mito,                              # the already-labeled volume
    spacing=spacing,
    properties=("label", "area", "centroid", "bbox", "equivalent_diameter"),
)
df = pd.DataFrame(props)
df["volume_nm_3"] = df["area"]
# centroid-{0,1,2} are in the SAME axis order as spacing — z,y,x.
# Convert to world-nm xyz, adding the per-axis offset.
ax = layers["mito"]["axes"]
def col(name):
    return f"centroid-{ax.index(name)}"
df["com_x_nm"] = df[col("x")] + offsets[ax.index("x")]
df["com_y_nm"] = df[col("y")] + offsets[ax.index("y")]
df["com_z_nm"] = df[col("z")] + offsets[ax.index("z")]
df["object_id"] = df["label"].astype(int)   # original segment ids preserved
_TG_TABLE = df[["object_id","volume_nm_3","com_x_nm","com_y_nm","com_z_nm","equivalent_diameter"]]
_TG_TABLE_NAME = "mito"
_TG_NARRATION = f"Measured {len(df)} mito components."
"""

      "erode mito by 50 nm" -> python_on_layers
        layers=["mito"]
        python="""
# Backend runtime auto-picks since fastmorph is imported.
import fastmorph
spacing = layers["mito"]["spacing"]   # array-axis order; pass directly
out = fastmorph.spherical_erode(mito, radius=50, anisotropy=spacing)
_TG_NEW_LAYER = {"array": out, "name": "mito_eroded_50nm",
                 "type": "segmentation",
                 "spacing": spacing,
                 "offsets": layers["mito"]["offsets"]}
"""

      "make meshes for mito + cilia" -> python_on_layers
        layers=["mito", "cilia"]
        python="""
# zmesh import → forces backend runtime. _TG_NEW_MESH_LAYER renders
# precomputed meshes one shot per layer.
import zmesh
results = []
for name in ["mito", "cilia"]:
    arr = layers[name]["array"]
    spacing = layers[name]["spacing"]   # array-axis order
    mesher = zmesh.Mesher(spacing)
    mesher.mesh(arr)
    # _TG_NEW_MESH_LAYER takes a labels volume and the harness writes
    # a precomputed mesh layer. One per loop iteration is fine; later
    # iterations overwrite earlier ones, so usually call this once per
    # turn — for two layers, two python_on_layers calls is cleanest.
_TG_NEW_MESH_LAYER = {"labels": layers["mito"]["array"],
                      "name": "mito_meshes",
                      "spacing": layers["mito"]["spacing"],
                      "offsets": layers["mito"]["offsets"]}
"""

      "length of the longest mito skeletons" ->
         (1) run_sql to pick the IDs (or use describe_dataset if no
             SQL table exists yet):
             SELECT object_id FROM mito ORDER BY volume_nm_3 DESC LIMIT 5
         (2) python_on_layers with ONLY skeletons (do NOT also pass
             layers=["mito"] — that loads the volume too and wastes
             hundreds of MB. The volume isn't needed for skeleton
             length.):
        skeletons=[{"layer":"mito","segment_ids":[...]}]
        # NO 'layers' field at all.
        python="""
# mito_skel = {seg_id: {"vertices": (N,3) nm, "edges": (M,2)}}
import numpy as np, pandas as pd
rows = []
for sid, sk in mito_skel.items():
    v = sk["vertices"]; e = sk["edges"]
    L = 0.0 if len(e) == 0 else float(
        np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).sum()
    )
    # Skeleton-vertex centroid as a stand-in COM. If the prior turn
    # already saved real volume-derived COMs, the merge step below
    # keeps those (the merged-by-object_id behaviour means existing
    # com_x_nm/y/z survive when we also write com_x_nm here, and
    # the merge picks new values column-by-column — i.e. our
    # centroid wins where it overlaps; harmless either way).
    c = v.mean(axis=0) if len(v) else np.array([0.0, 0.0, 0.0])
    rows.append({
        "object_id": int(sid),
        "length_nm": L,
        "com_x_nm": float(c[0]),
        "com_y_nm": float(c[1]),
        "com_z_nm": float(c[2]),
    })
df = pd.DataFrame(rows).sort_values("length_nm", ascending=False).reset_index(drop=True)
_TG_TABLE = df
_TG_TABLE_NAME = "mito"   # merges into the existing mito table by
                          # object_id — prior columns (volume_nm_3,
                          # surface_area_nm_2, ...) survive untouched.
top = df.iloc[0]
_TG_NARRATION = f"Longest mito by skeleton length: {int(top.object_id)} ({top.length_nm/1000:.1f} um)"
_TG_FLY = {"layer": "mito", "segment_id": str(int(top.object_id)),
           "pos": [float(top.com_x_nm), float(top.com_y_nm), float(top.com_z_nm)]}
"""

      "volume of these segments" on a precomputed-mesh-only layer
      (no zarr volume, no skeletons — describe_dataset reports
      has_mesh=true, mesh_analysis_supported=true) ->
        meshes=[{"layer":"segmentation","segment_ids":[...]}]
        # NO 'layers' field — that would try to read precomputed
        # volume voxels which isn't supported.
        python="""
# seg_mesh = {seg_id: {"vertices": (N,3) f32 nm, "faces": (M,3) u32}}
# Compute volume + surface area + centroid from the triangle mesh:
#   Volume via signed-tet integration: V = (1/6) * sum |v0 · (v1×v2)|
#     for each triangle (works for closed meshes; absolute value
#     handles non-CCW windings).
#   Surface area: sum of 0.5 * |(v1-v0) × (v2-v0)|.
#   Centroid: area-weighted triangle centroid.
import numpy as np, pandas as pd
rows = []
for sid, m in seg_mesh.items():
    v = m["vertices"]; f = m["faces"]
    if len(f) == 0:
        rows.append({"object_id": int(sid), "volume_nm_3": 0.0,
                     "surface_area_nm_2": 0.0,
                     "com_x_nm": 0.0, "com_y_nm": 0.0, "com_z_nm": 0.0})
        continue
    v0 = v[f[:, 0]]; v1 = v[f[:, 1]]; v2 = v[f[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_area = 0.5 * np.linalg.norm(cross, axis=1)
    surf_area = float(tri_area.sum())
    tet_vol = np.einsum("ij,ij->i", v0, cross) / 6.0
    vol = float(abs(tet_vol.sum()))
    tri_cent = (v0 + v1 + v2) / 3.0
    com = (tri_cent * tri_area[:, None]).sum(axis=0) / max(surf_area, 1e-12)
    rows.append({
        "object_id": int(sid),
        "volume_nm_3": vol,
        "surface_area_nm_2": surf_area,
        "com_x_nm": float(com[0]),
        "com_y_nm": float(com[1]),
        "com_z_nm": float(com[2]),
    })
df = pd.DataFrame(rows).sort_values("volume_nm_3", ascending=False).reset_index(drop=True)
_TG_TABLE = df
_TG_TABLE_NAME = "segmentation"   # merges into existing table by object_id
top = df.iloc[0]
_TG_NARRATION = f"Largest segment by mesh volume: {int(top.object_id)} ({top.volume_nm_3/1e9:.2f} um^3)"
"""

  answer(text: string)
    Deliver a final text answer to the user. Use for counts, means,
    informational questions. Keep it concise.

  done()
    End this turn. You MUST call this after delivering any answer / plot / fly_to.

CRITICAL — TABLE AND COLUMN NAMES VARY PER DATASET. The schema above is
the single source of truth. Tourguide normalizes incoming columns at
ingest, so most tables use canonical names with explicit nm units:
  - com_x_nm / com_y_nm / com_z_nm        (center of mass in nm,
                                           = voxel_centroid * voxel_size + offset)
  - volume_nm_3 / surface_area_nm_2       (sizes)
  - bbox_min_x_nm / bbox_max_x_nm / etc.  (bounding box, nm)
  - equivalent_diameter_nm                (sphere diameter, nm)
  - object_id                             (segment id)
All world coordinates are nanometers in Neuroglancer's reference frame:
positions account for the layer's voxel size AND its offset. Pass
them straight to fly_to without scaling.

…but older / hand-authored / shared tables can have other names
('position_x', 'centroid_x', 'volume', etc.). ALWAYS pick names from
the schema listed above. NEVER invent column names from a typical-
flow example — those are illustrative, not literal. The same goes
for table names: the actual table for a class might be 'mito',
'mito_computed_s0', 'mitochondria', or whatever's listed.

CRITICAL — fly_to expects POSITION coordinates in nanometers, NOT a
volume / surface_area / id. ALWAYS include position columns (whatever
they're called in the schema) in your SQL when you plan to fly_to a
result. Pass the row's position values as fly_to's position argument.

TYPICAL FLOWS (placeholders in <angle brackets> — substitute with the
actual schema names):
  "show me the largest <class>"      -> run_sql (SELECT object_id, <pos_x>, <pos_y>, <pos_z>, <size>
                                                  FROM <table> ORDER BY <size> DESC LIMIT 1)
                                        -> fly_to (position = [<pos_x>, <pos_y>, <pos_z>] from row,
                                                   object_id from row) -> done
  "how many <class>?"                 -> run_sql (SELECT COUNT(*) FROM <table>) -> answer -> done
  "plot <class> volumes"              -> make_plot -> done
  "fly through the 3 biggest <class>" -> run_sql (... ORDER BY <size> DESC LIMIT 3)
                                        -> fly_to (row 1) -> fly_to (row 2) -> fly_to (row 3) -> done
  "show only the largest 10 <class>"  -> run_sql (SELECT object_id FROM <table>
                                                  ORDER BY <size> DESC LIMIT 10)
                                        -> highlight_segments (ids) -> done
  "densest region of <class>"         -> run_python on com_<class> (already an Nx3 numpy array
                                        of COMs in nm — DO NOT reconstruct from df_<class>):
                                          bins = 20
                                          coms = com_<class>
                                          idx = np.floor((coms - coms.min(0)) /
                                                         ((coms.max(0) - coms.min(0)) / bins)
                                                        ).astype(int).clip(0, bins-1)
                                          flat = idx[:,0]*bins*bins + idx[:,1]*bins + idx[:,2]
                                          counts = np.bincount(flat, minlength=bins**3)
                                          b = np.unravel_index(counts.argmax(), (bins, bins, bins))
                                          cell = (coms.max(0) - coms.min(0)) / bins
                                          center = coms.min(0) + (np.array(b) + 0.5) * cell
                                          print(center)
                                          _out = center.tolist()
                                        -> fly_to (position = center) -> done
  "median volume of <class>"          -> run_python (df_<class>[<size_col>].median(); print)
                                        -> answer -> done
  "two closest <class> pairs"         -> run_python on com_<class>: pdist + argmin; set
                                        _out = {"ids": [id1, id2], "coms": [[x,y,z],[x,y,z]]}
                                        so the next tool uses both without another SQL lookup.
                                        -> highlight_segments (ids) -> done

WHEN TO USE WHICH TOOL — IMPORTANT:
  - Filter / sort / count / "the N biggest/smallest"           -> run_sql
  - Distribution shape, scatter, histogram, "show on a chart"  -> make_plot
  - Anything spatial (density, clusters, neighbors, distance,
    "where are most of the X", "region", "near", bounding box,
    convex hull) or anything statistical beyond simple aggregates
    (median, percentile, std, IQR, correlation, regression)    -> run_python
  - Operate on layer voxels (erode, dilate, threshold,
    skeletonize, mesh, contact area between two label volumes)
    OR persist a NEW LAYER ("add an eroded mito layer",
    "show me the boundaries of X")                              -> python_on_layers
  - Never reach for run_sql when the question implies geometric
    reasoning over position columns — SQL can't compute density
    or pairwise distances. ORDER BY <size> DESC LIMIT 1 is NOT
    "the densest <class>".
  - run_python operates on the SQL-derived DataFrames (df_<class>)
    and pre-extracted COMs (com_<class>); it does NOT have access
    to layer voxels. If you need pixels, use python_on_layers.
  - Whenever python_on_layers computes per-object metrics (volume,
    COM, surface area, regionprops of any kind), ALWAYS set _TG_TABLE
    so the metrics persist in the SQL DB for the rest of the
    session. The next turn (or a follow-up question) can then run_sql
    against them instead of re-running the heavy computation.
    If the user asked to "plot X", the natural flow is: compute the
    table once (set _TG_TABLE) AND set _TG_PLOT in the same call —
    don't compute and discard.
  - REQUIRED COLUMNS for any _TG_TABLE you save: at minimum
        object_id, com_x_nm, com_y_nm, com_z_nm
    on top of whatever metric you computed (volume_nm_3,
    surface_area_nm_2, etc.). The structured browser uses these to
    let the user click a row and fly the viewer to that segment;
    skipping them breaks click-to-fly silently.
  - TABLES MERGE BY object_id, NOT REPLACE: when you set
    _TG_TABLE_NAME to a name that already exists in the SQL DB AND
    both the existing table and your new DataFrame have an
    object_id column, the harness MERGES on object_id — new columns
    are added to existing rows, new values overwrite old ones for
    columns that overlap, and existing rows whose object_id isn't
    in your new DataFrame are preserved untouched. Two consequences:
      (a) When you compute a NEW metric for an existing organelle
          class (e.g. you've already saved 'mito' with volume + COM,
          and now you're computing skeleton length), DON'T re-emit
          the old columns — just object_id + the new metric. The
          merge brings everything together.
      (b) Re-running the same computation is safe — same columns +
          object_ids → in-place update, no row duplication.
  - REQUIRED _TG_TABLE_NAME: when the table is per-object metrics
    for a layer X, name the table EXACTLY "X" (not "X_volumes" /
    "X_metrics" / "X_data"). The browser uses the table name as the
    Neuroglancer layer to highlight on click; mismatched names break
    the highlight. Set with: _TG_TABLE_NAME = "mito" (not
    "mito_volumes").
  - ALREADY-LABELED LAYERS — CRITICAL: Layers with type="segmentation"
    in describe_dataset are ALREADY instance-labeled (each unique
    non-zero value is a segment id, matching the precomputed mesh /
    skeleton sources for that layer). Do NOT run cc3d.connected_components
    on them — that reassigns ids 1..N, which:
      • breaks the link to existing meshes / skeletons (NG renders
        mesh for the OLD id, not the new one),
      • silently breaks click-to-fly (the row's object_id no longer
        matches the rendered segment).
    Pass the array directly to regionprops_table as the labels arg:
        props = regionprops_table(mito, spacing=spacing,
                                  properties=("label", "area", "centroid", ...))
    The "label" column IS the original segment id; copy it into
    object_id verbatim. Use np.unique(arr[arr > 0]) if you only need
    the id list without metrics.
    For ambiguous cases (image-typed binary mask, etc.), call
    ask_user with a yesno is_already_labeled (default true for
    type=segmentation, false otherwise).

BUDGET: You have AT MOST 5 tool calls per question. Plan accordingly — most flows above finish in 2-4. If you can't reach an answer in 5, end with answer() explaining what's missing rather than running out silently.

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
  // Models occasionally emit args as a bare string instead of an object —
  // {tool: "run_sql", args: "SELECT ..."} rather than args: {sql: "..."}.
  // Wrap it under the tool's primary key so the executor sees something
  // usable. Other tools that take just a string (answer, fly_to with
  // segmented args) keep their structure unchanged.
  if (typeof parsed.args === "string") {
    const primaryKey: Record<string, string> = {
      run_sql: "sql",
      run_python: "python",
      make_plot: "python",
      answer: "text",
    };
    const key = primaryKey[parsed.tool];
    if (key) {
      parsed.args = { [key]: parsed.args } as Record<string, unknown>;
    } else {
      parsed.args = {};
    }
  }
  parsed.args = parsed.args ?? {};
  return parsed;
}

const TERMINAL_TOOLS = new Set(["answer", "make_plot"]);
// Tools that produce something the user actually sees — viewer
// changes count, since the camera move IS the answer to "show me X".
// Used to decide whether `done` is acceptable (vs. an empty-trace bail).
const DELIVERED_BY_TOOL = new Set(["answer", "make_plot", "fly_to", "highlight_segments"]);

function guardSqlReadOnly(sql: string): void {
  const banned = /\b(create|drop|alter|insert|update|delete|attach|pragma)\b/i;
  if (banned.test(sql)) throw new Error(`SQL must be SELECT only`);
}

// Accept a code-like argument under any of the common key names a
// model might pick (python, code, script, source, query, sql), and
// stitch back arrays-of-lines that some models emit instead of a
// single string. Returns "" if nothing usable is found.
function coerceCodeArg(args: Record<string, unknown>, keys: string[]): string {
  for (const k of keys) {
    const v = args[k];
    if (v === undefined || v === null) continue;
    if (typeof v === "string") return v.trim();
    if (Array.isArray(v) && v.every((x) => typeof x === "string")) {
      return (v as string[]).join("\n").trim();
    }
    // Last-ditch: stringify whatever it is. Only useful if the model
    // wrapped the code in {body: "..."} or similar.
    return String(v).trim();
  }
  return "";
}

async function execRunSql(
  args: Record<string, unknown>,
  ctx: AgentContext,
): Promise<{ columns: string[]; rows: unknown[][] }> {
  if (!ctx.db) throw new Error("No database loaded");
  const sql = coerceCodeArg(args, ["sql", "query", "statement", "code"]);
  if (!sql) throw new Error("run_sql requires 'sql'");
  guardSqlReadOnly(sql);
  const limitedSql = /\blimit\b/i.test(sql) ? sql : `${sql.replace(/;$/, "")} LIMIT 50`;
  const result = runQuery(ctx.db.db, limitedSql);
  return { columns: result.columns, rows: result.rows };
}

// Sanity bound for fly_to positions. EM volumes top out at ~1 mm
// (1e6 nm) per axis — anything above this is the model confusing a
// volume / surface_area / object_id value for a coordinate. Reject
// loudly so the model can self-correct (typical mistake: SELECT'd
// volume but not com_x/y/z, then put volume into position[0]).
const FLY_TO_MAX_NM = 1e7; // 10 mm — generous upper bound

async function execFlyTo(
  args: Record<string, unknown>,
  ctx: AgentContext,
): Promise<{ ok: boolean }> {
  const pos = args.position as unknown;
  if (!Array.isArray(pos) || pos.length !== 3 || pos.some((n) => typeof n !== "number")) {
    throw new Error("fly_to requires position: [x, y, z] numbers");
  }
  // Reject absurd magnitudes — these come from the model substituting
  // a volume / surface_area / object_id into a position slot. A 12-
  // billion-nm 'x' is 12 meters; obviously not a dataset coord.
  for (let i = 0; i < 3; i++) {
    const v = (pos as number[])[i];
    if (!Number.isFinite(v) || Math.abs(v) > FLY_TO_MAX_NM) {
      throw new Error(
        `fly_to position[${i}] = ${v} nm is out of dataset range. Positions must come from the schema's center-of-mass columns (typically com_x_nm / com_y_nm / com_z_nm — check the schema), in nanometers. SELECT those columns alongside object_id, then pass them as fly_to's position. Do NOT pass volume / surface_area / object_id as position.`,
      );
    }
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
  const code = coerceCodeArg(args, ["python", "code", "script", "source", "body"]);
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
  // See execRunPython for why we auto-load imports — same problem
  // here (e.g. 'import seaborn' would otherwise fail).
  try {
    await py.loadPackagesFromImports(code);
  } catch (err) {
    console.warn("[agent] loadPackagesFromImports failed:", (err as Error).message);
  }
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

// python_on_layers — heavy-lift counterpart to run_python / make_plot.
// Routes through AnalysisClient.customAnalyze (the same worker engine the
// Custom Python dialog uses), so the agent can:
//   - load actual layer voxels as numpy arrays
//   - persist a derived volume via _TG_NEW_LAYER (e.g. eroded mito)
//   - render a mesh layer via _TG_NEW_MESH_LAYER
//   - add a remote source layer via _TG_ADD_SOURCE_LAYER
//   - emit annotations / fly / highlight / plot / narration the same way
// Auto-picks the coarsest scale under SAFE_INPUT_BYTES so the model never
// has to think about scale paths. Errors back with an explicit message
// when a requested layer isn't a zarr source (skeleton / precomputed
// mesh layers aren't yet supported by the underlying engine).
async function execPythonOnLayers(
  args: Record<string, unknown>,
  ctx: AgentContext,
): Promise<{ summary: string; produced: string[] }> {
  if (!ctx.descriptor) throw new Error("No dataset loaded");
  const code = coerceCodeArg(args, ["python", "code", "script", "source", "body"]);
  if (!code) throw new Error("python_on_layers requires 'python'");
  const layersArg = args.layers;
  const skeletonsArg = args.skeletons;
  const meshesArgEarly = args.meshes;
  const hasLayers = Array.isArray(layersArg) && layersArg.length > 0;
  const hasSkeletons = Array.isArray(skeletonsArg) && skeletonsArg.length > 0;
  const hasMeshesEarly = Array.isArray(meshesArgEarly) && meshesArgEarly.length > 0;
  if (!hasLayers && !hasSkeletons && !hasMeshesEarly) {
    throw new Error(
      "python_on_layers requires at least one of: 'layers' (zarr layer names), 'skeletons' ([{layer, segment_ids}]), or 'meshes' ([{layer, segment_ids}]).",
    );
  }
  const layerNames = hasLayers ? (layersArg as unknown[]).map((v) => String(v)) : [];

  // Resolve each name to a descriptor entry. Reject up front rather than
  // letting the worker fail on a bad URL — the model will recover faster
  // from a clear "no such layer" than from a worker error.
  const resolved: { name: string; layer: DatasetLayer }[] = [];
  for (const name of layerNames) {
    const layer = ctx.descriptor.layers.find((l) => l.name === name);
    if (!layer) {
      const known = ctx.descriptor.layers.map((l) => l.name).join(", ");
      throw new Error(
        `No layer named '${name}'. Loaded layers: ${known || "(none)"}. Call describe_dataset to list them.`,
      );
    }
    if (!isZarrSource(layer.source)) {
      // Differentiate precomputed-volume from other non-zarr sources
      // so the error tells the model what's actually possible, not
      // just what failed. Precomputed segmentations (e.g. hemibrain
      // at precomputed://gs://...) host volume + mesh + skeleton
      // inside the same dir — the volume voxels aren't readable by
      // python_on_layers (no precomputed-volume browser reader yet),
      // but bundled meshes / skeletons may still be accessible.
      const sources = Array.isArray(layer.source) ? layer.source : [layer.source];
      const isPrecomputed = sources.some((s) => /^precomputed:\/\//.test(s));
      // Path-based skeleton detection first.
      let skelHint = "";
      const pathSkel = sources.find((s) => /\/skeleton\b|\/skeletons?\//i.test(s));
      if (pathSkel) {
        skelHint = ` This layer DOES have a skeleton source — try skeletons=[{"layer":"${name}","segment_ids":[...]}] for length / connectivity / branching metrics.`;
      } else if (isPrecomputed) {
        // Probe the precomputed info file (cached) to see if
        // skeletons / meshes are bundled. For meshes, also check
        // the format — only legacy non-sharded is currently
        // analyzable. Tells the agent yes/no up front so it
        // doesn't have to call describe_dataset separately just
        // to discover this.
        let bundledSkel: string | null = null;
        let bundledMesh: string | null = null;
        for (const s of sources) {
          if (!/^precomputed:\/\//i.test(s)) continue;
          if (!bundledSkel) bundledSkel = await resolveBundledSubpath(s, "skeletons");
          if (!bundledMesh) bundledMesh = await resolveBundledSubpath(s, "mesh");
          if (bundledSkel && bundledMesh) break;
        }
        let meshAnalyzable = false;
        if (bundledMesh) {
          const minfo = await probeMeshInfo(bundledMesh);
          if (minfo) {
            meshAnalyzable = minfo.atType === "neuroglancer_legacy_mesh" && !minfo.isSharded;
          }
        }
        if (bundledSkel && meshAnalyzable) {
          skelHint = ` This precomputed dir bundles BOTH skeletons and (legacy-format) meshes. Pass skeletons=[{"layer":"${name}","segment_ids":[...]}] for length / branching, OR meshes=[{"layer":"${name}","segment_ids":[...]}] for volume / surface area.`;
        } else if (bundledSkel) {
          skelHint = ` This precomputed dir bundles skeletons (per its info file) — pass skeletons=[{"layer":"${name}","segment_ids":[...]}] for length / branching metrics.`;
        } else if (meshAnalyzable) {
          skelHint = ` This precomputed dir bundles (legacy-format) meshes — pass meshes=[{"layer":"${name}","segment_ids":[...]}] for volume / surface area.`;
        } else if (bundledMesh) {
          skelHint = ` This precomputed dir bundles meshes, but the on-disk format isn't 'neuroglancer_legacy_mesh' or is sharded — Tourguide can't analyze it yet (no draco / shard reader).`;
        } else {
          skelHint = ` Probed the precomputed info file — no bundled meshes or skeletons.`;
        }
      }
      if (isPrecomputed) {
        throw new Error(
          `Layer '${name}' is a precomputed segmentation, not a zarr volume. python_on_layers can't read precomputed-volume voxels directly (no in-browser reader for that format), so volume / regionprops on the voxels isn't available here.${skelHint}`,
        );
      }
      throw new Error(
        `Layer '${name}' is not a zarr source — pass skeleton sources via 'skeletons' instead of 'layers'.`,
      );
    }
    resolved.push({ name, layer });
  }

  // Resolve skeleton entries. Each entry maps a layer name + segment IDs
  // to the precomputed-skeleton source URL on that layer, suffixing the
  // varName with "_skel" so it doesn't collide if the same name also
  // appears in 'layers' (volume).
  interface SkelResolved {
    varName: string;
    source: string;
    segmentIds: string[];
    offsetNm?: [number, number, number];
  }
  const resolvedSkeletons: SkelResolved[] = [];
  if (hasSkeletons) {
    for (const raw of skeletonsArg as unknown[]) {
      if (!raw || typeof raw !== "object") {
        throw new Error("Each entry in 'skeletons' must be an object with 'layer' and 'segment_ids'.");
      }
      const r = raw as Record<string, unknown>;
      const layerName = String(r.layer ?? r.name ?? "").trim();
      if (!layerName) throw new Error("skeletons[].layer is required");
      const idsRaw = r.segment_ids ?? r.ids;
      if (!Array.isArray(idsRaw) || idsRaw.length === 0) {
        throw new Error(`skeletons['${layerName}'].segment_ids is required (non-empty list).`);
      }
      const segmentIds = (idsRaw as unknown[]).map((v) => String(v));
      const layer = ctx.descriptor.layers.find((l) => l.name === layerName);
      if (!layer) {
        throw new Error(`No layer named '${layerName}' for skeleton input.`);
      }
      const sources = Array.isArray(layer.source) ? layer.source : [layer.source];
      // 1. Path-based: a separate source URL whose path contains
      //    /skeleton/ or /skeletons/ (Janelia / cellmap convention).
      let skelSource = sources.find(
        (s) => /\/skeleton\b|\/skeletons?\//i.test(s) || /^precomputed:\/\/.*skeleton/i.test(s),
      );
      // 2. Bundled-in-precomputed: hemibrain-style segmentations
      //    declare a `skeletons` subkey inside their info file. Probe
      //    each precomputed source's info — first one that reports a
      //    skeletons subpath wins. Cached per session.
      if (!skelSource) {
        for (const s of sources) {
          if (!/^precomputed:\/\//i.test(s)) continue;
          const resolved = await resolveBundledSubpath(s, "skeletons");
          if (resolved) {
            skelSource = resolved;
            break;
          }
        }
      }
      if (!skelSource) {
        throw new Error(
          `Layer '${layerName}' has no skeleton source — no source URL contains /skeleton/, and no precomputed source declared a 'skeletons' subkey in its info file. Call describe_dataset to confirm.`,
        );
      }
      const safeVar = layerName.replace(/[^a-zA-Z0-9_]/g, "_") || "layer";
      resolvedSkeletons.push({
        varName: `${safeVar}_skel`,
        source: skelSource,
        segmentIds,
        // Same NG-state translation we apply to the volume offset.
        // The worker shifts every skeleton vertex by this so the
        // skeleton stays aligned with the volume in analysis math.
        offsetNm: layer.transform_offset_nm,
      });
    }
  }

  // Resolve mesh entries — parallel to the skeleton loop above. Only
  // legacy single-LOD unsharded meshes are readable right now; format
  // probe gates this so the agent gets a clear error for multilod_draco
  // / sharded variants instead of fetching binary that won't parse.
  const meshesArg = args.meshes;
  const hasMeshes = Array.isArray(meshesArg) && meshesArg.length > 0;
  interface MeshResolved {
    varName: string;
    source: string;
    segmentIds: string[];
    offsetNm?: [number, number, number];
  }
  const resolvedMeshes: MeshResolved[] = [];
  if (hasMeshes) {
    for (const raw of meshesArg as unknown[]) {
      if (!raw || typeof raw !== "object") {
        throw new Error("Each entry in 'meshes' must be an object with 'layer' and 'segment_ids'.");
      }
      const r = raw as Record<string, unknown>;
      const layerName = String(r.layer ?? r.name ?? "").trim();
      if (!layerName) throw new Error("meshes[].layer is required");
      const idsRaw = r.segment_ids ?? r.ids;
      if (!Array.isArray(idsRaw) || idsRaw.length === 0) {
        throw new Error(`meshes['${layerName}'].segment_ids is required (non-empty list).`);
      }
      const segmentIds = (idsRaw as unknown[]).map((v) => String(v));
      const layer = ctx.descriptor.layers.find((l) => l.name === layerName);
      if (!layer) {
        throw new Error(`No layer named '${layerName}' for mesh input.`);
      }
      const sources = Array.isArray(layer.source) ? layer.source : [layer.source];
      // 1. Path-based mesh source
      let meshSource: string | null =
        sources.find((s) => /\/mesh\b|\/meshes?\//i.test(s)) ?? null;
      // 2. Bundled-in-precomputed (hemibrain pattern)
      if (!meshSource) {
        for (const s of sources) {
          if (!/^precomputed:\/\//i.test(s)) continue;
          const resolved = await resolveBundledSubpath(s, "mesh");
          if (resolved) {
            meshSource = resolved;
            break;
          }
        }
      }
      if (!meshSource) {
        throw new Error(
          `Layer '${layerName}' has no mesh source. Call describe_dataset; mesh_format will be null when no mesh is available.`,
        );
      }
      // 3. Probe the mesh info to confirm format is supported.
      const minfo = await probeMeshInfo(meshSource);
      if (!minfo) {
        throw new Error(
          `Couldn't read mesh info for '${layerName}' at ${meshSource}. Network error or unsupported format.`,
        );
      }
      if (minfo.atType !== "neuroglancer_legacy_mesh" || minfo.isSharded) {
        throw new Error(
          `Mesh format '${minfo.atType || "(unknown)"}'${minfo.isSharded ? " (sharded)" : ""} on '${layerName}' isn't supported yet. Only 'neuroglancer_legacy_mesh' (unsharded) can be loaded; multi-LOD draco + sharded variants need a draco decoder + shard reader (not yet implemented). Skeletons or zarr volumes are the alternatives.`,
        );
      }
      const safeVar = layerName.replace(/[^a-zA-Z0-9_]/g, "_") || "layer";
      resolvedMeshes.push({
        varName: `${safeVar}_mesh`,
        source: meshSource,
        segmentIds,
        offsetNm: layer.transform_offset_nm,
      });
    }
  }

  // Decide which runtime (local Pyodide worker vs HF backend) to use.
  // 'auto' is the default, but a few signals force one or the other:
  //   - skeletons present → must run local (HF backend doesn't accept
  //     skeletonLayers yet; sending would NameError on <var>_skel)
  //   - code imports any Seung-lab pkg or sets _TG_NEW_MESH_LAYER →
  //     must run on backend (zmesh / fastmorph / cc3d / kimimaro / edt
  //     / fastremap aren't in Pyodide)
  // When neither is forced, we prefer local for speed/quota and only
  // fall through to backend when the input is too big for Pyodide's
  // ~4 GB WASM ceiling.
  const requestedRuntime = String(args.runtime ?? "auto").toLowerCase();
  const settings = loadSettings();
  const backendUrl = settings.analysisBackendUrl.trim();
  const codeImportsSeungLab = /\b(?:import|from)\s+(?:cc3d|fastmorph|fastremap|edt|kimimaro|zmesh)\b/.test(code);
  const codeNeedsMeshLayer = /\b_TG_NEW_MESH_LAYER\s*=/.test(code);
  const mustBeLocal = resolvedSkeletons.length > 0 || resolvedMeshes.length > 0;
  const mustBeBackend = codeImportsSeungLab || codeNeedsMeshLayer;
  if (mustBeLocal && mustBeBackend) {
    throw new Error(
      "Conflict: this code uses Seung-lab packages or _TG_NEW_MESH_LAYER (backend-only) but also has skeleton inputs (local-only). Split into two python_on_layers calls — one for skeleton work, one for mesh/heavy ops.",
    );
  }
  // The size-vs-budget decision needs inspect results, so we run inspect
  // first, then resolve effective runtime. ESTIMATED total bytes feeds
  // both the runtime choice and the per-layer scale pick.
  ctx.callbacks.onProgress?.(
    resolved.length > 0
      ? "python_on_layers: inspecting layers…"
      : "python_on_layers: preparing skeletons…",
  );
  // Pyodide WASM hits OOM around ~1.5 GB of input (intermediates blow
  // up). HF backend has 16 GB but we want headroom; targeting ≤5 GB of
  // input keeps regionprops + ndi + a couple of intermediates safe.
  const LOCAL_TARGET_BYTES = SAFE_INPUT_BYTES; // 1.5 GB
  const BACKEND_TARGET_BYTES = 5 * 1024 ** 3;
  const client = new AnalysisClient();
  const layerInspections: { name: string; layer: DatasetLayer; url: string; insp: Awaited<ReturnType<typeof client.inspect>> }[] = [];
  const finestBytes: number[] = [];
  try {
    for (const { name, layer } of resolved) {
      const url = normalizeZarrUrl(layer.source);
      const insp = await client.inspect(url, ctx.descriptor.voxel_size_nm, (m) =>
        ctx.callbacks.onProgress?.(`Inspecting ${name}: ${m}`),
      );
      layerInspections.push({ name, layer, url, insp });
      // Index 0 is the finest scale. Track to gauge whether even the
      // finest fits in our target — if yes for everyone, no need to
      // route through backend purely for size.
      if (insp.scales[0]) finestBytes.push(insp.scales[0].approxBytes);
    }

    // Effective runtime resolution.
    let runtime: "local" | "backend";
    let runtimeReason: string;
    if (requestedRuntime === "local" || requestedRuntime === "backend") {
      runtime = requestedRuntime;
      runtimeReason = `requested explicitly`;
    } else if (mustBeLocal) {
      runtime = "local";
      runtimeReason = "skeleton inputs require local runtime";
    } else if (mustBeBackend) {
      if (!backendUrl) {
        throw new Error(
          "This code needs the HF backend (Seung-lab packages or _TG_NEW_MESH_LAYER) but no analysis backend URL is configured. Open Settings → Advanced → Analysis backend.",
        );
      }
      runtime = "backend";
      runtimeReason = "code uses Seung-lab packages or _TG_NEW_MESH_LAYER";
    } else {
      const totalFinest = finestBytes.reduce((a, b) => a + b, 0);
      if (totalFinest <= LOCAL_TARGET_BYTES) {
        runtime = "local";
        runtimeReason = `${humanBytes(totalFinest)} fits Pyodide`;
      } else if (backendUrl) {
        runtime = "backend";
        runtimeReason = `finest scale total ${humanBytes(totalFinest)} > Pyodide budget`;
      } else {
        runtime = "local";
        runtimeReason = "no backend URL configured; will downsample";
      }
    }
    const targetBytes = runtime === "backend" ? BACKEND_TARGET_BYTES : LOCAL_TARGET_BYTES;

    // Per-layer scale: pick the FINEST scale that still fits the
    // chosen runtime's byte budget. If even the coarsest is bigger
    // than the budget, fall back to coarsest (will likely OOM but
    // surfaces the issue with a clear runtime+scale message).
    const layersForRequest: {
      varName: string;
      url: string;
      scalePath: string;
      axesOrder: string[];
      voxelNm: [number, number, number];
      offsetNm: [number, number, number];
      // Diagnostics we surface in the result narration.
      _scalePath: string;
      _approxBytes: number;
      _layerName: string;
      _shape: number[];
      _voxelNm: [number, number, number];
    }[] = [];
    for (const { layer, url, insp } of layerInspections) {
      let idx = insp.scales.length - 1; // coarsest fallback
      for (let i = 0; i < insp.scales.length; i++) {
        if (insp.scales[i].approxBytes <= targetBytes) {
          idx = i;
          break;
        }
      }
      const scale = insp.scales[idx];
      const varName = (layer.organelle_class ?? layer.name).replace(/[^a-zA-Z0-9_]/g, "_") || "layer";
      // Add the user-applied NG-state translation on top of the
      // zarr's intrinsic offset. Without this, per-object positions
      // come back in the raw zarr frame and fly_to lands in the
      // wrong place visually when the user has moved the layer in
      // NG. transform_offset_nm is xyz; scale.offsetNm is also xyz.
      const tx = layer.transform_offset_nm ?? [0, 0, 0];
      const effectiveOffset: [number, number, number] = [
        scale.offsetNm[0] + tx[0],
        scale.offsetNm[1] + tx[1],
        scale.offsetNm[2] + tx[2],
      ];
      layersForRequest.push({
        varName,
        url,
        scalePath: scale.path,
        axesOrder: insp.axes.map((a) => a.name),
        voxelNm: scale.voxelNm,
        offsetNm: effectiveOffset,
        _scalePath: scale.path,
        _approxBytes: scale.approxBytes,
        _layerName: layer.name,
        _shape: scale.shape,
        _voxelNm: scale.voxelNm,
      });
    }

    // Tables already in the SQL DB are exposed as df_<class> just like
    // run_python does, so a python_on_layers script can mix layer
    // voxels with organelle metadata (e.g. erode mito where volume > X).
    const tables = (ctx.db?.tables ?? []).map((t) => {
      const result = runQuery(ctx.db!.db, `SELECT * FROM "${t.table_name}"`);
      return {
        name: t.organelle_class,
        columns: result.columns,
        rows: result.rows as (number | string | null)[][],
      };
    });

    const fmtVoxel = (v: [number, number, number]): string =>
      v.map((x) => (Number.isInteger(x) ? String(x) : x.toFixed(1))).join("×");
    const scaleBlurb = layersForRequest
      .map((l) => `${l._layerName}@${l._scalePath || "root"} (${humanBytes(l._approxBytes)})`)
      .join(", ");
    // Detailed per-layer line surfaces shape + voxel size + total
    // voxel count alongside the scale path. Shows up under the
    // backend/local + scale blurb so the user can tell whether
    // they got coarse or fine data without expanding the trace.
    const detailLines = layersForRequest.map((l) => {
      const numVox = l._shape.reduce((a, b) => a * b, 1);
      return `   shape ${l._shape.join("×")} · ${fmtVoxel(l._voxelNm)} nm/vox · ${numVox.toLocaleString()} voxels`;
    });
    const runtimeLabel = runtime === "backend"
      ? "🖥️ started on HF backend"
      : "💻 started locally (in-browser Pyodide)";
    const metaLine = `${runtimeLabel} · ${scaleBlurb}${
      resolvedSkeletons.length > 0 ? ` · ${resolvedSkeletons.length} skeleton input${resolvedSkeletons.length === 1 ? "" : "s"}` : ""
    }\n${detailLines.join("\n")}`;
    ctx.callbacks.onMeta?.(metaLine);
    ctx.callbacks.onProgress?.(
      `python_on_layers: running on ${runtime} — ${runtimeReason}. ${scaleBlurb}${
        resolvedSkeletons.length > 0 ? ` + ${resolvedSkeletons.length} skeleton input${resolvedSkeletons.length === 1 ? "" : "s"}` : ""
      }`,
    );

    const requestLayers = layersForRequest.map(({ _scalePath, _approxBytes, _layerName, ...rest }) => {
      void _scalePath;
      void _approxBytes;
      void _layerName;
      return rest;
    });

    let result: CustomAnalysisResult;
    if (runtime === "backend") {
      // Cold-start tolerance — if the Space is asleep, this can take a
      // few minutes. waitForBackendReady streams progress messages.
      await waitForBackendReady(backendUrl, {
        onProgress: (_state, msg) => ctx.callbacks.onProgress?.(msg),
      });
      const sessionId =
        typeof crypto?.randomUUID === "function"
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      result = await postAnalysisRequest(
        backendUrl,
        {
          layers: requestLayers,
          tables,
          code,
          timeoutMs: 300_000,
          sessionId,
        },
        ctx.signal,
      );
    } else {
      result = await client.customAnalyze(
        {
          kind: "custom",
          layers: requestLayers,
          skeletonLayers: resolvedSkeletons.length > 0 ? resolvedSkeletons : undefined,
          meshLayers: resolvedMeshes.length > 0 ? resolvedMeshes : undefined,
          tables,
          code,
          timeoutMs: 60000,
        },
        (m) => ctx.callbacks.onProgress?.(m),
      );
    }

    const applied = await applyCustomResult(result, code, ctx);
    // Prepend runtime + scale info so the next agent turn (and the
    // visible answer) reflect what we actually did. The user's choice
    // explicitly: "let em know which".
    const runtimeLine = `Ran on ${runtime} (${runtimeReason}); scales: ${scaleBlurb}.`;
    return {
      summary: `${runtimeLine}\n${applied.summary}`,
      produced: applied.produced,
    };
  } finally {
    client.terminate();
  }
}

function humanBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

// Wire a CustomAnalysisResult through the agent's callbacks + viewer.
// Mirrors what custom_analysis_ui's renderOutput does for layer-add
// effects — keeps the two surfaces behaviourally consistent without
// re-extracting the renderer in this commit.
async function applyCustomResult(
  result: CustomAnalysisResult,
  code: string,
  ctx: AgentContext,
): Promise<{ summary: string; produced: string[] }> {
  const produced: string[] = [];

  if (result.narration) {
    ctx.callbacks.onAnswer?.(result.narration);
    produced.push("narration");
  }
  if (result.plotPngDataUrl) {
    ctx.callbacks.onPlot?.(result.plotPngDataUrl, code);
    produced.push("plot");
  }
  if (result.fly) {
    const pos = result.fly.pos;
    const layer = result.fly.layer ?? "";
    ctx.viewer.flyTo(pos, result.fly.segmentId, layer);
    ctx.callbacks.onFly?.(pos, layer, result.fly.segmentId);
    produced.push("fly");
  }
  if (result.highlight) {
    ctx.viewer.highlightSegments(result.highlight.layer, result.highlight.ids);
    ctx.callbacks.onHighlight?.(result.highlight.layer, result.highlight.ids);
    produced.push("highlight");
  }
  if (result.annotations) {
    ctx.viewer.addAnnotationLayer(result.annotations.layerName, result.annotations.points);
    produced.push(`annotations(${result.annotations.points.length})`);
  }
  if (result.addSourceLayer) {
    const { name, type, source } = result.addSourceLayer;
    ctx.viewer.addLayerFromSpec({ type, name, source });
    registerLayer(ctx.descriptor, { name, type, source });
    produced.push(`addSourceLayer(${name})`);
  }
  if (result.meshLayer) {
    ctx.viewer.addMeshOnlyLayer({
      name: result.meshLayer.name,
      source: result.meshLayer.source,
      segments: result.meshLayer.meshIds,
    });
    registerLayer(ctx.descriptor, {
      name: result.meshLayer.name,
      type: "segmentation",
      source: result.meshLayer.source,
    });
    produced.push(`meshLayer(${result.meshLayer.name})`);
  }
  if (result.newLayer) {
    const { synthesizedId, name, type, shape, dtype } = result.newLayer;
    const url = new URL(`synthesized/${synthesizedId}/`, window.location.href).toString();
    const source = `zarr://${url}`;
    ctx.viewer.addLayerFromSpec({ type, name, source });
    registerLayer(ctx.descriptor, { name, type, source });
    produced.push(`newLayer(${name}, ${shape.join("×")} ${dtype})`);
  }
  // Tables emitted by python_on_layers go straight into the SQL DB
  // when ctx.setDB is wired (it is, from query_ui). The next agent
  // turn can run_sql against them — e.g. python_on_layers produces a
  // table of mito sizes, then run_sql ORDER BY size DESC LIMIT 1.
  if (result.table) {
    if (ctx.setDB) {
      try {
        await ingestTableIntoDB(
          { getDB: () => ctx.db, setDB: ctx.setDB },
          result.table,
          ctx.descriptor?.layers.map((l) => l.name),
        );
        produced.push(`table(${result.table.name}, ${result.table.rows.length} rows; ingested)`);
      } catch (err) {
        produced.push(`table(${result.table.name}; ingest failed: ${(err as Error).message})`);
      }
    } else {
      produced.push(`table(${result.table.name}, ${result.table.rows.length} rows; not ingested — no setDB)`);
    }
    ctx.callbacks.onTable?.(result.table);
  }

  const summary = produced.length > 0 ? `Produced: ${produced.join(", ")}` : "No output channels set.";
  return { summary: result.stdout ? `${summary}\nstdout: ${result.stdout.slice(0, 1000)}` : summary, produced };
}

function registerLayer(d: DatasetDescriptor | null, layer: DatasetLayer): void {
  if (!d) return;
  const i = d.layers.findIndex((l) => l.name === layer.name);
  if (i >= 0) d.layers[i] = layer;
  else d.layers.push(layer);
}

// describe_dataset — list loaded layers (no arg) or fetch per-layer scale
// info (with name). The summary form is cheap; the per-layer form runs
// AnalysisClient.inspect, which fetches zarr metadata. Mirrors the same
// logic Custom Analysis uses for its scale picker so the answers line up.
async function execDescribeDataset(
  args: Record<string, unknown>,
  ctx: AgentContext,
): Promise<unknown> {
  if (!ctx.descriptor) throw new Error("No dataset loaded");
  const name = args.layer_name !== undefined ? String(args.layer_name).trim() : "";

  if (!name) {
    // Summary: enough for the model to pick a layer and call back with
    // a name for scale detail. Keep it terse — this returns into the
    // agent context on every describe_dataset() call.
    //
    // For layers whose ONLY source is a precomputed segmentation (e.g.
    // hemibrain at precomputed://gs://.../segmentation), the URL alone
    // doesn't reveal whether bundled meshes / skeletons exist —
    // they're declared inside the info file's `mesh` / `skeletons`
    // subkeys. Probe each precomputed source once (cached) so
    // has_mesh / has_skeleton + bundled URLs reflect reality.
    const probedLayers = await Promise.all(
      ctx.descriptor.layers.map(async (l) => {
        const sourceList = Array.isArray(l.source) ? l.source : [l.source];
        // Path-based detection (auxiliary sources whose URL itself
        // contains /mesh/ or /skeleton/) is still the primary signal.
        let hasMesh = sourceList.some((s) => /\/mesh\b|\/meshes?\//i.test(s));
        let hasSkeleton = sourceList.some((s) => /\/skeleton\b|\/skeletons?\//i.test(s));
        const bundled: { mesh?: string; skeletons?: string } = {};
        // Bundled-in-precomputed detection: only run when neither
        // mesh nor skeleton is path-detected yet, to avoid the
        // extra HTTP for layers that already declared them.
        if (!hasMesh || !hasSkeleton) {
          for (const s of sourceList) {
            if (!/^precomputed:\/\//i.test(s)) continue;
            const info = await probePrecomputedInfo(s);
            if (!info) continue;
            if (info.mesh && !hasMesh) {
              hasMesh = true;
              bundled.mesh = `${s.replace(/\/$/, "")}/${info.mesh}`;
            }
            if (info.skeletons && !hasSkeleton) {
              hasSkeleton = true;
              bundled.skeletons = `${s.replace(/\/$/, "")}/${info.skeletons}`;
            }
            if (hasMesh && hasSkeleton) break;
          }
        }
        // For any discovered mesh source (path-based OR bundled),
        // probe its info file to learn the on-disk format. Surfaces
        // whether mesh-based analysis is currently supported (legacy
        // single-fragment = yes; multilod_draco / sharded = not yet).
        let meshFormat: string | null = null;
        let meshSharded = false;
        let meshSupported = false;
        const meshUrl =
          bundled.mesh ??
          sourceList.find((s) => /\/mesh\b|\/meshes?\//i.test(s));
        if (meshUrl) {
          const minfo = await probeMeshInfo(meshUrl);
          if (minfo) {
            meshFormat = minfo.atType || null;
            meshSharded = minfo.isSharded;
            meshSupported =
              minfo.atType === "neuroglancer_legacy_mesh" && !minfo.isSharded;
          }
        }
        return {
          name: l.name,
          type: l.type,
          organelle_class: l.organelle_class ?? null,
          source_kinds: sourceList.map(sourceKind),
          has_volume: sourceList.some((s) => /^(zarr2?|zarr3|n5|graphene)/.test(s) || /\.zarr(\/|$)/i.test(s) || /precomputed:\/\//.test(s)),
          has_mesh: hasMesh,
          has_skeleton: hasSkeleton,
          // When mesh / skeleton was discovered inside a precomputed
          // segmentation's info file (rather than as a separate
          // source URL), surface the resolved URL so the agent knows
          // where to fetch from. Absent when the path-based detection
          // already found the source.
          bundled_mesh_url: bundled.mesh ?? null,
          bundled_skeleton_url: bundled.skeletons ?? null,
          // Mesh on-disk format (when a mesh source is present).
          // "neuroglancer_legacy_mesh" + non-sharded = volume / surface
          // area integration is supported via meshes=[...] on
          // python_on_layers. multilod_draco / sharded variants need
          // a draco decoder + shard reader (not yet implemented);
          // those still let click-to-fly + visualization work, just
          // not Python mesh analysis.
          mesh_format: meshFormat,
          mesh_sharded: meshSharded,
          mesh_analysis_supported: meshSupported,
          inspectable: isZarrSource(l.source),
          // User-applied translation from the pasted NG state, if any.
          // Tourguide folds this into per-object positions
          // automatically — no need to apply it in your code.
          ng_offset_nm: l.transform_offset_nm ?? null,
        };
      }),
    );
    return {
      dataset: ctx.descriptor.display_name,
      default_voxel_size_nm: ctx.descriptor.voxel_size_nm,
      layers: probedLayers,
    };
  }

  const layer = ctx.descriptor.layers.find((l) => l.name === name);
  if (!layer) {
    const known = ctx.descriptor.layers.map((l) => l.name).join(", ");
    throw new Error(`No layer named '${name}'. Loaded layers: ${known || "(none)"}.`);
  }
  if (!isZarrSource(layer.source)) {
    return {
      name: layer.name,
      type: layer.type,
      source: layer.source,
      note: "Source is not a zarr — no scale-level metadata available. Voxel size unknown without a per-format inspector.",
    };
  }

  ctx.callbacks.onProgress?.(`Inspecting layer '${name}'…`);
  const client = new AnalysisClient();
  try {
    const url = normalizeZarrUrl(layer.source);
    const insp = await client.inspect(url, ctx.descriptor.voxel_size_nm);
    const tx = layer.transform_offset_nm;
    return {
      name: layer.name,
      type: layer.type,
      source_url: url,
      axes: insp.axes.map((a) => a.name),
      is_multiscale: insp.isMultiscale,
      // offset_nm is the zarr's intrinsic offset. If the user has
      // applied an NG-state translation, ng_offset_nm shows it; the
      // analysis pipeline auto-applies it on top of the scale's
      // offset before running, so user code should NOT add it again.
      ng_offset_nm: tx ?? null,
      scales: insp.scales.map((s) => ({
        path: s.path,
        shape: s.shape,
        voxel_size_nm: s.voxelNm,
        offset_nm: s.offsetNm,
        downsample: s.downsample,
        approx_bytes: s.approxBytes,
      })),
    };
  } finally {
    client.terminate();
  }
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

// ask_user — pause for a structured user answer. Renders a form in
// the current turn card; the Promise resolves when the user submits.
// Defaults are applied if the UI isn't wired (so headless tests don't
// hang) — but the typical path is the UI returns the user's choices.
async function execAskUser(
  args: Record<string, unknown>,
  ctx: AgentContext,
): Promise<Record<string, unknown>> {
  const prompt = String(args.prompt ?? args.question ?? "").trim();
  const rawFields = args.fields;
  if (!prompt) throw new Error("ask_user requires 'prompt'");
  if (!Array.isArray(rawFields) || rawFields.length === 0) {
    throw new Error("ask_user requires 'fields' (array of {id, label, type, ...})");
  }
  // Validate + coerce. Bad fields raise — the agent learns not to
  // emit them via the error path.
  const fields: AskField[] = [];
  for (const f of rawFields as unknown[]) {
    if (!f || typeof f !== "object") throw new Error("ask_user field must be an object");
    const r = f as Record<string, unknown>;
    const id = String(r.id ?? "").trim();
    const label = String(r.label ?? id).trim();
    const type = String(r.type ?? "").trim();
    if (!id) throw new Error("ask_user field requires 'id'");
    if (type === "select" || type === "multi") {
      if (!Array.isArray(r.options) || r.options.length === 0) {
        throw new Error(`ask_user field '${id}' (${type}) requires 'options'`);
      }
      const options = (r.options as unknown[]).map((o) => {
        if (typeof o === "string") return { label: o, value: o };
        if (o && typeof o === "object") {
          const oo = o as Record<string, unknown>;
          const value = String(oo.value ?? oo.label ?? "");
          return { label: String(oo.label ?? value), value };
        }
        return { label: String(o), value: String(o) };
      });
      if (type === "select") {
        fields.push({
          id, label, type: "select", options,
          default: r.default !== undefined ? String(r.default) : undefined,
        });
      } else {
        fields.push({
          id, label, type: "multi", options,
          default: Array.isArray(r.default) ? (r.default as unknown[]).map(String) : undefined,
        });
      }
    } else if (type === "yesno") {
      fields.push({
        id, label, type: "yesno",
        default: r.default !== undefined ? Boolean(r.default) : undefined,
      });
    } else if (type === "text") {
      fields.push({
        id, label, type: "text",
        default: r.default !== undefined ? String(r.default) : undefined,
        placeholder: r.placeholder !== undefined ? String(r.placeholder) : undefined,
      });
    } else {
      throw new Error(`ask_user field '${id}' has unknown type '${type}' (use select / multi / yesno / text)`);
    }
  }

  // Headless fallback: if the UI didn't implement onAskUser, resolve
  // immediately with each field's default. Keeps the agent loop
  // unblocked (e.g. for unit tests or future SSR scenarios).
  if (!ctx.callbacks.onAskUser) {
    const out: Record<string, unknown> = {};
    for (const f of fields) {
      if (f.type === "yesno") out[f.id] = f.default ?? true;
      else if (f.type === "multi") out[f.id] = f.default ?? [];
      else out[f.id] = f.default ?? "";
    }
    return out;
  }

  // Stop button propagation: if the abort signal fires while the form
  // is open, reject with AbortError so the agent loop unwinds the
  // same way as a network abort. The UI's form renderer also listens
  // to the signal to disable inputs visually.
  return await ctx.callbacks.onAskUser(prompt, fields);
}

async function execRunPython(
  args: Record<string, unknown>,
  ctx: AgentContext,
): Promise<{ stdout: string; result?: string }> {
  if (!ctx.db) throw new Error("No database loaded");
  const code = coerceCodeArg(args, ["python", "code", "script", "source", "body"]);
  if (!code) throw new Error("run_python requires 'python'");

  ctx.callbacks.onProgress?.("Loading Pyodide…");
  const py = await loadPyodide((m) => ctx.callbacks.onProgress?.(m));
  // First use of Pyodide on this page: prime DataFrames + helpers. We
  // reuse the same setup make_plot does so globals (df_*, np, pd, plt)
  // are consistent regardless of which Python tool is called first.
  py.globals.set("_tables_json", tablesToJson(ctx.db));
  await py.runPythonAsync(SETUP_PY);
  await py.runPythonAsync(`
_dfs = _materialize_tables(_tables_json)
for _k, _v in _dfs.items():
    globals()[_k] = _v
import numpy as np
`);
  // Auto-load any prebuilt Pyodide packages the user code imports
  // (scipy, scikit-learn, sympy, networkx, statsmodels, etc.). Without
  // this the model has to know to call pyodide.loadPackage manually,
  // which it never does — it just writes `import scipy` and gets a
  // ModuleNotFoundError. Pyodide's loadPackagesFromImports is a
  // no-op for already-loaded packages, so this is cheap to call every
  // time.
  ctx.callbacks.onProgress?.("Loading required Python packages…");
  try {
    await py.loadPackagesFromImports(code);
  } catch (err) {
    console.warn("[agent] loadPackagesFromImports failed:", (err as Error).message);
  }
  ctx.callbacks.onProgress?.("Running Python…");
  // Run the user code inside a try/finally so we always restore stdout,
  // and capture (a) printed output, (b) whatever the model assigned to
  // `_out`. Indenting under `try:` keeps multi-line code intact;
  // imports / assignments inside still bind globally because
  // runPythonAsync runs at module scope.
  const indented = code.replace(/^/gm, "    ");
  await py.runPythonAsync(`
import sys, io, json as _json
_buf = io.StringIO()
_old_stdout = sys.stdout
sys.stdout = _buf
_out = None
try:
${indented}
finally:
    sys.stdout = _old_stdout
_PY_STDOUT = _buf.getvalue()
def _serialize_out(v):
    if v is None:
        return None
    try:
        if hasattr(v, "to_dict"):
            return _json.dumps(v.to_dict(), default=str)
    except Exception:
        pass
    try:
        return _json.dumps(v, default=str)
    except Exception:
        return repr(v)
_PY_OUT = _serialize_out(_out)
`);
  const stdoutRaw = String(py.globals.get("_PY_STDOUT") ?? "");
  // Cap stdout so a chatty `print(df)` doesn't blow the LLM context.
  const stdout =
    stdoutRaw.length > 4000 ? stdoutRaw.slice(0, 4000) + "\n…(truncated)" : stdoutRaw;
  const outVal = py.globals.get("_PY_OUT");
  return {
    stdout,
    result: outVal === null || outVal === undefined ? undefined : String(outVal),
  };
}

const SETUP_PY = `
import json
import io
import base64
import numpy as np
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
        # Convenience: pre-extract center-of-mass coordinates as an
        # (N, 3) numpy array under com_<class>. Saves the model from
        # writing 'np.array(df_x[["com_x_nm", "com_y_nm", "com_z_nm"]])'
        # every time, which is exactly where it confuses DataFrame and
        # numpy syntax. We accept either com_*_nm or position_* columns
        # — same physical meaning.
        com_cols = None
        for cands in [
            ["com_x_nm", "com_y_nm", "com_z_nm"],
            ["position_x_nm", "position_y_nm", "position_z_nm"],
            ["com_x", "com_y", "com_z"],
            ["position_x", "position_y", "position_z"],
        ]:
            if all(c in df.columns for c in cands):
                com_cols = cands
                break
        if com_cols is not None:
            out[f"com_{name}"] = df[com_cols].to_numpy(dtype=np.float64)
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
  describe_dataset: execDescribeDataset,
  run_sql: execRunSql,
  run_python: execRunPython,
  python_on_layers: execPythonOnLayers,
  fly_to: execFlyTo,
  highlight_segments: execHighlight,
  make_plot: execMakePlot,
  ask_user: execAskUser,
  answer: execAnswer,
};

// Translate raw SQLite / Pyodide errors into a short, directive hint
// the model can act on. WebLLM small models in particular don't
// recover from "unrecognized token" or "no such column" without one
// — and crucially, they can't re-read the schema in the system prompt
// once it's pushed back in context. So for column / table errors we
// reinject the actual schema as part of the hint.
// Returns "" when no useful hint is available.
function synthesizeErrorHint(
  tool: string,
  args: Record<string, unknown>,
  errMsg: string,
  ctx: AgentContext,
): string {
  const lower = errMsg.toLowerCase();
  if (tool === "run_sql") {
    const sql = String(args.sql ?? "");
    if (lower.includes("unrecognized token")) {
      // SQLite gives this when a column name has a special char (^, /,
      // parens) and isn't quoted. The schema lists the real column
      // names — re-emphasize that they go between double quotes.
      return `Column names with special characters like '(', ')', '^' must be wrapped in double quotes: "volume_(nm^3)" not volume_(nm^3). Or use only the alphanumeric prefix if it's unique.`;
    }
    if (lower.includes("no such column") && ctx.db) {
      const colMatch = errMsg.match(/no such column[:\s]+([^\s,;]+)/i);
      const bad = colMatch ? colMatch[1] : "that column";
      // Try to figure out which table the SQL targeted so we can list
      // ITS columns specifically (more focused than dumping every
      // table). Falls back to listing all tables' columns if FROM
      // can't be parsed.
      const fromMatch = sql.match(/from\s+"?([a-zA-Z0-9_]+)"?/i);
      const targetTable = fromMatch?.[1]?.toLowerCase();
      const targets = targetTable
        ? ctx.db.tables.filter((t) => t.table_name === targetTable)
        : ctx.db.tables;
      const schema = (targets.length > 0 ? targets : ctx.db.tables)
        .map((t) => `"${t.table_name}": ${t.columns.join(", ")}`)
        .join("\n");
      return `Column ${bad} doesn't exist. The actual columns are:\n${schema}\nUse one of these names verbatim.`;
    }
    if (lower.includes("no such table") && ctx.db) {
      const names = ctx.db.tables.map((t) => `"${t.table_name}"`).join(", ");
      return `That table doesn't exist. Available tables: ${names}.`;
    }
    if (lower.includes("syntax error")) {
      return `SQL syntax error. Re-check the SELECT clause, table name, and any quoted identifiers. Note: SQLite dialect.`;
    }
    if (sql.length === 0) {
      return `run_sql expects {"sql": "SELECT ..."}.`;
    }
    if (lower.includes("requires 'sql'")) {
      return `Pass the SQL under the key "sql" — e.g. {"tool": "run_sql", "args": {"sql": "SELECT * FROM mito LIMIT 10"}}. Don't use "query" / "statement".`;
    }
  }
  if (tool === "run_python") {
    if (lower.includes("requires 'python'")) {
      return `Pass the code under the key "python" — e.g. {"tool": "run_python", "args": {"python": "df_mitochondria['volume_nm_3'].mean()"}}. Don't use "code" / "script" — just "python".`;
    }
    if (lower.includes("nameerror") || lower.includes("name '") && lower.includes("' is not defined")) {
      return `That variable doesn't exist. The DataFrames are named df_<organelle_class> (see the system prompt schema). np and pd are imported.`;
    }
    if (lower.includes("keyerror") || lower.includes("not in index")) {
      return `That column isn't on the DataFrame. Use df.columns to discover names — they include the suffix (e.g. 'volume_(nm^3)' is a literal column key in pandas, write df["volume_(nm^3)"]).`;
    }
    if (lower.includes("attributeerror")) {
      return `Wrong type / method. If you're calling a Series method on a DataFrame (or vice versa), index into the column first: df["col"].mean(), not df.mean()["col"].`;
    }
    if (lower.includes("indexerror") && lower.includes("only integers")) {
      return `You're using DataFrame syntax (arr["col_name"]) on a numpy array. Numpy arrays index by integer/slice: arr[:, 0] for the first column, arr[i] for the i-th row. Use the pre-extracted com_<class> arrays for COM data — they're already numpy. com_mito[:, 0] = x coords, com_mito[:, 1] = y, com_mito[:, 2] = z.`;
    }
    if (lower.includes("singular data covariance") || (lower.includes("dimensions") && lower.includes("samples"))) {
      return `gaussian_kde expects (D, N) shape — D dimensions, N samples. Pass com_<class>.T not com_<class>. Example: kde = gaussian_kde(com_mito.T) where com_mito is (N, 3).`;
    }
    if (lower.includes("unterminated string literal") || lower.includes("unterminated f-string")) {
      return `Don't put raw \\n inside a Python string literal — it becomes an actual line break and breaks the string. Either drop the \\n (use a separate print call), use a triple-quoted string ('''text''' or """text"""), or write \\\\n in the JSON so the Python source has the literal escape \\n.`;
    }
    if (lower.includes("indentationerror") || lower.includes("syntaxerror")) {
      return `Python syntax error. Watch indentation; the harness wraps your code under a try block, so it must be valid as-is. If the error is about an unterminated string, you have an unescaped newline inside a string literal — split into multiple print() calls instead.`;
    }
    if (lower.includes("modulenotfounderror")) {
      return `That package isn't in Pyodide's prebuilt list. Stick to numpy / pandas / matplotlib / scipy / scikit-learn / sympy / networkx / statsmodels — and DON'T call micropip.install or pyodide.loadPackage; the harness auto-loads anything you import. If you really need a Seung-lab / heavyweight package (cc3d, fastmorph, kimimaro, zmesh, edt, fastremap), switch to python_on_layers with the Run-on-backend toggle on — those live on the HF Space. Otherwise pick a different approach with the available ones.`;
    }
  }
  if (tool === "make_plot") {
    if (lower.includes("did not produce a figure")) {
      return `Your code ran but no matplotlib figure was open. Make sure you call plt.figure() / plt.plot() / plt.bar() etc. — at least one plot command before the script ends.`;
    }
  }
  if (tool === "python_on_layers") {
    if (lower.includes("requires 'python'")) {
      return `python_on_layers needs the code under the key "python" in args. Example: {"tool": "python_on_layers", "args": {"python": "import numpy as np\\n_TG_NARRATION = str(mito.shape)", "layers": ["mito"]}}. Don't use "code" / "script" / "source" — only "python". Don't omit it.`;
    }
    if (lower.includes("modulenotfounderror") || lower.includes("no module named")) {
      const m = errMsg.match(/No module named ['"]?([\w.]+)['"]?/i);
      const pkg = m ? m[1] : "<that module>";
      return `Module '${pkg}' isn't available in this analysis runtime. Available locally (Pyodide): numpy, scipy, scikit-image, pandas, matplotlib. Available on the HF backend (auto-routing kicks in when you import these): cc3d, fastmorph, fastremap, edt, kimimaro, zmesh. To add a new package permanently, edit hf-space/requirements.txt and redeploy. Pick a different approach with the available libraries for now.`;
    }
    if (lower.includes("not a zarr source")) {
      return `python_on_layers currently only loads zarr layers. Skeleton / precomputed-mesh inputs aren't yet supported through this tool.`;
    }
    // KeyError on the layers dict — should be rare now that we alias
    // the common variants, but if the model invents something
    // genuinely outside the set we surface the canonical keys.
    if (lower.includes("keyerror")) {
      const m = errMsg.match(/KeyError:\s*['"]?([^'"\n]+)['"]?/);
      const bad = m ? m[1] : "<that key>";
      return `KeyError on '${bad}'. layers["<name>"] has: "array", "spacing" (alias "voxel_size_nm"), "offsets" (alias "offset_nm"), "axes". For shape/dtype, use array.shape and array.dtype on the bare variable.`;
    }
  }
  return "";
}

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
  // Compress prior turns of this session into the system prompt so
  // follow-ups ("now do the same for ER") can resolve. We deliberately
  // do NOT replay full tool traces — that would balloon context
  // immediately. One Q + one short summary per turn is enough.
  const priorBlock = (ctx.priorTurns ?? [])
    .map((t, i) => `(${i + 1}) user: ${t.question}\n    you: ${t.summary}`)
    .join("\n");
  const systemContent = priorBlock
    ? `${SYSTEM_PROMPT(ctx.db, ctx.descriptor)}\n\nEARLIER IN THIS SESSION (most recent last) — refer back when the user says "now do the same for X", "redo with …", or "the previous":\n${priorBlock}`
    : SYSTEM_PROMPT(ctx.db, ctx.descriptor);

  const messages: LLMMessage[] = [
    { role: "system", content: systemContent },
    { role: "user", content: question },
  ];

  // Snapshot the cloud-API call counter so we can show "this turn used
  // N calls" — the actual number that matters when you're hitting a
  // free-tier quota. Local backends (WebLLM) leave this at 0.
  const startReqCount = GeminiBackend.requestCount;

  // Track whether the model has delivered any user-visible output
  // (answer text or a plot). If it tries to `done` without one — which
  // small models do after a single tool error — we push back once,
  // asking for a written answer summarizing what happened. Avoids the
  // 'Agent finished without delivering an answer' dead end.
  let deliveredOutput = false;
  let nudgedForAnswer = false;

  for (let i = 0; i < MAX_ITERATIONS; i++) {
    const stepStart = performance.now();
    let tokenCount = 0;
    let lastUpdate = 0;
    let detectedTool: string | null = null;
    ctx.callbacks.onProgress?.(`Step ${i + 1}: thinking…`);
    if (ctx.signal?.aborted) {
      throw new DOMException("Agent stopped by user", "AbortError");
    }
    const raw = await ctx.backend.complete(messages, {
      temperature: 0.1,
      jsonMode: true,
      maxTokens: 1500,
      signal: ctx.signal,
      onToken: (_t, accumulated) => {
        tokenCount += 1;
        // Pull the tool name out of the streamed JSON as soon as it's
        // there. Once we have it, the user-facing status flips from
        // opaque "thinking…" to a friendly "Step N: querying tables".
        // Token-rate / preview info goes to console.debug so curious
        // devs can still see it without polluting the visible status.
        if (!detectedTool) {
          const m = /"tool"\s*:\s*"([a-z_]+)"/.exec(accumulated);
          if (m) {
            detectedTool = m[1];
            const label = TOOL_LABELS[detectedTool] ?? detectedTool;
            ctx.callbacks.onProgress?.(`Step ${i + 1}: ${label}`);
          }
        }
        const now = performance.now();
        if (now - lastUpdate > 120) {
          lastUpdate = now;
          const elapsed = ((now - stepStart) / 1000).toFixed(1);
          const tps = tokenCount / Math.max(0.001, (now - stepStart) / 1000);
          const preview = accumulated.replace(/\s+/g, " ").slice(-90);
          console.debug(
            `[agent] step ${i + 1}: ${tokenCount} tokens · ${tps.toFixed(0)} tok/s · ${elapsed}s — ${preview}`,
          );
        }
      },
    });
    const stepElapsed = ((performance.now() - stepStart) / 1000).toFixed(1);
    const apiCalls = GeminiBackend.requestCount - startReqCount;
    const apiNote = apiCalls > 0 ? ` · ${apiCalls} API call${apiCalls === 1 ? "" : "s"} this turn` : "";
    console.debug(`[agent] step ${i + 1}: done in ${stepElapsed}s (${tokenCount} tokens)${apiNote}`);

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
    // Explicit "thinking done — now running X" transition. Without
    // this beat between LLM stream end and the executor's onProgress,
    // the user sees "thinking…" followed by executor messages with
    // no signal that the LLM phase finished. Note: each tool's own
    // executor will overwrite this within milliseconds with its own
    // progress messages — that's fine; this just bridges the gap
    // when there's a brief pause (Pyodide spin-up, fetch latency).
    if (call.tool !== "done") {
      const label = TOOL_LABELS[call.tool] ?? call.tool;
      ctx.callbacks.onProgress?.(`Step ${i + 1}: ${label} …`);
    }

    if (call.tool === "done") {
      // If the model bails without ever calling answer / make_plot,
      // push back once and ask it to summarize. Some flows are valid
      // visual-only (fly_to → done) but most "I errored, give up"
      // flows silently leave the user with nothing on screen.
      if (!deliveredOutput && !nudgedForAnswer) {
        nudgedForAnswer = true;
        messages.push({ role: "assistant", content: JSON.stringify(call) });
        messages.push({
          role: "user",
          content: `You called done without delivering an answer or plot. Call answer(text) summarizing what you found, or — if a tool kept failing — explain in one sentence what blocked you. Don't call done again until you've called answer.`,
        });
        continue;
      }
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
        content: `Error: tool "${call.tool}" does not exist. Pick from: describe_dataset, run_sql, run_python, python_on_layers, fly_to, highlight_segments, make_plot, ask_user, answer, done.`,
      });
      continue;
    }

    messages.push({ role: "assistant", content: JSON.stringify(call) });
    const trace: AgentTraceItem = { tool: call.tool, args: call.args ?? {} };
    try {
      const result = await executor(call.args ?? {}, ctx);
      trace.result = result;
      ctx.callbacks.onTrace?.(trace);
      // answer / make_plot are clearly user-visible. fly_to and
      // highlight_segments also produce a visible viewer change, so
      // count them as "delivered" — the user sees the camera move
      // even if no text answer follows. This means a single fly_to
      // without an answer() is still considered a successful turn.
      if (DELIVERED_BY_TOOL.has(call.tool)) {
        deliveredOutput = true;
      }
      if (TERMINAL_TOOLS.has(call.tool)) {
        // answer / make_plot are inherently terminal — no need for a separate done call.
        return;
      }
      // ask_user is a user-mediated pause, not "thinking" work — don't
      // count it against the iteration budget. Without this, an agent
      // that asks one question (very common for ambiguous prompts)
      // would have only 4 of its 5 calls left to actually execute.
      if (call.tool === "ask_user") i--;
      // After fly_to / highlight_segments, the user already sees the
      // result on screen — push back a directive next-step message
      // instead of the generic 'call next or done'. Without this,
      // small models keep running more SQL queries trying to "do
      // more", burning the iteration budget on work the user didn't
      // ask for.
      const isVisualDelivery =
        call.tool === "fly_to" || call.tool === "highlight_segments";
      const isAskUser = call.tool === "ask_user";
      const nextPrompt = isVisualDelivery
        ? `tool_result: ${summarizeResult(result)}\n\nThe user has now seen the result on the viewer. End the turn now: emit {"tool":"done"} unless the user's question explicitly asked for additional info beyond what's visible.`
        : isAskUser
          ? `user_answer: ${summarizeResult(result)}\n\nProceed with these values. The user has answered — do NOT ask the same question again. Move to the next tool call (typically the operation you were going to perform once you knew the answer).`
          : `tool_result: ${summarizeResult(result)}\n\nCall the next tool, or {"tool":"done"} when finished.`;
      messages.push({ role: "user", content: nextPrompt });
    } catch (err) {
      const errMsg = (err as Error).message;
      trace.error = errMsg;
      ctx.callbacks.onTrace?.(trace);
      // Synthesize a hint specific to common error shapes so the model
      // can self-correct. Small WebLLM models can't translate raw
      // SQLite / Python errors into the right fix on their own — they
      // just trip into 'done' if the message ends with "give up".
      // Don't offer 'done' as an option here at all; the model can
      // still emit it, but biasing toward retry-with-hint catches the
      // common 80%.
      const hint = synthesizeErrorHint(call.tool, call.args ?? {}, errMsg, ctx);
      messages.push({
        role: "user",
        content: `tool_error: ${errMsg}${hint ? `\n\nHint: ${hint}` : ""}\n\nFix the call and try again. If you've tried twice with no progress, call answer() to explain what's blocking you.`,
      });
    }
  }
  throw new Error(`Agent exceeded ${MAX_ITERATIONS} iterations — giving up.`);
}
