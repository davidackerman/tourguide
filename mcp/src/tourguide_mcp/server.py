"""tourguide-mcp — stdio MCP server proxying to the Tourguide Workspace API.

Run:  uv run tourguide-mcp        (or: tourguide-mcp once installed)

The server launches/attaches to a Tourguide workspace session via the local
bridge and exposes high-level workspace tools to any MCP-capable agent.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .launcher import LauncherConfig
from .session import WorkspaceSession
from .tools import register_tools


def _ancestors(pid: int) -> set[int]:
    """Walk up the parent chain (our uv-run / Desktop wrappers) so we never
    kill a process that launched us."""
    chain: set[int] = set()
    cur = pid
    for _ in range(20):
        try:
            ppid = int(
                subprocess.run(
                    ["ps", "-o", "ppid=", "-p", str(cur)],
                    capture_output=True, text=True, timeout=3,
                ).stdout.strip()
                or 0
            )
        except Exception:
            break
        if ppid <= 1 or ppid in chain:
            break
        chain.add(ppid)
        cur = ppid
    return chain


def _proc_age_seconds(pid: int) -> int | None:
    """Elapsed seconds since `pid` started, or None if unknown. Uses `ps -o
    etime` ([[DD-]hh:]mm:ss), which works on both macOS and Linux (etimes is
    Linux-only)."""
    try:
        out = subprocess.run(
            ["ps", "-o", "etime=", "-p", str(pid)],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
    except Exception:
        return None
    if not out:
        return None
    days = 0
    if "-" in out:
        d, out = out.split("-", 1)
        days = int(d)
    try:
        parts = [int(p) for p in out.split(":")]
    except ValueError:
        return None
    secs = 0
    for p in parts:  # mm:ss or hh:mm:ss
        secs = secs * 60 + p
    return days * 86400 + secs


# Only reap processes clearly older than any concurrent Desktop spawn. Claude
# Desktop sometimes launches a second server seconds after the first (and
# briefly runs both); reaping those siblings makes Desktop report a spurious
# "Server disconnected". A genuine orphan from a previous session is minutes-to-
# hours old, so an age gate cleanly separates the two.
_REAP_MIN_AGE_S = int(os.environ.get("TG_MCP_REAP_MIN_AGE", "600"))


def _reap_stale_servers() -> None:
    """Kill clearly-orphaned older tourguide-mcp processes on startup so one from
    a previous Desktop session can't linger and serve stale code. Skips its own
    process chain and any process younger than _REAP_MIN_AGE_S (likely a sibling
    Desktop just spawned). Opt out entirely with TG_MCP_NO_REAP=1."""
    if os.environ.get("TG_MCP_NO_REAP"):
        return
    me = os.getpid()
    keep = {me} | _ancestors(me)
    try:
        pids = subprocess.run(
            ["pgrep", "-f", "tourguide-mcp"], capture_output=True, text=True, timeout=3
        ).stdout.split()
    except Exception:
        return
    for raw in pids:
        try:
            pid = int(raw)
        except ValueError:
            continue
        if pid in keep:
            continue
        age = _proc_age_seconds(pid)
        if age is None or age < _REAP_MIN_AGE_S:
            continue  # too young to be a stale orphan — leave Desktop's spawns alone
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass  # already gone / not ours to kill

# Surfaced to the model by MCP clients (e.g. Claude Desktop, which — unlike a
# coding agent — never reads the repo's CLAUDE.md). This is where the
# measure-and-show workflow has to live for a Desktop-driven session. Speed
# matters: the in-browser version was fast because nothing round-tripped
# through an LLM, so these rules exist to keep the agent path nearly as fast.
INSTRUCTIONS_TEMPLATE = """\
Tourguide is a 3D microscopy visual workspace. You own reasoning and compute;
Tourguide owns visual state. You read data and run analysis in YOUR OWN
environment (you have a real shell), then push results into the workspace.
Tourguide is an artifact sink, not a compute runtime.

USE THE PRE-LOADED ANALYSIS ENVIRONMENT — do not pip-install:
  A ready Python env with the measurement stack (tensorstore, zarr, s3fs,
  connected-components-3d (cc3d), scikit-image, scipy, numpy, dask, pandas,
  matplotlib, trimesh) lives at:
      {analysis_dir}
  Run compute with:  uv run --project {analysis_dir} python your_script.py
  It's already synced — installing libraries yourself just wastes time.

MEASURING PROPERTIES — use the predesigned tools, not shell scripts:
  When asked to measure/quantify objects in a segmentation (volume, count,
  centroid, …), call the `measure` TOOL with the layer's source URL (from
  get_session). It runs the predesigned recipe server-side and ingests the
  table for you — fast, consistent coordinates, and (unlike shelling out
  yourself) NO file-permission prompt and no Pyodide. Tell the user you used
  the predesigned measurement and they can ask for a custom one.

  For other/standard analyses, `list_recipes` shows what's available
  (built-in + the user's templates in ~/.tourguide/recipes) and `run_recipe`
  runs one by name over a source URL. When the user has a custom analysis
  they'll reuse, save it as ~/.tourguide/recipes/<name>.py (a CLI taking a
  source URL and writing a CSV via --out) so it shows up in list_recipes.
  Only write throwaway analysis code yourself if no recipe fits and the user
  asks. Do NOT substitute published/precomputed segment_properties or meshes
  unless the user explicitly asks for the published values.

LOADING WHAT THE USER DROPS IN — discern the type, don't demand a format:
  When the user says "open this in tourguide" / "load these", infer the type
  from the content or extension and route it — don't require a yaml descriptor:
  - Neuroglancer URL (has a '#!{...}' state fragment, often URL-encoded): call
    load_url(url) — ONE step, the server decodes + applies it. Do NOT decode or
    percent-unescape it yourself in a shell or read it into context (that's the
    slow path; load_url is instant).
  - Data source (zarr / n5 / precomputed over s3/gcs/http): add_layer with that
    source — segmentation vs image by dtype/intent.
  - Table file (.csv / .json): ingest_table(name, path="…").
  - Tourguide descriptor (.yaml / .json with layers + voxel size):
    load_descriptor.
  Only ask the user if the type is genuinely ambiguous.

INGEST BIG TABLES BY PATH — never stream rows through your tokens:
  Re-emitting hundreds of rows as a tool argument is the single slowest thing
  you can do (it cost minutes in testing). The recipes already write a CSV —
  just call ingest_table(name, path="objects.csv"). The server reads the file
  and forwards it directly: fast, no token cost. Use inline columns+rows only
  for a handful of rows.

The loop:
  1. get_session → the target layer's data source URL.
  2. measure(source=…) → measures every object and ingests the table
     (object_id, volume_nm_3, com_x/y/z_nm for click-to-fly). For other
     analyses use run_recipe. Surface area is resolution-sensitive and isn't
     measured — use the meshes for that.
  3. (measure/run_recipe already ingested the table; only call ingest_table
     yourself for a table you computed with custom code — then prefer path=.)
  4. add_narration_note describing what you measured and how (source layer,
     scale, method, object count). Your measuring happens in your own env and
     is otherwise INVISIBLE in the workspace — this note is the record that
     the measurement was done, and it shows in the Agent Actions panel.
  5. Optionally fly_to the largest object, select_segments, or show a figure.

PLOTS — render to a .png file, pass show_plot(png_path=…):
  Render the figure in the analysis env (matplotlib savefig to a .png) and call
  show_plot(png_path="fig.png"). The server reads + encodes the file, so the
  image never goes through your tokens. Do NOT base64-encode it yourself, and
  do NOT read the png/base64 back into your context (that cost ~2 minutes in
  testing). Do ALL compute and rendering in your own environment — there is no
  in-browser (Pyodide) path, by design.

Other notes:
  - Call launch_or_attach first. If it returns a `shareUrl`, tell the user —
    it's how others on their network view the same workspace.
  - Keep to one workspace tab unless the user wants parallel ones; when
    several are open, launch_or_attach asks which to drive.
"""


def _analysis_dir() -> str:
    """Path to the pre-loaded analysis env (sibling of this package's repo)."""
    return str(Path(__file__).resolve().parents[3] / "analysis")


def build_server() -> FastMCP:
    # Plain replace, NOT str.format: the template is full of literal braces
    # ({...}, {columns, rows}, '#!{...}') that .format would misparse.
    instructions = INSTRUCTIONS_TEMPLATE.replace("{analysis_dir}", _analysis_dir())
    mcp = FastMCP("tourguide", instructions=instructions)
    session = WorkspaceSession(LauncherConfig())
    register_tools(mcp, session)
    return mcp


def main() -> None:
    _reap_stale_servers()  # evict orphaned older instances first
    build_server().run()  # stdio transport by default


if __name__ == "__main__":
    main()
