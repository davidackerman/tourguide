"""tourguide-mcp — stdio MCP server proxying to the Tourguide Workspace API.

Run:  uv run tourguide-mcp        (or: tourguide-mcp once installed)

The server launches/attaches to a Tourguide workspace session via the local
bridge and exposes high-level workspace tools to any MCP-capable agent.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .launcher import LauncherConfig
from .session import WorkspaceSession
from .tools import register_tools

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

MEASURING PROPERTIES — compute them yourself, by default:
  When asked to measure/quantify objects in a segmentation (volume, count,
  centroid, surface area, …), DEFAULT TO COMPUTING IT YOURSELF from the raw
  data. Do NOT substitute published/precomputed `segment_properties`, meshes,
  or external tables unless the user EXPLICITLY asks for the published values.

INGEST BIG TABLES BY PATH — never stream rows through your tokens:
  Re-emitting hundreds of rows as a tool argument is the single slowest thing
  you can do (it cost minutes in testing). Instead, WRITE the table to a file
  in your env (.csv or .json) and call ingest_table(name, path="…"). The
  server reads the file and forwards it directly — fast, no token cost. Use
  inline columns+rows only for a handful of rows.

The loop:
  1. get_session → the target layer's data source URL + voxel size.
  2. In the analysis env, open that zarr/n5 (tensorstore/zarr) and compute
     per object — cc3d.statistics gives volume + centroid + bbox in one pass.
     For a volume too big to load whole, read it in blocks and accumulate;
     a downsampled scale is fine for a quick volume/centroid distribution, but
     surface area is resolution-sensitive (use native res or the meshes).
     Write the result to results.csv.
  3. ingest_table(name, path="results.csv")  — include object_id and
     com_x_nm/com_y_nm/com_z_nm so click-to-fly works.
  4. Optionally fly_to the largest object, select_segments, or show_plot(png=…).

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
    instructions = INSTRUCTIONS_TEMPLATE.format(analysis_dir=_analysis_dir())
    mcp = FastMCP("tourguide", instructions=instructions)
    session = WorkspaceSession(LauncherConfig())
    register_tools(mcp, session)
    return mcp


def main() -> None:
    build_server().run()  # stdio transport by default


if __name__ == "__main__":
    main()
