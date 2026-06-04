"""MCP tool surface for Tourguide.

Each tool maps closely to a Workspace API operation. The tools are
intentionally high-level (semantic viewer ops, tables, plots, saved states,
recording) rather than low-level Neuroglancer primitives — set_viewer_state
is the escape hatch. Keep this adapter thin: if a convenience belongs
anywhere, it belongs in the Workspace API, not here.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .session import WorkspaceSession

# Where analysis recipes live: built-ins ship with the repo; the user drops
# their own templates in ~/.tourguide/recipes so they show up alongside.
_ANALYSIS_DIR = Path(__file__).resolve().parents[3] / "analysis"
_BUILTIN_RECIPES = _ANALYSIS_DIR / "recipes"
_USER_RECIPES = Path.home() / ".tourguide" / "recipes"


def _recipe_dirs() -> list[Path]:
    return [_BUILTIN_RECIPES, _USER_RECIPES]


def _find_recipe(name: str) -> Path | None:
    """Resolve a recipe by name (with or without .py); user recipes shadow built-ins."""
    stem = name[:-3] if name.endswith(".py") else name
    for d in reversed(_recipe_dirs()):  # user dir first → user templates win
        p = d / f"{stem}.py"
        if p.is_file():
            return p
    return None


def _parse_ng_state(url: str) -> dict | None:
    """Decode a Neuroglancer state from a URL/fragment/JSON. Handles the
    `#!{...}` (often percent-encoded) fragment, a bare `{...}` blob, or a
    `#!middleauth+...` / json_url style left for the browser. Returns the state
    dict, or None if no inline state is present."""
    s = url.strip()
    frag = s.split("#!", 1)[1] if "#!" in s else (s if s.startswith(("{", "%7B")) else None)
    if frag is None:
        return None
    frag = urllib.parse.unquote(frag)
    if not frag.startswith("{"):
        return None  # e.g. a json_url/state-server reference, not inline JSON
    return json.loads(frag)


def _recipe_doc(path: Path) -> str:
    """First meaningful line of the recipe's module docstring, for listings."""
    try:
        for line in path.read_text().splitlines():
            s = line.strip().strip('"').strip("'").strip()
            if s and not s.startswith(("#", "import", "from", "#!")):
                return s
    except Exception:
        pass
    return ""


def _coerce(v: Any) -> Any:
    """CSV cells arrive as strings; turn numeric-looking ones into numbers so
    columns like com_x_nm stay numeric (needed for SQL + click-to-fly)."""
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return v


def _load_table(path: str) -> tuple[list[str], list[list]]:
    """Read a table the agent wrote to disk into (columns, rows) — so large
    tables flow server→bridge instead of through the model's tokens. Supports
    .csv and .json (list-of-records, {columns, rows}, or list-of-lists)."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"ingest_table: no file at {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        with p.open(newline="") as f:
            reader = csv.reader(f)
            all_rows = [r for r in reader]
        if not all_rows:
            raise ValueError(f"ingest_table: {p} is empty")
        columns = all_rows[0]
        rows = [[_coerce(c) for c in r] for r in all_rows[1:]]
        return columns, rows
    if suffix in (".json", ".jsonl", ".ndjson"):
        if suffix in (".jsonl", ".ndjson"):
            records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        else:
            records = json.loads(p.read_text())
        # {columns, rows}
        if isinstance(records, dict) and "columns" in records and "rows" in records:
            return list(records["columns"]), list(records["rows"])
        if not isinstance(records, list) or not records:
            raise ValueError(f"ingest_table: {p} must be a non-empty list or {{columns, rows}}")
        # list-of-records
        if isinstance(records[0], dict):
            columns = list(records[0].keys())
            rows = [[rec.get(c) for c in columns] for rec in records]
            return columns, rows
        # list-of-lists — first row is the header
        return list(records[0]), [list(r) for r in records[1:]]
    raise ValueError(f"ingest_table: unsupported file type '{suffix}' (use .csv or .json)")


async def _run_recipe_to_csv(recipe: Path, args: list[str]) -> str:
    """Run a recipe in the pre-loaded analysis env, writing a CSV to a temp
    file, and return its path. The MCP server (not the agent) runs it, so there
    is no sandbox/Documents prompt and no Pyodide. Raises on failure."""
    out = tempfile.NamedTemporaryFile(prefix="tg_recipe_", suffix=".csv", delete=False)
    out.close()
    cmd = ["uv", "run", "--project", str(_ANALYSIS_DIR), "python", str(recipe),
           *args, "--out", out.name]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"recipe {recipe.name} failed:\n{stderr.decode(errors='replace')[-2000:]}"
        )
    return out.name


def register_tools(mcp: FastMCP, session: WorkspaceSession) -> None:
    async def _run_and_ingest(recipe: Path, source: str, name: str, extra: list[str]) -> dict:
        csv_path = await _run_recipe_to_csv(recipe, [source, *extra])
        columns, rows = _load_table(csv_path)
        result = await session.call(
            "ingest_table", {"name": name, "columns": columns, "rows": rows}
        )
        if isinstance(result, dict):
            result["rowCount"] = len(rows)
        return result

    @mcp.tool()
    async def load_url(url: str) -> dict:
        """Load a Neuroglancer state URL into the viewer in ONE step. Pass the
        full link (or just its '#!...' fragment); the server decodes the state
        and applies it. Do NOT decode/percent-unescape it yourself in a shell or
        read it into your context — that's slow; this is instant. For a data
        source URL (zarr/n5/precomputed) use add_layer; for a .csv use measure/
        ingest_table."""
        state = _parse_ng_state(url)
        if state is None:
            raise ValueError(
                "load_url: no inline Neuroglancer '#!{...}' state in that URL. "
                "If it's a data source, use add_layer instead."
            )
        return await session.call("set_viewer_state", {"state": state})

    @mcp.tool()
    async def list_recipes() -> dict:
        """List the analysis recipes available to run server-side: the built-in
        ones that ship with Tourguide plus any the user has added as templates
        in ~/.tourguide/recipes. Each runs in the pre-loaded analysis env (no
        Pyodide, no file-permission prompt) and ingests its result as a table."""
        out = []
        for d in _recipe_dirs():
            kind = "builtin" if d == _BUILTIN_RECIPES else "user"
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.py")):
                if p.name.startswith("_"):
                    continue
                out.append({"name": p.stem, "source": kind, "description": _recipe_doc(p)})
        return {"recipes": out, "userRecipeDir": str(_USER_RECIPES)}

    @mcp.tool()
    async def measure(source: str, name: str = "measurements", scale: str | None = None) -> dict:
        """Measure every object in a segmentation layer and ingest the result as
        a table (the predesigned `measure_objects` recipe). `source` is the
        layer's data URL from get_session (e.g.
        n5://s3://.../labels/mito_seg). Runs server-side in the analysis env —
        fast, consistent coordinates, no Pyodide, no permission prompt. The
        table has object_id, volume_nm_3, voxel_count, com_x/y/z_nm (click-to-
        fly). Optional `scale` forces a multiscale level (e.g. "s2")."""
        recipe = _find_recipe("measure_objects")
        if recipe is None:
            raise FileNotFoundError("measure_objects recipe not found in the analysis env")
        extra = ["--scale", scale] if scale else []
        return await _run_and_ingest(recipe, source, name, extra)

    @mcp.tool()
    async def run_recipe(
        recipe: str, source: str, name: str | None = None, args: list[str] | None = None
    ) -> dict:
        """Run any recipe by name (see list_recipes) — a built-in or a user
        template in ~/.tourguide/recipes — server-side over a layer `source`
        URL, and ingest its CSV output as a table. `args` are extra CLI flags
        passed to the recipe. Use this for meshes/contact-sites/custom recipes;
        for the common object measurement, `measure` is the shortcut."""
        path = _find_recipe(recipe)
        if path is None:
            raise FileNotFoundError(
                f"recipe {recipe!r} not found. Call list_recipes to see available ones."
            )
        return await _run_and_ingest(path, source, name or recipe, args or [])

    @mcp.tool()
    async def launch_or_attach(new: bool = False, session_id: str | None = None) -> dict:
        """Launch Tourguide if needed, or attach to a workspace tab. Call this
        first; other tools auto-attach but this surfaces launch problems and
        the tab choice explicitly.

        Returns the bound session record (includes `sessionId` and `label`).
        Selection never silently guesses among multiple open tabs:
          - default: attach to the sole live tab, or open one if none exist.
          - if several tabs are open: returns `{ambiguous: true, sessions: […]}`
            — show the labels, ask the user which, then call again with
            `session_id` (or `new=true` for a fresh dedicated tab).
          - `new=true`: open a fresh tab and bind to it (parallel workspace).
          - `session_id="…"`: bind to that specific tab.

        When the result has a `shareUrl`, ALWAYS report it to the user: it is
        the address others on the same network can open to view this same
        workspace in their own browser."""
        result = await session.launch_or_attach(new=new, session=session_id)
        # Make the network-shareable address impossible to miss: a live tab's
        # `url` is localhost-only, so promote the LAN URL to `shareUrl` with a
        # human-facing note the agent relays verbatim.
        if isinstance(result, dict) and result.get("lanUrl"):
            result["shareUrl"] = result["lanUrl"]
            result["shareNote"] = (
                f"Others on your network can view this workspace at {result['lanUrl']}"
            )
        return result

    @mcp.tool()
    async def get_session() -> dict:
        """Get a summary of the current workspace: layers, selected segments,
        camera position, tables, plots, saved states, and recording status."""
        return await session.call("get_session")

    @mcp.tool()
    async def load_descriptor(descriptor: dict) -> dict:
        """Load a Tourguide dataset descriptor (layers + voxel size + metadata)
        into the viewer."""
        return await session.call("load_descriptor", {"descriptor": descriptor})

    @mcp.tool()
    async def get_viewer_state() -> dict:
        """Get the raw Neuroglancer viewer state (escape hatch; prefer
        get_session for a summary)."""
        return await session.call("get_viewer_state")

    @mcp.tool()
    async def set_viewer_state(state: dict) -> dict:
        """Apply a raw Neuroglancer viewer-state blob (escape hatch). Prefer
        select_segments / fly_to / add_layer for routine work."""
        return await session.call("set_viewer_state", {"state": state})

    @mcp.tool()
    async def get_selection() -> dict:
        """Get currently selected (visible) segment IDs per segmentation layer."""
        return await session.call("get_selection")

    @mcp.tool()
    async def select_segments(layer: str, segment_ids: list[str]) -> dict:
        """Set which segment IDs are highlighted/visible in a segmentation layer."""
        return await session.call(
            "select_segments", {"layer": layer, "segmentIds": segment_ids}
        )

    @mcp.tool()
    async def fly_to(
        position: list[float], segment_id: str | None = None, layer: str | None = None
    ) -> dict:
        """Move the camera to a world-space position [x, y, z] in nanometers,
        optionally highlighting a segment in a layer."""
        params: dict[str, Any] = {"position": position}
        if segment_id is not None:
            params["segmentId"] = segment_id
        if layer is not None:
            params["layer"] = layer
        return await session.call("fly_to", params)

    @mcp.tool()
    async def add_layer(layer: dict) -> dict:
        """Add or replace a single Neuroglancer layer (spec must include 'name')."""
        return await session.call("add_layer", {"layer": layer})

    @mcp.tool()
    async def add_annotations(annotations: list[dict], layer_name: str | None = None) -> dict:
        """Add annotations to an annotation layer. Each is one of:
        {type:'point', position:[x,y,z], label?}, {type:'line', points:[[x,y,z],...], label?},
        {type:'bbox', min:[x,y,z], max:[x,y,z], label?}. Coordinates in nm."""
        params: dict[str, Any] = {"annotations": annotations}
        if layer_name is not None:
            params["layerName"] = layer_name
        return await session.call("add_annotations", params)

    @mcp.tool()
    async def list_tables() -> dict:
        """List the data tables available in the workspace (id, name, row count,
        columns)."""
        return await session.call("list_tables")

    @mcp.tool()
    async def get_table_schema(table: str) -> dict:
        """Get the column names and types for a table."""
        return await session.call("get_table_schema", {"table": table})

    @mcp.tool()
    async def run_sql(sql: str) -> dict:
        """Run a read-only SQL query against the workspace's in-memory tables.
        Returns columns + rows (capped); for a persisted, browsable result use
        show_table instead."""
        return await session.call("run_sql", {"sql": sql})

    @mcp.tool()
    async def ingest_table(
        name: str,
        path: str | None = None,
        columns: list[str] | None = None,
        rows: list[list] | None = None,
    ) -> dict:
        """Push a table YOU computed into Tourguide. This is the core of the
        workspace model: you read the data and compute in your OWN environment,
        then send the result here to display. The table appears in the
        structured browser with click-to-fly (include an 'object_id' column and
        'com_x_nm'/'com_y_nm'/'com_z_nm' for navigation). Returns the table id.

        PREFER `path` for anything beyond a few rows: write the table to a file
        in your environment (.csv, or .json as list-of-records / {columns,rows})
        and pass its path. The server reads + forwards it directly, so the data
        never has to be serialized through your token stream — this is FAR
        faster than passing `rows` inline (which makes you re-emit every value
        as text). Use inline `columns`+`rows` only for tiny tables."""
        if path is not None:
            columns, rows = _load_table(path)
        if not columns or rows is None:
            raise ValueError("ingest_table: provide `path`, or both `columns` and `rows`")
        return await session.call("ingest_table", {"name": name, "columns": columns, "rows": rows})

    @mcp.tool()
    async def show_table(sql: str, name: str | None = None) -> dict:
        """Run SQL against tables already in Tourguide and open the result as a
        new table. (To display a table you computed yourself, use ingest_table.)"""
        params: dict[str, Any] = {"sql": sql}
        if name is not None:
            params["name"] = name
        return await session.call("show_table", params)

    @mcp.tool()
    async def show_plot(
        png_path: str | None = None,
        png: str | None = None,
        title: str | None = None,
        kind: str | None = None,
        source_table: str | None = None,
    ) -> dict:
        """Display a plot in Tourguide. Render the figure YOURSELF in your own
        environment (matplotlib savefig) to a .png file and pass its `png_path`
        — the server reads + encodes it, so the image never goes through your
        token stream (do NOT read the file or a base64 string into your context;
        that's slow and cost minutes in testing). `png` (an inline base64/data
        URL) also works for small images. There is intentionally no in-browser
        (Pyodide) code path: do all compute + rendering in your environment.
        Returns the plot artifact id."""
        params: dict[str, Any] = {}
        if png_path is not None:
            p = Path(png_path).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"show_plot: no file at {p}")
            png = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
        if png is None:
            raise ValueError("show_plot: provide png_path (preferred) or png")
        params["png"] = png
        if title is not None:
            params["title"] = title
        if kind is not None:
            params["kind"] = kind
        if source_table is not None:
            params["sourceTable"] = source_table
        return await session.call("show_plot", params)

    @mcp.tool()
    async def save_session_state(name: str | None = None) -> dict:
        """Save the current workspace state (viewer + descriptor + table/plot
        refs) as a named, restorable saved state. Returns its id."""
        params: dict[str, Any] = {}
        if name is not None:
            params["name"] = name
        return await session.call("save_session_state", params)

    @mcp.tool()
    async def restore_session_state(id: str) -> dict:
        """Restore a previously saved workspace state by id."""
        return await session.call("restore_session_state", {"id": id})

    @mcp.tool()
    async def list_saved_states() -> dict:
        """List saved workspace states (id, name, createdAt)."""
        return await session.call("list_saved_states")

    @mcp.tool()
    async def start_recording() -> dict:
        """Start a workspace recording session (tracks narration notes)."""
        return await session.call("start_recording")

    @mcp.tool()
    async def stop_recording() -> dict:
        """Stop the active workspace recording."""
        return await session.call("stop_recording")

    @mcp.tool()
    async def add_narration_note(
        text: str, position: list[float] | None = None, segment_id: str | None = None
    ) -> dict:
        """Add a narration note (a workspace-side annotation of what's
        interesting), optionally tied to a position/segment."""
        params: dict[str, Any] = {"text": text}
        if position is not None:
            params["position"] = position
        if segment_id is not None:
            params["segmentId"] = segment_id
        return await session.call("add_narration_note", params)

    @mcp.tool()
    async def export_session_summary() -> dict:
        """Export the session: saved states, plots, tables, recording status,
        and narration notes."""
        return await session.call("export_session_summary")
