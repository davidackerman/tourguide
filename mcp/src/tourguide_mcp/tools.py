"""MCP tool surface for Tourguide.

Each tool maps closely to a Workspace API operation. The tools are
intentionally high-level (semantic viewer ops, tables, plots, saved states,
recording) rather than low-level Neuroglancer primitives — set_viewer_state
is the escape hatch. Keep this adapter thin: if a convenience belongs
anywhere, it belongs in the Workspace API, not here.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .session import WorkspaceSession


def register_tools(mcp: FastMCP, session: WorkspaceSession) -> None:
    @mcp.tool()
    async def launch_or_attach() -> dict:
        """Launch Tourguide if needed, or attach to a running workspace session.
        Returns the session record. Call this first; other tools auto-attach
        but this surfaces launch problems explicitly."""
        return await session.launch_or_attach()

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
    async def show_table(sql: str, name: str | None = None) -> dict:
        """Run SQL and open the result as a Tourguide table in the structured
        browser. Returns the new table id."""
        params: dict[str, Any] = {"sql": sql}
        if name is not None:
            params["name"] = name
        return await session.call("show_table", params)

    @mcp.tool()
    async def show_plot(
        code: str | None = None,
        question: str | None = None,
        title: str | None = None,
        kind: str | None = None,
        source_table: str | None = None,
    ) -> dict:
        """Create a plot inside Tourguide. Provide matplotlib `code` (preferred;
        tables are preloaded as df_<class> DataFrames) or a natural-language
        `question` (needs Tourguide's AI backend). Returns the plot artifact id."""
        params: dict[str, Any] = {}
        if code is not None:
            params["code"] = code
        if question is not None:
            params["question"] = question
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
