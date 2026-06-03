"""Shared constants for the Tourguide Python SDK, mirroring the Workspace API
protocol (web-app/src/workspace_api/protocol.ts)."""

from __future__ import annotations


class WorkspaceError(RuntimeError):
    """A Workspace op returned ok:false, or the bridge was unreachable."""


WORKSPACE_OPS = (
    "launch_or_attach",
    "get_session",
    "load_descriptor",
    "get_viewer_state",
    "set_viewer_state",
    "get_selection",
    "select_segments",
    "fly_to",
    "add_layer",
    "add_annotations",
    "list_tables",
    "get_table_schema",
    "run_sql",
    "show_table",
    "show_plot",
    "save_session_state",
    "restore_session_state",
    "list_saved_states",
    "start_recording",
    "stop_recording",
    "add_narration_note",
    "export_session_summary",
)
