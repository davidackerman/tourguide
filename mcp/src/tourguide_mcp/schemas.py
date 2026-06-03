"""Shared constants mirroring web-app/src/workspace_api/protocol.ts.

The wire contract lives in TypeScript; this is the Python-side reflection so
the MCP tools and the launcher stay in lockstep with the Workspace API.
"""

from __future__ import annotations

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

# Workspace annotation shapes accepted by add_annotations (minimal v0):
#   {"type": "point", "position": [x,y,z], "label": "..."}
#   {"type": "line",  "points": [[x,y,z], ...], "label": "..."}
#   {"type": "bbox",  "min": [x,y,z], "max": [x,y,z], "label": "..."}
ANNOTATION_TYPES = ("point", "line", "bbox")
