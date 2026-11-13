"""Shared constants mirroring web-app/src/workspace_api/protocol.ts."""

from __future__ import annotations


class WorkspaceError(RuntimeError):
    """A Workspace operation returned ok:false, or the bridge was unreachable."""


WORKSPACE_OPS = (
    "launch_or_attach",
    "get_session",
    "load_descriptor",
    "wait_for_ready",
    "screenshot",
    "get_viewer_state",
    "set_viewer_state",
    "get_selection",
    "select_segments",
    "fly_to",
    "fly_to_segment",
    "add_layer",
    "add_annotations",
    "list_tables",
    "get_table_schema",
    "run_sql",
    "ingest_table",
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

# Annotation shapes accepted by add_annotations (map 1:1 to Neuroglancer):
#   {"type": "point", "position": [x,y,z], "label": "..."}
#   {"type": "line",  "points": [[x,y,z], [x,y,z]], "label": "..."}
#   {"type": "bbox",  "min": [x,y,z], "max": [x,y,z], "label": "..."}
ANNOTATION_TYPES = ("point", "line", "bbox")
