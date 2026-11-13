#!/usr/bin/env python3
"""
Test script to validate neuroglancer state before movie compilation.

This shows the new validation warnings for missing image layers.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from server.recording import RecordingSession, MovieCompiler, FrameRecord

def load_session(session_path: Path) -> RecordingSession:
    """Load a recording session from disk."""
    metadata_file = session_path / "metadata.json"

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    created_at = datetime.fromisoformat(metadata['created_at'])
    recording_started_at = (
        datetime.fromisoformat(metadata['recording_started_at'])
        if metadata.get('recording_started_at')
        else None
    )
    recording_stopped_at = (
        datetime.fromisoformat(metadata['recording_stopped_at'])
        if metadata.get('recording_stopped_at')
        else None
    )

    frames = []
    for frame_data in metadata.get('frames', []):
        frame = FrameRecord(
            frame_number=frame_data['frame_number'],
            timestamp=frame_data['timestamp'],
            frame_file=frame_data['frame_file'],
            state_file=frame_data['state_file'],
            urls_file=frame_data['urls_file'],
            has_narration=frame_data.get('has_narration', False),
            narration_file=frame_data.get('narration_file'),
            narration_text=frame_data.get('narration_text'),
            display_duration=frame_data.get('display_duration', 2.0),
        )
        frames.append(frame)

    transitions = metadata.get('transitions', {})

    session = RecordingSession(
        session_id=metadata['session_id'],
        base_dir=session_path,
        created_at=created_at,
        fps=metadata.get('fps', 1.0),
        status=metadata.get('status', 'stopped'),
        recording_started_at=recording_started_at,
        recording_stopped_at=recording_stopped_at,
        frames=frames,
        transition_type='interpolate',  # Force interpolate to trigger validation
        transition_duration=transitions.get('duration', 0.5),
        error_message=metadata.get('error_message'),
    )

    return session


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_validation.py <session_path>")
        sys.exit(1)

    session_path = Path(sys.argv[1])
    session = load_session(session_path)

    print("Testing validation for interpolation mode...")
    print("This will show warnings if frames are missing image layers.\n")

    try:
        compiler = MovieCompiler(session)
        # This will just create the neuroglancer script and show warnings
        # We'll catch the error when it tries to actually run video_tool
        compiler.compile_with_interpolation()
    except Exception as e:
        print(f"\nExpected error (video_tool would hang): {e}")
