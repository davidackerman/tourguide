#!/usr/bin/env python3
"""
Simple test script for generating movies using 'cut' transitions (fastest method).

This bypasses the neuroglancer video_tool entirely and just uses FFmpeg to stitch
together the existing screenshot frames with audio. Much faster for testing!

Usage:
    python test_movie_simple.py <session_path>

Example:
    python test_movie_simple.py recordings/session_selected_20251228_152416
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from server.recording import RecordingSession, MovieCompiler, FrameRecord

def load_session(session_path: Path) -> RecordingSession:
    """Load a recording session from disk."""
    metadata_file = session_path / "metadata.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"No metadata.json found in {session_path}")

    print(f"Loading session from {session_path}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Parse timestamps
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

    # Parse frames
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

    # Create session object - force "cut" transitions
    session = RecordingSession(
        session_id=metadata['session_id'],
        base_dir=session_path,
        created_at=created_at,
        fps=metadata.get('fps', 1.0),
        status=metadata.get('status', 'stopped'),
        recording_started_at=recording_started_at,
        recording_stopped_at=recording_stopped_at,
        frames=frames,
        transition_type='cut',  # Force cut transitions
        transition_duration=0.5,
        error_message=metadata.get('error_message'),
    )

    return session


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    session_path = Path(sys.argv[1])

    if not session_path.exists():
        print(f"Error: Session path does not exist: {session_path}")
        sys.exit(1)

    # Load session
    try:
        session = load_session(session_path)
    except Exception as e:
        print(f"Error loading session: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print session info
    print(f"\n{'='*80}")
    print(f"SESSION: {session.session_id}")
    print(f"{'='*80}")
    print(f"Total frames: {len(session.frames)}")

    narrated_frames = [f for f in session.frames if f.has_narration]
    print(f"Frames with narration: {len(narrated_frames)}")

    print(f"\nTransition type: {session.transition_type} (fast, no interpolation)")
    print(f"\nThis will create a movie with:")
    print(f"  - Direct cuts between frames")
    print(f"  - 2 second transitions")
    print(f"  - Audio narration aligned with frames")

    total_duration = sum(f.display_duration + 2.0 for f in narrated_frames) - 2.0
    print(f"\nEstimated duration: {total_duration:.1f} seconds")

    print(f"\nFrame details:")
    print("-" * 80)
    for i, frame in enumerate(session.frames, 1):
        if frame.has_narration:
            narration_text = frame.narration_text[:60] + "..." if frame.narration_text and len(frame.narration_text) > 60 else (frame.narration_text or "")
            print(f"🎤 Frame {i:3d}: {frame.display_duration:.1f}s  {narration_text}")
    print(f"{'='*80}\n")

    if not narrated_frames:
        print("ERROR: No frames with narration found!")
        sys.exit(1)

    # Compile movie
    print(f"Starting movie compilation (this should take < 30 seconds)...\n")

    try:
        compiler = MovieCompiler(session)
        output_file = compiler.compile()

        print(f"\n{'='*80}")
        print(f"SUCCESS!")
        print(f"{'='*80}")
        print(f"Output: {output_file}")
        print(f"Size: {output_file.stat().st_size / (1024*1024):.1f} MB")
        print(f"\nView with: ffplay {output_file}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"FAILED")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
