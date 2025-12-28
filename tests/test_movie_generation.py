#!/usr/bin/env python3
"""
Test script for generating movies from existing recording sessions.

This script allows you to test movie generation from a previously recorded session
without needing to run the server and capture new screenshots. It loads the session
metadata and runs the movie compiler with progress reporting.

Usage:
    python test_movie_generation.py <session_path> [transition_type]

Examples:
    python test_movie_generation.py recordings/session_selected_20251228_152416
    python test_movie_generation.py recordings/session_selected_20251228_152416 interpolate
    python test_movie_generation.py recordings/session_selected_20251228_152416 crossfade
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from server.recording import RecordingSession, MovieCompiler, FrameRecord

def load_session(session_path: Path) -> RecordingSession:
    """
    Load a recording session from disk.

    Args:
        session_path: Path to the session directory

    Returns:
        RecordingSession object

    Raises:
        FileNotFoundError: If metadata.json doesn't exist
        ValueError: If metadata is invalid
    """
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

    # Get transition settings
    transitions = metadata.get('transitions', {})
    transition_type = transitions.get('type', 'cut')
    transition_duration = transitions.get('duration', 0.5)

    # Create session object
    session = RecordingSession(
        session_id=metadata['session_id'],
        base_dir=session_path,
        created_at=created_at,
        fps=metadata.get('fps', 1.0),
        status=metadata.get('status', 'stopped'),
        recording_started_at=recording_started_at,
        recording_stopped_at=recording_stopped_at,
        frames=frames,
        transition_type=transition_type,
        transition_duration=transition_duration,
        error_message=metadata.get('error_message'),
    )

    return session


def print_session_info(session: RecordingSession):
    """Print session information."""
    print("\n" + "="*80)
    print(f"SESSION: {session.session_id}")
    print("="*80)
    print(f"Created: {session.created_at}")
    print(f"Status: {session.status}")
    print(f"Total frames: {len(session.frames)}")

    narrated_frames = [f for f in session.frames if f.has_narration]
    print(f"Frames with narration: {len(narrated_frames)}")

    if session.error_message:
        print(f"Previous error: {session.error_message}")

    print(f"\nTransition type: {session.transition_type}")
    print(f"Transition duration: {session.transition_duration}s")

    # Print frame details
    print(f"\nFrame details:")
    print("-" * 80)
    for i, frame in enumerate(session.frames, 1):
        narration_indicator = "🎤" if frame.has_narration else "  "
        duration_str = f"{frame.display_duration:.1f}s" if frame.has_narration else "N/A"
        narration_text = frame.narration_text[:60] + "..." if frame.narration_text and len(frame.narration_text) > 60 else (frame.narration_text or "")
        print(f"{narration_indicator} Frame {i:3d}: duration={duration_str:6s}  {narration_text}")

    print("="*80 + "\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    session_path = Path(sys.argv[1])

    if not session_path.exists():
        print(f"Error: Session path does not exist: {session_path}")
        sys.exit(1)

    if not session_path.is_dir():
        print(f"Error: Session path is not a directory: {session_path}")
        sys.exit(1)

    # Load session
    try:
        session = load_session(session_path)
    except Exception as e:
        print(f"Error loading session: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Override transition type if provided
    if len(sys.argv) >= 3:
        transition_type = sys.argv[2].lower()
        if transition_type not in ['cut', 'crossfade', 'interpolate']:
            print(f"Error: Invalid transition type '{transition_type}'")
            print("Valid options: cut, crossfade, interpolate")
            sys.exit(1)
        print(f"\nOverriding transition type to: {transition_type}")
        session.transition_type = transition_type

    # Print session info
    print_session_info(session)

    # Check for narrated frames
    narrated_frames = [f for f in session.frames if f.has_narration]
    if not narrated_frames:
        print("WARNING: No frames with narration found!")
        print("Movie compilation requires at least one frame with narration.")
        sys.exit(1)

    # Confirm before proceeding
    print(f"Ready to compile movie with {len(narrated_frames)} narrated frames")
    print(f"Transition type: {session.transition_type}")
    print(f"Output will be saved to: {session.base_dir / 'output' / 'movie.mp4'}")

    response = input("\nProceed with compilation? [y/N]: ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Compile movie
    print("\n" + "="*80)
    print("STARTING MOVIE COMPILATION")
    print("="*80 + "\n")

    try:
        compiler = MovieCompiler(session)
        output_file = compiler.compile()

        print("\n" + "="*80)
        print("COMPILATION SUCCESSFUL!")
        print("="*80)
        print(f"Output file: {output_file}")
        print(f"File size: {output_file.stat().st_size / (1024*1024):.1f} MB")
        print(f"\nYou can view the movie with:")
        print(f"  ffplay {output_file}")
        print(f"  or open {output_file}")
        print("="*80 + "\n")

    except Exception as e:
        print("\n" + "="*80)
        print("COMPILATION FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
