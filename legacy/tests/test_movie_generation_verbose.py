#!/usr/bin/env python3
"""
Test script for generating movies from existing recording sessions with verbose progress.

This script provides real-time progress monitoring for the neuroglancer video_tool
rendering process and FFmpeg compilation.

Usage:
    python test_movie_generation_verbose.py <session_path> [transition_type]

Examples:
    python test_movie_generation_verbose.py recordings/session_selected_20251228_152416
    python test_movie_generation_verbose.py recordings/session_selected_20251228_152416 interpolate
    python test_movie_generation_verbose.py recordings/session_selected_20251228_152416 crossfade
"""

import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from server.recording import RecordingSession, FrameRecord, generate_public_url

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


def compile_with_interpolation_verbose(session: RecordingSession) -> Path:
    """
    Compile movie with neuroglancer state interpolation with verbose progress.

    This is a modified version that shows real-time progress.
    """
    print(f"[COMPILER] Compiling with neuroglancer state interpolation", flush=True)

    session_dir = session.base_dir
    narrated_frames = [f for f in session.frames if f.has_narration]

    if not narrated_frames:
        raise RuntimeError("No frames with narration found.")

    print(f"[COMPILER] Using {len(narrated_frames)} frames with narration", flush=True)

    # Create neuroglancer script file
    script_file = session_dir / "neuroglancer_script.txt"

    print(f"[COMPILER] Creating neuroglancer script", flush=True)

    valid_frames = []
    with open(script_file, "w") as f:
        for i, frame in enumerate(narrated_frames):
            state_file = session_dir / frame.state_file
            try:
                with open(state_file, "r") as state_f:
                    state_json = json.load(state_f)

                if "layers" not in state_json or not state_json["layers"]:
                    print(f"[COMPILER] Warning: Frame {i + 1} has no layers, skipping", flush=True)
                    continue

                has_source = any("source" in layer for layer in state_json["layers"])
                if not has_source:
                    print(f"[COMPILER] Warning: Frame {i + 1} has no layer sources, skipping", flush=True)
                    continue

                public_url = generate_public_url(state_json)

            except Exception as e:
                print(f"[COMPILER] Warning: Failed to read state for frame {i + 1}: {e}", flush=True)
                continue

            if len(valid_frames) > 0:
                transition_duration = 2.0
                f.write(f"{transition_duration}\n")

            valid_frames.append(frame)
            f.write(f"{public_url}\n")

    if len(valid_frames) < 2:
        raise RuntimeError(f"Need at least 2 valid frames, found {len(valid_frames)}")

    print(f"[COMPILER] Script created with {len(valid_frames)} keyframes", flush=True)

    # Create output directory
    frames_dir = session_dir / "interpolated_frames"
    frames_dir.mkdir(exist_ok=True)

    # Render frames with neuroglancer video_tool
    print(f"\n{'='*80}", flush=True)
    print(f"RENDERING FRAMES WITH NEUROGLANCER (this may take several minutes)", flush=True)
    print(f"{'='*80}\n", flush=True)

    render_cmd = [
        sys.executable,
        "-m",
        "neuroglancer.tool.video_tool",
        "render",
        "--fps", "30",
        "--width", "960",
        "--height", "540",
        "--scale-bar-scale", "1.0",
        str(script_file),
        str(frames_dir),
    ]

    print(f"Running: {' '.join(str(x) for x in render_cmd)}\n", flush=True)

    # Run with real-time output
    start_time = time.time()
    process = subprocess.Popen(
        render_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Stream output in real-time
    last_progress = ""
    for line in process.stdout:
        line = line.rstrip()
        if line:
            # Show progress updates on same line if possible
            if "Screenshot in progress:" in line or "Requesting screenshot" in line:
                print(f"\r{line}", end='', flush=True)
                last_progress = line
            else:
                if last_progress:
                    print()  # New line after progress
                    last_progress = ""
                print(line, flush=True)

    if last_progress:
        print()  # Final newline

    return_code = process.wait()
    elapsed = time.time() - start_time

    if return_code != 0:
        raise RuntimeError(f"Neuroglancer video_tool failed with code {return_code}")

    print(f"\n[COMPILER] Rendering completed in {elapsed:.1f} seconds", flush=True)

    # Count rendered frames
    frame_files = sorted(frames_dir.glob("*.png"))
    print(f"[COMPILER] Found {len(frame_files)} rendered frames", flush=True)

    if not frame_files:
        raise RuntimeError("No frames were rendered")

    # Create FFmpeg concat file and compile
    print(f"\n{'='*80}", flush=True)
    print(f"COMPILING VIDEO WITH FFMPEG", flush=True)
    print(f"{'='*80}\n", flush=True)

    concat_file = session_dir / "concat_interpolated.txt"
    frames_per_keyframe_pair = len(frame_files) // (len(narrated_frames) - 1) if len(narrated_frames) > 1 else len(frame_files)

    print(f"[COMPILER] Approximately {frames_per_keyframe_pair} frames per keyframe transition", flush=True)

    with open(concat_file, "w") as f:
        for i, narrated_frame in enumerate(narrated_frames):
            if len(narrated_frames) > 1:
                keyframe_render_idx = i * frames_per_keyframe_pair
            else:
                keyframe_render_idx = 0

            keyframe_render_idx = min(keyframe_render_idx, len(frame_files) - 1)

            # Hold on keyframe for narration duration
            audio_duration = narrated_frame.display_duration
            f.write(f"file '{frame_files[keyframe_render_idx].absolute()}'\n")
            f.write(f"duration {audio_duration}\n")

            # Add transition to next keyframe
            if i < len(narrated_frames) - 1:
                transition_start = keyframe_render_idx
                transition_end = min(transition_start + 60, len(frame_files))
                transition_frames = frame_files[transition_start:transition_end]

                frame_duration = 2.0 / len(transition_frames) if transition_frames else 0.033
                for trans_frame in transition_frames:
                    f.write(f"file '{trans_frame.absolute()}'\n")
                    f.write(f"duration {frame_duration}\n")

        # Duplicate last frame
        if frame_files:
            f.write(f"file '{frame_files[-1].absolute()}'\n")

    # Create audio timeline
    from server.recording import MovieCompiler
    compiler = MovieCompiler(session)
    audio_file = compiler._create_audio_timeline(add_transition_silence=True)

    # Compile final video
    output_file = session_dir / "output" / "movie.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-i", str(audio_file),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        str(output_file),
    ]

    print(f"[COMPILER] Running FFmpeg...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"FFmpeg stderr: {result.stderr}", flush=True)
        raise RuntimeError(f"FFmpeg failed with code {result.returncode}")

    return output_file


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

    # Override transition type if provided
    if len(sys.argv) >= 3:
        transition_type = sys.argv[2].lower()
        if transition_type not in ['cut', 'crossfade', 'interpolate']:
            print(f"Error: Invalid transition type '{transition_type}'")
            sys.exit(1)
        session.transition_type = transition_type

    # Print session info
    print_session_info(session)

    # Check for narrated frames
    narrated_frames = [f for f in session.frames if f.has_narration]
    if not narrated_frames:
        print("ERROR: No frames with narration found!")
        sys.exit(1)

    # Compile movie
    print(f"\nStarting movie compilation with {session.transition_type} transitions...")

    try:
        if session.transition_type == 'interpolate':
            output_file = compile_with_interpolation_verbose(session)
        else:
            # Use standard compiler for other types
            from server.recording import MovieCompiler
            compiler = MovieCompiler(session)
            output_file = compiler.compile()

        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"Output: {output_file}")
        print(f"Size: {output_file.stat().st_size / (1024*1024):.1f} MB")
        print(f"\nView with: ffplay {output_file}")
        print("="*80 + "\n")

    except Exception as e:
        print("\n" + "="*80)
        print("FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
