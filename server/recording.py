"""
Recording and movie compilation module for Neuroglancer tour guide.

This module handles:
- Recording sessions with frame capture, state saving, and URL generation
- Movie compilation using FFmpeg with multiple transition types
- Audio timeline creation and synchronization
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import uuid
import subprocess
import urllib.parse
import time
import shutil


def generate_public_url(state_json: Dict[str, Any]) -> str:
    """
    Generate a public Neuroglancer URL from state JSON.

    The neuroglancer-demo.appspot.com service uses a JSON state format
    that can be URL-encoded and passed as a fragment identifier.

    Args:
        state_json: Neuroglancer state as dictionary

    Returns:
        Public URL that can be shared to view this exact state
    """
    # Compact JSON format (no whitespace)
    json_str = json.dumps(state_json, separators=(",", ":"))

    # URL-encode the JSON
    encoded = urllib.parse.quote(json_str, safe="")

    # Build the full URL with fragment identifier
    public_url = f"https://neuroglancer-demo.appspot.com/#!{encoded}"

    return public_url


@dataclass
class FrameRecord:
    """Represents a single recorded frame with associated metadata."""

    frame_number: int
    timestamp: float
    frame_file: str  # relative path from session directory
    state_file: str  # relative path from session directory
    urls_file: str  # relative path from session directory
    has_narration: bool = False
    narration_file: Optional[str] = None  # relative path from session directory
    narration_text: Optional[str] = None
    display_duration: float = 2.0  # default duration if no audio (seconds)


@dataclass
class RecordingSession:
    """Manages a single recording session."""

    session_id: str
    base_dir: Path
    created_at: datetime
    fps: float
    status: str = (
        "initialized"  # initialized|recording|stopped|compiling|completed|error
    )
    recording_started_at: Optional[datetime] = None
    recording_stopped_at: Optional[datetime] = None
    frames: List[FrameRecord] = field(default_factory=list)
    transition_type: str = "cut"  # cut|crossfade|interpolate
    transition_duration: float = 0.5  # for crossfade/interpolate

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "recording_started_at": (
                self.recording_started_at.isoformat()
                if self.recording_started_at
                else None
            ),
            "recording_stopped_at": (
                self.recording_stopped_at.isoformat()
                if self.recording_stopped_at
                else None
            ),
            "fps": self.fps,
            "frame_count": len(self.frames),
            "narration_count": sum(1 for f in self.frames if f.has_narration),
            "status": self.status,
            "transitions": {
                "type": self.transition_type,
                "duration": self.transition_duration,
            },
            "frames": [asdict(frame) for frame in self.frames],
        }

    def save_metadata(self):
        """Save metadata to JSON file."""
        metadata_file = self.base_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[RECORDING] Metadata saved: {len(self.frames)} frames", flush=True)


class RecordingManager:
    """Manages recording sessions and movie compilation."""

    def __init__(self, base_recordings_dir: Path = None):
        if base_recordings_dir is None:
            base_recordings_dir = Path("./recordings")
        self.base_recordings_dir = base_recordings_dir
        self.base_recordings_dir.mkdir(exist_ok=True)
        self.current_session: Optional[RecordingSession] = None
        self.is_recording = False

    def start_recording(
        self, fps: float, transition_type: str = "cut", transition_duration: float = 0.5
    ) -> RecordingSession:
        """
        Start a new recording session.

        Args:
            fps: Screenshot capture rate
            transition_type: Type of transition (cut, crossfade, interpolate)
            transition_duration: Duration of transitions in seconds

        Returns:
            RecordingSession object

        Raises:
            ValueError: If recording already in progress
        """
        if self.is_recording:
            raise ValueError("Recording already in progress")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp}"
        session_dir = self.base_recordings_dir / session_id

        # Create directory structure
        session_dir.mkdir(exist_ok=True)
        (session_dir / "frames").mkdir(exist_ok=True)
        (session_dir / "audio").mkdir(exist_ok=True)
        (session_dir / "states").mkdir(exist_ok=True)
        (session_dir / "urls").mkdir(exist_ok=True)
        (session_dir / "output").mkdir(exist_ok=True)

        self.current_session = RecordingSession(
            session_id=session_id,
            base_dir=session_dir,
            created_at=datetime.now(),
            fps=fps,
            transition_type=transition_type,
            transition_duration=transition_duration,
        )

        self.current_session.recording_started_at = datetime.now()
        self.current_session.status = "recording"
        self.is_recording = True
        self.current_session.save_metadata()

        print(f"[RECORDING] Started session {session_id} at {fps} fps", flush=True)
        return self.current_session

    def stop_recording(self):
        """
        Stop the current recording session.

        Raises:
            ValueError: If no recording in progress
        """
        if not self.is_recording or not self.current_session:
            raise ValueError("No recording in progress")

        self.current_session.recording_stopped_at = datetime.now()
        self.current_session.status = "stopped"
        self.is_recording = False
        self.current_session.save_metadata()

        print(
            f"[RECORDING] Stopped session {self.current_session.session_id} with {len(self.current_session.frames)} frames",
            flush=True,
        )

    def add_frame(
        self,
        jpeg_bytes: bytes,
        state_json: Dict[str, Any],
        timestamp: float,
        narration_text: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
        audio_format: str = "mp3",
    ):
        """
        Add a frame to the current recording.

        This saves the screenshot, state JSON, and public URL. Audio can be
        added separately after frame creation via update_frame_narration().

        Args:
            jpeg_bytes: JPEG image data
            state_json: Neuroglancer state dictionary
            timestamp: Unix timestamp
            narration_text: Optional narration text (for metadata)
            audio_bytes: Optional audio file bytes
            audio_format: Audio file format extension (mp3 or wav)
        """
        if not self.is_recording or not self.current_session:
            return  # Silently skip if not recording

        frame_number = len(self.current_session.frames) + 1
        session_dir = self.current_session.base_dir

        # Save frame image
        frame_file = f"frame_{frame_number:06d}.jpg"
        frame_path = session_dir / "frames" / frame_file
        with open(frame_path, "wb") as f:
            f.write(jpeg_bytes)

        # Save state JSON
        state_file = f"state_{frame_number:06d}.json"
        state_path = session_dir / "states" / state_file
        with open(state_path, "w") as f:
            json.dump(state_json, f, indent=2)

        # Generate and save public URL
        public_url = generate_public_url(state_json)
        urls_file = f"urls_{frame_number:06d}.txt"
        urls_path = session_dir / "urls" / urls_file
        with open(urls_path, "w") as f:
            f.write(f"Neuroglancer Public URL:\n")
            f.write("=" * 80 + "\n")
            f.write(f"{public_url}\n\n")
            f.write(f"Instructions:\n")
            f.write("1. Copy the URL above\n")
            f.write("2. Paste into browser to view this exact state\n")
            f.write("3. The state JSON can be loaded programmatically\n\n")
            f.write(f"Raw State JSON:\n")
            f.write("=" * 80 + "\n")
            f.write(json.dumps(state_json, indent=2))

        # Save audio if present
        narration_file = None
        display_duration = 2.0  # default
        if audio_bytes:
            narration_file = f"narration_{frame_number:06d}.{audio_format}"
            audio_path = session_dir / "audio" / narration_file
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            # Calculate audio duration
            display_duration = self._get_audio_duration(audio_path)

        # Create frame record
        frame_record = FrameRecord(
            frame_number=frame_number,
            timestamp=timestamp,
            frame_file=f"frames/{frame_file}",
            state_file=f"states/{state_file}",
            urls_file=f"urls/{urls_file}",
            has_narration=narration_text is not None,
            narration_file=f"audio/{narration_file}" if narration_file else None,
            narration_text=narration_text,
            display_duration=display_duration,
        )

        self.current_session.frames.append(frame_record)

        # Save metadata every 10 frames to avoid excessive I/O
        if len(self.current_session.frames) % 10 == 0:
            self.current_session.save_metadata()

        print(
            f"[RECORDING] Frame {frame_number} saved (narration: {narration_text is not None})",
            flush=True,
        )

    def update_frame_narration(
        self,
        frame_number: int,
        narration_text: str,
        audio_bytes: bytes,
        audio_format: str = "mp3",
    ):
        """
        Update a frame with narration audio after it was created.

        This is used when narration is generated after the frame is captured.

        Args:
            frame_number: Frame number to update (1-indexed)
            narration_text: Narration text
            audio_bytes: Audio file bytes
            audio_format: Audio file format (mp3 or wav)
        """
        if not self.current_session:
            return

        if frame_number < 1 or frame_number > len(self.current_session.frames):
            print(
                f"[RECORDING] Warning: Frame {frame_number} not found for narration update",
                flush=True,
            )
            return

        session_dir = self.current_session.base_dir
        frame_record = self.current_session.frames[frame_number - 1]

        # Save audio file
        narration_file = f"narration_{frame_number:06d}.{audio_format}"
        audio_path = session_dir / "audio" / narration_file
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # Update frame record
        frame_record.has_narration = True
        frame_record.narration_file = f"audio/{narration_file}"
        frame_record.narration_text = narration_text
        frame_record.display_duration = self._get_audio_duration(audio_path)

        self.current_session.save_metadata()

        print(
            f"[RECORDING] Frame {frame_number} updated with narration ({frame_record.display_duration:.1f}s)",
            flush=True,
        )

    def _get_audio_duration(self, audio_path: Path) -> float:
        """
        Get duration of audio file in seconds using ffprobe.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds, or 2.0 if unable to determine
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            duration = float(result.stdout.strip())
            return max(duration, 0.5)  # Minimum 0.5 seconds
        except Exception as e:
            print(f"[RECORDING] Error getting audio duration: {e}", flush=True)
            return 2.0  # fallback default

    def cleanup_old_sessions(self, keep_days: int = 7):
        """
        Remove recording sessions older than keep_days.

        Args:
            keep_days: Number of days to keep recordings
        """
        cutoff = datetime.now() - timedelta(days=keep_days)

        for session_dir in self.base_recordings_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith("session_"):
                # Check metadata for age
                metadata_file = session_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        created_at = datetime.fromisoformat(metadata["created_at"])
                        if created_at < cutoff:
                            print(
                                f"[CLEANUP] Removing old session: {session_dir.name}",
                                flush=True,
                            )
                            shutil.rmtree(session_dir)
                    except Exception as e:
                        print(
                            f"[CLEANUP] Error checking session {session_dir.name}: {e}",
                            flush=True,
                        )

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all recording sessions.

        Returns:
            List of session metadata dictionaries, sorted by creation time (newest first)
        """
        sessions = []
        for session_dir in self.base_recordings_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith("session_"):
                metadata_file = session_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        sessions.append(metadata)
                    except Exception as e:
                        print(
                            f"[SESSION LIST] Error reading {session_dir.name}: {e}",
                            flush=True,
                        )

        return sorted(sessions, key=lambda x: x["created_at"], reverse=True)


class MovieCompiler:
    """Compiles recorded frames into a movie using FFmpeg."""

    def __init__(self, session: RecordingSession):
        self.session = session
        self.session_dir = session.base_dir

    def compile(self) -> Path:
        """
        Compile the movie based on session settings.

        Returns:
            Path to the output movie file

        Raises:
            Exception: If compilation fails
        """
        self.session.status = "compiling"
        self.session.save_metadata()

        try:
            if self.session.transition_type == "cut":
                output_file = self.compile_with_cuts()
            elif self.session.transition_type == "crossfade":
                output_file = self.compile_with_crossfade()
            elif self.session.transition_type == "interpolate":
                output_file = self.compile_with_interpolation()
            else:
                raise ValueError(
                    f"Unknown transition type: {self.session.transition_type}"
                )

            self.session.status = "completed"
            self.session.save_metadata()

            print(f"[COMPILER] Movie created: {output_file}", flush=True)
            return output_file

        except Exception as e:
            self.session.status = "error"
            self.session.save_metadata()
            print(f"[COMPILER] Error: {e}", flush=True)
            raise

    def compile_with_cuts(self) -> Path:
        """
        Compile movie with direct cuts and 2-second transitions.

        For each narrated frame:
        - Display the frame for the duration of the audio narration
        - Add a 2-second transition to the next frame (in silence)

        Uses FFmpeg concat demuxer to combine frames with their durations.
        Only includes frames that have narration.

        Returns:
            Path to output movie file
        """
        print(f"[COMPILER] Compiling with direct cuts and 2s transitions", flush=True)

        # Filter to only frames with narration
        narrated_frames = [f for f in self.session.frames if f.has_narration]

        if not narrated_frames:
            raise RuntimeError(
                "No frames with narration found. Cannot create movie without audio."
            )

        print(
            f"[COMPILER] Using {len(narrated_frames)} frames with narration (out of {len(self.session.frames)} total)",
            flush=True,
        )

        concat_file = self.session_dir / "concat_list.txt"
        output_file = self.session_dir / "output" / "movie.mp4"

        # Create concat file with narrated frames
        # Each frame is shown for its audio duration, then for an additional 2 seconds for transition
        with open(concat_file, "w") as f:
            for i, frame in enumerate(narrated_frames):
                frame_path = self.session_dir / frame.frame_file
                abs_path = frame_path.absolute()

                # Display frame for the duration of narration
                audio_duration = frame.display_duration
                f.write(f"file '{abs_path}'\n")
                f.write(f"duration {audio_duration}\n")

                # Add 2-second transition (except after the last frame)
                if i < len(narrated_frames) - 1:
                    f.write(f"file '{abs_path}'\n")
                    f.write(f"duration 2.0\n")

            # Duplicate last frame (ffmpeg concat requirement)
            if narrated_frames:
                last_frame = self.session_dir / narrated_frames[-1].frame_file
                abs_last_frame = last_frame.absolute()
                f.write(f"file '{abs_last_frame}'\n")

        # Create audio timeline with 2s silence between narrations for transitions
        audio_file = self._create_audio_timeline(add_transition_silence=True)

        # FFmpeg command with audio
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-i",
            str(audio_file),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",  # Match shortest stream
            str(output_file),
        ]

        print(f"[COMPILER] Running FFmpeg (cuts with 2s transitions)...", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"[COMPILER] FFmpeg stderr: {result.stderr}", flush=True)
            raise RuntimeError(f"FFmpeg failed with code {result.returncode}")

        return output_file

    def compile_with_crossfade(self) -> Path:
        """
        Compile movie with crossfade transitions.

        Uses FFmpeg xfade filter for smooth transitions between frames.
        This is more complex and slower than direct cuts.

        Returns:
            Path to output movie file
        """
        print(f"[COMPILER] Compiling with crossfade transitions", flush=True)

        # Filter to only frames with narration
        narrated_frames = [f for f in self.session.frames if f.has_narration]

        if not narrated_frames:
            raise RuntimeError(
                "No frames with narration found. Cannot create movie without audio."
            )

        print(
            f"[COMPILER] Using {len(narrated_frames)} frames with narration (out of {len(self.session.frames)} total)",
            flush=True,
        )

        output_file = self.session_dir / "output" / "movie.mp4"
        duration = self.session.transition_duration

        if len(narrated_frames) < 2:
            # Fall back to cuts if only one frame
            print(f"[COMPILER] Only one frame, using direct cuts instead", flush=True)
            return self.compile_with_cuts()

        # For crossfade, we'll create a simpler approach:
        # Use concat with fade in/out on each frame
        # This is more reliable than complex xfade filter graphs

        # Create individual clips with fade in/out
        temp_dir = self.session_dir / "temp_clips"
        temp_dir.mkdir(exist_ok=True)

        clip_files = []
        for i, frame in enumerate(narrated_frames):
            frame_path = self.session_dir / frame.frame_file
            clip_file = temp_dir / f"clip_{i:06d}.mp4"

            # Create video clip with fade in/out
            fade_duration = min(duration, frame.display_duration / 2)
            cmd = [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-i",
                str(frame_path),
                "-vf",
                f"fade=t=in:st=0:d={fade_duration},fade=t=out:st={frame.display_duration - fade_duration}:d={fade_duration}",
                "-t",
                str(frame.display_duration),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                str(clip_file),
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode != 0:
                print(f"[COMPILER] Warning: Failed to create clip {i}", flush=True)
                continue

            clip_files.append(clip_file)

        # Concat all clips
        concat_file = temp_dir / "concat_list.txt"
        with open(concat_file, "w") as f:
            for clip in clip_files:
                f.write(f"file '{clip}'\n")

        # Final concat
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_file),
        ]

        # Add audio if present
        has_audio = any(frame.has_narration for frame in self.session.frames)
        if has_audio:
            # Create audio timeline with 2s silence between narrations for transitions
            audio_file = self._create_audio_timeline(add_transition_silence=True)
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-i",
                str(audio_file),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(output_file),
            ]

        print(f"[COMPILER] Running FFmpeg (crossfade)...", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"[COMPILER] FFmpeg stderr: {result.stderr}", flush=True)
            raise RuntimeError(f"FFmpeg failed with code {result.returncode}")

        return output_file

    def compile_with_interpolation(self) -> Path:
        """
        Compile movie with neuroglancer state interpolation.

        Uses neuroglancer's built-in video_tool to generate smooth camera movements
        between keyframes. This properly interpolates positions (linearly),
        orientations (spherical linear interpolation), and zoom factors (exponentially).

        Returns:
            Path to output movie file
        """
        print(f"[COMPILER] Compiling with neuroglancer state interpolation", flush=True)

        try:
            import neuroglancer

            # Filter to only frames with narration
            narrated_frames = [f for f in self.session.frames if f.has_narration]

            if not narrated_frames:
                raise RuntimeError(
                    "No frames with narration found. Cannot create movie without audio."
                )

            print(
                f"[COMPILER] Using {len(narrated_frames)} frames with narration (out of {len(self.session.frames)} total)",
                flush=True,
            )

            # Create neuroglancer script file from saved states
            script_file = self.session_dir / "neuroglancer_script.txt"

            print(
                f"[COMPILER] Creating neuroglancer script from {len(narrated_frames)} keyframes",
                flush=True,
            )

            # Build script with alternating URLs and transition durations
            with open(script_file, "w") as f:
                for i, frame in enumerate(narrated_frames):
                    # Read the public URL we already generated
                    url_file = self.session_dir / frame.urls_file
                    with open(url_file, "r") as url_f:
                        lines = url_f.readlines()
                        # URL is on line 3 (after header and separator)
                        public_url = lines[2].strip()

                    f.write(f"{public_url}\n")

                    # Transition duration (2 second smooth transition to next state)
                    if i < len(narrated_frames) - 1:
                        transition_duration = 2.0
                        f.write(f"{transition_duration}\n")

            print(f"[COMPILER] Script created at {script_file}", flush=True)

            # Create output directory for rendered frames
            frames_dir = self.session_dir / "interpolated_frames"
            frames_dir.mkdir(exist_ok=True)

            # Use neuroglancer's video tool CLI to render frames
            print(
                f"[COMPILER] Rendering frames with neuroglancer video_tool at 30 fps",
                flush=True,
            )

            # Call neuroglancer video_tool as subprocess
            render_cmd = [
                "python",
                "-m",
                "neuroglancer.tool.video_tool",
                "render",
                "--fps",
                "30",
                "--width",
                "960",
                "--height",
                "540",
                "--scale-bar-scale",
                "1.0",
                str(script_file),
                str(frames_dir),
            ]

            result = subprocess.run(
                render_cmd, capture_output=True, text=True, timeout=600
            )

            if result.returncode != 0:
                print(
                    f"[COMPILER] Neuroglancer render failed: {result.stderr}",
                    flush=True,
                )
                raise RuntimeError(
                    f"Neuroglancer video_tool failed with code {result.returncode}"
                )

            print(f"[COMPILER] Frames rendered, creating video with FFmpeg", flush=True)

            # Get list of rendered PNG frames
            frame_files = sorted(frames_dir.glob("*.png"))

            if not frame_files:
                raise RuntimeError("No frames were rendered by neuroglancer video_tool")

            print(f"[COMPILER] Found {len(frame_files)} rendered frames", flush=True)

            # Neuroglancer rendered frames with 2s transitions between keyframes at 30fps
            # Expected: 60 frames per transition between keyframes
            # We want to slow down playback so each narration lasts for its audio duration

            # Calculate frames per segment (transition + hold on keyframe)
            # neuroglancer creates smooth transitions between keyframes
            frames_per_keyframe_pair = len(frame_files) // (len(narrated_frames) - 1) if len(narrated_frames) > 1 else len(frame_files)

            print(
                f"[COMPILER] Approximately {frames_per_keyframe_pair} frames per keyframe pair",
                flush=True,
            )

            # Create FFmpeg concat file
            # Strategy:
            # - For each narrated keyframe, hold on the first frame of that segment for audio duration
            # - Then play through the transition frames over 2 seconds
            concat_file = self.session_dir / "concat_interpolated.txt"
            fps = 30

            with open(concat_file, "w") as f:
                for i, narrated_frame in enumerate(narrated_frames):
                    # Calculate which rendered frame corresponds to this keyframe
                    if len(narrated_frames) > 1:
                        keyframe_render_idx = i * frames_per_keyframe_pair
                    else:
                        keyframe_render_idx = 0

                    keyframe_render_idx = min(keyframe_render_idx, len(frame_files) - 1)

                    # Hold on keyframe for narration duration
                    audio_duration = narrated_frame.display_duration
                    f.write(f"file '{frame_files[keyframe_render_idx].absolute()}'\n")
                    f.write(f"duration {audio_duration}\n")

                    # Add transition to next keyframe (except for last frame)
                    if i < len(narrated_frames) - 1:
                        # Use 60 frames for 2-second transition (30fps * 2s)
                        transition_start = keyframe_render_idx
                        transition_end = min(transition_start + 60, len(frame_files))
                        transition_frames = frame_files[transition_start:transition_end]

                        # Play transition frames at normal speed (2 seconds total)
                        frame_duration = 2.0 / len(transition_frames) if transition_frames else 0.033
                        for trans_frame in transition_frames:
                            f.write(f"file '{trans_frame.absolute()}'\n")
                            f.write(f"duration {frame_duration}\n")

                # Duplicate last frame
                if frame_files:
                    last_path = frame_files[-1].absolute()
                    f.write(f"file '{last_path}'\n")

            # Compile video with audio if available
            output_file = self.session_dir / "output" / "movie.mp4"
            has_audio = any(frame.has_narration for frame in self.session.frames)

            if has_audio:
                # Create audio timeline with 2s silence between narrations for transitions
                audio_file = self._create_audio_timeline(add_transition_silence=True)
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_file),
                    "-i",
                    str(audio_file),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "medium",
                    "-crf",
                    "23",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",
                    str(output_file),
                ]
            else:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_file),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "medium",
                    "-crf",
                    "23",
                    str(output_file),
                ]

            print(f"[COMPILER] Running FFmpeg to compile final video", flush=True)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                print(f"[COMPILER] FFmpeg stderr: {result.stderr}", flush=True)
                raise RuntimeError(f"FFmpeg failed with code {result.returncode}")

            return output_file

        except Exception as e:
            print(
                f"[COMPILER] Interpolation failed: {e}, falling back to cuts",
                flush=True,
            )
            import traceback

            traceback.print_exc()
            # Fall back to cuts if interpolation fails
            return self.compile_with_cuts()

    def _create_audio_timeline(self, add_transition_silence: bool = False) -> Path:
        """
        Create a single audio file from all narrations.

        Only processes frames that have narration. For crossfade and interpolation modes,
        adds 2 seconds of silence between narrations for transitions.

        Args:
            add_transition_silence: If True, add 2s silence between narrations (for transitions)

        Returns:
            Path to combined audio file
        """
        print(
            f"[COMPILER] Creating audio timeline (transitions: {add_transition_silence})",
            flush=True,
        )

        output_audio = self.session_dir / "output" / "combined_audio.mp3"
        concat_list = self.session_dir / "audio_concat_list.txt"
        temp_dir = self.session_dir / "temp_audio"
        temp_dir.mkdir(exist_ok=True)

        # Filter to only frames with narration
        narrated_frames = [f for f in self.session.frames if f.has_narration]

        # Convert all audio files to mp3 format and add transition silences
        normalized_files = []
        for i, frame in enumerate(narrated_frames):
            audio_path = self.session_dir / frame.narration_file
            if audio_path.exists():
                # Convert to mp3 if it's wav or other format
                normalized_file = temp_dir / f"audio_{i:06d}.mp3"
                convert_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    str(normalized_file),
                ]
                result = subprocess.run(convert_cmd, capture_output=True, timeout=30)
                if result.returncode == 0:
                    normalized_files.append(normalized_file)

                    # Add transition silence after each narration (except the last one)
                    if add_transition_silence and i < len(narrated_frames) - 1:
                        silence_file = temp_dir / f"transition_{i:06d}.mp3"
                        self._create_silence(silence_file, 2.0)  # 2 seconds of silence
                        normalized_files.append(silence_file)
                else:
                    print(
                        f"[COMPILER] Warning: Failed to convert {audio_path}",
                        flush=True,
                    )
            else:
                print(
                    f"[COMPILER] Warning: Audio file not found: {audio_path}",
                    flush=True,
                )

        # Write concat list with normalized files (using absolute paths)
        with open(concat_list, "w") as f:
            for audio_file in normalized_files:
                # Use absolute path to avoid path resolution issues
                abs_path = audio_file.absolute()
                f.write(f"file '{abs_path}'\n")

        # Concat all audio segments
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",  # Use copy since all files are already mp3
            str(output_audio),
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            print(f"[COMPILER] Audio timeline creation failed", flush=True)
            print(f"[COMPILER] FFmpeg stderr: {result.stderr.decode()}", flush=True)
            print(f"[COMPILER] FFmpeg stdout: {result.stdout.decode()}", flush=True)
            raise RuntimeError("Failed to create audio timeline")

        return output_audio

    def _create_silence(self, output_path: Path, duration: float):
        """
        Create a silent audio file of specified duration.

        Args:
            output_path: Path to output file
            duration: Duration in seconds
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=44100:cl=stereo",
            "-t",
            str(duration),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(output_path),
        ]

        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
