"""
AI Narrator for Neuroglancer exploration.
Stage 4: Text narration based on viewer state.
"""

import time
from typing import Optional, Dict, Any, List
import os
import base64


class Narrator:
    """Generates contextual narration based on viewer state changes."""

    def __init__(self, api_key: Optional[str] = None, provider: Optional[str] = None):
        # Check for debug mode
        self.debug_mode = os.environ.get("DEBUG_MODE", "false").lower() == "true"

        # Auto-detect provider based on available API keys or USE_LOCAL flag
        self.provider = provider or os.environ.get("AI_PROVIDER", "auto")
        self.use_local = os.environ.get("USE_LOCAL", "false").lower() == "true"

        # In debug mode, skip API initialization
        if self.debug_mode:
            self.enabled = True
            self.provider = "debug"
            print("[NARRATOR] Debug mode enabled - using example narrations")
            # Initialize TTS engines (continue to TTS init below)
        else:
            # Check for API keys
            anthropic_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            google_key = os.environ.get("GOOGLE_API_KEY")

        if not self.debug_mode and self.provider == "auto":
            if self.use_local:
                self.provider = "local"
            elif google_key:
                self.provider = "gemini"
            elif anthropic_key:
                self.provider = "claude"
            else:
                print(
                    "[NARRATOR] WARNING: No API key found (GOOGLE_API_KEY or ANTHROPIC_API_KEY). Set USE_LOCAL=true for local mode."
                )
                self.enabled = False
                return

        # Initialize the appropriate client (skip in debug mode)
        if self.debug_mode:
            # Skip client initialization in debug mode
            pass
        elif self.provider == "local":
            try:
                import ollama

                # Test Ollama connection
                ollama.list()
                self.client = ollama
                self.enabled = True
                print("[NARRATOR] Using Local Mode (Ollama llama3.2-vision)")
            except ImportError:
                print("[NARRATOR] ERROR: ollama not installed. Run: pip install ollama")
                self.enabled = False
                return
            except Exception as e:
                print(
                    f"[NARRATOR] ERROR: Failed to connect to Ollama. Is it running? {e}"
                )
                print(
                    "[NARRATOR] Install from ollama.com and run: ollama pull llama3.2-vision"
                )
                self.enabled = False
                return
        elif self.provider == "gemini":
            if not google_key:
                print("[NARRATOR] ERROR: GOOGLE_API_KEY not found")
                self.enabled = False
                return
            try:
                import google.generativeai as genai

                genai.configure(api_key=google_key)
                self.client = genai.GenerativeModel("gemini-3-flash-preview")
                self.enabled = True
                print("[NARRATOR] Using Gemini 1.5 Flash 8B")
            except ImportError:
                print(
                    "[NARRATOR] ERROR: google-generativeai not installed. Run: pip install google-generativeai"
                )
                self.enabled = False
                return
        elif self.provider == "claude":
            if not anthropic_key:
                print("[NARRATOR] ERROR: ANTHROPIC_API_KEY not found")
                self.enabled = False
                return
            try:
                from anthropic import Anthropic

                self.client = Anthropic(api_key=anthropic_key)
                self.enabled = True
                print("[NARRATOR] Using Claude 3.5 Sonnet")
            except ImportError:
                print(
                    "[NARRATOR] ERROR: anthropic not installed. Run: pip install anthropic"
                )
                self.enabled = False
                return
        else:
            print(f"[NARRATOR] ERROR: Unknown provider '{self.provider}'")
            self.enabled = False
            return

        # Narration state
        self.last_narration_time = 0
        self.last_narrated_screenshot_ts = 0  # Track which screenshot we last narrated
        self.narration_history: List[Dict[str, Any]] = []
        self.max_history = 10
        self.last_state: Optional[Dict[str, Any]] = None
        self.generating_narration = False  # Track if generation is in progress

        # Initialize TTS engines
        self.tts_available = False
        self.tts_engine = None
        self.use_coqui = os.getenv("USE_COQUI", "false").lower() == "true"
        self.use_chatterbox = os.getenv("USE_CHATTERBOX", "false").lower() == "true"
        self.voice_reference = os.getenv("VOICE_REFERENCE_PATH", None)
        self.voice_embedding = None  # Store cloned voice embedding

        # Priority: Coqui > Chatterbox > edge-tts
        if self.use_coqui:
            try:
                from TTS.api import TTS
                import torch

                self.tts_engine = "coqui"

                # Initialize Coqui XTTS v2
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(
                    f"[NARRATOR] Initializing Coqui XTTS v2 on {device}...", flush=True
                )
                self.coqui = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

                # If a voice reference is provided, ensure it's a WAV file
                if self.voice_reference:
                    # Convert to WAV if not already WAV (Coqui expects WAV/PCM readable by soundfile/torchaudio)
                    if not self.voice_reference.lower().endswith(".wav"):
                        try:
                            import subprocess
                            import tempfile

                            wav_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                            wav_path = wav_temp.name
                            wav_temp.close()

                            # Use ffmpeg to convert to 16-bit PCM WAV
                            subprocess.run(
                                [
                                    "ffmpeg",
                                    "-y",
                                    "-i",
                                    self.voice_reference,
                                    "-acodec",
                                    "pcm_s16le",
                                    wav_path,
                                ],
                                check=True,
                                capture_output=True,
                            )

                            # Replace voice_reference with converted WAV path
                            self.voice_reference = wav_path
                            # Remember converted path for potential cleanup
                            self._voice_reference_converted = wav_path
                            print(f"[NARRATOR] Converted voice reference to WAV: {wav_path}", flush=True)
                        except Exception as conv_err:
                            print(
                                f"[NARRATOR] Warning: Could not convert audio format: {conv_err}",
                                flush=True,
                            )
                            print(f"[NARRATOR] Will try original file...", flush=True)

                    print(
                        f"[NARRATOR] Voice cloning enabled with: {self.voice_reference}",
                        flush=True,
                    )
                    print(
                        f"[NARRATOR] TTS engine initialized (Coqui XTTS with voice cloning)",
                        flush=True,
                    )
                else:
                    print(
                        "[NARRATOR] TTS engine initialized (Coqui XTTS - default voice)",
                        flush=True,
                    )

                self.tts_available = True
            except Exception as e:
                print(
                    f"[NARRATOR] Coqui TTS not available: {e}, trying Chatterbox...",
                    flush=True,
                )
                self.use_coqui = False

        if not self.use_coqui and self.use_chatterbox:
            try:
                from chatterbox import ChatterboxTTS
                import torch

                self.tts_engine = "chatterbox"

                # Initialize ChatterboxTTS with proper arguments
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(
                    f"[NARRATOR] Initializing Chatterbox TTS on {device}...", flush=True
                )
                self.chatterbox = ChatterboxTTS.from_pretrained(device=device)
                print(f"[NARRATOR] Chatterbox model loaded onto {device}", flush=True)

                # Prepare voice conditionals (embedding) ONCE at startup if reference provided
                if self.voice_reference:
                    print(
                        f"[NARRATOR] Preparing voice embedding from {self.voice_reference}...",
                        flush=True,
                    )

                    # Convert m4a/other formats to WAV if needed
                    voice_file = self.voice_reference
                    if not voice_file.lower().endswith(".wav"):
                        try:
                            import subprocess
                            import tempfile

                            wav_temp = tempfile.NamedTemporaryFile(
                                suffix=".wav", delete=False
                            )
                            wav_path = wav_temp.name
                            wav_temp.close()

                            # Use ffmpeg to convert to WAV (preserve original sample rate)
                            # Chatterbox will resample internally to 16kHz and 24kHz
                            subprocess.run(
                                [
                                    "ffmpeg",
                                    "-i",
                                    voice_file,
                                    "-acodec",
                                    "pcm_s16le",  # 16-bit PCM
                                    wav_path,
                                    "-y",
                                ],
                                check=True,
                                capture_output=True,
                            )
                            voice_file = wav_path
                            print(f"[NARRATOR] Converted to WAV: {wav_path}", flush=True)
                        except Exception as conv_err:
                            print(
                                f"[NARRATOR] Warning: Could not convert audio format: {conv_err}",
                                flush=True,
                            )
                            print(f"[NARRATOR] Will try original file...", flush=True)

                    # Prepare voice embedding - this caches in self.chatterbox.conds
                    self.chatterbox.prepare_conditionals(voice_file, exaggeration=0.7)
                    print(
                        f"[NARRATOR] TTS engine initialized (Chatterbox with cloned voice)",
                        flush=True,
                    )
                else:
                    print(
                        "[NARRATOR] TTS engine initialized (Chatterbox - default voice)",
                        flush=True,
                    )

                self.tts_available = True
            except Exception as e:
                print(
                    f"[NARRATOR] Chatterbox TTS not available: {e}, falling back to edge-tts",
                    flush=True,
                )
                self.use_chatterbox = False

        if not self.use_coqui and not self.use_chatterbox:
            try:
                import edge_tts

                self.tts_engine = "edge-tts"
                self.edge_voice = os.getenv("EDGE_VOICE", "en-GB-RyanNeural")
                self.tts_available = True
                print(
                    f"[NARRATOR] TTS engine initialized (edge-tts with {self.edge_voice})",
                    flush=True,
                )
            except Exception as e:
                print(f"[NARRATOR] TTS engine not available: {e}", flush=True)

        # Thresholds for triggering narration
        self.min_narration_interval = 3.0  # seconds between narrations
        self.position_threshold = (
            1000  # voxels - significant movement (lowered for easier triggering)
        )
        self.zoom_threshold = 0.2  # 20% zoom change (lowered for easier triggering)
        self.idle_threshold = 10.0  # seconds - narrate if idle after movement

    def should_narrate(
        self, current_state: Dict[str, Any], screenshot_ts: float = 0
    ) -> bool:
        """Determine if we should generate narration for this state.

        Args:
            current_state: Current viewer state
            screenshot_ts: Timestamp of the current screenshot (0 if no screenshot)
        """
        if not self.enabled:
            return False

        # Don't start new narration if one is already being generated
        if self.generating_narration:
            return False

        current_time = time.time()

        # Don't narrate too frequently
        if current_time - self.last_narration_time < self.min_narration_interval:
            return False

        # Only narrate if we have a NEW screenshot (not already narrated)
        if screenshot_ts > 0 and screenshot_ts <= self.last_narrated_screenshot_ts:
            return False

        # First state always gets narration
        if self.last_state is None:
            return True

        # Check for selection changes
        last_selection = self.last_state.get("selected_segments", [])
        curr_selection = current_state.get("selected_segments", [])
        if last_selection != curr_selection:
            return True

        # Check for layer visibility changes
        last_layers = {
            l["name"]: l["visible"] for l in self.last_state.get("layers", [])
        }
        curr_layers = {l["name"]: l["visible"] for l in current_state.get("layers", [])}
        if last_layers != curr_layers:
            return True

        # Check for significant position change
        last_pos = self.last_state.get("position")
        curr_pos = current_state.get("position")
        if last_pos and curr_pos and len(last_pos) >= 3 and len(curr_pos) >= 3:
            distance = (
                sum((a - b) ** 2 for a, b in zip(last_pos[:3], curr_pos[:3])) ** 0.5
            )
            if distance > self.position_threshold:
                return True

        # Check for significant zoom change
        last_scale = self.last_state.get("scale", 1)
        curr_scale = current_state.get("scale", 1)
        if last_scale > 0 and curr_scale > 0:
            zoom_ratio = abs(curr_scale - last_scale) / last_scale
            if zoom_ratio > self.zoom_threshold:
                return True

        return False

    def generate_narration(
        self,
        state: Dict[str, Any],
        screenshot_b64: Optional[str] = None,
        screenshot_ts: float = 0,
    ) -> Optional[str]:
        """Generate AI narration based on current viewer state and optional screenshot.

        Args:
            state: Current viewer state
            screenshot_b64: Base64-encoded screenshot (optional)
            screenshot_ts: Timestamp of the screenshot
        """
        if not self.enabled:
            return None

        # Set flag to prevent concurrent generation
        self.generating_narration = True

        try:
            # In debug mode, return example narration
            if self.debug_mode:
                narration = self._get_debug_narration(state)
            else:
                # Build context from state
                context = self._build_context(state)

                # Build prompt for AI
                prompt = self._build_prompt(context, state, screenshot_b64)

                # Call the appropriate API
                if self.provider == "gemini":
                    narration = self._call_gemini(prompt, screenshot_b64)
                elif self.provider == "claude":
                    narration = self._call_claude(prompt, screenshot_b64)
                elif self.provider == "local":
                    narration = self._call_local(prompt, screenshot_b64)
                else:
                    return None

            # Update state
            self.last_narration_time = time.time()
            self.last_narrated_screenshot_ts = screenshot_ts
            self.last_state = state
            self._add_to_history(narration, state)

            print(f"[NARRATOR] Generated: {narration}", flush=True)
            return narration

        except Exception as e:
            print(f"[NARRATOR] Error generating narration: {e}", flush=True)
            return None
        finally:
            # Always clear the flag when done
            self.generating_narration = False

    def _get_debug_narration(self, state: Dict[str, Any]) -> str:
        """Return example narration for debug mode."""
        # Cycle through a few different example narrations
        examples = [
            "We're exploring a mitochondrial network with intricate cristae folds, surrounded by the dense granular texture of the cytoplasm. The organelles appear as elongated ellipsoids, their internal membranes creating parallel ridges that maximize surface area for energy production.",
            "This region shows the endoplasmic reticulum forming an interconnected tubular network throughout the cell. The rough ER appears studded with ribosomes, while smooth ER regions branch off in sinuous curves, likely involved in lipid synthesis.",
            "A large spherical nucleus dominates the center of this view, its double membrane clearly delineated. Dark patches of heterochromatin cluster along the nuclear envelope, while lighter euchromatin regions suggest areas of active transcription.",
            "Multiple vesicles of varying sizes populate this area, appearing as small circular or ellipsoid structures. Some appear to be budding from larger membrane compartments, part of the cell's active transport machinery.",
        ]

        # Use the number of narrations to cycle through examples
        index = len(self.narration_history) % len(examples)
        return examples[index]

    def _build_context(self, state: Dict[str, Any]) -> str:
        """Build a human-readable context description from state."""
        parts = []

        # Position
        if "position" in state:
            pos = state["position"]
            parts.append(f"Position: [{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}] nm")

        # Zoom level
        if "scale" in state:
            scale = state["scale"]
            parts.append(f"Zoom level: {scale:.2f}")

        # Visible layers
        visible_layers = [
            l["name"] for l in state.get("layers", []) if l.get("visible", True)
        ]
        if visible_layers:
            parts.append(f"Visible layers: {', '.join(visible_layers)}")

        # Selected segments
        if "selected_segments" in state:
            segs = state["selected_segments"]
            parts.append(f"Selected segments: {segs}")

        return " | ".join(parts)

    def _build_prompt(
        self, context: str, state: Dict[str, Any], screenshot_b64: Optional[str] = None
    ) -> str:
        """Build the prompt for the AI narrator."""
        # Get recent narration history for context
        recent_history = (
            "\n".join(
                [f"- {item['narration']}" for item in self.narration_history[-3:]]
            )
            if self.narration_history
            else "No previous narrations."
        )

        # Get visible layers information
        visible_layers = [
            l["name"] for l in state.get("layers", []) if l.get("visible", True)
        ]
        layers_info = ", ".join(visible_layers) if visible_layers else "None"

        base_context = f"""Role: You are an expert cellular morphologist narrating a live EM data stream.

Task: Provide a 2-3 sentence description of the current view, focusing on the shapes, textures, and spatial organization of the biological structures.

Guidelines:

- Flexible Identification: If color segmentations are visible, use them for identification. If not, identify structures by their morphology (e.g., "ellipsoid bodies with internal folds" for mitochondria, or "large, spherical void" for the nucleus).

- Describe Shapes: Use descriptive geometry—mention if structures appear tubular, branched, stacked, or granular.

- Biological Context: Relate the shapes to their function.

- Pacing: Aim for 50–80 words (roughly 20-30 seconds of speech).

**Current Viewer State:**
{context}

**Visible Layers:** {layers_info}

**Recent narrations (avoid repeating):**
{recent_history}
"""

        if screenshot_b64:
            prompt = (
                base_context + "\n\nDescribe what you see in the image.\n\nNarration:"
            )
        else:
            prompt = (
                base_context
                + "\n\nBased on the current position and zoom level, narrate what the viewer might be seeing.\n\nNarration:"
            )

        return prompt

    def _call_gemini(
        self, prompt: str, screenshot_b64: Optional[str] = None
    ) -> Optional[str]:
        """Call Gemini API."""
        try:
            if screenshot_b64:
                # Gemini with vision
                import PIL.Image
                import io

                image_data = base64.b64decode(screenshot_b64)
                image = PIL.Image.open(io.BytesIO(image_data))

                # Debug: Save image to verify what's being sent
                debug_path = "/tmp/narrator_debug_image.jpg"
                image.save(debug_path, "JPEG")
                print(
                    f"[NARRATOR DEBUG] Saved image to {debug_path} - Size: {image.size}, Mode: {image.mode}",
                    flush=True,
                )

                response = self.client.generate_content([prompt, image])
            else:
                # Text only
                response = self.client.generate_content(prompt)

            return response.text.strip()
        except Exception as e:
            print(f"[NARRATOR] Gemini error: {e}", flush=True)
            return None

    def _call_claude(
        self, prompt: str, screenshot_b64: Optional[str] = None
    ) -> Optional[str]:
        """Call Claude API."""
        try:
            if screenshot_b64:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=150,
                    temperature=0.7,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": screenshot_b64,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )
            else:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=150,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}],
                )

            return response.content[0].text.strip()
        except Exception as e:
            print(f"[NARRATOR] Claude error: {e}", flush=True)
            return None

    def _call_local(
        self, prompt: str, screenshot_b64: Optional[str] = None
    ) -> Optional[str]:
        """Call local Ollama with gemma3:12b."""
        try:
            if screenshot_b64:
                # Save image temporarily for Ollama
                import tempfile
                import io
                import PIL.Image

                image_data = base64.b64decode(screenshot_b64)
                image = PIL.Image.open(io.BytesIO(image_data))

                # Debug: Save and log image details
                debug_path = "/tmp/narrator_debug_ollama.jpg"
                image.save(debug_path, "JPEG")
                print(
                    f"[NARRATOR DEBUG] Ollama image - Size: {image.size}, Mode: {image.mode}, Saved to: {debug_path}",
                    flush=True,
                )

                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    image.save(tmp.name, "JPEG")
                    tmp_path = tmp.name

                try:
                    # Call Ollama with vision
                    response = self.client.chat(
                        model="gemma3:12b",
                        messages=[
                            {"role": "user", "content": prompt, "images": [tmp_path]}
                        ],
                    )
                    return response["message"]["content"].strip()
                finally:
                    # Clean up temp file
                    import os

                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            else:
                # Text only
                response = self.client.chat(
                    model="gemma3:12b", messages=[{"role": "user", "content": prompt}]
                )
                return response["message"]["content"].strip()
        except Exception as e:
            print(f"[NARRATOR] Local (Ollama) error: {e}", flush=True)
            return None

    async def generate_audio_async(self, text: str) -> Optional[str]:
        """Generate audio from text and return as base64-encoded audio."""
        if not self.tts_available:
            print(
                f"[NARRATOR] TTS not available, skipping audio generation", flush=True
            )
            return None

        try:
            import tempfile
            import base64
            import os

            if self.tts_engine == "coqui":
                print(
                    f"[NARRATOR] Generating audio with Coqui XTTS for: {text[:50]}...",
                    flush=True,
                )
                
                # Generate audio with Coqui XTTS
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name

                # Generate with voice cloning if reference audio provided
                if self.voice_reference:
                    self.coqui.tts_to_file(
                        text=text,
                        file_path=tmp_path,
                        speaker_wav=self.voice_reference,
                        language="en"
                    )
                else:
                    self.coqui.tts_to_file(
                        text=text,
                        file_path=tmp_path,
                        language="en"
                    )
                
                print(f"[NARRATOR] Coqui wrote WAV to: {tmp_path}", flush=True)

                # Read and encode
                with open(tmp_path, "rb") as f:
                    audio_data = f.read()

                print(
                    f"[NARRATOR] Audio file size: {len(audio_data)} bytes", flush=True
                )
                encoded_audio = base64.b64encode(audio_data).decode("utf-8")

                # Clean up
                try:
                    os.unlink(tmp_path)
                except:
                    pass

                return encoded_audio

            elif self.tts_engine == "chatterbox":
                print(
                    f"[NARRATOR] Generating audio with Chatterbox for: {text[:50]}...",
                    flush=True,
                )
                # Use Chatterbox for TTS - voice embedding already cached from startup
                # generate() returns torch tensor at 24000 Hz (S3GEN_SR)
                audio_tensor = self.chatterbox.generate(text=text, exaggeration=0.7, cfg_weight=0.3)
                print(
                    f"[NARRATOR] Chatterbox generated audio tensor shape: {audio_tensor.shape}",
                    flush=True,
                )
                print(f"[NARRATOR] Audio tensor type: {type(audio_tensor)}", flush=True)

                # Convert torch tensor to numpy array
                import numpy as np

                audio_np = audio_tensor.squeeze().cpu().numpy()
                print(
                    f"[NARRATOR] Numpy audio shape: {audio_np.shape}, dtype: {audio_np.dtype}",
                    flush=True,
                )

                # Convert to int16 for WAV file
                audio_int16 = (audio_np * 32767).astype(np.int16)
                print(
                    f"[NARRATOR] Int16 audio shape: {audio_int16.shape}, dtype: {audio_int16.dtype}",
                    flush=True,
                )

                # Save to temp WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name

                import scipy.io.wavfile

                scipy.io.wavfile.write(tmp_path, 24000, audio_int16)
                print(f"[NARRATOR] Wrote WAV to: {tmp_path}", flush=True)

                # Read and encode
                with open(tmp_path, "rb") as f:
                    audio_data = f.read()

                print(
                    f"[NARRATOR] Audio file size: {len(audio_data)} bytes", flush=True
                )

                try:
                    os.unlink(tmp_path)
                except:
                    pass

                encoded = base64.b64encode(audio_data).decode("utf-8")
                print(
                    f"[NARRATOR] Base64 encoded audio: {len(encoded)} chars", flush=True
                )
                return encoded

            else:
                # Use edge-tts
                import edge_tts

                # Create temporary file for MP3
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp_path = tmp.name

                # Generate speech using edge-tts
                communicate = edge_tts.Communicate(text, self.edge_voice)
                await communicate.save(tmp_path)

                # Read and encode the file
                with open(tmp_path, "rb") as f:
                    audio_data = f.read()

                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

                # Return base64-encoded audio
                return base64.b64encode(audio_data).decode("utf-8")

        except Exception as e:
            print(f"[NARRATOR] Audio generation error: {e}", flush=True)
            import traceback

            traceback.print_exc()
            return None

    def _add_to_history(self, narration: str, state: Dict[str, Any]):
        """Add narration to history, maintaining max size."""
        self.narration_history.append(
            {"narration": narration, "state": state, "timestamp": time.time()}
        )

        # Keep only recent history
        if len(self.narration_history) > self.max_history:
            self.narration_history.pop(0)

    def get_recent_narrations(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get the N most recent narrations."""
        return self.narration_history[-count:]
