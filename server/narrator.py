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
        # Auto-detect provider based on available API keys or USE_LOCAL flag
        self.provider = provider or os.environ.get("AI_PROVIDER", "auto")
        self.use_local = os.environ.get("USE_LOCAL", "false").lower() == "true"

        # Check for API keys
        anthropic_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        google_key = os.environ.get("GOOGLE_API_KEY")

        if self.provider == "auto":
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

        # Initialize the appropriate client
        if self.provider == "local":
            try:
                import ollama
                # Test Ollama connection
                ollama.list()
                self.client = ollama
                self.enabled = True
                print("[NARRATOR] Using Local Mode (Ollama llama3.2-vision)")
                # Initialize Kokoro TTS if available
                try:
                    from kokoro import KPipeline
                    import sounddevice as sd
                    self.tts_pipeline = KPipeline(lang_code='a')  # American English
                    self.tts_available = True
                    print("[NARRATOR] Kokoro TTS enabled")
                except ImportError:
                    print("[NARRATOR] WARNING: Kokoro TTS not available (pip install kokoro soundfile sounddevice)")
                    self.tts_available = False
            except ImportError:
                print("[NARRATOR] ERROR: ollama not installed. Run: pip install ollama")
                self.enabled = False
                return
            except Exception as e:
                print(f"[NARRATOR] ERROR: Failed to connect to Ollama. Is it running? {e}")
                print("[NARRATOR] Install from ollama.com and run: ollama pull llama3.2-vision")
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
                self.client = genai.GenerativeModel("gemini-1.5-flash-8b")
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
        self.narration_history: List[Dict[str, Any]] = []
        self.max_history = 10
        self.last_state: Optional[Dict[str, Any]] = None
        self.generating_narration = False  # Track if generation is in progress

        # Thresholds for triggering narration
        self.min_narration_interval = 3.0  # seconds between narrations
        self.position_threshold = (
            1000  # voxels - significant movement (lowered for easier triggering)
        )
        self.zoom_threshold = 0.2  # 20% zoom change (lowered for easier triggering)
        self.idle_threshold = 10.0  # seconds - narrate if idle after movement

    def should_narrate(self, current_state: Dict[str, Any]) -> bool:
        """Determine if we should generate narration for this state."""
        if not self.enabled:
            return False

        # Don't start new narration if one is already being generated
        if self.generating_narration:
            return False

        current_time = time.time()

        # Don't narrate too frequently
        if current_time - self.last_narration_time < self.min_narration_interval:
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
        self, state: Dict[str, Any], screenshot_b64: Optional[str] = None
    ) -> Optional[str]:
        """Generate AI narration based on current viewer state and optional screenshot."""
        if not self.enabled:
            return None

        # Set flag to prevent concurrent generation
        self.generating_narration = True
        
        try:
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
                # Optionally speak using local TTS
                if narration and self.tts_available:
                    import threading
                    # Speak in background to not block
                    threading.Thread(target=self.speak_local, args=(narration,), daemon=True).start()
            else:
                return None

            # Update state
            self.last_narration_time = time.time()
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

        base_context = f"""You are an AI narrator for a scientific tour of electron microscopy (EM) data. You will receive a stream of Neuroglancer viewer snapshots showing navigation through 3D EM datasets of cells and tissues.

**Your Role:**
- Provide real-time narration as the viewer explores the dataset in Neuroglancer
- Each image is a snapshot from the Neuroglancer 3D viewer showing the current view
- Images may show raw EM data (grayscale) and/or colored segmentations of cellular structures
- Narrate what you observe in each frame to create an engaging scientific tour
- Be concise, accurate, and scientifically informative

**About Neuroglancer:**
- Interactive web-based viewer for volumetric data
- Displays EM imagery and segmentation layers that can be toggled on/off
- Supports 3D navigation: panning, zooming, rotating through the dataset
- Each snapshot shows the current cross-section and visible layers

**Dataset Information:**
- Type: High-resolution 3D EM imaging of cells and tissues
- Raw data: Grayscale EM showing cell membranes, organelles, and subcellular structures
- Segmentations: Colored overlays identifying specific organelles, cells, or structures
- Coordinate system: Position in nanometers (nm)
- Resolution: ~4-8 nm/pixel in XY, varies in Z

**Current Viewer State:**
{context}

**Visible Layers:** {layers_info}

**Recent narrations (avoid repeating):**
{recent_history}
"""

        if screenshot_b64:
            prompt = """\n**Context:** You are viewing electron microscopy (EM) data of cells and tissue cultures displayed as 2D orthogonal cross-sections in Neuroglancer. The image shows 4 panels representing different viewing planes through 3D volumetric data.

**What you're looking at:**
- EM data: Grayscale imagery showing cellular ultrastructure (membranes, organelles, subcellular details)
- Colored regions: These are segmented organelles overlaid on the grayscale EM data
- Each color represents a different segmented organelle or cellular structure

**CRITICAL: Describe ONLY what you literally see in the image - do not invent details.**

**What the data looks like when loaded:**
- Textured grayscale EM imagery with visible cellular structures
- Colored overlays (if present) marking segmented organelles
- Visible detail, patterns, and variation in brightness

**What indicates data is NOT loaded:**
- Flat uniform gray/black panels with no texture
- Only coordinate axes and scale bar visible
- No cellular structures or patterns

**Task:**
Generate ONE concise sentence (max 20 words) that:
1. Confirms if EM data is visible or still loading
2. If visible, describes the cellular structures you observe in the grayscale EM
3. Mentions colored segmented organelles if present

Examples:
- "Data is still loading."
- "Grayscale EM cross-sections show cellular membranes with colored organelle segmentations."
- "EM data displays textured cellular ultrastructure with various colored segmented regions."

Narration:"""
        else:
            prompt = (
                base_context
                + """\n**Task:**
Based on the viewer position and zoom level, generate a single, concise sentence (max 20 words) that:
1. Describes what the user might be looking at based on zoom level and any visible layers
2. Provides brief scientific context when relevant
3. Is engaging like a museum tour guide
4. Avoids hallucinating specific structures without visual confirmation

Focus on zoom-appropriate features:
- Scale <5: Subcellular details (organelles, synapses)
- Scale 5-50: Cellular level (individual cells, nuclei)  
- Scale >50: Tissue organization

Narration:"""
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
                image.save(debug_path, 'JPEG')
                print(f"[NARRATOR DEBUG] Saved image to {debug_path} - Size: {image.size}, Mode: {image.mode}", flush=True)
                
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
                image.save(debug_path, 'JPEG')
                print(f"[NARRATOR DEBUG] Ollama image - Size: {image.size}, Mode: {image.mode}, Saved to: {debug_path}", flush=True)
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    image.save(tmp.name, 'JPEG')
                    tmp_path = tmp.name
                
                try:
                    # Call Ollama with vision
                    response = self.client.chat(
                        model='gemma3:12b',
                        messages=[{
                            'role': 'user',
                            'content': prompt,
                            'images': [tmp_path]
                        }]
                    )
                    return response['message']['content'].strip()
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
                    model='gemma3:12b',
                    messages=[{'role': 'user', 'content': prompt}]
                )
                return response['message']['content'].strip()
        except Exception as e:
            print(f"[NARRATOR] Local (Ollama) error: {e}", flush=True)
            return None

    def speak_local(self, text: str):
        """Speak text using local Kokoro TTS."""
        if not self.tts_available:
            return
        
        try:
            import sounddevice as sd
            generator = self.tts_pipeline(text, voice='af_bella', speed=1.1)
            for _, _, audio in generator:
                sd.play(audio, 24000)
                sd.wait()
        except Exception as e:
            print(f"[NARRATOR] TTS error: {e}", flush=True)
    
    def _speak_kokoro(self, text: str):
        """Speak text using pyttsx3 TTS."""
        if not self.tts_available:
            return
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"[NARRATOR] TTS error: {e}", flush=True)

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
