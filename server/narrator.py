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
        # Auto-detect provider based on available API keys
        self.provider = provider or os.environ.get("AI_PROVIDER", "auto")

        # Check for API keys
        anthropic_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        google_key = os.environ.get("GOOGLE_API_KEY")

        if self.provider == "auto":
            if google_key:
                self.provider = "gemini"
            elif anthropic_key:
                self.provider = "claude"
            else:
                print(
                    "[NARRATOR] WARNING: No API key found (GOOGLE_API_KEY or ANTHROPIC_API_KEY). Narration disabled."
                )
                self.enabled = False
                return

        # Initialize the appropriate client
        if self.provider == "gemini":
            if not google_key:
                print("[NARRATOR] ERROR: GOOGLE_API_KEY not found")
                self.enabled = False
                return
            try:
                import google.generativeai as genai

                genai.configure(api_key=google_key)
                self.client = genai.GenerativeModel("gemini-2.5-flash")
                self.enabled = True
                print("[NARRATOR] Using Gemini 2.5 Flash")
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

        base_context = f"""You are viewing electron microscopy (EM) data and segmentations in a Neuroglancer viewer.

**Dataset Information:**
- Type: High-resolution 3D EM imaging of neural tissue
- Data: Grayscale EM showing cell membranes, organelles, and subcellular structures
- Segmentations: Colored overlays identifying specific cells, neurons, or structures
- Coordinate system: Position in nanometers (nm)
- Resolution: ~4-8 nm/pixel in XY, varies in Z

**Current Viewer State:**
{context}

**Recent narrations (avoid repeating):**
{recent_history}
"""

        if screenshot_b64:
            prompt = (
                base_context
                + """\n**Task:**
Look at the image showing the current view. Generate a single, concise sentence (max 25 words) that:
1. Describes what you actually SEE in the image (cellular structures, membranes, organelles, etc.)
2. Relates the zoom level to the scale of structures visible
3. Is engaging and informative, like a scientific tour guide
4. Only mentions specific structures you can clearly identify in the image
5. Avoids speculation - be cautious and scientific

Focus on observable features:
- At high zoom (scale <5): Synapses, mitochondria, vesicles, membrane details
- At medium zoom (5-50): Individual cells, nuclei, cellular boundaries
- At low zoom (>50): Tissue organization, cell populations

Narration:"""
            )
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
