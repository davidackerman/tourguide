"""
Neuroglancer viewer with state capture and screenshot streaming.
Stages 1-4: State callback + screenshot loop + AI narration
"""

import neuroglancer
import threading
import time
import io
import base64
from PIL import Image
from typing import Optional, Dict, Any, Callable
import json
import zarr
from narrator import Narrator


class NG_StateTracker:
    """Manages Neuroglancer viewer, state tracking, and screenshot capture."""

    def __init__(
        self,
        bind_address="127.0.0.1",
        port=9999,
        narrator_callback: Optional[Callable[[str], None]] = None,
        use_hela=False,
    ):
        self.bind_address = bind_address
        self.port = port
        self.use_hela = use_hela

        # Viewer setup
        neuroglancer.set_server_bind_address(bind_address, bind_port=port)
        self.viewer = neuroglancer.Viewer()

        # Add sample EM data (C. elegans by default, HeLa if flag set)
        if use_hela:
            self._add_hela_data()
        else:
            self._add_celegans_data()

        # State tracking
        self.last_state_summary = None
        self.current_state_summary = None
        self.state_dirty = False

        # Screenshot tracking
        self.screenshot_dirty = False
        self.latest_frame = None
        self.latest_frame_ts = None
        self.screenshot_thread = None
        self.running = False

        # Movement thresholds to avoid spam
        self.position_threshold = 100  # voxels
        self.scale_threshold = 0.15  # 15% change

        # AI Narrator (Stage 4)
        self.narrator = Narrator()
        self.narrator_callback = narrator_callback  # Called when narration is generated

        # Flag to temporarily disable narration (e.g., during query-based navigation)
        self.narration_enabled = True

        # Register state change callback (for config changes only)
        self.viewer.config_state.add_changed_callback(self._on_state_change)

    def _add_celegans_data(self):
        """Add C. elegans embryo EM data with organelle segmentations (default)."""
        # Load zarr to get dataset dimensions and calculate center
        zarr_path = "/nrs/cellmap/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1/em/fibsem-uint8"
        try:
            # zarr 3.x API: open directly with path
            z = zarr.open(zarr_path, mode="r")
            # Calculate center position from dataset dimensions
            # center_position = [dim // 2 for dim in z.shape]
            print(f"[NG] C. elegans dataset shape: {z.shape}", flush=True)
            print(
                f"[NG] Setting initial position to center: {center_position}",
                flush=True,
            )
        except Exception as e:
            print(f"[NG] Could not read zarr metadata: {e}", flush=True)
            print(f"[NG] Using default center position", flush=True)
            # center_position = [5000, 5000, 5000]

        with self.viewer.txn() as s:
            # C. elegans comma stage embryo EM data (via local cellmap-vm1 server)
            # Note: /nrs/cellmap/data/... maps to https://cellmap-vm1.int.janelia.org/nrs/data/...
            base_url = "zarr://https://cellmap-vm1.int.janelia.org/nrs/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1"

            s.layers["fibsem-uint8"] = neuroglancer.ImageLayer(
                source=f"{base_url}/em/fibsem-uint8/"
            )

            # Organelle segmentations
            seg_base = f"{base_url}/labels/inference/segmentations"

            s.layers["cell"] = neuroglancer.SegmentationLayer(
                source=f"{seg_base}/cell/"
            )
            s.layers["ld"] = neuroglancer.SegmentationLayer(source=f"{seg_base}/ld/")
            s.layers["lyso"] = neuroglancer.SegmentationLayer(
                source=f"{seg_base}/lyso/"
            )
            s.layers["mito"] = neuroglancer.SegmentationLayer(
                source=f"{seg_base}/mito/"
            )
            s.layers["nuc"] = neuroglancer.SegmentationLayer(source=f"{seg_base}/nuc/")
            s.layers["perox"] = neuroglancer.SegmentationLayer(
                source=f"{seg_base}/perox/"
            )
            s.layers["yolk"] = neuroglancer.SegmentationLayer(
                source=f"{seg_base}/yolk/"
            )

            # Set initial view position to center of dataset
            # s.position = center_position

            # Set 3D view orientation
            s.projection_orientation = [0, 0, 0, 1]

    def _add_hela_data(self):
        """Add HeLa cell EM data with organelle segmentations."""
        with self.viewer.txn() as s:
            # HeLa-2 cell EM data
            s.layers["fibsem-uint8"] = neuroglancer.ImageLayer(
                source="zarr://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8/"
            )

            # Organelle segmentations
            s.layers["endo_seg"] = neuroglancer.SegmentationLayer(
                source="n5://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/labels/endo_seg"
            )
            s.layers["er_seg"] = neuroglancer.SegmentationLayer(
                source="n5://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/labels/er_seg"
            )
            s.layers["golgi_seg"] = neuroglancer.SegmentationLayer(
                source="n5://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/labels/golgi_seg"
            )
            s.layers["mito_seg"] = neuroglancer.SegmentationLayer(
                source="n5://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/labels/mito_seg"
            )
            s.layers["nucleus_seg"] = neuroglancer.SegmentationLayer(
                source="n5://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/labels/nucleus_seg"
            )
            s.layers["pm_seg"] = neuroglancer.SegmentationLayer(
                source="n5://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/labels/pm_seg"
            )
            s.layers["vesicle_seg"] = neuroglancer.SegmentationLayer(
                source="n5://s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/labels/vesicle_seg"
            )

            # Set initial view position
            # HeLa-2 dataset has 4nm voxel resolution
            s.position = [24000.512 / 2, 3199.5 / 2, 16684.5 / 2]

            # Set 3D view orientation
            s.projection_orientation = [0, 0, 0, 1]

    def _on_state_change(self):
        """Called whenever viewer state changes."""
        try:
            with self.viewer.txn() as s:
                state_json = s.to_json()

            summary = self.summarize_state(state_json)
            self.current_state_summary = summary

            # Check if change is meaningful
            if self._is_meaningful_change(self.last_state_summary, summary):
                self._print_state_summary(summary)
                self.last_state_summary = summary
                self.state_dirty = True
                self.screenshot_dirty = True  # Trigger screenshot on meaningful change

                # Narration is handled in stream.py with screenshots
                # This callback just marks state as dirty
        except Exception as e:
            print(f"Error in state change callback: {e}")

    def summarize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information from state JSON."""
        summary = {}

        # Position
        if "position" in state:
            summary["position"] = state["position"]

        # Cross-section scale/zoom
        if "crossSectionScale" in state:
            summary["scale"] = state["crossSectionScale"]

        # Orientation
        if "crossSectionOrientation" in state:
            summary["orientation"] = state["crossSectionOrientation"]

        # Visible layers
        layers_info = []
        if "layers" in state:
            for layer in state["layers"]:
                layer_info = {
                    "name": layer.get("name", "unnamed"),
                    "type": layer.get("type", "unknown"),
                    "visible": layer.get("visible", True),
                }
                layers_info.append(layer_info)
        summary["layers"] = layers_info

        # Selected segments
        selected_segments = []
        if "layers" in state:
            for layer in state["layers"]:
                if "segments" in layer:
                    selected_segments.extend(layer["segments"])
        if selected_segments:
            summary["selected_segments"] = selected_segments

        return summary

    def _is_meaningful_change(self, prev: Optional[Dict], curr: Dict) -> bool:
        """Determine if state change is meaningful enough to report."""
        if prev is None:
            return True

        # Check position change
        if "position" in prev and "position" in curr:
            prev_pos = prev["position"]
            curr_pos = curr["position"]
            if len(prev_pos) >= 3 and len(curr_pos) >= 3:
                distance = (
                    sum((a - b) ** 2 for a, b in zip(prev_pos[:3], curr_pos[:3])) ** 0.5
                )
                if distance > self.position_threshold:
                    return True

        # Check scale change
        if "scale" in prev and "scale" in curr:
            scale_ratio = abs(curr["scale"] - prev["scale"]) / (prev["scale"] + 1e-10)
            if scale_ratio > self.scale_threshold:
                return True

        # Check selection change
        if prev.get("selected_segments") != curr.get("selected_segments"):
            return True

        # Check layer visibility change
        prev_layers = {l["name"]: l["visible"] for l in prev.get("layers", [])}
        curr_layers = {l["name"]: l["visible"] for l in curr.get("layers", [])}
        if prev_layers != curr_layers:
            return True

        return False

    def _print_state_summary(self, summary: Dict[str, Any]):
        """Print a compact state summary."""
        parts = []

        if "position" in summary:
            pos = summary["position"]
            parts.append(f"pos=[{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}]")

        if "scale" in summary:
            parts.append(f"scale={summary['scale']:.2f}")

        if "selected_segments" in summary:
            segs = summary["selected_segments"]
            parts.append(f"selected={segs}")

        visible_layers = [l["name"] for l in summary.get("layers", []) if l["visible"]]
        if visible_layers:
            parts.append(f"layers={visible_layers}")

        print(f"[STATE] {' | '.join(parts)}")

    def start_screenshot_loop(self, max_fps=0.1):
        """Start background thread for screenshot capture."""
        if self.screenshot_thread is not None:
            print("Screenshot loop already running")
            return

        self.running = True
        # Set dirty flag to capture initial frame
        self.screenshot_dirty = True

        self.screenshot_thread = threading.Thread(
            target=self._screenshot_loop, args=(max_fps,), daemon=True
        )
        self.screenshot_thread.start()
        print(f"Screenshot loop started at max {max_fps} fps")

    def _screenshot_loop(self, max_fps):
        """Background loop that captures screenshots when dirty."""
        frame_interval = 1.0 / max_fps
        idle_interval = 5.0  # Capture every 5 seconds when idle
        last_capture_time = 0

        while self.running:
            current_time = time.time()

            # Capture if dirty OR if it's been a while (idle refresh)
            time_since_last = current_time - last_capture_time
            should_capture = self.screenshot_dirty or (time_since_last >= idle_interval)

            if not should_capture:
                time.sleep(0.1)
                continue

            # Check if enough time has passed (respect frame rate)
            if self.screenshot_dirty and time_since_last < frame_interval:
                time.sleep(0.01)
                continue

            # Capture screenshot
            try:
                print(f"[DEBUG] Attempting screenshot capture...", flush=True)
                print(f"[DEBUG] Viewer object: {self.viewer}", flush=True)
                print(
                    f"[DEBUG] Viewer has screenshot method: {hasattr(self.viewer, 'screenshot')}",
                    flush=True,
                )

                # Try screenshot with timeout using threading
                screenshot_result = {"reply": None, "error": None}

                def get_screenshot():
                    try:
                        print(
                            f"[DEBUG] Inside screenshot thread, calling viewer.screenshot()...",
                            flush=True,
                        )
                        # Use browser's natural window size (no resize)
                        result = self.viewer.screenshot()
                        print(
                            f"[DEBUG] viewer.screenshot() returned: {result}",
                            flush=True,
                        )
                        screenshot_result["reply"] = result
                    except Exception as e:
                        print(
                            f"[DEBUG] Exception in screenshot thread: {e}", flush=True
                        )
                        screenshot_result["error"] = str(e)

                screenshot_thread = threading.Thread(target=get_screenshot, daemon=True)
                screenshot_thread.start()
                print(
                    f"[DEBUG] Screenshot thread started, waiting up to 30 seconds...",
                    flush=True,
                )
                screenshot_thread.join(
                    timeout=30.0
                )  # 30 second timeout (EM data takes time to render)

                if screenshot_thread.is_alive():
                    print(
                        f"[WARN] Screenshot call timed out (still blocking), will retry",
                        flush=True,
                    )
                    self.screenshot_dirty = False
                    time.sleep(1.0)
                    continue

                if screenshot_result["error"]:
                    print(
                        f"[ERROR] Screenshot failed: {screenshot_result['error']}",
                        flush=True,
                    )
                    self.screenshot_dirty = False
                    time.sleep(1.0)
                    continue

                if not screenshot_result["reply"]:
                    print(f"[WARN] Screenshot returned no data", flush=True)
                    self.screenshot_dirty = False
                    time.sleep(1.0)
                    continue

                print(f"[DEBUG] Screenshot method returned!", flush=True)

                # Extract PNG bytes from the reply object (using .image attribute)
                png_bytes = screenshot_result["reply"].screenshot.image
                print(
                    f"[DEBUG] Screenshot captured: {len(png_bytes)} bytes PNG",
                    flush=True,
                )

                # Convert PNG to JPEG to reduce bandwidth
                png_image = Image.open(io.BytesIO(png_bytes))
                jpeg_buffer = io.BytesIO()
                png_image.convert("RGB").save(jpeg_buffer, format="JPEG", quality=85)
                jpeg_bytes = jpeg_buffer.getvalue()

                # Store frame
                self.latest_frame = {
                    "jpeg_bytes": jpeg_bytes,
                    "jpeg_b64": base64.b64encode(jpeg_bytes).decode("utf-8"),
                    "timestamp": current_time,
                    "state": self.current_state_summary,
                }
                self.latest_frame_ts = current_time

                print(f"[FRAME] Captured: {len(jpeg_bytes)} bytes JPEG", flush=True)

                # Clear dirty flag
                self.screenshot_dirty = False
                last_capture_time = current_time

            except Exception as e:
                print(f"[ERROR] Failed to capture screenshot: {e}", flush=True)
                import traceback

                traceback.print_exc()
                self.screenshot_dirty = False
                time.sleep(0.5)

    def stop_screenshot_loop(self):
        """Stop the screenshot loop."""
        self.running = False
        if self.screenshot_thread is not None:
            self.screenshot_thread.join(timeout=2)
            self.screenshot_thread = None

    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """Get the latest captured frame with state."""
        return self.latest_frame

    def get_url(self) -> str:
        """Get the Neuroglancer viewer URL."""
        return str(self.viewer)

    def get_available_layers(self) -> Dict[str, str]:
        """
        Discover available segmentation layers from Neuroglancer.

        Returns:
            Dict mapping layer_name → layer_type (e.g., 'mito_filled' → 'SegmentationLayer')
        """
        try:
            with self.viewer.txn() as s:
                layers = {}
                # Neuroglancer layers is an OrderedDict-like object
                for layer in s.layers:
                    layer_name = layer.name
                    layer_type = type(layer).__name__
                    layers[layer_name] = layer_type
                print(f"[NG] Discovered {len(layers)} layers: {list(layers.keys())}")
                return layers
        except Exception as e:
            print(f"[NG] Error discovering layers: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def get_voxel_size(self) -> tuple:
        """
        Get voxel size from Neuroglancer viewer state.

        Returns:
            Tuple of (voxel_size_x, voxel_size_y, voxel_size_z) in nanometers
            Returns (1, 1, 1) if unable to determine
        """
        try:
            with self.viewer.txn() as s:
                state_json = s.to_json()

                # Try to get dimensions from first image layer
                if "layers" in state_json:
                    for layer in state_json["layers"]:
                        if layer.get("type") == "image":
                            # Check if layer has source with transform
                            source = layer.get("source")
                            if source and isinstance(source, dict):
                                transform = source.get("transform")
                                if transform and "outputDimensions" in transform:
                                    output_dims = transform["outputDimensions"]
                                    # Extract voxel sizes from dimensions
                                    voxel_sizes = []
                                    for dim_name, dim_info in output_dims.items():
                                        if (
                                            isinstance(dim_info, list)
                                            and len(dim_info) >= 2
                                        ):
                                            voxel_sizes.append(
                                                dim_info[1]
                                            )  # [units, size]
                                    if len(voxel_sizes) >= 3:
                                        print(
                                            f"[NG] Voxel size from state: {tuple(voxel_sizes[:3])}",
                                            flush=True,
                                        )
                                        return tuple(voxel_sizes[:3])

                # Fallback: check coordinate space in state
                if "dimensions" in state_json:
                    dims = state_json["dimensions"]
                    if isinstance(dims, dict):
                        voxel_sizes = []
                        for axis in ["x", "y", "z"]:
                            if axis in dims:
                                voxel_sizes.append(
                                    dims[axis][1] if isinstance(dims[axis], list) else 1
                                )
                        if len(voxel_sizes) == 3:
                            print(
                                f"[NG] Voxel size from dimensions: {tuple(voxel_sizes)}",
                                flush=True,
                            )
                            return tuple(voxel_sizes)

                print(
                    f"[NG] Could not determine voxel size from state, using default (8, 8, 8)",
                    flush=True,
                )
                return (8, 8, 8)  # Default for C. elegans dataset

        except Exception as e:
            print(f"[NG] Error getting voxel size: {e}", flush=True)
            import traceback

            traceback.print_exc()
            return (8, 8, 8)  # Default fallback


# Module-level instance for easy access
_tracker_instance = None


def get_tracker() -> NG_StateTracker:
    """Get or create the global tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = NG_StateTracker()
    return _tracker_instance
