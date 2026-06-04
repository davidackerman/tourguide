"""In-process Neuroglancer viewer (Python), driven directly by the MCP.

Spike: an alternative to the bundled-JS-NG-in-the-browser viewer. Because the
MCP is already Python, the viewer state is a live object here — fly_to /
set_state / select_segments are DIRECT calls (no bridge → browser relay), and
there's no custom browser viewer layer to maintain.

    v = NgViewer()
    v.set_state(ng_state_dict)      # e.g. decoded from a Neuroglancer link
    v.url()                         # open this in a browser
    v.fly_to([x, y, z])            # instant, no relay
    v.select_segments("mito_seg", ["1", "2", ...])
    v.get_state()                   # live read-back

Lazy: the viewer/server starts on first use. Bind address via TG_NG_BIND
(default 127.0.0.1; set 0.0.0.0 to share on the LAN like our preview server).
"""

from __future__ import annotations

import os
from typing import Any


class NgViewer:
    def __init__(self, bind: str | None = None):
        self._bind = bind or os.environ.get("TG_NG_BIND", "127.0.0.1")
        self._ng = None
        self._viewer = None

    def _ensure(self):
        if self._viewer is None:
            import neuroglancer  # imported lazily so the dep is only needed in this mode

            neuroglancer.set_server_bind_address(self._bind)
            self._ng = neuroglancer
            self._viewer = neuroglancer.Viewer()
        return self._viewer

    # --- url / state ---------------------------------------------------------

    def url(self) -> str:
        return self._ensure().get_viewer_url()

    def set_state(self, state: dict) -> None:
        v = self._ensure()
        v.set_state(self._ng.ViewerState(json_data=state))

    def get_state(self) -> dict:
        return self._ensure().state.to_json()

    # --- viewer ops (all direct, no relay) -----------------------------------

    def fly_to(self, position: list[float]) -> None:
        v = self._ensure()
        with v.txn() as s:
            s.position = [float(p) for p in position]

    def select_segments(self, layer: str, segment_ids: list) -> dict:
        v = self._ensure()
        with v.txn() as s:
            lyr = s.layers[layer]
            lyr.segments = {int(i) for i in segment_ids}
        return {"layer": layer, "count": len(segment_ids)}

    def add_layer(self, spec: dict) -> None:
        """Add or replace a layer by name (format-agnostic JSON merge)."""
        if not spec.get("name"):
            raise ValueError("add_layer: spec needs a 'name'")
        st = self.get_state()
        layers = [l for l in st.get("layers", []) if l.get("name") != spec["name"]]
        layers.append(spec)
        st["layers"] = layers
        self.set_state(st)

    def get_selection(self) -> dict:
        st = self.get_state()
        out: dict[str, Any] = {}
        for l in st.get("layers", []):
            if l.get("type") == "segmentation":
                out[l.get("name")] = [str(s) for s in (l.get("segments") or [])]
        return out
