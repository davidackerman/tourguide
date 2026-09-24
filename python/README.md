# tourguide-client

Python SDK for the **Tourguide Workspace API**. Drive a running Tourguide
visual session from a script or notebook — the same HTTP `/op` contract the
MCP adapter uses, synchronous.

```bash
cd python
uv pip install -e .       # or: pip install -e .
```

```python
from tourguide_client import TourguideSession

# Stack running (`cd web-app && npm run workspace:preview`, or let the MCP
# launch_or_attach start it) and a workspace tab open. The bridge token is
# read from its token file automatically; pass token=... to override.
s = TourguideSession.attach()

info = s.get_session()                 # layers with source URLs, voxel size, tables…
s.ingest_dataframe("mito", df)         # DataFrame with object_id + com_*_nm → click-to-fly
s.fly_to_segment("mito_seg", 4312)     # camera + selection via the table's centroid
s.show_figure(fig, title="Volumes")    # a matplotlib Figure, rendered here
s.screenshot("view.png")               # PNG of the current view
s.add_annotations([{"type": "bbox", "min": [0,0,0], "max": [500,500,500], "label": "ROI"}])

# React to the person at the screen
ev = s.wait_for_user_action(timeout_ms=30000)
for e in ev["events"]:
    if e["type"] == "selection_changed":
        print("user selected", e["selectedSegmentsByLayer"])

state = s.save_session_state("interesting state")
s.restore_session_state(state["id"])
```

All methods map 1:1 to Workspace API operations; `set_viewer_state` is the
escape hatch for raw Neuroglancer blobs. Configure the bridge URL with
`TourguideSession.attach(bridge_url=...)`.
