# tourguide-client

Python SDK for the **Tourguide Workspace API**. Drive a running Tourguide
visual session from a script or notebook — the same HTTP `/op` contract the
MCP adapter uses, just synchronous.

```bash
cd python
uv pip install -e .       # or: pip install -e .
```

```python
from tourguide_client import TourguideSession

# Bridge must be running (cd web-app && npm run bridge) and a workspace tab
# open (http://localhost:5173/?mode=workspace).
s = TourguideSession.attach()

print(s.get_session())                          # summary: layers, selection, tables...
s.select_segments("mito", ["12", "34"])         # highlight segments
s.fly_to([12000, 8000, 4000], layer="mito")     # camera to a point (nm)

s.run_sql("SELECT object_id, volume_nm_3 FROM mito ORDER BY volume_nm_3 DESC LIMIT 10")
s.show_table("SELECT * FROM mito WHERE volume_nm_3 > 1e9", name="big_mito")
s.show_plot(code="plt.hist(df_mitochondria['volume_nm_3']/1e9, bins=40)", title="Volumes (µm³)")

state = s.save_session_state("interesting state")
s.restore_session_state(state["id"])
```

All methods map 1:1 to Workspace API operations; `set_viewer_state` is the
escape hatch for raw Neuroglancer blobs. Configure the bridge URL with
`TourguideSession.attach(bridge_url=...)`.
