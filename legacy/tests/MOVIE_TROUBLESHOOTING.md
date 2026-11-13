# Movie Generation Troubleshooting Guide

## Summary

I've created test scripts to help you generate movies from existing sessions without needing to run the server:

1. **test_movie_simple.py** - Fast, reliable movie generation using "cut" transitions
2. **test_movie_generation.py** - Supports all transition types
3. **test_movie_generation_verbose.py** - Real-time progress monitoring

## Key Findings

### Why Neuroglancer Interpolation Hangs

The neuroglancer `video_tool` hangs when:

1. **Missing Image Layers**: If the neuroglancer state only has segmentation layers (no EM/image layer), WebGL rendering never completes
   - Segmentation layers with empty `segments: []` won't render anything
   - Without an image layer, there's nothing to actually display

2. **Inaccessible Data Sources**: If the data sources can't be loaded:
   - Internal servers (`cellmap-vm1.int.janelia.org`) may not be accessible to headless Chrome
   - "No zarr metadata found" error means the source URL is invalid or inaccessible
   - The chunks load but WebGL never finishes rendering

3. **Network/Browser Issues**:
   - Headless Chrome may have different network access than your shell
   - SSL certificate issues with internal servers
   - DNS resolution problems for `.int.janelia.org` domains

### What Works

**Public datasets work perfectly**:
```bash
# Example with public Google Cloud dataset
timeout 60 pixi run python -m neuroglancer.tool.video_tool render \
  --fps 30 --width 480 --height 270 \
  /tmp/test_public_ng.txt /tmp/test_public_frames
```

**Private datasets that worked before** used public S3 buckets:
- `s3://janelia-cosem-datasets/` (publicly accessible)
- NOT `cellmap-vm1.int.janelia.org` (internal server)

## Solutions

### Option 1: Use "Cut" Transitions (Recommended)
```bash
pixi run python test_movie_simple.py recordings/session_selected_20251228_152416
```
- Uses existing screenshots (no re-rendering needed)
- Fast (<30 seconds)
- Always works

### Option 2: Fix Data Sources for Interpolation

To use interpolation, you need:
1. **Image layer must be present** in neuroglancer state
2. **Data must be publicly accessible** OR headless Chrome must be able to reach it
3. **At least 2 keyframes** with valid state

Check your session's state files:
```bash
# Check if image layers are present
python3 << 'EOF'
import json
from pathlib import Path

session_dir = Path("recordings/session_selected_20251228_152416")
for state_file in sorted(session_dir.glob("states/*.json")):
    with open(state_file) as f:
        state = json.load(f)
    layers = state.get("layers", [])
    image_layers = [l for l in layers if l.get("type") == "image"]
    seg_layers = [l for l in layers if l.get("type") == "segmentation"]
    print(f"{state_file.name}: {len(image_layers)} image, {len(seg_layers)} segmentation")
    if image_layers:
        print(f"  Image source: {image_layers[0].get('source', 'NO SOURCE')[:80]}")
EOF
```

### Option 3: Make Data Publicly Accessible

1. Use S3/public cloud storage
2. Use a public web server
3. Set up VPN/tunnel for internal servers

## Code Changes Made

Added validation in [server/recording.py](server/recording.py:760-828):
- Warns if frames have no image layers (only segmentation)
- Shows which layer types were found
- Recommends using "cut" or "crossfade" instead
- Displays critical warning banner if interpolation will likely hang

The warnings appear during movie compilation and clearly explain the issue and solutions.
