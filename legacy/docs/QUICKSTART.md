# Quick Start Guide

## 🚀 One-Step Launch

```bash
pixi run start
```

Then open the web panel URL shown in the terminal output.

**Example output:**
```
Open web panel at:
  - http://ackermand-ws2:8090/
  - http://localhost:8090/ (local only)
```

Use the first URL to access from any device on your network!

That's it! Everything is in one interface now.

## What You'll See

The web panel now has:

### Left Side: Neuroglancer Viewer (Embedded)
- Interactive EM viewer with sample data pre-loaded
- Fly brain EM data from Janelia FlyEM FAFB dataset
- Navigate, zoom, select segments directly here

### Right Side: Live Stream + State
- **Top**: Live screenshots of what you're viewing
- **Bottom**: Current viewer state (position, zoom, layers, selections)

## How It Works

1. **Navigate in Neuroglancer** (left panel)
   - Pan around with mouse
   - Zoom in/out with scroll wheel
   - Click to select segments

2. **Watch the updates** (right panels)
   - Screenshots update as you move (debounced to 2 fps)
   - State panel shows current position/zoom/selection
   - All happens automatically!

## Sample Data Included

The viewer starts with:
- **EM Layer**: FAFB v14 CLAHE (fly brain electron microscopy)
- **Segmentation**: FAFB v14 segmentation v2
- **Initial Position**: An interesting region of the fly brain

## Next: Add AI Narration (Stage 4)

AI narration is implemented! To enable it:

1. Get an Anthropic API key from https://console.anthropic.com/
2. Create `.env` file: `cp .env.example .env`
3. Add your key: `ANTHROPIC_API_KEY=sk-ant-api03-...`
4. Restart the server: `pixi run start`

The AI will narrate as you navigate, appearing in the "AI Narration" panel!

See [STAGE4_COMPLETE.md](STAGE4_COMPLETE.md) for full documentation.

## Troubleshooting

### Screenshots not updating?
- Wait a few seconds - they timeout initially until Neuroglancer fully loads
- Try navigating around in the viewer to trigger captures

### Neuroglancer not loading?
- Check the console for errors
- Verify ports 8090 and 9999 are not in use

### Performance issues?
- Reduce FPS: `pixi run python server/main.py --fps 1`
- Screenshots are debounced and only capture when needed
