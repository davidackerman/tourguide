# Usage Guide - Neuroglancer Live Stream

## Quick Start

1. **Start the server:**
   ```bash
   pixi run start
   ```

2. **Open Neuroglancer in your browser:**
   - Copy the Neuroglancer URL from the terminal output
   - Example: `http://127.0.0.1:9999/v/...`
   - **IMPORTANT**: You MUST open this URL first before screenshots will work

3. **Open the web panel in a SECOND browser tab/window:**
   - Open `http://localhost:8090/`
   - This shows the live stream view

## Why Do I Need To Open Neuroglancer?

Neuroglancer's screenshot functionality requires an active browser connection. The screenshots are actually rendered by the browser viewing Neuroglancer, then sent back to the server to be streamed to the web panel.

**Workflow:**
```
Browser Tab 1: Neuroglancer Viewer (renders the data)
       ↓
   Server (captures screenshots, tracks state)
       ↓
Browser Tab 2: Web Panel (shows live stream + state)
```

## What You'll See

- **Web Panel**: Live updating screenshots of what's in Neuroglancer
- **State Display**: Current position, zoom level, visible layers, selected segments
- **Console Logs**: State changes and frame captures in the terminal

## Troubleshooting

### "Waiting for frames" in web panel
- Make sure you've opened the Neuroglancer URL in a browser
- Navigate around in Neuroglancer to trigger screenshots
- Wait 5 seconds for an idle refresh

### No state updates
- Make sure you're moving enough (threshold: 100 voxels)
- Or changing zoom by >15%
- Or selecting/deselecting segments

### Screenshots timing out
- Check that Neuroglancer is fully loaded in the browser
- Try refreshing the Neuroglancer page
- Check browser console for errors

## Tips

- Keep both browser tabs/windows visible side-by-side
- The web panel updates automatically as you explore in Neuroglancer
- Screenshots are debounced to 2 fps (configurable with `--fps` flag)
- Idle screenshots happen every 5 seconds even without movement
