# Neuroglancer Live Stream

A sidecar service that streams live screenshots and viewer state from Neuroglancer to a browser panel, with optional AI narration.

## Features

- **Live Screenshot Streaming**: Debounced 1-3 fps JPEG streaming
- **State Tracking**: Position, zoom, orientation, layer visibility, and segment selection
- **WebSocket Updates**: Real-time updates to browser panel
- **Responsive UI**: Clean dark theme with status indicators

## Quick Start

### Installation with pixi (recommended)

```bash
# Install dependencies with pixi
pixi install

# Start the server
pixi run start

# Or with custom settings
pixi run python server/main.py --ng-port 9999 --web-port 8090 --fps 2
```

### Alternative: Installation with pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r server/requirements.txt

# Start the server
python server/main.py
```

### Usage

**Just open one URL:** `http://localhost:8090/`

The web panel now includes:
- **Embedded Neuroglancer viewer** (left) with sample EM data pre-loaded
- **Live screenshots** (right top) updating as you navigate
- **State tracking** (right bottom) showing position, zoom, layers, selections

Navigate in the embedded viewer and watch the live stream update automatically!

See [QUICKSTART.md](QUICKSTART.md) for detailed usage guide.

## Architecture

### Stage 1: State Capture ✅

- Neuroglancer viewer with state change callbacks
- Summarizes position, zoom, orientation, layers, and selections
- Filters meaningful changes to avoid spam

### Stage 2: Screenshot Loop ✅

- Background thread captures screenshots when viewer state is "dirty"
- Converts PNG to JPEG for bandwidth efficiency
- Debounced to max 2 fps (configurable)

### Stage 3: WebSocket Streaming ✅

- FastAPI server with WebSocket endpoint
- Sends `{type: "frame", jpeg_b64: "...", state: {...}}` messages
- Browser displays live frames and state summary

### Stage 4: AI Narrator ✅

- Triggers narration on meaningful state changes
- Uses Claude 3.5 Sonnet to describe current view based on state
- Context-aware prompts for EM/neuroanatomy
- Real-time WebSocket broadcasting to all clients
- Configurable thresholds and intervals

### Stage 5: Voice (TODO)

- Text-to-speech for narration
- Queue management to avoid overlap
- Rate limiting during fast navigation

## Project Structure

```
tourguide/
├── server/
│   ├── main.py          # Entry point
│   ├── ng.py            # Neuroglancer viewer + state tracking
│   ├── stream.py        # FastAPI WebSocket server
│   ├── narrator.py      # (Stage 4) AI narration
│   ├── tts.py          # (Stage 5) Text-to-speech
│   └── requirements.txt # Legacy pip requirements
├── web/
│   ├── index.html      # Web UI
│   ├── app.js          # WebSocket client
│   └── style.css       # Styling
├── pixi.toml           # Pixi environment config
└── README.md
```

## Configuration

### Command-line Arguments

```
--ng-host HOST        Neuroglancer bind address (default: 127.0.0.1)
--ng-port PORT        Neuroglancer port (default: 9999)
--web-host HOST       Web server bind address (default: 0.0.0.0)
--web-port PORT       Web server port (default: 8090)
--fps FPS             Maximum screenshot frame rate (default: 2)
```

## Development Stages

- [x] **Stage 0**: Repository structure
- [x] **Stage 1**: Neuroglancer state capture
- [x] **Stage 2**: Screenshot loop
- [x] **Stage 3**: WebSocket streaming
- [x] **Stage 4**: AI narrator
- [ ] **Stage 5**: Voice/TTS
- [ ] **Stage 6**: Quality upgrades (ROI crop, UI controls, recording)

## Using AI Narration (Stage 4)

The AI narrator is now implemented and ready to use! To enable it:

1. **Get an Anthropic API key** from https://console.anthropic.com/

2. **Create a `.env` file** in the project root:
   ```bash
   cp .env.example .env
   ```

3. **Add your API key** to `.env`:
   ```bash
   ANTHROPIC_API_KEY=sk-ant-api03-...
   ```

4. **Restart the server**:
   ```bash
   pixi run start
   ```

5. **Navigate in Neuroglancer** - the AI will narrate as you explore!

See [STAGE4_COMPLETE.md](STAGE4_COMPLETE.md) for detailed documentation.

## Requirements

- Python 3.10+
- FastAPI
- Uvicorn
- Pillow
- Neuroglancer

## License

BSD 3-Clause License - see [LICENSE](LICENSE) file for details.
