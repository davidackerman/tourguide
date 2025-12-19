# Neuroglancer Live Stream

A sidecar service that streams live screenshots and viewer state from Neuroglancer to a browser panel, with AI narration powered by Gemini, Claude, or local Ollama.

## Features

- **Live Screenshot Streaming**: Debounced 0.1-5 fps JPEG streaming
- **State Tracking**: Position, zoom, orientation, layer visibility, and segment selection
- **WebSocket Updates**: Real-time updates to browser panel
- **AI Narration**: Context-aware descriptions using cloud (Gemini/Claude) or local (Ollama) AI
- **Local TTS**: Optional voice narration with Kokoro (local mode only)
- **Responsive UI**: Clean dark theme with status indicators and narration history

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

## Using AI Narration

### Option 1: Cloud AI (Gemini - Recommended)

1. **Get a free API key** from https://aistudio.google.com/app/apikey

2. **Create a `.env` file**:
   ```bash
   cp .env.example .env
   ```

3. **Add your API key** to `.env`:
   ```bash
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Start the server**:
   ```bash
   pixi run start
   ```

### Option 2: Local AI (Ollama + Kokoro TTS - No API Key!)

For completely local, private, and free AI narration with voice:

1. **Install Ollama** from [ollama.com](https://ollama.com)

2. **Download the vision model**:
   ```bash
   ollama pull llama3.2-vision
   ```

3. **Install TTS** (optional):
   ```bash
   pixi run pip install kokoro soundfile sounddevice
   ```

4. **Enable local mode** in `.env`:
   ```bash
   USE_LOCAL=true
   ```

5. **Start the server**:
   ```bash
   pixi run start
   ```

See [LOCAL_SETUP.md](LOCAL_SETUP.md) for detailed local setup instructions.

### Option 3: Cloud AI (Claude/Anthropic)

Use `ANTHROPIC_API_KEY` in `.env` instead of `GOOGLE_API_KEY`.

---

Navigate in Neuroglancer and watch the AI narrate your exploration in real-time!

## Requirements

- Python 3.10+
- FastAPI
- Uvicorn
- Pillow
- Neuroglancer

## License

BSD 3-Clause License - see [LICENSE](LICENSE) file for details.
