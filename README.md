# Neuroglancer Live Stream

A sidecar service that streams live screenshots and viewer state from Neuroglancer to a browser panel, with AI narration and movie recording capabilities powered by Gemini, Claude, or local Ollama.

## Features

- **Live Screenshot Streaming**: Debounced 0.1-5 fps JPEG streaming
- **State Tracking**: Position, zoom, orientation, layer visibility, and segment selection
- **WebSocket Updates**: Real-time updates to browser panel
- **AI Narration**: Context-aware descriptions using cloud (Gemini/Claude) or local (Ollama) AI
- **Natural Language Query**: Ask questions about organelles in plain English
- **Agent-Driven Visualization**: AI interprets queries to show/hide segments intelligently
- **AI-Powered Analysis Mode**: Generate and execute Python code for data analysis via natural language
- **Voice Synthesis**: Browser-based TTS or edge-tts with multiple voices
- **Movie Recording**: Record navigation sessions with synchronized narration
- **Multiple Transition Modes**: Direct cuts, crossfade, or smooth state interpolation
- **Responsive UI**: Clean dark theme with status indicators and narration history
- **Explore Mode with Verbose Logging**: Real-time progress tracking shows screenshot capture, AI narration generation, and audio synthesis status

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
- **Explore Mode** (default, right panel):
  - **Screenshots tab**: Live screenshots with AI narrations as you navigate
  - **Verbose Log tab**: Real-time progress tracking (📸 Screenshot captured → 📤 Sent to AI → ⏳ Waiting → ✅ Narration received → 🔊 Audio generated)
- **Query Mode**: Natural language questions about organelles with AI-driven visualization
- **State tracking**: Position, zoom, layers, selections
- **Recording controls**: Capture and compile narrated tours with multiple transition modes

Navigate in the embedded viewer and watch the live stream update automatically!

### Natural Language Queries

Ask questions about organelles in plain English:

**Examples:**
- "show the largest mitochondrion"
- "how many nuclei are there?"
- "take me to the smallest peroxisome"
- "show mitochondria larger than 1e11 nm³"
- "also show nucleus 5" (adds to current selection)
- "hide all mitochondria" (removes from view)

The AI agent:
1. Converts your question to SQL
2. Queries the organelle database
3. Interprets the results based on query semantics
4. Updates the visualization intelligently
5. Provides a natural language answer with voice narration

See [AGENT_DRIVEN_VISUALIZATION.md](AGENT_DRIVEN_VISUALIZATION.md) for technical details.

### Analysis Mode

Switch to **Analysis Mode** to generate and execute Python code for data analysis using natural language:

**Examples:**
- "Plot the volume distribution of mitochondria"
- "Show me a histogram of nucleus sizes"
- "Create a scatter plot comparing mitochondria volume vs surface area"

The AI analysis agent:
1. Converts your question to Python code
2. Executes the code in a sandboxed container (Docker or Apptainer)
3. Displays generated plots and statistics
4. Tracks session metadata and timing information

**Container Support:**
- **Docker**: Default for most systems
- **Apptainer**: Automatic fallback for HPC/cluster environments

See [ANALYSIS_MODE.md](ANALYSIS_MODE.md) for technical details and API documentation.

### Recording Tours

1. **Start Recording**: Click "Start Recording" to begin capturing frames
2. **Navigate**: Explore the dataset - narration triggers automatically on significant view changes
3. **Stop Recording**: Click "Stop Recording" when done
4. **Create Movie**: Choose transition style and click "Create Movie"
   - **Direct Cuts**: Instant transitions with 2-second silent pauses
   - **Crossfade**: Smooth dissolve transitions between views
   - **State Interpolation**: Neuroglancer renders smooth camera movements

Movies are saved to `recordings/<session_id>/output/movie.mp4` with:
- 960x540 resolution
- Frame duration matches audio narration length
- 2-second silent transitions between narrations
- Synchronized audio track

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
- Uses Gemini, Claude, or local Ollama to describe current view
- Context-aware prompts for EM/neuroanatomy
- Real-time WebSocket broadcasting to all clients
- Configurable thresholds and intervals

### Stage 5: Voice & TTS ✅

- Browser-based TTS or edge-tts with multiple voices
- Automatic audio playback in browser
- Audio synchronized with narration display
- Saved to recordings for movie compilation

### Stage 6: Movie Recording ✅

- Record navigation sessions with frame capture
- Three transition modes: cuts, crossfade, interpolation
- Frame duration matches narration audio length
- 2-second silent transitions between narrations
- FFmpeg-based video compilation with audio sync
- Neuroglancer video_tool integration for smooth camera movements

### Stage 7: Natural Language Query System ✅

- SQLite database for organelle metadata (volume, position, etc.)
- AI-powered natural language to SQL conversion
- Multi-query support with automatic splitting
- Intent classification: navigation, visualization, or informational
- Agent-driven visualization state updates
- Semantic understanding: "show X" vs "also show X" vs "hide X"
- Context-aware command generation using current viewer state

### Stage 8: Analysis Mode ✅

- Natural language to Python code generation
- Sandboxed code execution (Docker/Apptainer)
- Interactive plot generation and visualization
- Session metadata tracking with timing breakdown
- Comprehensive results management with REST API
- Automatic container detection for HPC environments

## Project Structure

```
tourguide/
├── server/
│   ├── main.py             # Entry point
│   ├── ng.py               # Neuroglancer viewer + state tracking
│   ├── stream.py           # FastAPI WebSocket server + query/analysis endpoints
│   ├── narrator.py         # AI narration engine
│   ├── query_agent.py      # Natural language query agent
│   ├── analysis_agent.py   # Natural language to Python code agent
│   ├── docker_sandbox.py   # Docker container sandbox
│   ├── apptainer_sandbox.py # Apptainer container sandbox
│   ├── analysis_results.py # Analysis session metadata manager
│   ├── organelle_db.py     # SQLite database for organelle metadata
│   ├── recording.py        # Movie recording and compilation
│   └── requirements.txt    # Legacy pip requirements
├── web/
│   ├── index.html      # Web UI with recording and analysis controls
│   ├── app.js          # WebSocket client + recording + analysis logic
│   ├── style.css       # Styling with spinner animations
│   └── ng-screenshot-handler.js  # Neuroglancer screenshot capture
├── organelle_data/     # Organelle CSV files and database (gitignored)
├── analysis_results/   # Analysis session outputs (gitignored)
├── containers/         # Container images (gitignored)
├── recordings/         # Recorded sessions (auto-created)
├── pixi.toml           # Pixi environment config
├── AGENT_DRIVEN_VISUALIZATION.md  # Agent visualization docs
├── ANALYSIS_MODE.md    # Analysis mode documentation
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
- [x] **Stage 5**: Voice/TTS
- [x] **Stage 6**: Movie recording and compilation
- [x] **Stage 7**: Natural language query system with agent-driven visualization
- [x] **Stage 8**: Analysis mode with AI code generation and sandboxed execution
- [ ] **Stage 9**: Quality upgrades (ROI crop, advanced UI controls)

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

## Running on GPU Cluster (LSF/H100)

To run on a GPU cluster node, use `mode=shared` when requesting GPUs:

```bash
bsub -P cellmap -n 12 -gpu "num=1:mode=shared" -q gpu_h100 -Is /bin/bash
```

**Important**: The `mode=shared` parameter is required! Without it, the GPU will be in exclusive mode, preventing both PyTorch (Chatterbox) and Ollama from using the GPU simultaneously.

Once on the node, run the application normally:
```bash
pixi run start
```

See [CLUSTER_TROUBLESHOOTING.md](CLUSTER_TROUBLESHOOTING.md) for detailed cluster setup and troubleshooting.

## Requirements

- Python 3.10+
- FastAPI & Uvicorn
- Pillow
- Neuroglancer
- FFmpeg (for movie compilation)
- edge-tts (for voice synthesis, optional)

## License

BSD 3-Clause License - see [LICENSE](LICENSE) file for details.
