# Neuroglancer TourGuide

An intelligent exploration platform for Neuroglancer that combines AI-powered narration, natural language queries, and interactive visualization with live progress tracking.

## Features

### 🔍 Dual Mode Interface
- **Exploration Mode**: Navigate and learn with AI narration that describes what you're seeing
- **Query Mode**: Ask questions in plain English to find, visualize, and navigate to structures

### 🤖 AI-Powered Intelligence
- **Natural Language Query**: "show the largest mitochondrion", "how many nuclei are there?"
- **Context-Aware Narration**: Real-time descriptions using Gemini, Claude, or local Ollama
- **Smart Visualization**: AI interprets "show", "also show", and "hide" commands semantically
- **Multi-Query Support**: Handles complex questions automatically split into sub-queries

### 📊 Live Progress Tracking
- **Real-time Query Updates**: See each step as queries are processed (classification, SQL generation, execution, response)
- **Exploration Progress**: Track narration generation, voice synthesis, and audio creation
- **Detailed Timing**: Monitor performance with timing information for each operation
- **Transparent Operations**: Know exactly what's happening behind the scenes

### 🎬 Recording & Playback
- **Session Recording**: Capture navigation tours with synchronized narration
- **Multiple Transitions**: Direct cuts, crossfade, or smooth state interpolation
- **Movie Export**: Create MP4 videos with audio narration tracks

### 💬 Voice & Audio
- **Text-to-Speech**: Browser-based, edge-tts, or advanced voice cloning (Coqui/Chatterbox)
- **Custom Voices**: Clone your own voice for personalized narration
- **Toggle Control**: Enable/disable voice output on the fly

### 🎨 Modern UI
- **Embedded Viewer**: Neuroglancer integrated directly in the interface
- **Live Screenshots**: Automatic capture with configurable frame rate
- **Dark Theme**: Clean, responsive interface with status indicators
- **Progress Log**: Detailed activity feed for debugging and monitoring

## Quick Start

### Installation

```bash
# Install with pixi (recommended)
pixi install

# Start the server
pixi run start

# Open in browser
http://localhost:8090/
```

### Configuration

Create a `.env` file:

```bash
cp .env.example .env
```

**For cloud AI (Gemini - recommended):**
```bash
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3-flash-preview  # or gemini-1.5-flash-8b
```

Get a free API key from https://aistudio.google.com/app/apikey

**For local AI (no API key needed):**
```bash
USE_LOCAL=true
```

Then install Ollama and run: `ollama pull llama3.2-vision`

**For query mode (organelle database):**
```bash
ORGANELLE_DB_PATH=./organelle_data/organelles.db
ORGANELLE_CSV_PATHS=/path/to/mito.csv,/path/to/nuc.csv
QUERY_AI_MODEL=nemotron-3-nano  # Local Ollama model for queries
```

**For voice cloning:**
```bash
USE_CHATTERBOX=true  # or USE_COQUI=true
VOICE_REFERENCE_PATH=/path/to/voice_sample.m4a
```

## Usage

### Exploration Mode

1. Navigate in the embedded Neuroglanger viewer
2. AI generates narration describing what you see
3. Watch live progress: "Generating narration..." → "Generating audio..." → "Audio generated"
4. Narration appears in the panel with optional voice playback
5. Toggle voice on/off with the microphone button

### Query Mode

1. Switch to Query Mode using the mode selector
2. Type natural language questions:
   - "show the largest mitochondrion"
   - "how many nuclei are there?"
   - "take me to the smallest peroxisome"
   - "show mitochondria larger than 1e11 nm³"
   - "also show nucleus 5" (adds to selection)
   - "hide all mitochondria" (removes from view)

3. Watch live progress updates:
   - Analyzing query...
   - Classifying query type
   - Generating SQL query
   - Executing database query: X rows
   - Generating response...
   - Applying visualization

4. See verbose details in the Progress tab:
   - Full AI prompts and responses
   - Generated SQL queries
   - Query results
   - Timing information
   - Model used

### Recording Tours

1. Click **Start Recording** to begin capturing
2. Navigate and explore - narration triggers automatically
3. Click **Stop Recording** when done
4. Choose transition style and click **Create Movie**:
   - **Direct Cuts**: Instant transitions with pauses
   - **Crossfade**: Smooth dissolve between views
   - **State Interpolation**: Neuroglancer renders smooth camera movements

Movies are saved to `recordings/<session_id>/output/movie.mp4`

## Architecture

### Core Components

- **Neuroglancer Integration** (`ng.py`): State tracking, screenshot capture, viewer management
- **Query Agent** (`query_agent.py`): Natural language to SQL, intent classification, multi-query handling
- **Narrator** (`narrator.py`): AI narration generation with Gemini/Claude/Ollama support
- **Streaming Server** (`stream.py`): WebSocket + SSE for live updates, REST API for queries
- **Organelle Database** (`organelle_db.py`): SQLite database with metadata indexing
- **Recording Manager** (`recording.py`): Session capture, movie compilation with FFmpeg

### Data Flow

**Query Mode:**
```
User Query → Query Detection → Classification → SQL Generation →
Execution → Response Generation → Visualization Update → Voice Synthesis
         ↓ (live progress via SSE)
    Progress Updates in UI
```

**Exploration Mode:**
```
State Change → Screenshot Capture → Narration Generation →
Voice Synthesis → WebSocket Broadcast
         ↓ (live progress messages)
    Progress Log Updates
```

## Project Structure

```
tourguide/
├── server/
│   ├── main.py              # Entry point, CLI args
│   ├── ng.py                # Neuroglancer viewer + state tracking
│   ├── stream.py            # FastAPI server (WebSocket + SSE + REST)
│   ├── narrator.py          # AI narration engine
│   ├── query_agent.py       # Natural language query processing
│   ├── organelle_db.py      # SQLite database for organelle metadata
│   └── recording.py         # Movie recording and compilation
├── web/
│   ├── index.html           # Main UI with dual-mode interface
│   ├── app.js               # WebSocket/SSE client, query handling
│   ├── style.css            # Dark theme styling
│   └── ng-screenshot-handler.js  # Screenshot capture logic
├── organelle_data/          # CSV files and database (gitignored)
├── recordings/              # Recorded sessions (auto-created)
├── .env                     # Configuration (gitignored)
├── .env.example             # Configuration template
└── pixi.toml                # Pixi environment config
```

## Configuration Options

### Command-line Arguments

```bash
pixi run python server/main.py \
  --ng-host 127.0.0.1 \
  --ng-port 9999 \
  --web-host 0.0.0.0 \
  --web-port 8090 \
  --fps 2
```

### Environment Variables

**AI Provider Selection:**
- `AI_PROVIDER`: Force specific provider (gemini|claude|local|auto)
- `USE_LOCAL`: Set to `true` for local Ollama mode

**Gemini Configuration:**
- `GOOGLE_API_KEY`: Gemini API key
- `GEMINI_MODEL`: Model name (gemini-3-flash-preview, gemini-1.5-flash-8b, gemini-1.5-flash)

**Claude Configuration:**
- `ANTHROPIC_API_KEY`: Claude API key

**Query Mode:**
- `ORGANELLE_DB_PATH`: SQLite database path
- `ORGANELLE_CSV_PATHS`: Comma-separated CSV file paths
- `QUERY_AI_MODEL`: Local Ollama model for query processing (e.g., nemotron-3-nano, qwen2.5-coder:7b)

**Voice/TTS:**
- `USE_COQUI`: Enable Coqui XTTS voice cloning
- `USE_CHATTERBOX`: Enable Chatterbox TTS voice cloning
- `VOICE_REFERENCE_PATH`: Path to voice sample (6-30 seconds)
- `EDGE_VOICE`: edge-tts voice name (en-GB-RyanNeural, en-US-AriaNeural)

## Advanced Setup

### GPU Cluster (LSF/H100)

Request GPU in shared mode to allow both PyTorch and Ollama:

```bash
bsub -P cellmap -n 12 -gpu "num=1:mode=shared" -q gpu_h100 -Is /bin/bash
pixi run start
```

See [CLUSTER_TROUBLESHOOTING.md](CLUSTER_TROUBLESHOOTING.md) for details.

### Local AI Setup

Complete local setup with no API keys:

1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull models:
   ```bash
   ollama pull llama3.2-vision      # For narration
   ollama pull nemotron-3-nano       # For queries (or qwen2.5-coder:7b)
   ```
3. Install local TTS:
   ```bash
   pixi run pip install kokoro soundfile sounddevice
   ```
4. Configure `.env`:
   ```bash
   USE_LOCAL=true
   QUERY_AI_MODEL=nemotron-3-nano
   ```

See [LOCAL_SETUP.md](LOCAL_SETUP.md) for detailed instructions.

### Voice Cloning

For custom voice narration:

1. Record 6-30 seconds of clear speech
2. Save as `.m4a` or `.wav` file
3. Configure `.env`:
   ```bash
   USE_CHATTERBOX=true  # Recommended for quality
   VOICE_REFERENCE_PATH=/path/to/voice.m4a
   ```

Chatterbox requires GPU. Use edge-tts for CPU-only systems.

## Development Roadmap

- [x] **Stage 1**: Neuroglancer state capture
- [x] **Stage 2**: Screenshot loop
- [x] **Stage 3**: WebSocket streaming
- [x] **Stage 4**: AI narrator
- [x] **Stage 5**: Voice/TTS
- [x] **Stage 6**: Movie recording and compilation
- [x] **Stage 7**: Natural language query system
- [x] **Stage 8**: Live progress tracking (query + exploration)
- [ ] **Stage 9**: Quality upgrades (ROI crop, advanced UI controls)

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - Detailed usage guide
- [AGENT_DRIVEN_VISUALIZATION.md](AGENT_DRIVEN_VISUALIZATION.md) - Query system architecture
- [LOCAL_SETUP.md](LOCAL_SETUP.md) - Complete local AI setup
- [CLUSTER_TROUBLESHOOTING.md](CLUSTER_TROUBLESHOOTING.md) - GPU cluster setup

## Requirements

- Python 3.10+
- FastAPI & Uvicorn
- Neuroglanger
- Pillow
- FFmpeg (for movie compilation)
- SQLite (built-in with Python)
- Optional: Ollama, edge-tts, Coqui/Chatterbox TTS

## License

BSD 3-Clause License - see [LICENSE](LICENSE) file for details.
