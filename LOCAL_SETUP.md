# Local AI Setup Guide

This guide shows how to set up the **Local Mode** using Ollama (for vision) and Kokoro (for TTS) instead of cloud APIs.

## Why Local Mode?

- **No API costs**: Run completely offline
- **Privacy**: Your data never leaves your machine
- **Lower latency**: No network round-trips
- **Voice output**: Built-in local TTS with Kokoro

## Prerequisites

- ~8GB disk space for the vision model
- 8GB+ RAM recommended
- Python 3.10+

## Installation Steps

### 1. Install Ollama

Download and install Ollama from [ollama.com](https://ollama.com):

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:**
Download the installer from ollama.com

### 2. Download the Vision Model

```bash
ollama pull llama3.2-vision
```

This downloads the ~7.9GB Llama 3.2 Vision model.

### 3. Install Local TTS (Optional but Recommended)

```bash
pixi run pip install kokoro soundfile sounddevice
```

**Linux users also need:**
```bash
sudo apt-get install espeak-ng
```

**Windows users:**
Download and install espeak-ng from the releases page.

### 4. Configure Environment

Edit your `.env` file:

```bash
# Enable local mode
USE_LOCAL=true

# Optional: No API keys needed for local mode
# GOOGLE_API_KEY=...  (can be removed or left commented)
```

### 5. Start the Server

```bash
pixi run start
```

You should see:
```
[NARRATOR] Using Local Mode (Ollama llama3.2-vision)
[NARRATOR] Kokoro TTS enabled
```

## Testing

1. Open http://localhost:8090
2. Navigate in the embedded Neuroglancer viewer
3. Watch for AI narration in the panel
4. Listen for voice narration (if Kokoro is installed)

## Troubleshooting

### "Failed to connect to Ollama"

Make sure Ollama is running:
```bash
ollama serve
```

Then restart the tourguide server.

### "llama3.2-vision model not found"

Pull the model:
```bash
ollama pull llama3.2-vision
```

### No voice output

Check if Kokoro installed correctly:
```bash
pixi run python -c "from kokoro import KPipeline; print('Kokoro OK')"
```

If it fails, reinstall:
```bash
pixi run pip install --upgrade kokoro soundfile sounddevice
```

## Switching Back to Cloud Mode

Edit `.env`:
```bash
USE_LOCAL=false
GOOGLE_API_KEY=your_api_key_here
```

## Performance Notes

- **First narration**: May take 5-10 seconds (model loading)
- **Subsequent narrations**: ~1-3 seconds
- **TTS**: ~1-2 seconds per sentence
- **GPU acceleration**: Ollama automatically uses GPU if available (much faster)

## Model Comparison

| Feature | Gemini 2.5 Flash (Cloud) | Llama 3.2 Vision (Local) |
|---------|--------------------------|--------------------------|
| Cost | Free tier, then paid | Completely free |
| Speed | ~1 second | ~2-3 seconds (CPU), ~1s (GPU) |
| Quality | Excellent | Very good |
| Privacy | Data sent to Google | Fully local |
| Voice | Gemini Native Audio (cloud) | Kokoro TTS (local) |

## Advanced: GPU Acceleration

If you have an NVIDIA GPU, Ollama will automatically use it for much faster inference.

Check if GPU is being used:
```bash
ollama ps
```

Look for GPU memory usage in the output.

## Disk Space

- Ollama llama3.2-vision: ~7.9GB
- Kokoro TTS models: ~200MB
- Total: ~8.1GB
