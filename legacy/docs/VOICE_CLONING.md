# Voice Cloning with Chatterbox

This project supports voice cloning using Chatterbox TTS, allowing you to use any voice for narration.

## Setup

### 1. Install Base Dependencies

```bash
pixi install
```

### 2. Install Chatterbox (Optional - for voice cloning)

Chatterbox has build issues with the `pkuseg` dependency. If you want voice cloning, install it manually:

```bash
# Enter pixi environment
pixi shell

# Install with build isolation disabled to work around pkuseg issues
pip install --no-build-isolation chatterbox-tts

# Or try with numpy pre-installed
pip install numpy
pip install chatterbox-tts
```

**Note:** If installation fails, you can use edge-tts instead (works everywhere, no voice cloning).

### 3. Configure Voice Cloning

Add these environment variables to your `.env` file:

```bash
# Enable Chatterbox instead of edge-tts
USE_CHATTERBOX=true

# Optional: Path to reference audio file for voice cloning
VOICE_REFERENCE_PATH=/path/to/your/voice/sample.wav

# Optional: Customize edge-tts voice (if not using Chatterbox)
EDGE_VOICE=en-GB-RyanNeural
```

## Usage

### Using Default Voice

Simply set `USE_CHATTERBOX=true` in your `.env` file:

```bash
USE_CHATTERBOX=true
```

The system will use Chatterbox's default voice for narration.

### Voice Cloning from Audio Sample

1. **Prepare a voice sample**:
   - Record a clear audio sample (WAV format recommended)
   - 10-30 seconds of speech is typically sufficient
   - Ensure good audio quality (minimal background noise)

2. **Configure the voice reference**:
   ```bash
   USE_CHATTERBOX=true
   VOICE_REFERENCE_PATH=/path/to/voice/sample.wav
   ```

3. **Start the server**:
   ```bash
   pixi run start
   ```

The narration will now use the cloned voice from your reference audio!

## Switching Between TTS Engines

### Use Chatterbox (with voice cloning)
```bash
USE_CHATTERBOX=true
VOICE_REFERENCE_PATH=/path/to/sample.wav
```

### Use edge-tts (cloud-based, no cloning)
```bash
USE_CHATTERBOX=false
EDGE_VOICE=en-GB-RyanNeural
```

Available edge-tts voices:
- `en-GB-RyanNeural` - British male (default)
- `en-US-AriaNeural` - American female
- `en-AU-WilliamNeural` - Australian male
- `en-IN-NeerjaNeural` - Indian female
- Many more available at [Microsoft TTS voices](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts)

## Audio Format

- **Chatterbox**: Outputs WAV format
- **edge-tts**: Outputs MP3 format

The browser automatically detects and plays both formats.

## Troubleshooting

### Chatterbox not working

If Chatterbox fails to load, the system automatically falls back to edge-tts:

```
[NARRATOR] Chatterbox TTS not available: [error], falling back to edge-tts
[NARRATOR] TTS engine initialized (edge-tts with en-GB-RyanNeural)
```

### Voice reference not found

Ensure the path in `VOICE_REFERENCE_PATH` is absolute and the file exists:

```bash
# Check if file exists
ls -lh /path/to/voice/sample.wav
```

### Audio not playing

Check the browser console for errors. The system supports both WAV and MP3 formats with automatic detection.
