# kitten-tts-rs 🐱🦀

Rust implementation of [KittenTTS](https://github.com/KittenML/KittenTTS). KittenTTS delivers high-quality voice synthesis with models ranging from **15M to 80M parameters** (25–80 MB on disk). This Rust implementation provides self-contained binaries with no Python dependency.

* A [Rust CLI program](#4-generate-speech-cli). It is ideally suited for AI agent skills.
* An [OpenAI compatible API server](#5-run-the-api-server). It supports [SSE streaming](#sse-streaming) for realtime audio applications like the [EchoKit](https://echokit.dev/).

> **Adapted from:** [KittenML/KittenTTS](https://github.com/KittenML/KittenTTS) (Apache-2.0). All model weights are from the original project.

## Key Features of KittenTTS

- **Ultra-lightweight** — 15M to 80M parameters; smallest model is just 25 MB (int8)
- **CPU-optimized** — ONNX-based inference runs efficiently without a GPU
- **8 built-in voices** — Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, and Leo
- **Adjustable speech speed** — control playback rate via the `--speed` parameter
- **Text preprocessing** — built-in pipeline handles numbers, currencies, units, and more
- **24 kHz output** — high-quality audio at a standard sample rate
- **Edge-ready** — small enough to run on embedded devices, Raspberry Pi, phones

## What This Rust Port Adds

- **Two binaries** — `kitten-tts` CLI and `kitten-tts-server` (OpenAI-compatible API)
- **Fast startup** — ~100ms vs ~2s for Python import overhead
- **Tiny footprint** — ~10 MB binary (+ model weights) vs ~500 MB Python environment
- **GPU acceleration** — optional CUDA, TensorRT, CoreML, or DirectML via Cargo features
- **Cross-platform** — builds for Linux (x86_64, aarch64) and macOS (aarch64)

## Available Models

| Model | Parameters | Size | Download |
|---|---|---|---|
| kitten-tts-mini | 80M | 80 MB | [KittenML/kitten-tts-mini-0.8](https://huggingface.co/KittenML/kitten-tts-mini-0.8) |
| kitten-tts-micro | 40M | 41 MB | [KittenML/kitten-tts-micro-0.8](https://huggingface.co/KittenML/kitten-tts-micro-0.8) |
| kitten-tts-nano | 15M | 56 MB | [KittenML/kitten-tts-nano-0.8](https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32) |
| kitten-tts-nano (int8) | 15M | 25 MB | [KittenML/kitten-tts-nano-0.8-int8](https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8) |

## Quick Start

### 1. Install Dependencies

**espeak-ng** is required for phonemization:

```bash
# macOS
brew install espeak-ng

# Ubuntu/Debian
sudo apt-get install -y espeak-ng

# Fedora/RHEL
sudo dnf install espeak-ng

# Arch
sudo pacman -S espeak-ng
```

### 2. Download Binaries

Download the pre-built binaries for your platform from the [Releases](https://github.com/second-state/kitten_tts_rs/releases) page:

```bash
# Example: Linux x86_64
curl -LO https://github.com/second-state/kitten_tts_rs/releases/latest/download/kitten-tts-x86_64-linux.tar.gz
tar xzf kitten-tts-x86_64-linux.tar.gz

# Example: macOS Apple Silicon
curl -LO https://github.com/second-state/kitten_tts_rs/releases/latest/download/kitten-tts-aarch64-macos.tar.gz
tar xzf kitten-tts-aarch64-macos.tar.gz
```

Each archive contains two binaries:
- `kitten-tts` — CLI tool for one-off speech generation
- `kitten-tts-server` — OpenAI-compatible API server

### 3. Download Models

```bash
curl -LO https://github.com/second-state/kitten_tts_rs/releases/latest/download/kitten-tts-models.tar.gz
tar xzf kitten-tts-models.tar.gz
```

This extracts a `models/` directory with all available models:

```
models/
├── kitten-tts-mini/         # 80M params, 80 MB — highest quality
├── kitten-tts-micro/        # 40M params, 41 MB — balanced
├── kitten-tts-nano/         # 15M params, 56 MB (fp32)
└── kitten-tts-nano-int8/    # 15M params, 25 MB — smallest
```

### 4. Generate Speech (CLI)

```bash
# Basic usage (outputs output.wav)
./kitten-tts ./models/kitten-tts-mini 'Hello, world!' Bruno

# Specify output file and speed
./kitten-tts ./models/kitten-tts-mini 'Hello, world!' --voice Luna --speed 1.2 --output hello.wav

# List available voices
./kitten-tts ./models/kitten-tts-mini "" --list-voices
```

### 5. Run the API Server

Start the server with a model directory:

```bash
./kitten-tts-server ./models/kitten-tts-mini --host 0.0.0.0 --port 8080
```

The server exposes an OpenAI-compatible `/v1/audio/speech` endpoint:

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kitten-tts",
    "input": "Hello, world! This is KittenTTS running as an API server.",
    "voice": "alloy"
  }' \
  --output speech.mp3
```

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `input` | string | *(required)* | Text to synthesize |
| `voice` | string | *(required)* | Voice name (OpenAI or KittenTTS names) |
| `model` | string | `""` | Accepted for compatibility; ignored |
| `response_format` | string | `"mp3"` | Output audio format (see below) |
| `speed` | float | `1.0` | Speech speed multiplier (0.25–4.0) |
| `stream` | bool | `false` | Enable SSE streaming (requires `"pcm"` format) |

**Supported audio formats:**

| Format | Content-Type | Description |
|---|---|---|
| `mp3` | `audio/mpeg` | MP3 128 kbps CBR (default, resampled to 44.1 kHz) |
| `opus` | `audio/ogg` | Opus in OGG container (resampled to 48 kHz) |
| `flac` | `audio/flac` | FLAC lossless (24 kHz native) |
| `wav` | `audio/wav` | WAV 16-bit PCM (24 kHz native) |
| `pcm` | `audio/pcm` | Raw 16-bit signed little-endian PCM (24 kHz) |
| `aac` | — | Not yet supported (returns error) |

**API endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/speech` | Generate speech from text |
| `GET` | `/v1/models` | List loaded model |
| `GET` | `/health` | Health check |

**Voice mapping (OpenAI → KittenTTS):**

| OpenAI | KittenTTS | Gender |
|---|---|---|
| alloy | Bella | Female |
| echo | Jasper | Male |
| fable | Luna | Female |
| onyx | Bruno | Male |
| nova | Rosie | Female |
| shimmer | Hugo | Male |

All 8 KittenTTS voices (Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo) can also be used directly by name.

### SSE Streaming

For lower time-to-first-audio on longer texts, set `"stream": true` with `"response_format": "pcm"`. The server returns [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) with base64-encoded PCM audio chunks, compatible with the OpenAI streaming TTS format:

```bash
curl -N -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kitten-tts",
    "input": "Hello, this is a streaming test. Each sentence is sent as a separate audio chunk.",
    "voice": "alloy",
    "response_format": "pcm",
    "stream": true
  }'
```

Each event is a JSON object on a `data:` line:

```
data: {"type":"speech.audio.delta","delta":"<base64-encoded-pcm>"}
data: {"type":"speech.audio.delta","delta":"<base64-encoded-pcm>"}
data: {"type":"speech.audio.done"}
```

The `delta` field contains 16-bit signed little-endian PCM at 24 kHz, base64-encoded. The first chunk is split at the earliest clause boundary for fast initial playback.

## Building from Source

### CPU Only (recommended for most users)

```bash
git clone https://github.com/second-state/kitten_tts_rs.git
cd kitten_tts_rs

cargo build --release
# Binaries at: target/release/kitten-tts and target/release/kitten-tts-server
```

### With CUDA (NVIDIA GPU)

Requires CUDA toolkit and cuDNN installed on the system. Linux and Windows only.

```bash
cargo build --release --features cuda
```

### With TensorRT (NVIDIA GPU, optimized)

Requires TensorRT runtime. Linux and Windows only.

```bash
cargo build --release --features tensorrt
```

### With CoreML (Apple Silicon / macOS)

```bash
cargo build --release --features coreml
```

### With DirectML (Windows GPU)

```bash
cargo build --release --features directml
```

## Building and Testing Locally

After building from source, download models directly from Hugging Face to test:

```bash
# Download the nano-int8 model (smallest, 25 MB — good for testing)
mkdir -p models/kitten-tts-nano-int8
for FILE in config.json kitten_tts_nano_v0_8.onnx voices.npz; do
  curl -L -o "models/kitten-tts-nano-int8/$FILE" \
    "https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8/resolve/main/$FILE"
done
```

For other models, replace the directory name and URL with the appropriate values from the [Available Models](#available-models) table:

```bash
# Mini (80M params, highest quality)
mkdir -p models/kitten-tts-mini
for FILE in config.json kitten_tts_mini_v0_8.onnx voices.npz; do
  curl -L -o "models/kitten-tts-mini/$FILE" \
    "https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main/$FILE"
done

# Micro (40M params, balanced)
mkdir -p models/kitten-tts-micro
for FILE in config.json kitten_tts_micro_v0_8.onnx voices.npz; do
  curl -L -o "models/kitten-tts-micro/$FILE" \
    "https://huggingface.co/KittenML/kitten-tts-micro-0.8/resolve/main/$FILE"
done

# Nano fp32 (15M params)
mkdir -p models/kitten-tts-nano
for FILE in config.json kitten_tts_nano_v0_8.onnx voices.npz; do
  curl -L -o "models/kitten-tts-nano/$FILE" \
    "https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32/resolve/main/$FILE"
done
```

Test the CLI:

```bash
./target/release/kitten-tts ./models/kitten-tts-nano-int8 'Hello, world!' Bruno
```

Test the API server:

```bash
./target/release/kitten-tts-server ./models/kitten-tts-nano-int8 --port 8080
# In another terminal:
curl -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Hello from the API", "voice": "alloy"}' \
  --output test.mp3
```

## CoreML on Apple Silicon: Pros and Cons

### Pros
- Uses Apple's **Neural Engine** and **Metal GPU** for compatible operations
- No additional software installation needed (built into macOS)
- Can accelerate larger models where GPU compute outweighs overhead

### Cons
- **Slower than CPU for small models** — In our benchmarks on Apple Silicon (Mac mini M4 Pro), the 80M-parameter mini model ran **~1.7x slower** with CoreML (8.2s) than CPU-only (4.8s)
- **Dynamic shape limitations** — CoreML requires static tensor shapes; KittenTTS uses dynamic output shapes, so CoreML can only accelerate a subset of operations while the rest falls back to CPU
- **Model compilation overhead** — CoreML compiles the ONNX model to its internal format on first load, adding latency
- **First build is slower** — The `ort` crate may need to build ONNX Runtime from source with CoreML support

### Recommendation

For KittenTTS models (15M–80M params), **CPU-only is faster and simpler**. ONNX Runtime's CPU backend already uses SIMD (NEON on ARM, AVX on x86) and multi-threading effectively. CoreML would be more beneficial for models with 1B+ parameters where GPU compute dominates.

## Architecture

```
src/
├── main.rs                          # CLI binary (clap)
├── lib.rs                           # Library root
├── model.rs                         # ONNX session, inference, text chunking
├── phonemize.rs                     # espeak-ng → IPA phonemes → token IDs
├── preprocess.rs                    # Text normalization (numbers, currency, etc.)
├── voices.rs                        # NPZ voice embedding loader
└── bin/kitten-tts-server/           # API server binary
    ├── main.rs                      # Axum/Tokio server, model loading
    ├── error.rs                     # OpenAI-style error responses
    ├── state.rs                     # Shared model state (Arc<Mutex>)
    └── routes/{health,models,speech,encode}.rs
```

### How It Works

1. **Text preprocessing** — Expands numbers ("42" → "forty-two"), currencies ("$10.50" → "ten dollars and fifty cents"), and normalizes whitespace
2. **Phonemization** — Converts English text to IPA phonemes via `espeak-ng`
3. **Token encoding** — Maps IPA phonemes to integer token IDs using a symbol table matching the original Python implementation
4. **Voice selection** — Loads style embeddings from the NPZ voice file
5. **ONNX inference** — Runs the model with input tokens, voice style, and speed parameters
6. **Audio encoding** — Outputs MP3 (default), Opus, FLAC, WAV, or raw PCM

### Compared to Python KittenTTS

| | Python | Rust |
|---|---|---|
| Dependencies | onnxruntime, misaki, phonemizer, numpy, soundfile, spacy | ort, hound, espeak-ng (system) |
| Install size | ~500 MB (with venv) | ~10 MB binary |
| Startup time | ~2s (Python import) | ~100ms |
| Deployment | pip install + venv | Single binary (CLI + API server) |
| GPU support | onnxruntime-gpu pip package | Cargo feature flags |

## Performance

Benchmarked on Apple Mac M4 Pro (CPU only, no GPU acceleration). WAV output format. RTF (Real-Time Factor) = execution time / audio length — lower is better, < 1.0 means faster than real-time.

**Test inputs:**
- Short: `"Hello, world!"` (14 chars)
- Long: 5-sentence paragraph (571 chars, multiple chunks)

### kitten-tts-nano-int8 (15M params, 25 MB)

| Test | Exec Time | Audio Length | RTF |
|---|---|---|---|
| CLI short | 0.93s | 1.47s | 0.63 |
| CLI long | 5.73s | 47.18s | 0.12 |
| API short | 0.24s | 1.94s | 0.12 |
| API long | 6.78s | 61.91s | 0.11 |

### kitten-tts-mini (80M params, 80 MB)

| Test | Exec Time | Audio Length | RTF |
|---|---|---|---|
| CLI short | 0.71s | 1.52s | 0.47 |
| CLI long | 11.33s | 44.11s | 0.26 |
| API short | 0.64s | 2.04s | 0.31 |
| API long | 14.35s | 55.91s | 0.26 |

CLI times include model loading (~0.3s). API server times are warm (model pre-loaded, after warmup requests).

The nano-int8 model runs **8–9x faster than real-time** and is recommended for interactive and real-time applications. The mini model produces higher quality audio at **~4x faster than real-time**.

## License

Apache-2.0 (same as KittenTTS)

## Acknowledgments

- [KittenML](https://kittenml.com) for the original KittenTTS models and Python library
- [pyke/ort](https://crates.io/crates/ort) for the excellent ONNX Runtime Rust bindings
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) for phonemization
