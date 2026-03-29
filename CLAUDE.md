# CLAUDE.md

## Project Overview

kitten-tts-rs is a Rust port of [KittenTTS](https://github.com/KittenML/KittenTTS) — an ultra-lightweight ONNX-based text-to-speech engine. It produces two binaries: `kitten-tts` (CLI) and `kitten-tts-server` (OpenAI-compatible API server).

## Build & Test

```bash
cargo build --release
# GPU variants: cargo build --release --features cuda|coreml|tensorrt|directml

# Run CLI
./target/release/kitten-tts ./models/kitten-tts-nano-int8 'Hello world' Bruno

# Run API server
./target/release/kitten-tts-server ./models/kitten-tts-nano-int8 --port 8080
```

Requires `espeak-ng` installed on the system for phonemization.

## Architecture

- `src/model.rs` — Core TTS engine: ONNX inference, text chunking (`chunk_text`, `chunk_text_streaming`)
- `src/phonemize.rs` — Calls `espeak-ng` binary for IPA phoneme conversion
- `src/preprocess.rs` — Text normalization (numbers, currencies, whitespace)
- `src/voices.rs` — NPZ voice embedding loader (8 built-in voices)
- `src/main.rs` — CLI binary
- `src/bin/kitten-tts-server/` — API server (Axum + Tokio)
  - `routes/speech.rs` — POST /v1/audio/speech handler, validation, streaming
  - `routes/encode.rs` — Audio format encoders (MP3, Opus, FLAC, WAV, PCM)
  - `state.rs` — Shared model state (`Arc<Mutex<KittenTTS>>`)

## Audio Format Implementation Notes

### MP3 (`mp3lame-encoder` crate)
- **Must resample to 44100 Hz** before encoding. The LAME C library (via `mp3lame-sys`) segfaults on macOS arm64 when given 24 kHz input directly. Resampling to the standard MPEG-1 rate of 44100 Hz avoids this and ensures universal player compatibility.
- Use `MonoPcm` (not `InterleavedPcm`) for mono audio — `InterleavedPcm` divides sample count by 2, producing half-duration chipmunk audio.
- Use `encode_to_vec()` / `flush_to_vec::<FlushNoGap>()` — the non-vec variants require unsafe `MaybeUninit` buffers.

### Opus (`audiopus` + `ogg` crates)
- Resample to 48 kHz (Opus native rate) before encoding.
- 20ms frames = 960 samples at 48 kHz.
- OGG container requires OpusHead + OpusTags headers per RFC 7845.
- Granule positions are in 48 kHz sample units.
- `audiopus_sys` uses `features = ["static"]` to statically link libopus.

### FLAC (`flacenc` crate)
- Pure Rust, no C dependencies. Encodes at native 24 kHz.
- Samples must be `i32` (16-bit range values stored as i32).
- Error types don't implement `std::error::Error` — use `anyhow::anyhow!("{:?}", e)`.

### Text Chunking
- Long input is split at sentence boundaries (max 400 chars per chunk).
- Each chunk is independently phonemized, inferred, and the audio is concatenated.
- Streaming mode uses `chunk_text_streaming()` which splits the first chunk at the earliest clause boundary (max 100 chars) for lower time-to-first-audio.

## Coding Conventions

- Default audio output format is MP3 (matching OpenAI API default).
- All audio is generated at 24 kHz mono internally; resampling is done only at the encoding stage when a format requires it.
- Shell examples in README use single quotes to avoid zsh history expansion issues with `!`.
