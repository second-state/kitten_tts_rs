use std::convert::Infallible;

use axum::extract::State;
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine;
use serde::Deserialize;
use tokio_stream::wrappers::ReceiverStream;

use kitten_tts::model;
use kitten_tts::preprocess;

use crate::error::ApiError;
use crate::state::AppState;

/// OpenAI-compatible speech request body.
#[derive(Deserialize)]
pub struct SpeechRequest {
    /// Model identifier (accepted for compatibility; ignored since one model is loaded at startup)
    #[serde(default)]
    #[allow(dead_code)]
    pub model: String,

    /// Text to synthesize (max 4096 characters)
    pub input: String,

    /// Voice name — accepts OpenAI names (alloy, echo, fable, onyx, nova, shimmer)
    /// or KittenTTS names (Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo)
    pub voice: String,

    /// Response audio format: "wav" (default) or "pcm"
    #[serde(default = "default_format")]
    pub response_format: String,

    /// Speech speed multiplier (0.25–4.0, default 1.0)
    #[serde(default = "default_speed")]
    pub speed: f32,

    /// Enable SSE streaming (default false). When true, response_format must be "pcm".
    #[serde(default)]
    pub stream: bool,
}

fn default_format() -> String {
    "wav".to_string()
}

fn default_speed() -> f32 {
    1.0
}

/// Map OpenAI voice names to KittenTTS voice names.
/// Unknown names are passed through so KittenTTS can resolve them directly.
fn map_voice(voice: &str) -> &str {
    match voice.to_lowercase().as_str() {
        "alloy" => "Bella",
        "echo" => "Jasper",
        "fable" => "Luna",
        "onyx" => "Bruno",
        "nova" => "Rosie",
        "shimmer" => "Hugo",
        _ => voice,
    }
}

/// Validate common request fields. Returns (voice, format) on success.
fn validate_request(req: &SpeechRequest) -> Result<(String, String), ApiError> {
    if req.input.is_empty() {
        return Err(ApiError::bad_request("'input' must not be empty"));
    }
    if req.input.len() > 4096 {
        return Err(ApiError::bad_request(
            "'input' must be at most 4096 characters",
        ));
    }
    if !(0.25..=4.0).contains(&req.speed) {
        return Err(ApiError::bad_request(
            "'speed' must be between 0.25 and 4.0",
        ));
    }

    let format = req.response_format.to_lowercase();
    if format != "wav" && format != "pcm" {
        return Err(ApiError::bad_request(format!(
            "Unsupported response_format '{}'. Supported formats: wav, pcm",
            req.response_format
        )));
    }

    if req.stream && format != "pcm" {
        return Err(ApiError::bad_request(
            "Streaming only supports response_format 'pcm'",
        ));
    }

    let voice = map_voice(&req.voice).to_string();
    Ok((voice, format))
}

/// POST /v1/audio/speech — generate speech from text.
/// Returns either a complete audio response or an SSE stream depending on `stream` field.
pub async fn speech(
    State(state): State<AppState>,
    Json(req): Json<SpeechRequest>,
) -> Result<Response, ApiError> {
    let (voice, format) = validate_request(&req)?;
    let input = req.input;
    let speed = req.speed;
    let stream = req.stream;

    tracing::info!(
        voice = %voice,
        speed = speed,
        len = input.len(),
        stream = stream,
        "Generating speech"
    );

    if stream {
        speech_stream(state, input, voice, speed).await
    } else {
        speech_full(state, input, voice, speed, format).await
    }
}

/// Non-streaming: generate all audio at once and return as wav/pcm.
async fn speech_full(
    state: AppState,
    input: String,
    voice: String,
    speed: f32,
    format: String,
) -> Result<Response, ApiError> {
    let audio = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, ApiError> {
        let mut model = state
            .lock()
            .map_err(|e| ApiError::internal(e.to_string()))?;
        model
            .generate(&input, &voice, speed, true)
            .map_err(|e| ApiError::internal(e.to_string()))
    })
    .await
    .map_err(|e| ApiError::internal(format!("inference task failed: {e}")))??;

    tracing::info!(samples = audio.len(), "Speech generated");

    let (bytes, content_type) = match format.as_str() {
        "pcm" => (encode_pcm(&audio), "audio/pcm"),
        _ => {
            let wav = encode_wav(&audio).map_err(|e| ApiError::internal(e.to_string()))?;
            (wav, "audio/wav")
        }
    };

    Ok(([(header::CONTENT_TYPE, content_type)], bytes).into_response())
}

/// SSE streaming: generate audio chunk-by-chunk and send each as an SSE event.
async fn speech_stream(
    state: AppState,
    input: String,
    voice: String,
    speed: f32,
) -> Result<Response, ApiError> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(8);

    tokio::task::spawn_blocking(move || {
        let text = preprocess::preprocess(&input);
        let chunks = model::chunk_text_streaming(&text, 100, 400);

        tracing::info!(num_chunks = chunks.len(), "Streaming speech");

        let mut model = match state.lock() {
            Ok(m) => m,
            Err(e) => {
                let err = serde_json::json!({
                    "type": "error",
                    "error": { "message": e.to_string() }
                });
                let _ = tx.blocking_send(Ok(Event::default().data(err.to_string())));
                return;
            }
        };

        let b64 = base64::engine::general_purpose::STANDARD;

        for (i, chunk) in chunks.iter().enumerate() {
            let audio = match model.generate_chunk(chunk, &voice, speed) {
                Ok(a) => a,
                Err(e) => {
                    let err = serde_json::json!({
                        "type": "error",
                        "error": { "message": e.to_string() }
                    });
                    let _ = tx.blocking_send(Ok(Event::default().data(err.to_string())));
                    return;
                }
            };

            let pcm = encode_pcm(&audio);
            let delta = b64.encode(&pcm);

            tracing::debug!(
                chunk = i,
                text_len = chunk.len(),
                samples = audio.len(),
                "Chunk generated"
            );

            let data = serde_json::json!({
                "type": "speech.audio.delta",
                "delta": delta,
            });

            let event = Event::default().data(data.to_string());
            if tx.blocking_send(Ok(event)).is_err() {
                tracing::info!("Client disconnected during streaming");
                return;
            }
        }

        // Send done event
        let done = serde_json::json!({ "type": "speech.audio.done" });
        let _ = tx.blocking_send(Ok(Event::default().data(done.to_string())));

        tracing::info!("Streaming complete");
    });

    let stream = ReceiverStream::new(rx);
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

/// Encode f32 samples as raw 16-bit signed little-endian PCM.
fn encode_pcm(samples: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let i = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        buf.extend_from_slice(&i.to_le_bytes());
    }
    buf
}

/// Encode f32 samples as a WAV file (mono, 24 kHz, 16-bit PCM).
fn encode_wav(samples: &[f32]) -> anyhow::Result<Vec<u8>> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for &s in samples {
            let i = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(i)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}
