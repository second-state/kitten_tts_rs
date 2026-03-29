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
use crate::routes::encode::{self, AudioFormat};
use crate::state::AppState;

/// OpenAI-compatible speech request body.
#[derive(Deserialize)]
pub struct SpeechRequest {
    /// Model identifier (accepted for compatibility; ignored since one model is loaded at startup)
    #[serde(default)]
    #[allow(dead_code)]
    pub model: String,

    /// Text to synthesize
    pub input: String,

    /// Voice name — accepts OpenAI names (alloy, echo, fable, onyx, nova, shimmer)
    /// or KittenTTS names (Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo)
    pub voice: String,

    /// Response audio format: "mp3" (default), "opus", "flac", "wav", "pcm", or "aac"
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
    "mp3".to_string()
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
fn validate_request(req: &SpeechRequest) -> Result<(String, AudioFormat), ApiError> {
    if req.input.is_empty() {
        return Err(ApiError::bad_request("'input' must not be empty"));
    }
    if !(0.25..=4.0).contains(&req.speed) {
        return Err(ApiError::bad_request(
            "'speed' must be between 0.25 and 4.0",
        ));
    }

    let format = AudioFormat::parse(&req.response_format).ok_or_else(|| {
        ApiError::bad_request(format!(
            "Unsupported response_format '{}'. Supported formats: mp3, opus, flac, wav, pcm, aac",
            req.response_format
        ))
    })?;

    if req.stream && format != AudioFormat::Pcm {
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
        format = %format,
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

/// Non-streaming: generate all audio at once and return in the requested format.
async fn speech_full(
    state: AppState,
    input: String,
    voice: String,
    speed: f32,
    format: AudioFormat,
) -> Result<Response, ApiError> {
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, ApiError> {
        let audio = {
            let mut model = state
                .lock()
                .map_err(|e| ApiError::internal(e.to_string()))?;
            model
                .generate(&input, &voice, speed, true)
                .map_err(|e| ApiError::internal(e.to_string()))?
        };

        tracing::info!(
            samples = audio.len(),
            "Speech generated, encoding as {format}"
        );

        encode::encode(&audio, format).map_err(|e| ApiError::internal(e.to_string()))
    })
    .await
    .map_err(|e| ApiError::internal(format!("inference task failed: {e}")))??;

    Ok(([(header::CONTENT_TYPE, format.content_type())], bytes).into_response())
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

            let pcm = encode::encode_pcm(&audio);
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
