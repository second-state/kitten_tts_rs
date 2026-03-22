use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

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

/// POST /v1/audio/speech — generate speech from text.
pub async fn speech(
    State(state): State<AppState>,
    Json(req): Json<SpeechRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // --- Validate ---
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

    let voice = map_voice(&req.voice).to_string();
    let input = req.input;
    let speed = req.speed;

    tracing::info!(
        voice = %voice,
        speed = speed,
        len = input.len(),
        "Generating speech"
    );

    // --- Inference (blocking) ---
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

    // --- Encode ---
    let (bytes, content_type) = match format.as_str() {
        "pcm" => (encode_pcm(&audio), "audio/pcm"),
        _ => {
            let wav = encode_wav(&audio).map_err(|e| ApiError::internal(e.to_string()))?;
            (wav, "audio/wav")
        }
    };

    Ok(([(header::CONTENT_TYPE, content_type)], bytes))
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
