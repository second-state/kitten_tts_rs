//! kitten-tts-server — OpenAI-compatible TTS API server
//!
//! Loads a KittenTTS model at startup and serves an
//! OpenAI-style `/v1/audio/speech` endpoint.

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use clap::Parser;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing_subscriber::EnvFilter;

use kitten_tts::model::KittenTTS;

mod error;
mod routes;
mod state;

use state::AppState;

#[derive(Parser)]
#[command(
    name = "kitten-tts-server",
    about = "OpenAI-compatible TTS API server powered by KittenTTS"
)]
struct Cli {
    /// Path to model directory (containing config.json, .onnx, voices.npz)
    model_dir: PathBuf,

    /// Server host address
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Server port
    #[arg(long, default_value = "8080")]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    tracing::info!("Loading model from {} ...", cli.model_dir.display());
    let model = KittenTTS::from_dir(&cli.model_dir)?;
    tracing::info!("Model loaded successfully");

    let state: AppState = Arc::new(Mutex::new(model));

    let app = Router::new()
        .route("/health", get(routes::health::health))
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/audio/speech", post(routes::speech::speech))
        .with_state(state);

    let addr = format!("{}:{}", cli.host, cli.port);
    tracing::info!("Listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
