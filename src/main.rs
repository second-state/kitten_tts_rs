//! kitten-tts CLI — Rust port of KittenTTS
//!
//! Usage:
//!   kitten-tts <model_dir> "Hello world" --voice Bruno --output output.wav
//!   kitten-tts <model_dir> "Hello world" Bruno         # positional shorthand

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use kitten_tts::model::KittenTTS;

#[derive(Parser)]
#[command(name = "kitten-tts", about = "Ultra-lightweight ONNX text-to-speech")]
struct Cli {
    /// Path to model directory (containing config.json, .onnx, voices.npz)
    model_dir: PathBuf,

    /// Text to synthesize
    text: String,

    /// Voice name (positional shorthand). Overridden by --voice if both given.
    #[arg(default_value = "Bruno")]
    voice_pos: Option<String>,

    /// Voice name
    #[arg(short, long)]
    voice: Option<String>,

    /// Speech speed multiplier (1.0 = normal)
    #[arg(short, long, default_value = "1.0")]
    speed: f32,

    /// Output WAV file path
    #[arg(short, long, default_value = "output.wav")]
    output: PathBuf,

    /// Disable text preprocessing (number expansion, etc.)
    #[arg(long)]
    no_clean: bool,

    /// List available voices and exit
    #[arg(long)]
    list_voices: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let mut model = KittenTTS::from_dir(&cli.model_dir)?;

    if cli.list_voices {
        println!("Available voices:");
        for v in model.available_voices() {
            println!("  {v}");
        }
        return Ok(());
    }

    let voice = cli
        .voice
        .or(cli.voice_pos)
        .unwrap_or_else(|| "Bruno".to_string());

    eprintln!("Generating speech: voice={voice}, speed={}", cli.speed);
    let audio = model.generate(&cli.text, &voice, cli.speed, !cli.no_clean)?;

    // Write WAV (mono, 24 kHz, f32)
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&cli.output, spec)?;
    for &sample in &audio {
        let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(s)?;
    }
    writer.finalize()?;

    let duration = audio.len() as f32 / 24000.0;
    eprintln!(
        "Done! {} samples ({:.2}s) written to {}",
        audio.len(),
        duration,
        cli.output.display()
    );

    Ok(())
}
