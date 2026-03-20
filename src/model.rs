//! ONNX model inference for KittenTTS.

use anyhow::{Context, Result};
use ndarray::Array2;
use ort::session::Session;
use ort::value::TensorRef;
use std::collections::HashMap;
use std::path::Path;

use crate::phonemize;
use crate::preprocess;
use crate::voices;

/// Model configuration loaded from config.json alongside the ONNX model.
#[derive(Debug, serde::Deserialize)]
pub struct ModelConfig {
    #[serde(rename = "type")]
    pub model_type: String,
    pub model_file: String,
    pub voices: String,
    #[serde(default)]
    pub speed_priors: HashMap<String, f32>,
    #[serde(default)]
    pub voice_aliases: HashMap<String, String>,
}

/// KittenTTS model ready for inference.
pub struct KittenTTS {
    session: Session,
    voices: HashMap<String, Array2<f32>>,
    speed_priors: HashMap<String, f32>,
    voice_aliases: HashMap<String, String>,
}

impl KittenTTS {
    /// Load model from a directory containing config.json, the ONNX model, and voices.npz.
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_data = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Cannot read {}", config_path.display()))?;
        let config: ModelConfig = serde_json::from_str(&config_data)?;

        anyhow::ensure!(
            config.model_type == "ONNX1" || config.model_type == "ONNX2",
            "Unsupported model type: {}",
            config.model_type
        );

        let model_path = model_dir.join(&config.model_file);
        let voices_path = model_dir.join(&config.voices);

        Self::load(
            &model_path,
            &voices_path,
            config.speed_priors,
            config.voice_aliases,
        )
    }

    /// Load from explicit ONNX model and voices file paths.
    pub fn load(
        model_path: &Path,
        voices_path: &Path,
        speed_priors: HashMap<String, f32>,
        voice_aliases: HashMap<String, String>,
    ) -> Result<Self> {
        eprintln!("Loading ONNX model from {}...", model_path.display());

        let eps: Vec<ort::execution_providers::ExecutionProviderDispatch> = Vec::new();

        #[cfg(feature = "tensorrt")]
        {
            eps.push(ort::ep::TensorRT::default().build());
            eprintln!("TensorRT execution provider registered");
        }
        #[cfg(feature = "cuda")]
        {
            eps.push(ort::ep::CUDA::default().build());
            eprintln!("CUDA execution provider registered");
        }
        #[cfg(feature = "coreml")]
        {
            eps.push(
                ort::ep::CoreML::default()
                    .with_subgraphs(true)
                    .with_static_input_shapes(true)
                    .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                    .build(),
            );
            eprintln!("CoreML execution provider registered");
        }
        #[cfg(feature = "directml")]
        {
            eps.push(ort::ep::DirectML::default().build());
            eprintln!("DirectML execution provider registered");
        }

        let session = if eps.is_empty() {
            Session::builder()?
                .commit_from_file(model_path)
                .context("Failed to load ONNX model")?
        } else {
            {
                let builder = Session::builder()
                    .map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?;
                let mut builder = builder
                    .with_execution_providers(eps)
                    .map_err(|e| anyhow::anyhow!("Failed to configure execution providers: {e}"))?;
                builder
                    .commit_from_file(model_path)
                    .context("Failed to load ONNX model")?
            }
        };

        eprintln!("Loading voices from {}...", voices_path.display());
        let voice_data = voices::load_voices(voices_path)?;
        eprintln!("Loaded {} voices", voice_data.len());

        Ok(Self {
            session,
            voices: voice_data,
            speed_priors,
            voice_aliases,
        })
    }

    /// List available voice names (friendly names).
    pub fn available_voices(&self) -> Vec<&'static str> {
        voices::VOICE_NAMES.to_vec()
    }

    /// Generate audio from text. Returns f32 samples at 24 kHz.
    pub fn generate(
        &mut self,
        text: &str,
        voice: &str,
        speed: f32,
        clean_text: bool,
    ) -> Result<Vec<f32>> {
        let text = if clean_text {
            preprocess::preprocess(text)
        } else {
            text.to_string()
        };

        let chunks = chunk_text(&text, 400);
        let mut all_audio = Vec::new();

        for chunk in &chunks {
            let audio = self.generate_chunk(chunk, voice, speed)?;
            all_audio.extend_from_slice(&audio);
        }

        Ok(all_audio)
    }

    fn generate_chunk(&mut self, text: &str, voice: &str, mut speed: f32) -> Result<Vec<f32>> {
        // Resolve voice name
        let internal_voice =
            voices::resolve_voice_name(voice, &self.voice_aliases).with_context(|| {
                format!(
                    "Unknown voice '{}'. Available: {:?}",
                    voice,
                    voices::VOICE_NAMES
                )
            })?;

        // Apply speed prior
        if let Some(&prior) = self.speed_priors.get(&internal_voice) {
            speed *= prior;
        }

        // Phonemize
        let phonemes = phonemize::phonemize(text)?;
        let token_ids = phonemize::text_to_token_ids(&phonemes);

        // Get voice style embedding
        let voice_data = self
            .voices
            .get(&internal_voice)
            .with_context(|| format!("Voice data for '{}' not found in NPZ", internal_voice))?;

        let ref_idx = token_ids.len().min(voice_data.nrows().saturating_sub(1));
        let style = voice_data.row(ref_idx).to_owned();

        // Prepare ONNX inputs
        let n_tokens = token_ids.len();
        let style_cols = voice_data.ncols();

        let input_ids_array = ndarray::Array2::from_shape_vec((1, n_tokens), token_ids)?;
        let style_array = style.into_shape_with_order((1, style_cols))?;
        let speed_array = ndarray::Array1::from_vec(vec![speed]);

        let input_ids_tensor = TensorRef::from_array_view(&input_ids_array)?;
        let style_tensor = TensorRef::from_array_view(&style_array)?;
        let speed_tensor = TensorRef::from_array_view(&speed_array)?;

        let outputs =
            self.session
                .run(ort::inputs![input_ids_tensor, style_tensor, speed_tensor])?;

        // Extract audio from first output
        let output = &outputs[0];
        let tensor_data = output.try_extract_tensor::<f32>()?;
        let audio: Vec<f32> = tensor_data.1.to_vec();

        // Trim trailing silence (last 5000 samples, matching Python)
        let trim_amount = 5000.min(audio.len());
        let audio = &audio[..audio.len() - trim_amount];

        Ok(audio.to_vec())
    }
}

/// Split text into chunks for processing, splitting on sentence boundaries.
fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    let sentences: Vec<&str> = regex::Regex::new(r"[.!?]+").unwrap().split(text).collect();

    let mut chunks = Vec::new();

    for sentence in sentences {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }

        if sentence.len() <= max_len {
            chunks.push(ensure_punctuation(sentence));
        } else {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let mut temp = String::new();
            for word in words {
                if temp.len() + word.len() < max_len {
                    if !temp.is_empty() {
                        temp.push(' ');
                    }
                    temp.push_str(word);
                } else {
                    if !temp.is_empty() {
                        chunks.push(ensure_punctuation(&temp));
                    }
                    temp = word.to_string();
                }
            }
            if !temp.is_empty() {
                chunks.push(ensure_punctuation(&temp));
            }
        }
    }

    chunks
}

fn ensure_punctuation(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return text.to_string();
    }
    let last = text.chars().last().unwrap();
    if ".!?,;:".contains(last) {
        text.to_string()
    } else {
        format!("{text},")
    }
}
