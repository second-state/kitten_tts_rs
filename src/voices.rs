//! Voice loading from NPZ files.
//!
//! KittenTTS voices are stored as NumPy .npz archives.
//! Each voice is a 2D float32 array (style embeddings indexed by text length).

use anyhow::{Context, Result};
use ndarray::Array2;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Voice name aliases (friendly → internal)
pub const VOICE_ALIASES: &[(&str, &str)] = &[
    ("Bella", "expr-voice-2-f"),
    ("Jasper", "expr-voice-2-m"),
    ("Luna", "expr-voice-3-f"),
    ("Bruno", "expr-voice-3-m"),
    ("Rosie", "expr-voice-4-f"),
    ("Hugo", "expr-voice-4-m"),
    ("Kiki", "expr-voice-5-f"),
    ("Leo", "expr-voice-5-m"),
];

/// All available friendly voice names.
pub const VOICE_NAMES: &[&str] = &[
    "Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo",
];

/// Resolve a voice name (friendly or internal) to the internal key.
pub fn resolve_voice_name(name: &str, config_aliases: &HashMap<String, String>) -> Option<String> {
    // Check config aliases first
    if let Some(internal) = config_aliases.get(name) {
        return Some(internal.clone());
    }
    // Check built-in aliases
    for &(friendly, internal) in VOICE_ALIASES {
        if friendly.eq_ignore_ascii_case(name) {
            return Some(internal.to_string());
        }
    }
    // Maybe it's already an internal name
    let internal_names = [
        "expr-voice-2-m",
        "expr-voice-2-f",
        "expr-voice-3-m",
        "expr-voice-3-f",
        "expr-voice-4-m",
        "expr-voice-4-f",
        "expr-voice-5-m",
        "expr-voice-5-f",
    ];
    if internal_names.contains(&name) {
        return Some(name.to_string());
    }
    None
}

/// Load voices from an NPZ file.
/// Returns a map of voice_key → Array2<f32>.
///
/// NPZ format: ZIP archive containing .npy files.
/// Each .npy file is named like "expr-voice-2-m.npy" and contains a 2D float32 array.
pub fn load_voices(path: &Path) -> Result<HashMap<String, Array2<f32>>> {
    let file = File::open(path).context("Failed to open voices NPZ file")?;
    let mut archive = zip::ZipArchive::new(file).context("Failed to read NPZ as ZIP")?;

    let mut voices = HashMap::new();

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry
            .name()
            .strip_suffix(".npy")
            .unwrap_or(entry.name())
            .to_string();

        let mut buf = Vec::new();
        entry.read_to_end(&mut buf)?;

        let array = parse_npy_f32(&buf)
            .with_context(|| format!("Failed to parse NPY for voice '{name}'"))?;
        voices.insert(name, array);
    }

    Ok(voices)
}

/// Minimal NPY parser for 2D float32 arrays (little-endian).
fn parse_npy_f32(data: &[u8]) -> Result<Array2<f32>> {
    // NPY format: magic "\x93NUMPY", major, minor, header_len, header (Python dict), then data
    anyhow::ensure!(data.len() >= 10, "NPY data too short");
    anyhow::ensure!(&data[..6] == b"\x93NUMPY", "Invalid NPY magic");

    let major = data[6];
    let (header_len, header_start) = if major == 1 {
        (u16::from_le_bytes([data[8], data[9]]) as usize, 10)
    } else {
        let len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        (len, 12)
    };

    let header = std::str::from_utf8(&data[header_start..header_start + header_len])?;

    // Parse shape from header: 'shape': (rows, cols)
    let shape_re = regex::Regex::new(r"'shape':\s*\((\d+),\s*(\d+)\)").unwrap();
    let caps = shape_re
        .captures(header)
        .context("Could not parse shape from NPY header")?;
    let rows: usize = caps[1].parse()?;
    let cols: usize = caps[2].parse()?;

    let data_start = header_start + header_len;
    let expected_bytes = rows * cols * 4;
    anyhow::ensure!(
        data.len() >= data_start + expected_bytes,
        "NPY data truncated"
    );

    let float_data: Vec<f32> = data[data_start..data_start + expected_bytes]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Array2::from_shape_vec((rows, cols), float_data).context("Shape mismatch in NPY data")
}
