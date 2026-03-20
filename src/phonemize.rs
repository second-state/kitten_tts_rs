//! Phonemizer: converts English text → IPA phonemes via espeak-ng.
//!
//! KittenTTS uses espeak-ng for phonemization. We shell out to the `espeak-ng`
//! binary (widely available on Linux/macOS/Windows) rather than linking the C
//! library, keeping the build simple and portable.

use anyhow::{Context, Result};
use std::process::Command;

/// Symbol table matching the Python TextCleaner.
/// Index 0 = padding "$", then punctuation, ASCII letters, IPA symbols.
fn build_symbol_table() -> Vec<char> {
    let pad = "$";
    let punctuation = ";:,.!?¡¿—…\"«»\"\" ";
    let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    let letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

    let mut symbols: Vec<char> = Vec::new();
    for c in pad.chars() {
        symbols.push(c);
    }
    for c in punctuation.chars() {
        symbols.push(c);
    }
    for c in letters.chars() {
        symbols.push(c);
    }
    for c in letters_ipa.chars() {
        symbols.push(c);
    }
    symbols
}

/// Convert a phoneme string to token IDs matching the Python TextCleaner.
pub fn text_to_token_ids(phonemes: &str) -> Vec<i64> {
    let symbols = build_symbol_table();
    let mut char_to_idx = std::collections::HashMap::new();
    for (i, &c) in symbols.iter().enumerate() {
        char_to_idx.entry(c).or_insert(i as i64);
    }

    // Tokenize: split on whitespace-like boundaries (word chars vs punctuation)
    let re = regex::Regex::new(r"\w+|[^\w\s]").unwrap();
    let tokens: Vec<&str> = re.find_iter(phonemes).map(|m| m.as_str()).collect();
    let joined = tokens.join(" ");

    let mut ids: Vec<i64> = Vec::new();
    ids.push(0); // start token (padding)
    for c in joined.chars() {
        if let Some(&idx) = char_to_idx.get(&c) {
            ids.push(idx);
        }
        // skip unknown chars (same as Python)
    }
    ids.push(10); // end token
    ids.push(0); // padding
    ids
}

/// Call espeak-ng to phonemize English text.
/// Returns IPA phoneme string.
pub fn phonemize(text: &str) -> Result<String> {
    let output = Command::new("espeak-ng")
        .args([
            "--ipa", "-q",
            "--sep=", // no separator between phonemes within a word
            "-v", "en-us",
            text,
        ])
        .output()
        .context("Failed to run espeak-ng. Is it installed? (brew install espeak-ng / apt install espeak-ng)")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("espeak-ng failed: {stderr}");
    }

    let phonemes = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(phonemes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_table_size() {
        let symbols = build_symbol_table();
        assert!(symbols.len() > 100);
    }

    #[test]
    fn test_token_ids_basic() {
        // Just verify it doesn't panic and produces start/end tokens
        let ids = text_to_token_ids("hello");
        assert_eq!(ids[0], 0); // start
        assert_eq!(ids[ids.len() - 1], 0); // end padding
        assert_eq!(ids[ids.len() - 2], 10); // end token
    }
}
