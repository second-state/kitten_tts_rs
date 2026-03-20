//! Text preprocessing: number-to-words, currency, ordinals, abbreviations, etc.
//! Ported from KittenTTS Python preprocess.py

use regex::Regex;

// ── Number → Words ──

const ONES: &[&str] = &[
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

const TENS: &[&str] = &[
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

const SCALE: &[&str] = &["", "thousand", "million", "billion", "trillion"];

fn three_digits_to_words(n: u64) -> String {
    if n == 0 {
        return String::new();
    }
    let mut parts = Vec::new();
    let hundreds = n / 100;
    let remainder = n % 100;
    if hundreds > 0 {
        parts.push(format!("{} hundred", ONES[hundreds as usize]));
    }
    if remainder < 20 {
        if remainder > 0 {
            parts.push(ONES[remainder as usize].to_string());
        }
    } else {
        let tens_word = TENS[(remainder / 10) as usize];
        let ones_word = ONES[(remainder % 10) as usize];
        if ones_word.is_empty() {
            parts.push(tens_word.to_string());
        } else {
            parts.push(format!("{tens_word}-{ones_word}"));
        }
    }
    parts.join(" ")
}

pub fn number_to_words(n: i64) -> String {
    if n == 0 {
        return "zero".to_string();
    }
    if n < 0 {
        return format!("negative {}", number_to_words(-n));
    }
    let n = n as u64;

    // Special: 100-9999 exact hundreds (not multiples of 1000) → "twelve hundred"
    if (100..=9999).contains(&n) && n.is_multiple_of(100) && !n.is_multiple_of(1000) {
        let h = n / 100;
        if h < 20 {
            return format!("{} hundred", ONES[h as usize]);
        }
    }

    let mut parts = Vec::new();
    let mut val = n;
    for scale in SCALE {
        let chunk = val % 1000;
        if chunk > 0 {
            let chunk_words = three_digits_to_words(chunk);
            if scale.is_empty() {
                parts.push(chunk_words);
            } else {
                parts.push(format!("{chunk_words} {scale}"));
            }
        }
        val /= 1000;
        if val == 0 {
            break;
        }
    }
    parts.reverse();
    parts.join(" ")
}

/// Expand currency symbols: "$42.50" → "forty-two dollars and fifty cents"
pub fn expand_currency(text: &str) -> String {
    let re = Regex::new(r"\$(\d+)(?:\.(\d{2}))?").unwrap();
    re.replace_all(text, |caps: &regex::Captures| {
        let dollars: i64 = caps[1].parse().unwrap_or(0);
        let cents: i64 = caps.get(2).map_or(0, |m| m.as_str().parse().unwrap_or(0));
        let mut result = number_to_words(dollars);
        result.push_str(if dollars == 1 { " dollar" } else { " dollars" });
        if cents > 0 {
            result.push_str(" and ");
            result.push_str(&number_to_words(cents));
            result.push_str(if cents == 1 { " cent" } else { " cents" });
        }
        result
    })
    .to_string()
}

/// Replace standalone numbers with words
pub fn expand_numbers(text: &str) -> String {
    let re = Regex::new(r"\b(\d+)\b").unwrap();
    re.replace_all(text, |caps: &regex::Captures| {
        let n: i64 = caps[1].parse().unwrap_or(0);
        number_to_words(n)
    })
    .to_string()
}

/// Full text preprocessing pipeline
pub fn preprocess(text: &str) -> String {
    let mut s = text.to_string();
    s = expand_currency(&s);
    s = expand_numbers(&s);
    // Normalize whitespace
    let re_ws = Regex::new(r"\s+").unwrap();
    s = re_ws.replace_all(&s, " ").trim().to_string();
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_to_words() {
        assert_eq!(number_to_words(0), "zero");
        assert_eq!(number_to_words(42), "forty-two");
        assert_eq!(number_to_words(1200), "twelve hundred");
        assert_eq!(number_to_words(1000), "one thousand");
        assert_eq!(number_to_words(1_000_000), "one million");
    }
}
