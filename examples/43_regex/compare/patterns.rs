//! The same four passes with Rust's `regex` crate over bytes.

use regex::bytes::{Regex, RegexBuilder};
use std::time::Instant;

fn make_text(lines: u64) -> Vec<u8> {
    let words: [&[u8]; 8] = [
        b"alpha", b"beta", b"gamma", b"delta", b"epsilon", b"zeta", b"eta", b"theta",
    ];
    let mut out = Vec::new();
    let mut state: u64 = 12345;
    for i in 0..lines {
        state = (state * 1103515245 + 12345) % (1 << 31);
        let word = words[(state % 8) as usize];
        out.extend_from_slice(word);
        out.extend_from_slice(b" = ");
        out.extend_from_slice((state % 1000).to_string().as_bytes());
        out.extend_from_slice(b"  ");
        out.extend_from_slice(word.to_ascii_uppercase().as_slice());
        if i % 7 == 0 {
            out.extend_from_slice(format!("0x{:x}", state).as_bytes());
        }
        out.push(b'\n');
    }
    out
}

fn count_words(word: &Regex, text: &[u8]) -> u64 {
    word.find_iter(text).count() as u64
}

fn longest_word(word: &Regex, text: &[u8]) -> u64 {
    word.find_iter(text).map(|m| m.len() as u64).max().unwrap_or(0)
}

fn sum_values(pair: &Regex, text: &[u8]) -> u64 {
    pair.captures_iter(text)
        .map(|c| std::str::from_utf8(&c[2]).unwrap().parse::<u64>().unwrap())
        .sum()
}

fn count_hex(hex: &Regex, text: &[u8]) -> u64 {
    hex.find_iter(text).count() as u64
}

fn timed(label: &str, mut run: impl FnMut() -> u64) -> u64 {
    let mut best = f64::INFINITY;
    let mut answer = 0;
    for _ in 0..5 {
        let started = Instant::now();
        answer = run();
        best = best.min(started.elapsed().as_secs_f64());
    }
    println!("# {}: {:.2} ms", label, best * 1000.0);
    answer
}

fn main() {
    let word = Regex::new(r"[A-Za-z]+").unwrap();
    let pair = Regex::new(r"(?P<key>\w+)\s*=\s*(\d+)").unwrap();
    let hex = RegexBuilder::new(r"0x[0-9a-f]+").case_insensitive(true).build().unwrap();
    let text = make_text(400_000);
    println!("{}", text.len());
    println!("{}", timed("count_words", || count_words(&word, &text)));
    println!("{}", timed("longest_word", || longest_word(&word, &text)));
    println!("{}", timed("sum_values", || sum_values(&pair, &text)));
    println!("{}", timed("count_hex", || count_hex(&hex, &text)));
}
