//! Sequence and N-gram Text Encoding
//!
//! Demonstrates encoding ordered sequences and text using:
//! 1. Positional permutation: sum_i permute(word_i, i)
//! 2. Sliding N-grams: sum_i (word_i * permute(word_{i+1}, 1) * ...)
//!
//! Run with:
//!   `cargo run --example 04_sequence_and_text_ngrams`

use guddalm_vsa::prelude::*;

fn main() {
    println!("=== GuddaLM VSA: Sequence and N-Gram Encoding Demo ===\n");

    const DIM: usize = 8192;
    let mut mem: ItemMemory<HDVector> = ItemMemory::new(DIM);

    // Sentence 1: "the quick brown fox"
    let sentence1_tokens = ["the", "quick", "brown", "fox"];
    let s1_vectors: Vec<HDVector> = sentence1_tokens.iter().map(|&w| mem.get_or_create(w)).collect();

    // Sentence 2: "the fast brown fox" (semantically similar)
    let sentence2_tokens = ["the", "fast", "brown", "fox"];
    let s2_vectors: Vec<HDVector> = sentence2_tokens.iter().map(|&w| mem.get_or_create(w)).collect();

    // Sentence 3: "fox brown quick the" (exact same words, but reversed order)
    let sentence3_tokens = ["fox", "brown", "quick", "the"];
    let s3_vectors: Vec<HDVector> = sentence3_tokens.iter().map(|&w| mem.get_or_create(w)).collect();

    // 1. Positional sequence encoding
    let pos_s1 = encode_positional_sequence(&s1_vectors);
    let pos_s2 = encode_positional_sequence(&s2_vectors);
    let pos_s3 = encode_positional_sequence(&s3_vectors);

    println!("Positional Sequence Similarity:");
    println!("  sim('the quick brown fox', 'the fast brown fox')  = {:.4}", pos_s1.cosine_similarity(&pos_s2));
    println!("  sim('the quick brown fox', 'fox brown quick the') = {:.4}", pos_s1.cosine_similarity(&pos_s3));
    assert!(pos_s1.cosine_similarity(&pos_s2) > pos_s1.cosine_similarity(&pos_s3));

    // 2. Bigram (N=2) encoding
    let ngram_s1 = encode_ngram_sequence(&s1_vectors, 2);
    let ngram_s2 = encode_ngram_sequence(&s2_vectors, 2);
    let ngram_s3 = encode_ngram_sequence(&s3_vectors, 2);

    println!("\nBigram (N=2) Sequence Similarity:");
    println!("  sim('the quick brown fox', 'the fast brown fox')  = {:.4}", ngram_s1.cosine_similarity(&ngram_s2));
    println!("  sim('the quick brown fox', 'fox brown quick the') = {:.4}", ngram_s1.cosine_similarity(&ngram_s3));

    println!("\nSequence and N-Gram encoding demo completed successfully!");
}
