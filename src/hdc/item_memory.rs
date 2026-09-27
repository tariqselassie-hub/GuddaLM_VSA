// Copyright (C) 2025-2026 guddalm_vsa contributors.
// SPDX-License-Identifier: AGPL-3.0
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::hdc::vector::{BinaryHDVector, HDVector};
use crate::hdc::fhrr::FHRRVector;
use crate::hdc::ghrr::GHRRVector;
use crate::hdc::vsa_trait::{IndexVector, VsaVector};
use crate::seed::{
    deterministic_binary_hd_vector,
    deterministic_fhrr_vector,
    deterministic_ghrr_vector as deterministic_ghrr_vector_seed,
    deterministic_hd_vector,
};

/// Trait for generating deterministic hypervectors from a symbol string.
pub trait DeterministicSymbol: VsaVector {
    fn generate_symbol(base_seed: u64, symbol: &str, dim: usize) -> Self;
}

impl DeterministicSymbol for HDVector {
    fn generate_symbol(base_seed: u64, symbol: &str, dim: usize) -> Self {
        deterministic_hd_vector(base_seed, symbol, dim)
    }
}

impl DeterministicSymbol for BinaryHDVector {
    fn generate_symbol(base_seed: u64, symbol: &str, dim: usize) -> Self {
        deterministic_binary_hd_vector(base_seed, symbol, dim)
    }
}

impl DeterministicSymbol for FHRRVector {
    fn generate_symbol(base_seed: u64, symbol: &str, dim: usize) -> Self {
        deterministic_fhrr_vector(base_seed, symbol, dim)
    }
}

impl DeterministicSymbol for GHRRVector {
    fn generate_symbol(base_seed: u64, symbol: &str, dim: usize) -> Self {
        deterministic_ghrr_vector_seed(base_seed, symbol, dim)
    }
}

impl DeterministicSymbol for IndexVector {
    fn generate_symbol(base_seed: u64, symbol: &str, dim: usize) -> Self {
        IndexVector::new(deterministic_binary_hd_vector(base_seed, symbol, dim))
    }
}

/// Item Memory (Symbol Table) for Vector Symbolic Architectures.
///
/// An `ItemMemory` associates discrete symbolic entities (e.g. words, field names,
/// categories, atoms) with quasi-orthogonal high-dimensional vectors.
///
/// When an unknown symbol is queried via [`ItemMemory::get_or_create`], a deterministic
/// hypervector is automatically generated using the configured base seed.
///
/// # Example
///
/// ```
/// use guddalm_vsa::hdc::item_memory::ItemMemory;
/// use guddalm_vsa::hdc::vector::HDVector;
/// use guddalm_vsa::hdc::vsa_trait::VsaVector;
///
/// let mut mem: ItemMemory<HDVector> = ItemMemory::new(10000);
/// let country = mem.get_or_create("country");
/// let usa = mem.get_or_create("USA");
/// let capital = mem.get_or_create("capital");
/// let dc = mem.get_or_create("Washington_DC");
///
/// // Encode record: (country * USA) + (capital * Washington_DC)
/// let record = country.bind(&usa).bundle(&capital.bind(&dc));
///
/// // Unbind with capital key to query the value
/// let query = record.unbind(&capital);
/// let (best_match, similarity) = mem.cleanup(&query).unwrap();
/// assert_eq!(best_match, "Washington_DC");
/// assert!(similarity > 0.5);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ItemMemory<V> {
    symbols: Vec<String>,
    vectors: Vec<V>,
    indices: HashMap<String, usize>,
    dim: usize,
    base_seed: u64,
}

impl<V: DeterministicSymbol> ItemMemory<V> {
    /// Default base seed for deterministic hypervector generation.
    pub const DEFAULT_BASE_SEED: u64 = 42;

    /// Create a new empty item memory with the specified dimension and default seed.
    pub fn new(dim: usize) -> Self {
        Self::with_seed(dim, Self::DEFAULT_BASE_SEED)
    }

    /// Create a new empty item memory with a custom base seed.
    pub fn with_seed(dim: usize, base_seed: u64) -> Self {
        Self {
            symbols: Vec::new(),
            vectors: Vec::new(),
            indices: HashMap::new(),
            dim,
            base_seed,
        }
    }

    /// Get the vector for `symbol`, or deterministically generate and store it if absent.
    pub fn get_or_create(&mut self, symbol: &str) -> V {
        if let Some(&idx) = self.indices.get(symbol) {
            self.vectors[idx].clone()
        } else {
            let vec = V::generate_symbol(self.base_seed, symbol, self.dim);
            self.insert(symbol, vec.clone());
            vec
        }
    }
}

impl<V: VsaVector> ItemMemory<V> {
    /// Dimension of hypervectors in this memory.
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of stored symbols.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// True if no symbols are currently stored.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    /// Get the base seed used for deterministic generation.
    #[inline(always)]
    pub fn base_seed(&self) -> u64 {
        self.base_seed
    }

    /// Check if a symbol is registered in item memory.
    pub fn contains(&self, symbol: &str) -> bool {
        self.indices.contains_key(symbol)
    }

    /// Retrieve the hypervector associated with `symbol`, if present.
    pub fn get(&self, symbol: &str) -> Option<&V> {
        self.indices.get(symbol).map(|&idx| &self.vectors[idx])
    }

    /// Insert an explicit symbol and hypervector. Overwrites if the symbol already exists.
    pub fn insert(&mut self, symbol: impl Into<String>, vector: V) {
        assert_eq!(vector.dim(), self.dim, "Vector dimension mismatch in ItemMemory");
        let sym = symbol.into();
        if let Some(&idx) = self.indices.get(&sym) {
            self.vectors[idx] = vector;
        } else {
            let idx = self.symbols.len();
            self.indices.insert(sym.clone(), idx);
            self.symbols.push(sym);
            self.vectors.push(vector);
        }
    }

    /// Clean up a noisy or unbound vector by finding the symbol with maximum cosine similarity.
    ///
    /// Returns `Some((symbol_name, similarity))` or `None` if memory is empty.
    pub fn cleanup(&self, query: &V) -> Option<(String, f64)> {
        if self.vectors.is_empty() {
            None
        } else {
            let mut best_idx = 0;
            let mut best_sim = f64::NEG_INFINITY;

            for (i, v) in self.vectors.iter().enumerate() {
                let sim = query.cosine_similarity(v);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }

            Some((self.symbols[best_idx].clone(), best_sim))
        }
    }

    /// Return the top `k` most similar symbols sorted in descending order of similarity.
    pub fn cleanup_top_k(&self, query: &V, k: usize) -> Vec<(String, f64)> {
        if self.vectors.is_empty() || k == 0 {
            Vec::new()
        } else {
            let mut scores: Vec<(String, f64)> = self.symbols
                .iter()
                .zip(self.vectors.iter())
                .map(|(s, v)| (s.clone(), query.cosine_similarity(v)))
                .collect();

            // Sort descending by similarity score
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.truncate(k);
            scores
        }
    }

    /// Slice of all registered symbol names.
    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }

    /// Slice of all registered hypervectors.
    pub fn vectors(&self) -> &[V] {
        &self.vectors
    }
}

impl<V: DeterministicSymbol> ItemMemory<V> {
    /// Encode a key-value record of strings into a composite hypervector.
    ///
    /// Computes: $\bigoplus_i (\text{field}_i \otimes \text{value}_i)$
    pub fn encode_record(&mut self, fields: &[(&str, &str)]) -> V {
        if fields.is_empty() {
            V::zero(self.dim)
        } else {
            let mut result = V::zero(self.dim);
            for &(field, value) in fields {
                let k = self.get_or_create(field);
                let v = self.get_or_create(value);
                let bound = k.bind(&v);
                result = result.bundle(&bound);
            }
            result
        }
    }

    /// Encode an ordered sequence of string symbols using positional permutation.
    ///
    /// Computes: $\bigoplus_i \Pi^i(\text{symbol}_i)$
    pub fn encode_sequence(&mut self, sequence: &[&str]) -> V {
        if sequence.is_empty() {
            V::zero(self.dim)
        } else {
            let mut result = V::zero(self.dim);
            for (i, &symbol) in sequence.iter().enumerate() {
                let v = self.get_or_create(symbol);
                let permuted = v.permute(i);
                result = result.bundle(&permuted);
            }
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_item_memory_get_or_create_determinism() {
        let mut mem: ItemMemory<HDVector> = ItemMemory::new(1024);
        let v1 = mem.get_or_create("alpha");
        let v2 = mem.get_or_create("alpha");
        assert_eq!(v1, v2);
        assert_eq!(mem.len(), 1);
        assert!(mem.contains("alpha"));
    }

    #[test]
    fn test_item_memory_cleanup_role_filler() {
        let mut mem: ItemMemory<HDVector> = ItemMemory::new(4096);
        let role = mem.get_or_create("role:agent");
        let filler = mem.get_or_create("entity:robot");
        let distractor = mem.get_or_create("entity:human");

        let bound = role.bind(&filler);
        let query = bound.unbind(&role);

        let (best, sim) = match mem.cleanup(&query) {
            Some(res) => res,
            None => panic!("cleanup failed"),
        };
        assert_eq!(best, "entity:robot");
        assert!(sim > 0.65, "similarity to correct entity must be high (got {sim})");

        let distractor_sim = query.cosine_similarity(&distractor);
        assert!(sim > distractor_sim + 0.4);
    }

    #[test]
    fn test_item_memory_binary_top_k() {
        let mut mem: ItemMemory<BinaryHDVector> = ItemMemory::new(4096);
        let s1 = mem.get_or_create("dog");
        let _s2 = mem.get_or_create("cat");
        let _s3 = mem.get_or_create("bird");

        let top = mem.cleanup_top_k(&s1, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "dog");
        assert!((top[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_item_memory_record_encoding() {
        let mut mem: ItemMemory<HDVector> = ItemMemory::new(4096);
        let record = mem.encode_record(&[
            ("country", "France"),
            ("capital", "Paris"),
            ("currency", "Euro"),
        ]);

        let capital_key = mem.get_or_create("capital");
        let query = record.unbind(&capital_key);

        let (best, _) = match mem.cleanup(&query) {
            Some(res) => res,
            None => panic!("cleanup failed"),
        };
        assert_eq!(best, "Paris");
    }
}
