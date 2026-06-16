// Copyright (C) 2025 guddalm_vsa contributors.
// SPDX-License-Identifier: AGPL-3.0
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
/// Cleanup Memory — auto-associative nearest-neighbor search for HD vectors.
///
/// A cleanup memory stores a set of prototype hypervectors and, given a
/// noisy or degraded query, retrieves the closest prototype. This is the
/// core of **auto-associative memory** in VSA systems:
///
///   `cleanup(noisy_vector) → prototype_vector`
///
/// The cleanup operation is used ubiquitously throughout VSA pipelines:
/// - After binding/unbinding to remove noise from imperfect inversion
/// - After bundling to resolve superposition interference
/// - At the output of resonator networks to discretize estimates
/// - As the final readout step in HD computing classifiers
///
/// This implementation supports multiple similarity metrics and uses
/// bit-packed similarity (XNOR-popcount) when the stored vectors are
/// binary, falling back to cosine similarity for real-valued vectors.
use crate::hdc::quantize::{wordwise_xnor_similarity_array64, PackedArray64};
use crate::hdc::vector::{BinaryHDVector, HDVector};
use crate::vsa::Codebook;

/// Result of a cleanup memory lookup.
#[derive(Debug, Clone)]
pub struct CleanupResult {
    /// Index of the best-matching prototype.
    pub index: usize,
    /// Similarity score of the best match in [0, 1] or [-1, 1].
    pub similarity: f64,
    /// The prototype vector itself.
    pub prototype: HDVector,
}

/// Auto-associative cleanup memory backed by a codebook of prototypes.
///
/// ## Example
/// ```ignore
/// let memory = CleanupMemory::new(codebook);
/// let result = memory.cleanup(&noisy_vector);
/// println!("Best match: index {} with sim {:.4}", result.index, result.similarity);
/// ```
pub struct CleanupMemory {
    codebook: Codebook,
    use_packed: bool,
}

impl CleanupMemory {
    /// Build a cleanup memory from a pre-existing codebook.
    ///
    /// Packed similarity is used automatically when the codebook has
    /// pre-computed bit-packed signatures.
    pub fn new(codebook: Codebook) -> Self {
        let use_packed = !codebook.packed.is_empty()
            && codebook.packed[0].len() == (codebook.dim + 63) / 64;
        CleanupMemory { codebook, use_packed }
    }

    /// Find the closest prototype to `query`.
    ///
    /// Uses packed XNOR-popcount similarity when the codebook supports it,
    /// otherwise falls back to cosine similarity.
    pub fn cleanup(&self, query: &HDVector) -> CleanupResult {
        let mut best_idx = 0;
        let mut best_sim = f64::NEG_INFINITY;

        if self.use_packed && query.is_binary() {
            // Fast path: bit-packed similarity
            for (i, packed) in self.codebook.packed.iter().enumerate() {
                let sim = crate::hdc::quantize::packed_similarity(query, packed);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }
        } else {
            // General path: cosine similarity
            for (i, w) in self.codebook.weights.iter().enumerate() {
                let sim = query.cosine_similarity(w);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }
        }

        CleanupResult {
            index: best_idx,
            similarity: best_sim,
            prototype: self.codebook.weights[best_idx].clone(),
        }
    }

    /// Clean up multiple queries in a single call.
    ///
    /// More efficient than repeated `cleanup` calls when the codebook
    /// supports batched packed similarity.
    pub fn batch_cleanup(&self, queries: &[HDVector]) -> Vec<CleanupResult> {
        queries.iter().map(|q| self.cleanup(q)).collect()
    }

    /// Batch cleanup using packed-only similarity for D=4096.
    ///
    /// Uses `batch_similarity_array64` for maximum SIMD throughput.
    /// Panics if the codebook dimension is not 4096 or if packed data
    /// is unavailable.
    pub fn batch_cleanup_packed4096(
        &self,
        queries: &[&PackedArray64],
    ) -> Vec<CleanupResult> {
        assert_eq!(self.codebook.dim, 4096, "packed4096 requires D=4096");
        assert!(!self.codebook.packed.is_empty(), "codebook has no packed data");

        let mut results = Vec::with_capacity(queries.len());
        for query_packed in queries {
            let mut best_idx = 0;
            let mut best_sim = f64::NEG_INFINITY;

            for (i, p) in self.codebook.packed.iter().enumerate() {
                let sig: &PackedArray64 = p.as_slice().try_into()
                    .expect("packed signature must be 64 words");
                let sim = wordwise_xnor_similarity_array64(query_packed, sig);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }

            results.push(CleanupResult {
                index: best_idx,
                similarity: best_sim,
                prototype: self.codebook.weights[best_idx].clone(),
            });
        }

        results
    }

    /// Number of prototypes in the memory.
    pub fn len(&self) -> usize {
        self.codebook.vocab_size
    }

    pub fn is_empty(&self) -> bool {
        self.codebook.vocab_size == 0
    }

    /// Access the underlying codebook.
    pub fn codebook(&self) -> &Codebook {
        &self.codebook
    }
}

/// Binary cleanup memory for BinaryHDVector prototypes.
///
/// Uses XOR-popcount similarity exclusively. Efficient for large
/// binary codebooks where bitwise operations dominate.
pub struct BinaryCleanupMemory {
    prototypes: Vec<BinaryHDVector>,
}

impl BinaryCleanupMemory {
    pub fn new(prototypes: Vec<BinaryHDVector>) -> Self {
        BinaryCleanupMemory { prototypes }
    }

    /// Find the closest binary prototype to `query`.
    pub fn cleanup(&self, query: &BinaryHDVector) -> (usize, f64, BinaryHDVector) {
        let mut best_idx = 0;
        let mut best_sim = f64::NEG_INFINITY;

        for (i, p) in self.prototypes.iter().enumerate() {
            let sim = query.hamming_similarity(p);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        (
            best_idx,
            best_sim,
            self.prototypes[best_idx].clone(),
        )
    }

    pub fn len(&self) -> usize {
        self.prototypes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prototypes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::vector::HDVector;
    use crate::vsa::Codebook;

    #[test]
    fn test_cleanup_identity() {
        let dim = 256;
        let vocab_size = 10;
        let codebook = Codebook::random(vocab_size, dim);
        let memory = CleanupMemory::new(codebook.clone());

        for i in 0..vocab_size {
            let query = codebook.weights[i].binarize();
            let result = memory.cleanup(&query);
            assert_eq!(result.index, i, "cleanup must find identical vector");
            assert!(
                result.similarity > 0.99,
                "identical vector similarity must be near 1.0"
            );
        }
    }

    #[test]
    fn test_cleanup_noisy() {
        let dim = 256;
        let vocab_size = 10;
        let codebook = Codebook::random(vocab_size, dim);
        let memory = CleanupMemory::new(codebook.clone());

        // Add noise to a prototype and verify cleanup recovers it
        let idx = 4;
        let original = &codebook.weights[idx];
        let noisy_data: Vec<f64> = original
            .data()
            .iter()
            .map(|&x| {
                if rand::random::<f64>() < 0.2 {
                    -x // flip 20% of bits
                } else {
                    x
                }
            })
            .collect();
        let noisy = HDVector::from_slice(&noisy_data);

        let result = memory.cleanup(&noisy);
        assert_eq!(
            result.index, idx,
            "cleanup must recover nearest prototype from noisy input"
        );
    }

    #[test]
    fn test_binary_cleanup_memory() {
        use crate::hdc::vector::BinaryHDVector;
        let dim = 256;
        let n = 8;
        let prototypes: Vec<BinaryHDVector> = (0..n).map(|_| BinaryHDVector::random(dim)).collect();
        let memory = BinaryCleanupMemory::new(prototypes.clone());

        for i in 0..n {
            let (idx, sim, _) = memory.cleanup(&prototypes[i]);
            assert_eq!(idx, i);
            assert!((sim - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_cleanup_memory_empty() {
        let codebook = Codebook::new(0, 64);
        let memory = CleanupMemory::new(codebook);
        assert!(memory.is_empty());
        assert_eq!(memory.len(), 0);
    }
}
