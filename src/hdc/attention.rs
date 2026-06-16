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
/// Multi-Head VSA Attention — batched selective bundling across subspaces.
///
/// This module implements multi-head attention entirely within the VSA
/// algebra, using selective bundling (`selective_bundle`) as the core
/// attention mechanism instead of softmax-weighted sums.
///
/// ## Architecture
///
/// Standard transformer attention computes:
///   `Attention(Q, K, V) = softmax(QK^T / √d) V`
///
/// VSA multi-head attention computes:
///   `output = bundle over heads of selective_bundle(Q_h, K_h, V_h, threshold)`
///
/// Each head projects queries, keys, and values into a subspace via
/// permutation (cyclic shift), runs selective bundling with an SDM-derived
/// threshold, then bundles across heads.
///
/// ## Advantages over softmax attention
/// - **O(n) per head** instead of O(n²) — selective bundling is linear
/// - **No learned projections** — permutation is deterministic and invertible
/// - **Inherently sparse** — threshold gates out irrelevant key-value pairs
/// - **Interpretable** — similarity threshold controls sparsity directly
use crate::hdc::bundle::selective_bundle;
use crate::hdc::sdm::sdm_snr_threshold_bipolar;
use crate::hdc::vector::HDVector;
use crate::vsa::VsaEngine;

/// Multi-head attention configuration.
pub struct MultiHeadAttention {
    /// Number of parallel attention heads.
    pub num_heads: usize,
    /// Dimensionality of the query/key/value space.
    pub dim: usize,
    /// Similarity threshold per head (typically SDM-optimal).
    pub thresholds: Vec<f64>,
    /// Per-head permutation engine for subspace projection.
    pub engine: VsaEngine,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention module.
    ///
    /// Each head uses a different permutation step for its subspace
    /// projection. The similarity threshold for each head is set to
    /// the SDM-optimal SNR threshold for `dim` by default.
    pub fn new(num_heads: usize, dim: usize) -> Self {
        let engine = VsaEngine::new(dim);
        let threshold = sdm_snr_threshold_bipolar(dim);
        let thresholds = vec![threshold; num_heads];
        MultiHeadAttention {
            num_heads,
            dim,
            thresholds,
            engine,
        }
    }

    /// Create heads with custom per-head thresholds.
    pub fn with_thresholds(num_heads: usize, dim: usize, thresholds: Vec<f64>) -> Self {
        assert_eq!(thresholds.len(), num_heads);
        let engine = VsaEngine::new(dim);
        MultiHeadAttention {
            num_heads,
            dim,
            thresholds,
            engine,
        }
    }

    /// Forward pass: apply multi-head attention.
    ///
    /// For each head `h`:
    ///   1. Project Q, K, V by permuting by `h` positions
    ///   2. Run `selective_bundle(Q_h, K_h, V_h, threshold_h)`
    ///   3. Unpermute the result back to the original space
    ///
    /// All head outputs are bundled into a single output vector.
    ///
    /// ## Parameters
    /// - `query`: the query vector (single)
    /// - `keys`: slice of key vectors
    /// - `values`: slice of value vectors (same length as keys)
    ///
    /// ## Returns
    /// A single HDVector representing the multi-head attention output.
    pub fn forward(
        &self,
        query: &HDVector,
        keys: &[HDVector],
        values: &[HDVector],
    ) -> HDVector {
        assert_eq!(keys.len(), values.len(), "keys and values must have same length");
        let mut output = HDVector::zeros(self.dim);

        for head in 0..self.num_heads {
            let threshold = self.thresholds[head];
            let step = head + 1; // each head gets a different shift

            // Project query into head subspace
            let q_proj = self.engine.permute(query, step);

            // Project keys and values (in bulk)
            let k_proj: Vec<HDVector> = keys
                .iter()
                .map(|k| self.engine.permute(k, step))
                .collect();
            let v_proj: Vec<HDVector> = values
                .iter()
                .map(|v| self.engine.permute(v, step))
                .collect();

            // Selective bundle in head subspace
            let head_out_proj = selective_bundle(&q_proj, &k_proj, &v_proj, threshold);

            // Unpermute back to original space and accumulate
            let head_out = self.engine.unpermute(&head_out_proj, step);
            output = output.bundle(&head_out);
        }

        output.binarize()
    }

    /// Batched forward: process multiple queries against the same keys/values.
    ///
    /// More efficient than repeated `forward` calls because key/value
    /// projection is shared across all queries.
    pub fn forward_batch(
        &self,
        queries: &[HDVector],
        keys: &[HDVector],
        values: &[HDVector],
    ) -> Vec<HDVector> {
        // Pre-project keys and values for each head (shared across queries)
        let k_proj_per_head: Vec<Vec<HDVector>> = (0..self.num_heads)
            .map(|head| {
                let step = head + 1;
                keys.iter()
                    .map(|k| self.engine.permute(k, step))
                    .collect()
            })
            .collect();

        let v_proj_per_head: Vec<Vec<HDVector>> = (0..self.num_heads)
            .map(|head| {
                let step = head + 1;
                values
                    .iter()
                    .map(|v| self.engine.permute(v, step))
                    .collect()
            })
            .collect();

        queries
            .iter()
            .map(|query| {
                let mut output = HDVector::zeros(self.dim);
                for head in 0..self.num_heads {
                    let step = head + 1;
                    let q_proj = self.engine.permute(query, step);
                    let head_out_proj =
                        selective_bundle(&q_proj, &k_proj_per_head[head], &v_proj_per_head[head], self.thresholds[head]);
                    let head_out = self.engine.unpermute(&head_out_proj, step);
                    output = output.bundle(&head_out);
                }
                output.binarize()
            })
            .collect()
    }
}

/// Single-head selective attention (convenience wrapper).
///
/// Equivalent to `selective_bundle` with the SDM-optimal SNR threshold
/// for the given dimension. This is the VSA analog of single-head
/// dot-product attention.
pub fn attention(
    query: &HDVector,
    keys: &[HDVector],
    values: &[HDVector],
) -> HDVector {
    let threshold = sdm_snr_threshold_bipolar(query.dim());
    selective_bundle(query, keys, values, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::vector::HDVector;

    #[test]
    fn test_single_head_attention() {
        let dim = 1024;
        let n_pairs = 8;

        let query = HDVector::random(dim);
        let keys: Vec<HDVector> = (0..n_pairs).map(|_| HDVector::random(dim)).collect();
        let values: Vec<HDVector> = (0..n_pairs).map(|_| HDVector::random(dim)).collect();

        let result = attention(&query, &keys, &values);
        assert_eq!(result.dim(), dim);
    }

    #[test]
    fn test_multi_head_attention_forward() {
        let dim = 512;
        let num_heads = 4;
        let n_pairs = 6;

        let mha = MultiHeadAttention::new(num_heads, dim);
        let query = HDVector::random(dim);
        let keys: Vec<HDVector> = (0..n_pairs).map(|_| HDVector::random(dim)).collect();
        let values: Vec<HDVector> = (0..n_pairs).map(|_| HDVector::random(dim)).collect();

        let output = mha.forward(&query, &keys, &values);
        assert_eq!(output.dim(), dim);
    }

    #[test]
    fn test_multi_head_batch() {
        let dim = 256;
        let num_heads = 2;
        let n_queries = 3;
        let n_pairs = 5;

        let mha = MultiHeadAttention::new(num_heads, dim);
        let queries: Vec<HDVector> = (0..n_queries).map(|_| HDVector::random(dim)).collect();
        let keys: Vec<HDVector> = (0..n_pairs).map(|_| HDVector::random(dim)).collect();
        let values: Vec<HDVector> = (0..n_pairs).map(|_| HDVector::random(dim)).collect();

        let outputs = mha.forward_batch(&queries, &keys, &values);
        assert_eq!(outputs.len(), n_queries);
        for out in &outputs {
            assert_eq!(out.dim(), dim);
        }
    }

    #[test]
    fn test_similar_query_preferred() {
        let dim = 512;
        let query = HDVector::random(dim);
        let similar_key = query.clone();
        let dissimilar_key = HDVector::random(dim);
        let value_for_similar = HDVector::random(dim);
        let value_for_dissimilar = HDVector::random(dim);

        let keys = vec![similar_key, dissimilar_key];
        let values = vec![value_for_similar.clone(), value_for_dissimilar.clone()];

        let result = attention(&query, &keys, &values);
        let sim_to_preferred = result.cosine_similarity(&value_for_similar);
        let sim_to_other = result.cosine_similarity(&value_for_dissimilar);

        assert!(
            sim_to_preferred > sim_to_other,
            "attention must favor value paired with similar key"
        );
    }
}
