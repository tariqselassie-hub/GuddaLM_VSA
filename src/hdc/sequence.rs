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
/// VSA State Machine / Sequence Learning.
///
/// This module provides tools for learning and predicting sequences of
/// hypervectors. The core idea is to treat each observed vector as a
/// **state** and to accumulate transition statistics as weighted bundles.
///
/// ## How it works
///
/// Given a sequence of observed HD vectors `s₁, s₂, ..., sₙ`:
///
/// 1. For each adjacent pair `(sᵢ, sᵢ₊₁)`, the transition `sᵢ → sᵢ₊₁`
///    is recorded by accumulating `sᵢ₊₁` into a bundle indexed by `sᵢ`.
///
/// 2. To predict the next state from a current state `sᵢ`, we retrieve
///    the accumulated bundle for `sᵢ` and clean it up against the
///    observed state codebook to find the most likely successor.
///
/// This is a **gradient-free, online** sequence learner — no backpropagation,
/// no stored gradients, just incremental bundling.
///
/// ## Extensions
/// - **N-gram models**: use `BundleAccumulator` with position-dependent
///   permutation (via `ngram_encode`) to learn higher-order transitions.
/// - **Weighted decay**: older observations can be downweighted by
///   periodically scaling down accumulator values.
use crate::hdc::cleanup::CleanupMemory;
use crate::hdc::quantize::packed_similarity;
use crate::hdc::stream::BundleAccumulator;
use crate::hdc::vector::HDVector;
use crate::vsa::Codebook;

/// Online sequence learner that accumulates transition statistics.
///
/// Maintains a `BundleAccumulator` for each observed state, collecting
/// successor vectors into a weighted bundle. Given a query state,
/// the accumulated successor bundle is cleaned up against a codebook
/// to predict the next state.
///
/// ## Example
/// ```ignore
/// let mut learner = SequenceLearner::new(codebook);
/// learner.observe(&state_a, &state_b);
/// learner.observe(&state_b, &state_c);
/// let (pred_idx, sim) = learner.predict(&state_b).unwrap();
/// ```
pub struct SequenceLearner {
    /// Codebook of known states for cleanup at prediction time.
    codebook: Codebook,
    /// Accumulators keyed by state index (index in codebook).
    /// Each accumulator bundles observed successor vectors.
    transitions: Vec<BundleAccumulator>,
    /// Total observation count per state.
    counts: Vec<usize>,
}

impl SequenceLearner {
    /// Create a new sequence learner over a given codebook of states.
    ///
    /// The codebook defines the discrete set of observable states.
    /// Transitions are accumulated internally and cleaned up against
    /// this codebook at prediction time.
    pub fn new(codebook: Codebook) -> Self {
        let dim = codebook.dim;
        let vocab_size = codebook.vocab_size;
        let transitions = (0..vocab_size)
            .map(|_| BundleAccumulator::new(dim))
            .collect();
        let counts = vec![0; vocab_size];
        SequenceLearner {
            codebook,
            transitions,
            counts,
        }
    }

    /// Observe a transition `from → to`.
    ///
    /// The `to` vector is accumulated into the bundle for `from`.
    /// To find the index, `from` is cleaned up against the codebook.
    /// Returns the index of `from` in the codebook.
    pub fn observe(&mut self, from: &HDVector, to: &HDVector) -> usize {
        let from_idx = self.cleanup_index(from);
        self.transitions[from_idx].add(to);
        self.counts[from_idx] += 1;
        from_idx
    }

    /// Observe a transition identified by codebook index.
    ///
    /// Faster than `observe` when the caller already knows the index.
    pub fn observe_indexed(&mut self, from_idx: usize, to: &HDVector) {
        self.transitions[from_idx].add(to);
        self.counts[from_idx] += 1;
    }

    /// Predict the next state given `current`.
    ///
    /// Cleans up `current` against the codebook to find the index,
    /// then retrieves the accumulated successor bundle for that index
    /// and cleans it up to find the most likely next state.
    ///
    /// Returns `Some((predicted_index, similarity))` if the current
    /// state was recognized and has observed successors, else `None`.
    pub fn predict(&self, current: &HDVector) -> Option<(usize, f64)> {
        let idx = self.cleanup_index(current);
        self.predict_indexed(idx)
    }

    /// Predict using a known codebook index (avoids cleanup step).
    pub fn predict_indexed(&self, state_idx: usize) -> Option<(usize, f64)> {
        if self.counts[state_idx] == 0 {
            return None;
        }
        let successor_bundle = self.transitions[state_idx].binarize();
        let memory = CleanupMemory::new(self.codebook.clone());
        let result = memory.cleanup(&successor_bundle);
        Some((result.index, result.similarity))
    }

    /// Get the accumulated successor bundle for a state (before cleanup).
    ///
    /// Useful for inspecting raw transition statistics or for composing
    /// predictions across multiple models.
    pub fn successor_bundle(&self, state_idx: usize) -> Option<HDVector> {
        if self.counts[state_idx] == 0 {
            return None;
        }
        Some(self.transitions[state_idx].binarize())
    }

    /// Number of times `state_idx` has been observed as a source state.
    pub fn observation_count(&self, state_idx: usize) -> usize {
        self.counts[state_idx]
    }

    /// Total number of transitions observed across all states.
    pub fn total_observations(&self) -> usize {
        self.counts.iter().sum()
    }

    /// Number of states in the codebook.
    pub fn num_states(&self) -> usize {
        self.codebook.vocab_size
    }

    /// Reset all accumulated transitions.
    pub fn reset(&mut self) {
        for acc in self.transitions.iter_mut() {
            acc.reset();
        }
        self.counts.fill(0);
    }

    /// Find the best-matching codebook index for a query vector.
    fn cleanup_index(&self, query: &HDVector) -> usize {
        let query_bin = if query.is_binary() {
            query.clone()
        } else {
            query.binarize()
        };

        let mut best_idx = 0;
        let mut best_sim = f64::NEG_INFINITY;

        if !self.codebook.packed.is_empty() {
            for (i, packed) in self.codebook.packed.iter().enumerate() {
                let sim = packed_similarity(&query_bin, packed);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }
        } else {
            for (i, w) in self.codebook.weights.iter().enumerate() {
                let sim = query_bin.cosine_similarity(w);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = i;
                }
            }
        }

        best_idx
    }
}

/// N-gram sequence model using position-dependent permutation.
///
/// Encodes n-gram contexts using `ngram_encode`-style binding with
/// permutation, then learns transitions from each n-gram to the next
/// observed element.
///
/// This captures higher-order dependencies that a bigram (first-order)
/// model cannot represent.
///
/// Encode an n-gram sequence into a single HD vector using
/// position-dependent permutation.
///
/// Equivalent to `ngram_encode` from `permute.rs`:
///   result = t₀ ⊛ permute(t₁, 1) ⊛ permute(t₂, 2) ⊛ ... ⊛ permute(tₙ, n)
pub fn encode_ngram(tokens: &[HDVector]) -> HDVector {
    if tokens.is_empty() {
        return HDVector::zeros(0);
    }
    let mut combined = tokens[0].clone();
    for (i, token) in tokens.iter().enumerate().skip(1) {
        let permuted = token.permute(i);
        combined = combined.bind(&permuted);
    }
    combined
}

/// Predict the next token given an n-gram context using a simple
/// similarity-weighted bundle of observed continuations.
///
/// `context` is an n-gram encoded as a single HD vector.
/// `continuations` is a slice of `(context_vector, continuation_vector)` pairs
/// collected from training data.
pub fn predict_from_context(
    context: &HDVector,
    continuations: &[(HDVector, HDVector)],
) -> HDVector {
    if continuations.is_empty() {
        return HDVector::zeros(context.dim());
    }
    let dim = context.dim();
    let mut weighted = vec![0.0; dim];
    for (ctx, cont) in continuations {
        let sim = context.cosine_similarity(ctx);
        for (d, val) in weighted.iter_mut().zip(cont.data().iter()) {
            *d += sim * val;
        }
    }
    HDVector::from_slice(&weighted).binarize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::vector::HDVector;
    use crate::vsa::Codebook;

    #[test]
    fn test_sequence_learner_identity_prediction() {
        let dim = 256;
        let vocab_size = 10;
        let codebook = Codebook::random(vocab_size, dim);

        let mut learner = SequenceLearner::new(codebook.clone());

        // Learn: state 2 → state 5
        let from = &codebook.weights[2];
        let to = &codebook.weights[5];
        learner.observe(from, to);

        // Predict from state 2
        let (pred_idx, sim) = learner.predict(from).unwrap();
        assert_eq!(pred_idx, 5, "must predict state 5 after state 2");
        assert!(sim > 0.3, "prediction similarity must be meaningful");
    }

    #[test]
    fn test_sequence_learner_multi_observation() {
        let dim = 256;
        let vocab_size = 8;
        let codebook = Codebook::random(vocab_size, dim);

        let mut learner = SequenceLearner::new(codebook.clone());

        // Learn: 0 → 1 → 2 → 3
        learner.observe(&codebook.weights[0], &codebook.weights[1]);
        learner.observe(&codebook.weights[1], &codebook.weights[2]);
        learner.observe(&codebook.weights[2], &codebook.weights[3]);

        assert_eq!(learner.observation_count(0), 1);
        assert_eq!(learner.observation_count(1), 1);
        assert_eq!(learner.total_observations(), 3);

        // Predict from state 2
        let (pred_idx, _) = learner.predict(&codebook.weights[2]).unwrap();
        assert_eq!(pred_idx, 3);

        // Predict from state 0
        let (pred_idx, _) = learner.predict(&codebook.weights[0]).unwrap();
        assert_eq!(pred_idx, 1);
    }

    #[test]
    fn test_ngram_encode() {
        let dim = 256;
        let a = HDVector::random(dim);
        let b = HDVector::random(dim);
        let c = HDVector::random(dim);

        let encoded = encode_ngram(&[a.clone(), b.clone(), c.clone()]);
        assert_eq!(encoded.dim(), dim);

        // ngram of different order should differ
        let different = encode_ngram(&[a, c, b]);
        let sim = encoded.cosine_similarity(&different);
        assert!(sim < 0.3, "different n-gram orders must yield different vectors");
    }

    #[test]
    fn test_predict_from_context() {
        let dim = 256;
        let ctx = HDVector::random(dim);
        let similar_ctx = ctx.clone();
        let dissimilar_ctx = HDVector::random(dim);
        let cont_a = HDVector::random(dim);
        let cont_b = HDVector::random(dim);

        let continuations = vec![(similar_ctx, cont_a.clone()), (dissimilar_ctx, cont_b.clone())];
        let prediction = predict_from_context(&ctx, &continuations);

        let sim_to_a = prediction.cosine_similarity(&cont_a);
        let sim_to_b = prediction.cosine_similarity(&cont_b);
        assert!(
            sim_to_a > sim_to_b,
            "prediction must favor continuation paired with similar context"
        );
    }

    #[test]
    fn test_sequence_learner_reset() {
        let dim = 128;
        let codebook = Codebook::random(5, dim);
        let mut learner = SequenceLearner::new(codebook.clone());
        learner.observe(&codebook.weights[0], &codebook.weights[1]);
        assert_eq!(learner.total_observations(), 1);
        learner.reset();
        assert_eq!(learner.total_observations(), 0);
        assert!(learner.predict(&codebook.weights[0]).is_none());
    }
}
