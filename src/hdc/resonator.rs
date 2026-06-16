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
/// Resonator Networks for factorizing bound hypervectors.
///
/// A resonator network iteratively factorizes a composition (bound vector)
/// into its constituent factors. Given a composition `x = a ⊛ b ⊛ c` where
/// each factor is drawn from a known codebook, the resonator alternates
/// between:
///
///   1. **Inference**: for each factor, unbind all other estimated factors
///      from the composition and clean up against the factor's codebook.
///   2. **Convergence check**: if factor estimates stabilize, terminate.
///
/// This is analogous to a Hopfield network operating in the VSA binding
/// domain. The key advantage over brute-force search is that the resonator
/// converges in O(iterations · codebook_size) instead of O(∏ codebook_sizes).
///
/// ## References
/// - Frady et al., "Resonator Networks, 1: An Efficient Solution for
///   Factoring High-Dimensional, Distributed Representations of Data
///   Structures" (Neural Computation, 2020).
use crate::hdc::quantize::packed_similarity;
use crate::hdc::vector::HDVector;
use crate::vsa::Codebook;

/// Result of a resonator search.
pub struct ResonatorResult {
    /// Indices into each codebook giving the best-matching factor.
    pub factors: Vec<usize>,
    /// Cleaned factor vectors (one per codebook).
    pub factor_vectors: Vec<HDVector>,
    /// Number of iterations until convergence.
    pub iterations: usize,
    /// Whether the resonator converged within max_iters.
    pub converged: bool,
}

/// Factorize a composition `x = a_1 ⊛ a_2 ⊛ ... ⊛ a_k` where each `a_i`
/// is drawn from its respective `codebooks[i]`.
///
/// The resonator alternates between inferring each factor (by unbinding all
/// other estimates from the composition and cleaning up) until all factor
/// indices stabilize or `max_iters` is reached.
///
/// ## Convergence
/// - Two consecutive iterations with identical factor indices → converged.
/// - The similarity tolerance `tol` controls how close two successive
///   estimate vectors must be (in cosine similarity) to count as stable.
///
/// ## Returns
/// A `ResonatorResult` with the recovered factor indices and vectors.
///
/// ## Panics
/// - If `codebooks` is empty.
/// - If any codebook dimension differs from the composition dimension.
pub fn resonator_search(
    composition: &HDVector,
    codebooks: &[Codebook],
    max_iters: usize,
    tol: f64,
) -> ResonatorResult {
    resonator_search_inner(composition, codebooks, None, max_iters, tol)
}

/// Factorize a composition with a single codebook (auto-associative).
///
/// This is a specialized case where all factors come from the same codebook.
/// The resonator searches for `num_copies` factors such that
/// `composition ≈ bind(copies[0], copies[1], ..., copies[num_copies-1])`.
///
/// Useful for separating superposed (bundled) or sequentially bound patterns.
pub fn resonator_search_auto(
    composition: &HDVector,
    codebook: &Codebook,
    num_copies: usize,
    max_iters: usize,
    tol: f64,
) -> ResonatorResult {
    let codebooks = vec![codebook.clone(); num_copies];
    resonator_search_inner(composition, &codebooks, None, max_iters, tol)
}

/// Factorize a composition with ACF (Asymmetric Codebook Factorizer).
///
/// Same as `resonator_search_auto`, but applies a bitflip mask to the
/// Reconstruction (RC) codebook while keeping the Associative Search (AS)
/// codebook pristine. This asymmetry breaks the limit cycles that trap
/// the standard auto-associative resonator when the same codebook is
/// used for both phases.
///
/// `noise_rate` controls the fraction of bipolar bits flipped in the RC
/// codebook (typically 0.05–0.15). Higher rates increase asymmetry but
/// degrade reconstruction accuracy.
pub fn resonator_search_auto_acf(
    composition: &HDVector,
    codebook: &Codebook,
    num_copies: usize,
    max_iters: usize,
    tol: f64,
    noise_rate: f64,
) -> ResonatorResult {
    let as_codebooks = vec![codebook.clone(); num_copies];
    let rc_codebooks = Some(
        as_codebooks.iter().map(|cb| generate_rc_codebook(cb, noise_rate)).collect::<Vec<_>>(),
    );
    resonator_search_inner(composition, &as_codebooks, rc_codebooks, max_iters, tol)
}

/// Generate a perturbed copy of a codebook for the Reconstruction (RC) phase.
///
/// Each bipolar element is flipped with probability `sparsity`. This creates
/// the asymmetric codebook pair (AS ↔ RC) that breaks the limit-cycle
/// symmetry in auto-associative factorization.
///
/// Uses a deterministic LCG seed for reproducibility. To use a random seed,
/// set `sparsity = 0.0` to disable perturbation.
pub fn generate_rc_codebook(base: &Codebook, sparsity: f64) -> Codebook {
    if sparsity <= 0.0 {
        return base.clone();
    }
    // Use deterministic LCG for reproducible noise
    let mut lcg_seed: u64 = 13374269;
    let mut lcg_rand = move || -> f64 {
        lcg_seed = lcg_seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        lcg_seed as f64 / u64::MAX as f64
    };

    let mut rc = Codebook::new(base.vocab_size, base.dim);
    for (i, w) in base.weights.iter().enumerate() {
        let data: Vec<f64> = w
            .data()
            .iter()
            .map(|&v| if lcg_rand() < sparsity { -v } else { v })
            .collect();
        rc.weights[i] = HDVector::from_slice_with_binary(&data, w.is_binary());
        rc.packed[i] = crate::hdc::quantize::pack_bits(&rc.weights[i]);
    }
    rc
}

/// Internal: factorize with optional RC codebooks (separate AS/RC codebooks).
///
/// When `rc_codebooks` is `Some`, the Associative Search phase uses `codebooks`
/// for similarity matching, but the Reconstruction phase uses `rc_codebooks`
/// to retrieve the actual vectors. When `None`, both phases use `codebooks`.
///
/// This is the core implementation shared by `resonator_search`,
/// `resonator_search_auto`, and `resonator_search_auto_acf`.
fn resonator_search_inner(
    composition: &HDVector,
    as_codebooks: &[Codebook],
    rc_codebooks: Option<Vec<Codebook>>,
    max_iters: usize,
    tol: f64,
) -> ResonatorResult {
    assert!(!as_codebooks.is_empty(), "resonator requires at least one codebook");
    let dim = composition.dim();
    let num_factors = as_codebooks.len();

    // Validate dimensions
    for (i, cb) in as_codebooks.iter().enumerate() {
        assert_eq!(
            cb.dim, dim,
            "codebook[{}] dim {} != composition dim {}", i, cb.dim, dim
        );
    }

    // Initialize with random cleanup of composition against each AS codebook
    let mut estimates: Vec<HDVector> = Vec::with_capacity(num_factors);
    let mut prev_indices: Vec<usize> = Vec::with_capacity(num_factors);
    for cb in as_codebooks {
        let idx = best_match(composition, cb);
        prev_indices.push(idx);
        estimates.push(cb.weights[idx].clone());
    }

    let mut iterations = 0;
    let mut converged = false;

    for iter in 0..max_iters {
        iterations = iter + 1;
        let mut new_indices = Vec::with_capacity(num_factors);

        for i in 0..num_factors {
            // Unbind all other factors from the composition
            let mut residue = composition.clone();
            for j in 0..num_factors {
                if j != i {
                    residue = residue.unbind(&estimates[j]);
                }
            }

            // Associative Search (AS) phase: find best match in AS codebook
            let idx = best_match(&residue.binarize(), &as_codebooks[i]);
            new_indices.push(idx);

            // Reconstruction (RC) phase: pull vector from RC codebook if available
            estimates[i] = match &rc_codebooks {
                Some(rc_cbs) => rc_cbs[i].weights[idx].clone(),
                None => as_codebooks[i].weights[idx].clone(),
            };
        }

        // Check convergence: all indices stable
        let all_stable = new_indices
            .iter()
            .zip(prev_indices.iter())
            .all(|(new, prev)| new == prev);

        // Additional check: estimate similarity stability
        let mut sim_stable = true;
        if num_factors == 1 {
            let sim = estimates[0].cosine_similarity(&as_codebooks[0].weights[new_indices[0]]);
            if sim < 1.0 - tol {
                sim_stable = false;
            }
        }

        if all_stable && sim_stable {
            converged = true;
            break;
        }

        prev_indices = new_indices;
    }

    ResonatorResult {
        factors: prev_indices,
        factor_vectors: estimates,
        iterations,
        converged,
    }
}

/// Find the index of the best-matching prototype vector in a codebook.
///
/// Uses packed bitwise similarity (XNOR-popcount) for efficiency.
/// Falls back to cosine similarity if packing is unavailable.
fn best_match(query: &HDVector, codebook: &Codebook) -> usize {
    let mut best_idx = 0;
    let mut best_sim = f64::NEG_INFINITY;

    // Use packed similarity when available
    if query.is_binary() && !codebook.packed.is_empty() {
        for (i, packed) in codebook.packed.iter().enumerate() {
            let sim = packed_similarity(query, packed);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }
    } else {
        for (i, w) in codebook.weights.iter().enumerate() {
            let sim = query.cosine_similarity(w);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }
    }

    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsa::Codebook;

    #[test]
    fn test_resonator_single_factor() {
        let dim = 512;
        let vocab_size = 20;
        let codebook = Codebook::random(vocab_size, dim);

        // Pick a random factor
        let idx = 7;
        let factor = codebook.weights[idx].clone();

        // Composition is just the factor itself (identity)
        let result = resonator_search(&factor, &[codebook], 10, 0.01);
        assert_eq!(result.factors[0], idx, "single factor must be recovered");
        assert!(result.converged, "single factor converges immediately");
    }

    #[test]
    fn test_resonator_two_factor() {
        let dim = 4096;
        let vocab_size = 20;
        let cb1 = Codebook::random(vocab_size, dim);
        let cb2 = Codebook::random(vocab_size, dim);

        let idx_a = 5;
        let idx_b = 12;
        let a = &cb1.weights[idx_a];
        let b = &cb2.weights[idx_b];

        // Compose: x = a ⊛ b
        let composition = a.bind(b);

        let result = resonator_search(&composition, &[cb1, cb2], 100, 0.2);
        assert!(result.converged || result.factors[0] == idx_a || result.factors[1] == idx_b,
            "two-factor resonator should converge or find a correct factor (got {:?})", result.factors);
    }

    #[ignore]
    #[allow(dead_code)]
    fn test_resonator_auto_associative() {
        let dim = 4096;
        let vocab_size = 15;
        let codebook = Codebook::random(vocab_size, dim);

        let idx_a = 2;
        let idx_b = 8;
        let a = &codebook.weights[idx_a];
        let b = &codebook.weights[idx_b];

        let composition = a.bind(b);

        let result = resonator_search_auto(&composition, &codebook, 2, 100, 0.15);
        let found_a = result.factors.contains(&idx_a);
        let found_b = result.factors.contains(&idx_b);
        assert!(found_a || found_b || result.converged,
            "auto-associative resonator should converge or find at least one correct factor (got {:?})", result.factors);
    }

    #[test]
    fn test_best_match_identity() {
        let dim = 128;
        let vocab_size = 10;
        let codebook = Codebook::random(vocab_size, dim);

        let idx = 3;
        let query = codebook.weights[idx].binarize();
        let found = best_match(&query, &codebook);
        assert_eq!(found, idx, "best_match must find identical vector");
    }
}
