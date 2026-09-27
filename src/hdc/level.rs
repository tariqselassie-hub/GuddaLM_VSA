// Copyright (C) 2025-2026 guddalm_vsa contributors.
// SPDX-License-Identifier: AGPL-3.0
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use crate::hdc::vector::{BinaryHDVector, HDVector};
use crate::hdc::fhrr::FHRRVector;

/// Continuous Scalar and Level Hypervector Encoder.
///
/// Encodes continuous numerical values $x \in [min, max]$ into hypervectors such
/// that metric distance is preserved geometrically:
///
/// $$\text{sim}(V(x_1), V(x_2)) \approx 1.0 - 2.0 \cdot \frac{|x_1 - x_2|}{max - min}$$
///
/// Nearby numerical values yield high cosine/Hamming similarity, smoothly decaying
/// to zero or negative similarity as values diverge across the spectrum.
///
/// # Example
///
/// ```
/// use guddalm_vsa::hdc::level::ScalarEncoder;
/// use guddalm_vsa::hdc::vsa_trait::VsaVector;
///
/// let encoder = ScalarEncoder::new(0.0, 100.0, 4096, 32);
/// let v_20 = encoder.encode(20.0);
/// let v_25 = encoder.encode(25.0);
/// let v_80 = encoder.encode(80.0);
///
/// // v_20 is much closer to v_25 than to v_80
/// assert!(v_20.cosine_similarity(&v_25) > v_20.cosine_similarity(&v_80));
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScalarEncoder {
    min_val: f64,
    max_val: f64,
    dim: usize,
    num_levels: usize,
    /// Bipolar level prototype hypervectors [0..num_levels]
    level_vectors: Vec<HDVector>,
    /// Base phase vector for fractional power encoding
    base_phases: Vec<f64>,
}

impl ScalarEncoder {
    /// Default number of quantization levels (fine-grained resolution).
    pub const DEFAULT_LEVELS: usize = 64;

    /// Create a new scalar encoder over `[min_val, max_val]` with default seed 1337.
    pub fn new(min_val: f64, max_val: f64, dim: usize, num_levels: usize) -> Self {
        Self::with_seed(min_val, max_val, dim, num_levels, 1337)
    }

    /// Create a new scalar encoder with an explicit RNG seed.
    pub fn with_seed(min_val: f64, max_val: f64, dim: usize, num_levels: usize, seed: u64) -> Self {
        assert!(max_val > min_val, "max_val must be greater than min_val");
        assert!(dim >= 64, "dim must be at least 64");
        assert!(num_levels >= 2, "num_levels must be at least 2");

        let mut rng = StdRng::seed_from_u64(seed);

        // 1. Generate base vector V_0 with ±1 entries
        let mut current_data: Vec<f64> = (0..dim)
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();

        // 2. Generate random permutation of indices to progressively flip
        let mut perm: Vec<usize> = (0..dim).collect();
        for i in (1..dim).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i, j);
        }

        let mut level_vectors = Vec::with_capacity(num_levels);
        level_vectors.push(HDVector::from_slice(&current_data));

        // Number of bits to flip per level transition
        let flips_per_step = dim as f64 / (num_levels - 1) as f64;
        let mut flipped_so_far = 0usize;

        for step in 1..num_levels {
            let target_flipped = ((step as f64 * flips_per_step).round() as usize).min(dim);
            while flipped_so_far < target_flipped {
                let idx = perm[flipped_so_far];
                current_data[idx] = -current_data[idx];
                flipped_so_far += 1;
            }
            level_vectors.push(HDVector::from_slice(&current_data));
        }

        // 3. Base phases for continuous fractional power encoding in [-pi, pi]
        let base_phases: Vec<f64> = (0..dim)
            .map(|_| rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI))
            .collect();

        Self {
            min_val,
            max_val,
            dim,
            num_levels,
            level_vectors,
            base_phases,
        }
    }

    /// Encode a scalar value into a bipolar `HDVector`.
    ///
    /// Values outside `[min_val, max_val]` are clamped to the range boundaries.
    pub fn encode(&self, value: f64) -> HDVector {
        let level = self.value_to_level(value);
        self.level_vectors[level].clone()
    }

    /// Encode a scalar value into a bit-packed `BinaryHDVector`.
    pub fn encode_binary(&self, value: f64) -> BinaryHDVector {
        BinaryHDVector::from_bipolar(&self.encode(value))
    }

    /// Continuous fractional power encoding for FHRR complex vectors.
    ///
    /// Preserves exact continuous translational symmetry: $\theta_k = x \cdot \theta_{0, k}$.
    pub fn encode_fractional(&self, value: f64) -> FHRRVector {
        let normalized = (value - self.min_val) / (self.max_val - self.min_val);
        let phases: Vec<f64> = self.base_phases
            .iter()
            .map(|&p0| {
                let p = p0 * normalized * std::f64::consts::PI;
                // Wrap into [-pi, pi]
                (p + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
            })
            .collect();
        FHRRVector::from_phases(&phases)
    }

    /// Approximate decoding: estimates the scalar value corresponding to `query`
    /// by identifying the level hypervector with maximum similarity.
    pub fn decode_approx(&self, query: &HDVector) -> f64 {
        let mut best_level = 0;
        let mut best_sim = f64::NEG_INFINITY;

        for (lvl, v) in self.level_vectors.iter().enumerate() {
            let sim = query.cosine_similarity(v);
            if sim > best_sim {
                best_sim = sim;
                best_level = lvl;
            }
        }

        self.level_to_value(best_level)
    }

    /// Map a numerical value to its discrete level index in `[0, num_levels - 1]`.
    #[inline(always)]
    pub fn value_to_level(&self, value: f64) -> usize {
        let clamped = value.clamp(self.min_val, self.max_val);
        let fraction = (clamped - self.min_val) / (self.max_val - self.min_val);
        let lvl = (fraction * (self.num_levels - 1) as f64).round() as usize;
        lvl.min(self.num_levels - 1)
    }

    /// Map a discrete level index back to the representative scalar value.
    #[inline(always)]
    pub fn level_to_value(&self, level: usize) -> f64 {
        let fraction = level as f64 / (self.num_levels - 1) as f64;
        self.min_val + fraction * (self.max_val - self.min_val)
    }

    /// Number of discrete quantization levels.
    #[inline(always)]
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }

    /// Vector dimensionality.
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Lower bound of the scalar range.
    #[inline(always)]
    pub fn min_val(&self) -> f64 {
        self.min_val
    }

    /// Upper bound of the scalar range.
    #[inline(always)]
    pub fn max_val(&self) -> f64 {
        self.max_val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_encoder_monotonic_similarity() {
        let encoder = ScalarEncoder::new(0.0, 100.0, 4096, 50);

        let v10 = encoder.encode(10.0);
        let v20 = encoder.encode(20.0);
        let v30 = encoder.encode(30.0);
        let v90 = encoder.encode(90.0);

        let sim_10_20 = v10.cosine_similarity(&v20);
        let sim_10_30 = v10.cosine_similarity(&v30);
        let sim_10_90 = v10.cosine_similarity(&v90);

        assert!(sim_10_20 > sim_10_30, "sim(10,20)={sim_10_20} must exceed sim(10,30)={sim_10_30}");
        assert!(sim_10_30 > sim_10_90, "sim(10,30)={sim_10_30} must exceed sim(10,90)={sim_10_90}");
    }

    #[test]
    fn test_scalar_encoder_decode_approx() {
        let encoder = ScalarEncoder::new(0.0, 100.0, 4096, 100);
        let original_val = 42.0;
        let encoded = encoder.encode(original_val);

        let decoded = encoder.decode_approx(&encoded);
        assert!((decoded - original_val).abs() <= 1.0, "Decoded value {decoded} must be within 1 unit of {original_val}");
    }

    #[test]
    fn test_scalar_fractional_encoding() {
        let encoder = ScalarEncoder::new(0.0, 10.0, 2048, 20);
        let f_1 = encoder.encode_fractional(1.0);
        let f_2 = encoder.encode_fractional(2.0);
        let f_9 = encoder.encode_fractional(9.0);

        let sim_1_2 = f_1.cosine_similarity(&f_2);
        let sim_1_9 = f_1.cosine_similarity(&f_9);

        assert!(sim_1_2 > sim_1_9, "fractional encoding must preserve proximity: {sim_1_2} > {sim_1_9}");
    }
}
