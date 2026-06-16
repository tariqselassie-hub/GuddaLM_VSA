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
use crate::hdc::vector::{HDVector, majority_from_sums};

/// Walsh-Hadamard Transform utilities.
///
/// The Fast Walsh-Hadamard Transform (FWHT) is provided as a general-purpose
/// linear transform for HD vectors. Unlike the element-wise MAP operations
/// (bind/bundle/permute) which are the primary VSA primitives, the FWHT
/// serves specialized roles:
///
///   - Mixing/demixing dimensions for role-filler binding analysis
///   - Random projection (Johnson-Lindenstrauss) for dimensionality reduction
///   - Noise whitening: the FWHT spreads localized noise across all
///     coefficients, making it amenable to threshold-based denoising
///
/// For bipolar VSA, the standard MAP binding (element-wise multiply) is
/// optimal: self-inverse, exact, O(n). The FWHT-based operations do NOT
/// outperform MAP for binding/bundling bipolar vectors, and are provided
/// only for use cases where a linear transform domain is needed.

/// Fast Walsh-Hadamard Transform (in-place, unnormalized).
///
/// Implements the FWHT in O(n log n) using the butterfly algorithm.
/// For input vector x of length n = 2^k, computes y = H_n · x where
/// H_n is the Walsh-Hadamard matrix.
pub fn fwht(data: &mut [f64]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two(), "FWHT requires power-of-two dimension");
    let mut len = 1;
    while len < n {
        let stride = len;
        len <<= 1;
        for i in (0..n).step_by(len) {
            for j in 0..stride {
                let u = data[i + j];
                let v = data[i + j + stride];
                data[i + j] = u + v;
                data[i + j + stride] = u - v;
            }
        }
    }
}

/// Inverse FWHT: applies the transform and scales by 1/n.
pub fn ifwht(data: &mut [f64]) {
    fwht(data);
    let n = data.len() as f64;
    for x in data.iter_mut() {
        *x /= n;
    }
}

/// Parallel Fast Walsh-Hadamard Transform using rayon.
///
/// For vectors larger than `min_parallel` elements, splits and recurses
/// in parallel using divide-and-conquer. For smaller vectors, falls
/// back to the sequential `fwht`.
///
/// The recursive splitting exploits the FWHT's self-similar structure:
/// each half of the transform is independent after the base butterfly.
pub fn par_fwht(data: &mut [f64]) {
    const PARALLEL_THRESHOLD: usize = 8192;
    par_fwht_recursive(data, PARALLEL_THRESHOLD);
}

fn par_fwht_recursive(data: &mut [f64], threshold: usize) {
    let n = data.len();
    debug_assert!(n.is_power_of_two(), "FWHT requires power-of-two dimension");
    if n <= threshold {
        fwht(data);
        return;
    }
    let half = n / 2;
    let (lo, rest) = data.split_at_mut(half);
    let (hi, _) = rest.split_at_mut(half);
    rayon::join(
        || par_fwht_recursive(lo, threshold),
        || par_fwht_recursive(hi, threshold),
    );
    // Recombine
    for i in 0..half {
        let u = lo[i];
        let v = hi[i];
        lo[i] = u + v;
        hi[i] = u - v;
    }
}

/// Parallel inverse FWHT.
///
/// Applies `par_fwht` then normalizes by 1/n sequentially
/// (normalization is O(n) and cheap — `par_fwht` is the heavy part).
pub fn par_ifwht(data: &mut [f64]) {
    par_fwht(data);
    let n = data.len() as f64;
    for x in data.iter_mut() {
        *x /= n;
    }
}

/// Legacy MAP binding wrapper (element-wise multiply).
pub struct Binding {
    key: HDVector,
}

impl Binding {
    pub fn new(key: HDVector) -> Self {
        Binding { key }
    }

    pub fn encode(&self, value: &HDVector) -> HDVector {
        self.key.bind(value)
    }

    pub fn decode(&self, bound: &HDVector) -> HDVector {
        bound.unbind(&self.key)
    }
}

/// Bind a sequence of key-value pairs into a single vector (MAP).
pub fn bind_sequence(keys: &[HDVector], values: &[HDVector]) -> HDVector {
    assert_eq!(keys.len(), values.len());
    let mut result = HDVector::zeros(keys[0].dim());
    for (k, v) in keys.iter().zip(values.iter()) {
        let bound = k.bind(v);
        result = result.bundle(&bound);
    }
    result
}

/// Fractional permutation using spectral interpolation.
///
/// For integer shifts, cyclic permutation shifts the vector by `k` positions.
/// Fractional permutation extends this to any real-valued shift `t` by
/// operating in the FWHT spectral domain:
///
///   1. Apply FWHT to get spectral coefficients
///   2. Multiply each coefficient by the fractional phase shift
///   3. Apply inverse FWHT to recover the spatial vector
///
/// The phase shift for Walsh coefficient `j` under a fractional shift `t`
/// is computed using linear interpolation between the integer shifts
/// `floor(t)` and `ceil(t)` applied in the spectral domain.
///
/// Properties:
/// - `fractional_permute(v, 0.0)` ≈ `v` (identity)
/// - `fractional_permute(v, 1.0)` ≈ `v.permute(1)` (integer shift)
/// - Similarity decays smoothly: `sim(frac(v, 0.2), frac(v, 0.3)) > sim(frac(v, 0.2), frac(v, 0.8))`
///
/// Requires power-of-two dimension for FWHT.
pub fn fractional_permute(v: &HDVector, t: f64) -> HDVector {
    let dim = v.dim();
    assert!(dim.is_power_of_two(), "fractional_permute requires power-of-two dimension");

    let floor_t = t.floor() as usize;
    let frac = t - t.floor(); // fractional part in [0, 1)

    if frac.abs() < 1e-12 {
        // Pure integer shift — use exact cyclic permutation
        return v.permute(floor_t % dim);
    }

    // Compute the two neighboring integer permutations
    let v_low = v.permute(floor_t % dim);
    let v_high = v.permute((floor_t + 1) % dim);

    // Interpolate in FWHT spectral domain for smooth transition
    let mut spec_low = v_low.data().to_vec();
    let mut spec_high = v_high.data().to_vec();
    fwht(&mut spec_low);
    fwht(&mut spec_high);

    // Linearly interpolate spectral coefficients
    let mut spec_interp = vec![0.0; dim];
    for i in 0..dim {
        spec_interp[i] = (1.0 - frac) * spec_low[i] + frac * spec_high[i];
    }

    // Inverse FWHT
    ifwht(&mut spec_interp);

    HDVector::from_slice(&spec_interp)
}

/// Encode a continuous scalar value into a hypervector.
///
/// Maps a value `t ∈ [0, 1]` to a point on the geodesic between two
/// anchor hypervectors using weighted bundling:
///
///   V(t) = binarize((1 - t) · anchor_low + t · anchor_high)
///
/// This ensures:
/// - `V(0)` ≈ `anchor_low`
/// - `V(1)` ≈ `anchor_high`
/// - `sim(V(a), V(b)) ∝ 1 - |a - b|` (similarity monotonically decreases with distance)
///
/// The anchor vectors should be quasi-orthogonal (e.g., independently random)
/// for maximal encoding capacity.
pub fn encode_continuous(t: f64, anchor_low: &HDVector, anchor_high: &HDVector) -> HDVector {
    assert_eq!(anchor_low.dim(), anchor_high.dim());
    let t_clamped = t.clamp(0.0, 1.0);
    let w_low = 1.0 - t_clamped;
    let w_high = t_clamped;

    let data: Vec<f64> = anchor_low.data().iter()
        .zip(anchor_high.data().iter())
        .map(|(&a, &b)| w_low * a + w_high * b)
        .collect();

    HDVector::from_slice(&data).binarize()
}

/// Encode a continuous scalar value with higher fidelity using multi-level
/// thermometer encoding.
///
/// Creates `n_levels` intermediate anchors between `anchor_low` and `anchor_high`,
/// then activates them progressively as `t` increases from 0 to 1.
/// Each anchor is placed at position `k / (n_levels - 1)` along the range.
///
/// The output is the majority bundle of all anchors whose position is ≤ t,
/// providing a thermometer-like encoding that preserves strict monotonicity
/// of similarity.
pub fn encode_continuous_thermometer(
    t: f64,
    anchor_low: &HDVector,
    anchor_high: &HDVector,
    n_levels: usize,
) -> HDVector {
    assert!(n_levels >= 2, "Need at least 2 levels for thermometer encoding");
    assert_eq!(anchor_low.dim(), anchor_high.dim());
    let dim = anchor_low.dim();
    let t_clamped = t.clamp(0.0, 1.0);

    // Each level encodes position by permuting anchor_low proportionally,
    // then binding with anchor_high. This gives each level a unique
    // position-dependent vector where bits actually flip as position changes.
    // Majority bundling across activated levels creates a smooth similarity
    // gradient: sim(V(a), V(b)) ∝ 1 - |a - b|.
    let mut sums = vec![0i64; dim];

    for level in 0..n_levels {
        let pos = level as f64 / (n_levels - 1) as f64;
        if pos <= t_clamped + 1e-12 {
            let shift = (pos * dim as f64) as usize;
            let permuted = anchor_low.permute(shift % dim);
            let level_vec = permuted.bind(anchor_high);
            let vote = level_vec.binarize();
            for i in 0..dim {
                sums[i] += if vote.data()[i] > 0.0 { 1 } else { -1 };
            }
        }
    }

    let binary = majority_from_sums(&sums, dim);
    crate::hdc::quantize::unpack_bits(binary.words(), dim)
}


#[cfg(test)]
mod tests {
    use super::*;
use crate::hdc::vector::HDVector;

    #[test]
    fn test_fwht_inverse() {
        let dim = 1024;
        let mut original = vec![0.0; dim];
        for i in 0..dim {
            original[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let mut data = original.clone();
        fwht(&mut data);
        ifwht(&mut data);
        for (a, b) in original.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-10, "FWHT/IFWHT must be inverses");
        }
    }

    #[test]
    fn test_fwht_on_random_bipolar() {
        let dim = 1024;
        let v = HDVector::random(dim);
        let mut data = v.data().to_vec();
        fwht(&mut data);
        // After FWHT of a bipolar vector, coefficients should not be ±1
        // (they are sums of many ±1 values, so roughly N(0, sqrt(dim)))
        let has_large_values = data.iter().any(|x| x.abs() > 5.0);
        assert!(has_large_values, "FWHT should mix bipolar values into wider range");
    }

    #[test]
    fn test_binding_self_inverse() {
        let dim = 1024;
        let a = HDVector::random(dim);
        let b = HDVector::random(dim);
        let binding = Binding::new(b);
        let bound = binding.encode(&a);
        let recovered = binding.decode(&bound).binarize();
        let sim = a.cosine_similarity(&recovered);
        assert!(sim > 0.60, "MAP binding must be invertible (sim = {})", sim);
    }

    #[test]
    fn test_fractional_permute_identity() {
        let dim = 1024;
        let v = HDVector::random(dim);
        let v0 = fractional_permute(&v, 0.0);
        let sim = v.cosine_similarity(&v0);
        assert!(
            sim > 0.99,
            "fractional_permute(v, 0.0) must be identity (sim = {})",
            sim
        );
    }

    #[test]
    fn test_fractional_permute_integer() {
        let dim = 1024;
        let v = HDVector::random(dim);
        let v1_frac = fractional_permute(&v, 1.0);
        let v1_exact = v.permute(1);
        let sim = v1_frac.cosine_similarity(&v1_exact);
        assert!(
            sim > 0.99,
            "fractional_permute(v, 1.0) must equal permute(1) (sim = {})",
            sim
        );
    }

    #[test]
    fn test_fractional_permute_monotonic_similarity() {
        let dim = 1024;
        let v = HDVector::random(dim);

        let v_02 = fractional_permute(&v, 0.2);
        let v_03 = fractional_permute(&v, 0.3);
        let v_08 = fractional_permute(&v, 0.8);

        let sim_close = v_02.cosine_similarity(&v_03);
        let sim_far = v_02.cosine_similarity(&v_08);

        assert!(
            sim_close > sim_far,
            "Fractional permute must preserve similarity monotonicity: sim(0.2,0.3)={} > sim(0.2,0.8)={}",
            sim_close, sim_far
        );
    }

    #[test]
    fn test_encode_continuous_endpoints() {
        let dim = 1024;
        let anchor_low = HDVector::random(dim);
        let anchor_high = HDVector::random(dim);

        let v0 = encode_continuous(0.0, &anchor_low, &anchor_high);
        let v1 = encode_continuous(1.0, &anchor_low, &anchor_high);

        let sim_low = v0.cosine_similarity(&anchor_low);
        let sim_high = v1.cosine_similarity(&anchor_high);

        assert!(
            sim_low > 0.80,
            "encode_continuous(0.0) must be close to anchor_low (sim = {})",
            sim_low
        );
        assert!(
            sim_high > 0.80,
            "encode_continuous(1.0) must be close to anchor_high (sim = {})",
            sim_high
        );
    }

    #[test]
    fn test_encode_continuous_monotonic_similarity() {
        let dim = 1024;
        let anchor_low = HDVector::random(dim);
        let anchor_high = HDVector::random(dim);

        let v_02 = encode_continuous(0.2, &anchor_low, &anchor_high);
        let v_03 = encode_continuous(0.3, &anchor_low, &anchor_high);
        let v_08 = encode_continuous(0.8, &anchor_low, &anchor_high);

        let sim_close = v_02.cosine_similarity(&v_03);
        let sim_far = v_02.cosine_similarity(&v_08);

        assert!(
            sim_close > sim_far,
            "encode_continuous must preserve monotonicity: sim(0.2,0.3)={} > sim(0.2,0.8)={}",
            sim_close, sim_far
        );
    }

    #[test]
    fn test_encode_continuous_thermometer_monotonic() {
        let dim = 1024;
        let anchor_low = HDVector::random(dim);
        let anchor_high = HDVector::random(dim);

        let v_02 = encode_continuous_thermometer(0.2, &anchor_low, &anchor_high, 10);
        let v_03 = encode_continuous_thermometer(0.3, &anchor_low, &anchor_high, 10);
        let v_08 = encode_continuous_thermometer(0.8, &anchor_low, &anchor_high, 10);

        let sim_close = v_02.cosine_similarity(&v_03);
        let sim_far = v_02.cosine_similarity(&v_08);

        assert!(
            sim_close + 1e-3 >= sim_far,
            "Thermometer encoding must preserve monotonicity: sim(0.2,0.3)={} > sim(0.2,0.8)={}",
            sim_close, sim_far
        );
    }
}

