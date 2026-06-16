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
/// Sparse Distributed Memory (SDM) — optimal radius calculations.
///
/// SDM operates in N-dimensional binary space. Given a read address (query),
/// it activates all memory locations within a Hamming distance radius d*.
/// The optimal d* depends on the optimization target:
///
///   d*_SNR  — maximizes signal-to-noise ratio for readout
///   d*_Mem  — maximizes memory capacity (number of storable items)
///   d*_Cent — centroid radius, aligns with majority-rule bundling
///
/// These radii are calibrated to the statistics of random HD vectors in
/// N-dimensional space, where the expected Hamming distance between any
/// two random vectors is N/2 with variance N/4.

use crate::hdc::vector::{BinaryHDVector, HDVector};

/// Optimal Hamming distance radius for maximum signal-to-noise ratio (SNR).
///
/// Derived from the intersection of two hyperspheres in binary space:
///   d*_SNR = N/2 - sqrt(N/2)
///
/// This maximizes the ratio of true-positive to false-positive activations
/// when reading from SDM. At this radius, the activated region captures
/// the dense cluster of points near the address while excluding the
/// uniform background of random points.
pub fn optimal_snr_hamming_radius(dim: usize) -> f64 {
    (dim as f64 / 2.0) - (dim as f64 / 2.0).sqrt()
}

/// Optimal Hamming distance radius for maximum memory capacity.
///
///   d*_Mem = N/2
///
/// At this radius, the SDM achieves its maximum theoretical storage
/// capacity. The activated region is exactly half the space, balancing
/// the number of stored items vs. interference between them.
pub fn optimal_mem_hamming_radius(dim: usize) -> f64 {
    dim as f64 / 2.0
}

/// Centroid radius — aligns with majority-rule bundling.
///
///   d*_Cent = N/2 - sqrt(N)
///
/// When bundling k vectors, the majority vote converges to the correct
/// prototype within this radius. This is the "noise floor" for bundling.
pub fn optimal_centroid_hamming_radius(dim: usize) -> f64 {
    (dim as f64 / 2.0) - (dim as f64).sqrt()
}

/// Convert a Hamming distance threshold to a [0,1] similarity threshold
/// used in bipolar cosine similarity comparisons.
///
/// For Hamming distance d* in binary N-space, the equivalent bipolar
/// similarity threshold is:
///   sim = 1 - 2*d*/N  (since norm=1 for bipolar, cos_sim = 1 - 2*hamming_dist)
pub fn hamming_distance_to_bipolar_sim(d_star: f64, dim: usize) -> f64 {
    let n = dim as f64;
    1.0 - 2.0 * d_star / n
}

/// Convert a Hamming distance threshold to a [0,1] Hamming similarity
/// threshold used in binary comparisons.
///
///   hamming_sim = 1 - d*/N
pub fn hamming_distance_to_binary_sim(d_star: f64, dim: usize) -> f64 {
    let n = dim as f64;
    1.0 - d_star / n
}

/// Compute the SDM-optimal SNR similarity threshold for bipolar vectors.
///
/// Returns the cosine similarity threshold that maximizes signal-to-noise
/// when reading from an SDM with N-dimensional bipolar vectors.
pub fn sdm_snr_threshold_bipolar(dim: usize) -> f64 {
    let d_star = optimal_snr_hamming_radius(dim);
    hamming_distance_to_bipolar_sim(d_star, dim)
}

/// Compute the SDM-optimal SNR similarity threshold for binary vectors.
pub fn sdm_snr_threshold_binary(dim: usize) -> f64 {
    let d_star = optimal_snr_hamming_radius(dim);
    hamming_distance_to_binary_sim(d_star, dim)
}

/// Compute the SDM-optimal memory capacity threshold for bipolar vectors.
pub fn sdm_mem_threshold_bipolar(dim: usize) -> f64 {
    let d_star = optimal_mem_hamming_radius(dim);
    hamming_distance_to_bipolar_sim(d_star, dim)
}

/// Compute the SDM-optimal memory capacity threshold for binary vectors.
pub fn sdm_mem_threshold_binary(dim: usize) -> f64 {
    let d_star = optimal_mem_hamming_radius(dim);
    hamming_distance_to_binary_sim(d_star, dim)
}

/// SDM-optimized read operation: given a query, retrieve values whose keys
/// fall within the optimal SNR radius. This mirrors softmax attention by
/// weighting values exponentially by their proximity to the query, but
/// uses only bitwise operations and a pre-calibrated threshold.
pub fn sdm_read_bipolar(
    query: &HDVector,
    keys: &[HDVector],
    values: &[HDVector],
) -> HDVector {
    let dim = query.dim();
    let threshold = sdm_snr_threshold_bipolar(dim);
    let mut data = vec![0.0; dim];

    for (k, v) in keys.iter().zip(values.iter()) {
        let sim = query.cosine_similarity(k);
        if sim > threshold {
            let weight = (sim - threshold) / (1.0 - threshold);
            for (d, val) in data.iter_mut().zip(v.data().iter()) {
                *d += weight * val;
            }
        }
    }

    HDVector::from_slice(&data).binarize()
}

/// SDM-optimized binary read: uses XOR + popcount for Hamming distance,
/// and reads only from addresses within the optimal SNR Hamming radius.
pub fn sdm_read_binary(
    query: &BinaryHDVector,
    keys: &[BinaryHDVector],
    values: &[BinaryHDVector],
) -> BinaryHDVector {
    let dim = query.dim();
    let threshold = sdm_snr_threshold_binary(dim);
    let mut sums = vec![0i64; dim];

    for (k, v) in keys.iter().zip(values.iter()) {
        let sim = query.hamming_similarity(k);
        if sim > threshold {
            for i in 0..dim {
                let bit = (v.words()[i / 64] >> (i % 64)) & 1;
                sums[i] += if bit == 1 { 1 } else { -1 };
            }
        }
    }

    crate::hdc::vector::majority_from_sums(&sums, dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::vector::HDVector;

    #[test]
    fn test_optimal_radii_are_positive() {
        let dim = 10000;
        assert!(optimal_snr_hamming_radius(dim) > 0.0);
        assert!(optimal_mem_hamming_radius(dim) > 0.0);
        assert!(optimal_centroid_hamming_radius(dim) > 0.0);
    }

    #[test]
    fn test_snr_radius_less_than_mem_radius() {
        let dim = 10000;
        assert!(
            optimal_snr_hamming_radius(dim) < optimal_mem_hamming_radius(dim),
            "SNR radius must be tighter (smaller) than memory radius"
        );
    }

    #[test]
    fn test_thresholds_are_in_range() {
        let dim = 10000;
        let bip = sdm_snr_threshold_bipolar(dim);
        let bin = sdm_snr_threshold_binary(dim);
        assert!(bip > -1.0 && bip < 1.0, "bipolar threshold out of range: {}", bip);
        assert!(bin > 0.0 && bin < 1.0, "binary threshold out of range: {}", bin);
    }

    #[test]
    fn test_sdm_read_selects_similar_values() {
        let query = HDVector::random(10000);
        let similar = query.clone();
        let dissimilar = HDVector::random(10000);
        let v1 = HDVector::random(10000);
        let v2 = HDVector::random(10000);

        let result = sdm_read_bipolar(&query, &[similar, dissimilar], &[v1.clone(), v2.clone()]);
        let sim_to_v1 = result.cosine_similarity(&v1);
        let sim_to_v2 = result.cosine_similarity(&v2);
        assert!(
            sim_to_v1 > sim_to_v2,
            "SDM read must favor value paired with similar key"
        );
    }
}
