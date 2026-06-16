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
use ndarray::{Array1, Array2, Axis};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

use crate::hdc::vector::HDVector;

/// Maps 2D coordinates (x, y) into a high-dimensional hyperspace using
/// Random Fourier Features (RFF).  The dot product of two such vectors
/// approximates a Gaussian RBF kernel:
///
///   similarity(pos1, pos2) ≈ exp(-d² / 2σ²)
///
/// # Example
///
/// ```ignore
/// use guddalm_vsa::ContinuousSpaceEncoder;
///
/// let encoder = ContinuousSpaceEncoder::new(1000, 1.0);
/// let pos = encoder.encode_coordinate(0.5, 0.5);
/// assert_eq!(pos.len(), 1000);
/// ```
pub struct ContinuousSpaceEncoder {
    projection_matrix: Array2<f32>,
    dimensions: usize,
}

impl ContinuousSpaceEncoder {
    /// Creates a new encoder.
    ///
    /// `dimensions` — total output dimensionality (must be even).
    /// `bandwidth` — RBF kernel standard deviation σ; larger values make
    ///   correlation decay more slowly with distance (a wider kernel).
    ///
    /// # Panics
    ///
    /// Panics if `dimensions` is odd or `bandwidth` ≤ 0.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let encoder = ContinuousSpaceEncoder::new(10000, 1.0);
    /// ```
    pub fn new(dimensions: usize, bandwidth: f32) -> Self {
        assert!(dimensions % 2 == 0, "Dimensions must be even");
        assert!(bandwidth > 0.0, "Bandwidth must be positive");

        let mut rng = thread_rng();
        let std_dev = 1.0 / bandwidth;
        let normal_dist = Normal::new(0.0, std_dev).expect("std_dev > 0 guaranteed by bandwidth assertion");

        let half_dims = dimensions / 2;
        let mut proj = Array2::zeros((half_dims, 2));
        for val in proj.iter_mut() {
            *val = normal_dist.sample(&mut rng);
        }

        Self {
            projection_matrix: proj,
            dimensions,
        }
    }

    /// Generates the RFF hypervector for coordinate (x, y), L2-normalized.
    ///
    /// Returns an `Array1<f32>` containing concatenated cosines and sines
    /// of the random Fourier features.  The normalized form satisfies
    /// Bochner's theorem: dot products approximate a Gaussian RBF kernel.
    ///
    /// Use this for similarity comparisons (e.g., nearest-neighbour searches).
    /// For element-wise binding with other vectors (e.g., intensity encoding),
    /// use [`encode_coordinate_raw`](Self::encode_coordinate_raw) instead to
    /// avoid signal attenuation from normalization.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let v = encoder.encode_coordinate(0.3, 0.7);
    /// assert!((v.dot(&v) - 1.0).abs() < 1e-5); // unit norm
    /// ```
    pub fn encode_coordinate(&self, x: f32, y: f32) -> Array1<f32> {
        let mut v = self.encode_coordinate_raw(x, y);
        let norm = v.dot(&v).sqrt();
        if norm > 0.0 {
            v /= norm;
        }
        v
    }

    /// Same as `encode_coordinate` but WITHOUT L2 normalization.
    ///
    /// Use this when you intend to bind (element-wise multiply) the
    /// position vector with other vectors (e.g., intensity levels).
    /// L2 normalization would shrink each element to ≈1/√D, and the
    /// product of two normalized vectors would have variance 1/D² —
    /// too weak for reliable binarization after accumulation.
    pub fn encode_coordinate_raw(&self, x: f32, y: f32) -> Array1<f32> {
        let coord = Array1::from_vec(vec![x, y]);
        let frequencies = self.projection_matrix.dot(&coord);

        let half_dims = self.dimensions / 2;
        let mut hdc_vector: Array1<f32> = Array1::zeros(self.dimensions);
        {
            let (mut cos_part, mut sin_part) = hdc_vector.view_mut().split_at(Axis(0), half_dims);
            cos_part.assign(&frequencies.mapv(|f| f.cos()));
            sin_part.assign(&frequencies.mapv(|f| f.sin()));
        }

        hdc_vector
    }

    /// Convenience adapter: returns an `HDVector` (bipolar f64) from an RFF
    /// encoding.  Intended for the Part-A similarity verification test where
    /// the standard `cosine_similarity` on `HDVector` suffices.
    ///
    /// For element-wise binding (used in the demo classification pipeline)
    /// you should work with the raw `Array1<f32>` returned by
    /// `encode_coordinate` directly.
    pub fn encode_hdvector(&self, x: f32, y: f32) -> HDVector {
        let arr = self.encode_coordinate(x, y);
        let data: Vec<f64> = arr.iter().map(|&v| v as f64).collect();
        HDVector::from_slice(&data)
    }

    pub fn dim(&self) -> usize {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ──────────────────────────────────────────────

    #[test]
    fn test_new_valid() {
        let e = ContinuousSpaceEncoder::new(1000, 1.0);
        assert_eq!(e.dim(), 1000);
    }

    #[test]
    fn test_new_multiple_dimensions() {
        for dim in [2, 100, 10000] {
            let e = ContinuousSpaceEncoder::new(dim, 0.5);
            assert_eq!(e.dim(), dim);
        }
    }

    #[test]
    fn test_new_multiple_bandwidths() {
        let e = ContinuousSpaceEncoder::new(100, 0.1);
        assert_eq!(e.dim(), 100);
        let e = ContinuousSpaceEncoder::new(100, 10.0);
        assert_eq!(e.dim(), 100);
        let e = ContinuousSpaceEncoder::new(100, 100.0);
        assert_eq!(e.dim(), 100);
    }

    #[test]
    #[should_panic(expected = "Dimensions must be even")]
    fn test_new_odd_dimension() {
        ContinuousSpaceEncoder::new(999, 1.0);
    }

    #[test]
    #[should_panic(expected = "Bandwidth must be positive")]
    fn test_new_zero_bandwidth() {
        ContinuousSpaceEncoder::new(100, 0.0);
    }

    #[test]
    #[should_panic(expected = "Bandwidth must be positive")]
    fn test_new_negative_bandwidth() {
        ContinuousSpaceEncoder::new(100, -1.0);
    }

    // ── Dimension / shape ────────────────────────────────────────

    #[test]
    fn test_encode_coordinate_output_dim() {
        let e = ContinuousSpaceEncoder::new(5000, 1.0);
        let v = e.encode_coordinate(0.5, 0.5);
        assert_eq!(v.len(), 5000);
    }

    #[test]
    fn test_encode_coordinate_raw_output_dim() {
        let e = ContinuousSpaceEncoder::new(5000, 1.0);
        let v = e.encode_coordinate_raw(0.5, 0.5);
        assert_eq!(v.len(), 5000);
    }

    #[test]
    fn test_encode_hdvector_output_dim() {
        let e = ContinuousSpaceEncoder::new(5000, 1.0);
        let v = e.encode_hdvector(0.5, 0.5);
        assert_eq!(v.dim(), 5000);
    }

    // ── Normalization ─────────────────────────────────────────────

    #[test]
    fn test_normalized_vector_is_unit_norm() {
        let e = ContinuousSpaceEncoder::new(10000, 1.0);
        let v = e.encode_coordinate(0.3, 0.7);
        let norm = v.dot(&v).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Normalized vector should have unit norm, got {}",
            norm
        );
    }

    #[test]
    fn test_raw_vector_not_unit_norm() {
        let e = ContinuousSpaceEncoder::new(10000, 1.0);
        let v = e.encode_coordinate_raw(0.3, 0.7);
        let norm = v.dot(&v).sqrt();
        // Raw cos/sin vector should have norm ≈ sqrt(D/2) ≈ 70.7 for D=10000
        assert!(
            (norm - 70.7).abs() < 10.0,
            "Raw vector norm should be ≈ sqrt(D/2) ≈ 70.7, got {}",
            norm
        );
    }

    // ── Self-similarity ──────────────────────────────────────────

    #[test]
    fn test_self_similarity_is_one() {
        let e = ContinuousSpaceEncoder::new(10000, 1.0);
        let a = e.encode_hdvector(0.5, 0.5);
        let b = e.encode_hdvector(0.5, 0.5);
        let sim = a.cosine_similarity(&b);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Self-similarity must be 1.0, got {}",
            sim
        );
    }

    // ── RBF kernel decay ─────────────────────────────────────────

    #[test]
    fn test_nearby_points_higher_similarity_than_far() {
        let e = ContinuousSpaceEncoder::new(10000, 1.0);
        let origin = e.encode_coordinate(0.0, 0.0);
        let near = e.encode_coordinate(0.05, 0.05);
        let far = e.encode_coordinate(0.5, 0.5);

        let sim_near = origin.dot(&near);
        let sim_far = origin.dot(&far);
        assert!(
            sim_near > sim_far,
            "Nearby points must have higher similarity than far points (near={}, far={})",
            sim_near,
            sim_far
        );
    }

    #[test]
    fn test_monotonic_decay_with_distance() {
        let e = ContinuousSpaceEncoder::new(10000, 1.0);
        let center = e.encode_coordinate(0.0, 0.0);
        let distances = [0.01, 0.05, 0.1, 0.2, 0.5];
        let mut prev_sim = center.dot(&center); // self = 1.0
        for &d in &distances {
            let pt = e.encode_coordinate(d, d);
            let sim = center.dot(&pt);
            assert!(
                sim <= prev_sim + 1e-6,
                "Similarity must monotonically decrease with distance (d={}, sim={}, prev={})",
                d,
                sim,
                prev_sim
            );
            prev_sim = sim;
        }
    }

    #[test]
    fn test_far_points_nearly_orthogonal() {
        let e = ContinuousSpaceEncoder::new(10000, 1.0);
        let a = e.encode_hdvector(0.0, 0.0);
        let b = e.encode_hdvector(5.0, 5.0);
        let sim = a.cosine_similarity(&b);
        assert!(
            sim.abs() < 0.05,
            "Far points should be nearly orthogonal (sim={})",
            sim
        );
    }

    #[test]
    fn test_rbf_approximation_reasonable() {
        // At distance d, the expected RBF similarity is exp(-d²/2).
        // The RFF approximation should be within ±0.1 at close range.
        let e = ContinuousSpaceEncoder::new(10000, 1.0);
        let origin = e.encode_coordinate(0.0, 0.0);
        let distances = [0.01, 0.05, 0.1];
        for &d in &distances {
            let pt = e.encode_coordinate(d, d);
            let sim = origin.dot(&pt);
            let expected = (-(d * d) / 2.0).exp();
            let diff = (sim - expected).abs();
            assert!(
                diff < 0.05,
                "RFF should approximate RBF kernel at d={} (sim={}, expected={}, diff={})",
                d,
                sim,
                expected,
                diff
            );
        }
    }

    // ── Bandwidth effect ─────────────────────────────────────────

    #[test]
    fn test_higher_bandwidth_gives_slower_decay() {
        let narrow = ContinuousSpaceEncoder::new(10000, 0.5);
        let wide = ContinuousSpaceEncoder::new(10000, 2.0);
        let d = 0.2;
        let p = (d, d);
        let sim_narrow = narrow.encode_coordinate(0.0, 0.0).dot(&narrow.encode_coordinate(p.0, p.1));
        let sim_wide = wide.encode_coordinate(0.0, 0.0).dot(&wide.encode_coordinate(p.0, p.1));
        // bandwidth = RBF σ; larger σ → wider kernel → slower decay → higher similarity
        assert!(
            sim_wide > sim_narrow + 0.05,
            "Higher bandwidth (wider kernel) must give slower decay (narrow={}, wide={})",
            sim_narrow,
            sim_wide
        );
    }

    // ── Raw encoding is deterministic per coordinate ─────────────

    #[test]
    fn test_encode_consistency() {
        let e = ContinuousSpaceEncoder::new(1000, 1.0);
        let a = e.encode_coordinate_raw(0.3, 0.7);
        let b = e.encode_coordinate_raw(0.3, 0.7);
        assert_eq!(a, b, "Same coordinate must produce identical encoding");
    }

    // ── Different coordinates produce different vectors ──────────

    #[test]
    fn test_different_coordinates_different_vectors() {
        let e = ContinuousSpaceEncoder::new(1000, 1.0);
        let a = e.encode_coordinate_raw(0.0, 0.0);
        let b = e.encode_coordinate_raw(0.5, 0.5);
        assert_ne!(a, b, "Different coordinates must produce different vectors");
    }
}
