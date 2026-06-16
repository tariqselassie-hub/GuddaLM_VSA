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
/// Fourier Holographic Reduced Representations (FHRR)
///
/// FHRR uses complex numbers on the unit circle. Each element is a phase
/// angle θ ∈ [0, 2π), represented as e^{iθ}.
///
/// Key properties over MAP (bipolar ±1) and BSC (binary 0/1):
///   - Capacity: ~D vectors per bundle (vs ~D/4 for MAP, ~D/8 for BSC)
///   - Binding: phase addition (not self-inverse; inverse = conjugate)
///   - Bundling: complex sum → normalize to unit circle
///   - Similarity: average of cos(Δθ) — continuous, not quantized
///
/// VSA operations:
///   bind(a, b):   θ_i = (a.θ_i + b.θ_i) mod 2π
///   inverse(a):   θ_i = (2π - a.θ_i) mod 2π
///   bundle(a, b): z_i = e^{i a.θ_i} + e^{i b.θ_i} → θ'_i = atan2(im, re)
///   similarity:   (1/D) Σ cos(a.θ_i - b.θ_i)
///   permute:      cyclic shift of phase array
use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::hdc::vector::{HDVector, Complex, fft};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FHRRVector {
    dim: usize,
    phases: Vec<f64>,
}

impl FHRRVector {
    /// Create a random FHRR vector with uniform phase angles in [0, 2π).
    pub fn random(dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let phases: Vec<f64> = (0..dim).map(|_| rng.gen::<f64>() * 2.0 * std::f64::consts::PI).collect();
        FHRRVector { dim, phases }
    }

    /// Create a random FHRR vector with uniform phase angles in [-max_phase, max_phase)
    /// to serve as a base vector for continuous/fractional positional encoding.
    pub fn random_continuous(dim: usize, max_phase: f64) -> Self {
        let mut rng = rand::thread_rng();
        let phases: Vec<f64> = (0..dim)
            .map(|_| (rng.gen::<f64>() * 2.0 - 1.0) * max_phase)
            .collect();
        FHRRVector { dim, phases }
    }

    /// Create an FHRR vector from a slice of phase angles (radians).
    pub fn from_phases(phases: &[f64]) -> Self {
        let dim = phases.len();
        let normalized: Vec<f64> = phases.iter().map(|&p| p.rem_euclid(2.0 * std::f64::consts::PI)).collect();
        FHRRVector { dim, phases: normalized }
    }

    /// Zero vector: all phases = 0 (i.e., e^{i0} = 1 + 0i).
    pub fn zeros(dim: usize) -> Self {
        FHRRVector { dim, phases: vec![0.0; dim] }
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Phase angles (read-only).
    pub fn phases(&self) -> &[f64] {
        &self.phases
    }

    /// Bind two FHRR vectors: phase addition.
    ///
    /// Unlike MAP (self-inverse multiply), FHRR binding adds phases.
    /// Inverse is the complex conjugate (negate phase).
    /// bind(a, b) = bind(b, a) = e^{i(θ_a + θ_b)}
    pub fn bind(&self, other: &FHRRVector) -> FHRRVector {
        assert_eq!(self.dim, other.dim);
        let two_pi = 2.0 * std::f64::consts::PI;
        let phases: Vec<f64> = self.phases.iter().zip(other.phases.iter())
            .map(|(&a, &b)| (a + b).rem_euclid(two_pi))
            .collect();
        FHRRVector { dim: self.dim, phases }
    }

    /// Inverse (complex conjugate): negate all phases.
    /// inverse(a) = e^{-iθ_a}
    /// bind(a, inverse(a)) = e^{i(θ_a - θ_a)} = e^{i0} = identity
    pub fn inverse(&self) -> FHRRVector {
        let two_pi = 2.0 * std::f64::consts::PI;
        let phases: Vec<f64> = self.phases.iter().map(|&p| (two_pi - p).rem_euclid(two_pi)).collect();
        FHRRVector { dim: self.dim, phases }
    }

    /// Bundle two FHRR vectors: complex sum → normalize to unit circle.
    ///
    /// For each dimension:
    ///   z = e^{iθ_a} + e^{iθ_b}
    ///   θ' = atan2(im(z), re(z))
    ///
    /// This is NOT idempotent (bundling a with itself gives a, not a⊕a = a).
    /// Bundling distributes continuously across the unit circle.
    pub fn bundle(&self, other: &FHRRVector) -> FHRRVector {
        assert_eq!(self.dim, other.dim);
        let phases: Vec<f64> = self.phases.iter().zip(other.phases.iter())
            .map(|(&a, &b)| {
                let re = a.cos() + b.cos();
                let im = a.sin() + b.sin();
                im.atan2(re).rem_euclid(2.0 * std::f64::consts::PI)
            })
            .collect();
        FHRRVector { dim: self.dim, phases }
    }

    /// Bundle multiple vectors: accumulate complex sum, then normalize.
    pub fn bundle_all(vectors: &[FHRRVector]) -> FHRRVector {
        if vectors.is_empty() {
            return FHRRVector::zeros(0);
        }
        let dim = vectors[0].dim();
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut sum_re = vec![0.0; dim];
        let mut sum_im = vec![0.0; dim];
        for v in vectors {
            assert_eq!(v.dim, dim);
            for d in 0..dim {
                sum_re[d] += v.phases[d].cos();
                sum_im[d] += v.phases[d].sin();
            }
        }
        let phases: Vec<f64> = sum_re.iter().zip(sum_im.iter())
            .map(|(&re, &im)| im.atan2(re).rem_euclid(two_pi))
            .collect();
        FHRRVector { dim, phases }
    }

    /// Cosine similarity in [0, 1]: (1/D) Σ cos(Δθ).
    ///
    /// Two identical vectors have similarity 1.0.
    /// Two opposite vectors (Δθ = π) have similarity -1.0.
    /// Two random vectors have expected similarity 0.0.
    pub fn cosine_similarity(&self, other: &FHRRVector) -> f64 {
        assert_eq!(self.dim, other.dim);
        let sum: f64 = self.phases.iter().zip(other.phases.iter())
            .map(|(&a, &b)| (a - b).cos())
            .sum();
        sum / self.dim as f64
    }

    /// Fractional power: raise this vector to a real exponent t.
    ///
    /// For FHRR, exponentiation is element-wise phase multiplication:
    ///   (e^{iθ})^t = e^{i(θ·t)}
    ///
    /// This enables continuous positional encoding:
    ///   position(t) = P^t where P is a base position hypervector.
    ///   P^{3.0} and P^{3.5} give smoothly interpolated positions.
    ///
    /// The operation is O(D) — just multiply each phase by t.
    pub fn power(&self, t: f64) -> FHRRVector {
        let two_pi = 2.0 * std::f64::consts::PI;
        let phases: Vec<f64> = self.phases.iter()
            .map(|&p| (p * t).rem_euclid(two_pi))
            .collect();
        FHRRVector { dim: self.dim, phases }
    }

    /// Create a positional encoding vector: P^position.
    ///
    /// Given a base position vector P and a position t (continuous),
    /// returns P^t with phases θ_i * t.
    /// Unlike discrete cyclic shift, this gives smooth interpolation
    /// between positions and works for fractional positions.
    pub fn position(t: f64, base: &FHRRVector) -> FHRRVector {
        base.power(t)
    }

    /// Permute (cyclic shift) the phases by `shift` positions.
    /// Used for discrete positional encoding.
    pub fn permute(&self, shift: usize) -> FHRRVector {
        let shift = shift % self.dim;
        let mut phases = vec![0.0; self.dim];
        for i in 0..self.dim {
            phases[i] = self.phases[(i + shift) % self.dim];
        }
        FHRRVector { dim: self.dim, phases }
    }

    /// Subsample: take the first `new_dim` phases (truncation).
    pub fn subsample(&self, new_dim: usize) -> FHRRVector {
        assert!(new_dim <= self.dim, "subsample must reduce dimension");
        FHRRVector { dim: new_dim, phases: self.phases[..new_dim].to_vec() }
    }

    /// Resample: pad with zeros (phase 0 = real 1.0) up to `new_dim`.
    pub fn resample(&self, new_dim: usize) -> FHRRVector {
        let mut phases = self.phases.clone();
        phases.resize(new_dim, 0.0);
        FHRRVector { dim: new_dim, phases }
    }

    /// Project this FHRRVector to ensure all phase components are normalized in [0, 2π).
    pub fn project(&self) -> FHRRVector {
        let two_pi = 2.0 * std::f64::consts::PI;
        let phases: Vec<f64> = self.phases.iter()
            .map(|&p| p.rem_euclid(two_pi))
            .collect();
        FHRRVector { dim: self.dim, phases }
    }

    /// Project a real-valued spatial vector (HDVector) onto FHRR space in the frequency domain.
    /// Transforms the real vector via FFT and projects onto unit circle to get phase angles.
    pub fn project_real(vec: &HDVector) -> FHRRVector {
        let dim = vec.dim();
        assert!(dim.is_power_of_two(), "FFT projection requires power-of-two dimension");
        let mut complex_data: Vec<Complex> = vec.data().iter()
            .map(|&x| Complex { re: x, im: 0.0 })
            .collect();
        fft(&mut complex_data, false);
        FHRRVector::project_complex(&complex_data)
    }

    /// Project a complex-valued frequency-domain vector onto the complex unit circle
    /// to construct an FHRRVector from the resulting phase angles.
    pub(crate) fn project_complex(complex_data: &[Complex]) -> FHRRVector {
        let dim = complex_data.len();
        let two_pi = 2.0 * std::f64::consts::PI;
        let phases: Vec<f64> = complex_data.iter()
            .map(|c| c.im.atan2(c.re).rem_euclid(two_pi))
            .collect();
        FHRRVector { dim, phases }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fhrr_random_dim() {
        let v = FHRRVector::random(256);
        assert_eq!(v.dim(), 256);
        for &p in v.phases() {
            assert!((0.0..=2.0 * std::f64::consts::PI).contains(&p), "phase must be in [0, 2π)");
        }
    }

    #[test]
    fn test_fhrr_bind_unbind() {
        let a = FHRRVector::random(64);
        let b = FHRRVector::random(64);
        let bound = a.bind(&b);
        let unbound = bound.bind(&b.inverse());
        let sim = a.cosine_similarity(&unbound);
        assert!(sim > 0.99, "bind+unbind should approx recover original (sim={sim})");
    }

    #[test]
    fn test_fhrr_bundle_then_cleanup() {
        let a = FHRRVector::random(64);
        let b = FHRRVector::random(64);
        let bundled = FHRRVector::bundle_all(&[a.clone(), b.clone()]);
        let sim_a = bundled.cosine_similarity(&a);
        let sim_b = bundled.cosine_similarity(&b);
        assert!(sim_a > 0.4 && sim_b > 0.4, "bundle must be similar to both components");
    }

    #[test]
    fn test_fhrr_permute_cycle() {
        let v = FHRRVector::random(256);
        let p1 = v.permute(1);
        let p_cycle = p1.permute(255);
        let sim = v.cosine_similarity(&p_cycle);
        assert!(sim > 0.99, "permute forward+back should recover (sim={sim})");
    }

    #[test]
    fn test_fhrr_similarity_self() {
        let v = FHRRVector::random(128);
        let sim = v.cosine_similarity(&v);
        assert!((sim - 1.0).abs() < 1e-12, "self-similarity must be 1");
    }

    #[test]
    fn test_fhrr_inverse() {
        let v = FHRRVector::random(64);
        let inv = v.inverse();
        let bound = v.bind(&inv);
        let identity = FHRRVector::from_phases(&vec![0.0; 64]);
        let identity_sim = bound.cosine_similarity(&identity);
        assert!(identity_sim > 0.99, "v ⊛ v⁻¹ should be identity (sim={identity_sim})");
    }

    #[test]
    fn test_fhrr_from_phases_zero() {
        let ident = FHRRVector::from_phases(&vec![0.0; 64]);
        for &p in ident.phases() {
            assert!((p - 0.0).abs() < 1e-12, "all phases must be 0");
        }
    }

    #[test]
    fn test_fhrr_resample() {
        let v = FHRRVector::random(64);
        let resampled = v.resample(128);
        assert_eq!(resampled.dim(), 128);
        for i in 0..64 {
            assert!((resampled.phases()[i] - v.phases()[i]).abs() < 1e-12, "first 64 phases unchanged");
        }
        for i in 64..128 {
            assert!((resampled.phases()[i] - 0.0).abs() < 1e-12, "new phases must be 0");
        }
    }

    #[test]
    fn test_fhrr_project() {
        let v = FHRRVector::random(64);
        let projected = v.project();
        let sim = v.cosine_similarity(&projected);
        assert!(sim > 0.99, "project should preserve approx similarity");
    }

    #[test]
    fn test_fhrr_project_real_roundtrip() {
        let dim = 256;
        let vec = HDVector::random(dim);
        let fhrr = FHRRVector::project_real(&vec);
        assert_eq!(fhrr.dim(), dim, "project_real dim stays same");
    }

    #[test]
    fn test_fhrr_bundle_all_single() {
        let v = FHRRVector::random(64);
        let result = FHRRVector::bundle_all(&[v.clone()]);
        let sim = result.cosine_similarity(&v);
        assert!(sim > 0.99, "bundle of one should approx equal itself (sim={sim})");
    }
}
