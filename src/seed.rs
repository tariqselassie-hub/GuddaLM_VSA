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
//! # Deterministic Seed & Vector Generation
//!
//! Unified module for generating reproducible hypervectors from string keys.
//! All crates in the GuddaLM workspace should use these functions instead of
//! rolling their own deterministic generators.
//!
//! ## Hash Algorithm
//!
//! Every function uses the same multiplicative hash to combine a base seed
//! with a string key:
//!
//! ```text
//! seed = base_seed + key.bytes().fold(0, |acc, b| acc.wrapping_mul(31).wrapping_add(b))
//! ```
//!
//! This ensures that the same `(base_seed, key)` pair always produces the
//! same vector, regardless of which crate calls it.

use rand::Rng;
use rand::SeedableRng;

use crate::hdc::vector::HDVector;
use crate::hdc::fhrr::FHRRVector;
use crate::hdc::ghrr::GHRRVector;
use crate::hdc::vector::Complex;

// ── Core Seed Computation ────────────────────────────────────

/// Compute a deterministic `u64` seed from a base seed and a string key.
///
/// Uses the same multiplicative hash (`wrapping_mul(31)`) used throughout
/// GuddaLM to guarantee cross-crate reproducibility.
///
/// # Examples
///
/// ```
/// use guddalm_vsa::seed::deterministic_seed;
/// let s1 = deterministic_seed(42, "hello");
/// let s2 = deterministic_seed(42, "hello");
/// assert_eq!(s1, s2);
/// ```
pub fn deterministic_seed(base_seed: u64, key: &str) -> u64 {
    key.bytes()
        .fold(base_seed, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
}

// ── HDVector (MAP / bipolar ±1) ──────────────────────────────

/// Generate a deterministic bipolar (±1) `HDVector` from a base seed and
/// string key.
///
/// Replaces the ad-hoc `deterministic_vector()` in `guddalm_ast::ast` and
/// `get_deterministic_vector()` in `guddalm_security::sentinel`.
///
/// # Examples
///
/// ```
/// use guddalm_vsa::seed::deterministic_hd_vector;
/// let v1 = deterministic_hd_vector(42, "token:hello", 1024);
/// let v2 = deterministic_hd_vector(42, "token:hello", 1024);
/// assert!((v1.cosine_similarity(&v2) - 1.0).abs() < 1e-9);
/// ```
pub fn deterministic_hd_vector(base_seed: u64, key: &str, dim: usize) -> HDVector {
    let seed = deterministic_seed(base_seed, key);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let data: Vec<f64> = (0..dim)
        .map(|_| if rng.gen::<f64>() < 0.5 { 1.0 } else { -1.0 })
        .collect();
    HDVector::from_slice(&data)
}

// ── BinaryHDVector (BSC / bit-packed binary) ──────────────────

/// Generate a deterministic `BinaryHDVector` from a base seed and string key.
///
/// Uses the same multiplicative hash as `deterministic_hd_vector` for
/// cross-representation reproducibility.
///
/// # Examples
///
/// ```
/// use guddalm_vsa::seed::deterministic_binary_hd_vector;
/// let v1 = deterministic_binary_hd_vector(42, "token:hello", 1024);
/// let v2 = deterministic_binary_hd_vector(42, "token:hello", 1024);
/// assert_eq!(v1, v2);
/// ```
pub fn deterministic_binary_hd_vector(base_seed: u64, key: &str, dim: usize) -> crate::hdc::vector::BinaryHDVector {
    let seed = deterministic_seed(base_seed, key);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let bits: Vec<u8> = (0..dim)
        .map(|_| if rng.gen::<f64>() < 0.5 { 1u8 } else { 0u8 })
        .collect();
    crate::hdc::vector::BinaryHDVector::from_bits(&bits)
}

// ── FHRRVector (phase angles) ────────────────────────────────

/// Generate a deterministic `FHRRVector` with uniform phase angles in
/// `[0, 2π)` from a base seed and string key.
///
/// Replaces `get_deterministic_phases()` in `guddalm_security::sentinel`.
pub fn deterministic_fhrr_vector(base_seed: u64, key: &str, dim: usize) -> FHRRVector {
    let seed = deterministic_seed(base_seed, key);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let two_pi = 2.0 * std::f64::consts::PI;
    let phases: Vec<f64> = (0..dim).map(|_| rng.gen::<f64>() * two_pi).collect();
    FHRRVector::from_phases(&phases)
}

/// Generate a deterministic `FHRRVector` with continuous phase angles in
/// `[-max_phase, max_phase)` from a base seed and string key.
///
/// Replaces `get_deterministic_continuous_phases()` in `guddalm_security::sentinel`.
pub fn deterministic_fhrr_continuous(
    base_seed: u64,
    key: &str,
    dim: usize,
    max_phase: f64,
) -> FHRRVector {
    let seed = deterministic_seed(base_seed, key);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let phases: Vec<f64> = (0..dim)
        .map(|_| (rng.gen::<f64>() * 2.0 - 1.0) * max_phase)
        .collect();
    FHRRVector::from_phases(&phases)
}

// ── GHRRVector (unitary 2×2 matrices) ────────────────────────

/// Generate a deterministic `GHRRVector` with random unitary 2×2 blocks
/// from a base seed and string key.
///
/// This is the canonical replacement for `guddalm_vsa::hdc::ghrr::deterministic_ghrr_vector`.
pub fn deterministic_ghrr_vector(base_seed: u64, key: &str, dim: usize) -> GHRRVector {
    let seed = deterministic_seed(base_seed, key);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(dim * 4);
    for _ in 0..dim {
        let theta = rng.gen::<f64>() * std::f64::consts::FRAC_PI_2;
        let psi = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
        let phi1 = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
        let phi2 = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let u00 = Complex {
            re: cos_t * (psi + phi1).cos(),
            im: cos_t * (psi + phi1).sin(),
        };
        let u01 = Complex {
            re: sin_t * (psi + phi2).cos(),
            im: sin_t * (psi + phi2).sin(),
        };
        let u10 = Complex {
            re: -sin_t * (psi - phi2).cos(),
            im: -sin_t * (psi - phi2).sin(),
        };
        let u11 = Complex {
            re: cos_t * (psi - phi1).cos(),
            im: cos_t * (psi - phi1).sin(),
        };

        data.push(u00);
        data.push(u01);
        data.push(u10);
        data.push(u11);
    }
    GHRRVector { dim, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_seed_reproducibility() {
        let s1 = deterministic_seed(42, "hello");
        let s2 = deterministic_seed(42, "hello");
        assert_eq!(s1, s2);

        let s3 = deterministic_seed(42, "world");
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_deterministic_hd_vector_reproducibility() {
        let v1 = deterministic_hd_vector(42, "token:hello", 1024);
        let v2 = deterministic_hd_vector(42, "token:hello", 1024);
        assert!((v1.cosine_similarity(&v2) - 1.0).abs() < 1e-9);

        let v3 = deterministic_hd_vector(42, "token:world", 1024);
        assert!(v1.cosine_similarity(&v3).abs() < 0.15);
    }

    #[test]
    fn test_deterministic_fhrr_vector_reproducibility() {
        let v1 = deterministic_fhrr_vector(42, "phase:test", 512);
        let v2 = deterministic_fhrr_vector(42, "phase:test", 512);
        assert!((v1.cosine_similarity(&v2) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_deterministic_ghrr_vector_reproducibility() {
        let v1 = deterministic_ghrr_vector(42, "ghrr:test", 128);
        let v2 = deterministic_ghrr_vector(42, "ghrr:test", 128);
        assert!((v1.cosine_similarity(&v2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_deterministic_fhrr_continuous() {
        let v = deterministic_fhrr_continuous(42, "time:base", 256, std::f64::consts::PI / 4.0);
        assert_eq!(v.dim(), 256);
    }

    #[test]
    fn test_backward_compat_with_old_security_algorithm() {
        // The security/general sentinel used: fold(base_seed)
        let base_seed: u64 = 1337;
        let key = "ip:192.168.1.1";
        let dim = 256;

        let old_seed = key.bytes().fold(base_seed, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });
        let mut old_rng = rand::rngs::StdRng::seed_from_u64(old_seed);
        let old_data: Vec<f64> = (0..dim)
            .map(|_| if old_rng.gen::<f64>() < 0.5 { 1.0 } else { -1.0 })
            .collect();
        let old_vec = HDVector::from_slice(&old_data);

        let new_vec = deterministic_hd_vector(base_seed, key, dim);
        assert!((old_vec.cosine_similarity(&new_vec) - 1.0).abs() < 1e-9);
    }
}
