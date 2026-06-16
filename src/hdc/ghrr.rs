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
use crate::hdc::vector::Complex;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Generalized Holographic Reduced Representations (GHRR)
///
/// GHRR extends FHRR by replacing U(1) complex scalars with U(m) unitary matrices.
/// This implementation uses m=2 (2x2 complex matrices) for non-commutative binding
/// to naturally encode graph topologies and hierarchical structures.
///
/// Each GHRRVector consists of `dim` independent 2x2 unitary matrix blocks.
///
/// VSA operations:
///   bind(a, b):   element-wise (block-wise) matrix multiplication: a_j * b_j
///   inverse(a):   element-wise (block-wise) conjugate transpose: a_j^*
///   bundle(a, b): element-wise (block-wise) addition, projected to closest unitary matrix
///   similarity:   average real trace product: (1 / (2*D)) * sum(Re(Tr(a_j * b_j^*)))
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GHRRVector {
    pub dim: usize,
    // Flat vector of length dim * 4.
    // Block j elements are at indices [4*j, 4*j + 1, 4*j + 2, 4*j + 3]
    // representing:
    // [[u00, u01],
    //  [u10, u11]]
    pub data: Vec<Complex>,
}

impl GHRRVector {
    /// Create a random GHRR vector with unitary blocks.
    pub fn random(dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(dim * 4);
        for _ in 0..dim {
            let theta = rng.gen::<f64>() * std::f64::consts::FRAC_PI_2; // [0, pi/2]
            let psi = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
            let phi1 = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
            let phi2 = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;

            let cos_t = theta.cos();
            let sin_t = theta.sin();

            let u00 = Complex { re: cos_t * (psi + phi1).cos(), im: cos_t * (psi + phi1).sin() };
            let u01 = Complex { re: sin_t * (psi + phi2).cos(), im: sin_t * (psi + phi2).sin() };
            let u10 = Complex { re: -sin_t * (psi - phi2).cos(), im: -sin_t * (psi - phi2).sin() };
            let u11 = Complex { re: cos_t * (psi - phi1).cos(), im: cos_t * (psi - phi1).sin() };

            data.push(u00);
            data.push(u01);
            data.push(u10);
            data.push(u11);
        }
        GHRRVector { dim, data }
    }

    /// Identity vector: all blocks are 2x2 identity matrices.
    pub fn identity(dim: usize) -> Self {
        let mut data = Vec::with_capacity(dim * 4);
        let one = Complex { re: 1.0, im: 0.0 };
        let zero = Complex { re: 0.0, im: 0.0 };
        for _ in 0..dim {
            data.push(one);
            data.push(zero);
            data.push(zero);
            data.push(one);
        }
        GHRRVector { dim, data }
    }

    /// Dimension (number of blocks).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Raw elements.
    pub fn data(&self) -> &[Complex] {
        &self.data
    }

    /// Non-commutative binding: block-wise matrix multiplication.
    /// bind(A, B)_j = A_j * B_j
    pub fn bind(&self, other: &GHRRVector) -> GHRRVector {
        assert_eq!(self.dim, other.dim);
        let mut data = Vec::with_capacity(self.dim * 4);
        for (a_chunk, b_chunk) in self.data.chunks_exact(4).zip(other.data.chunks_exact(4)) {
            let a00 = a_chunk[0];
            let a01 = a_chunk[1];
            let a10 = a_chunk[2];
            let a11 = a_chunk[3];

            let b00 = b_chunk[0];
            let b01 = b_chunk[1];
            let b10 = b_chunk[2];
            let b11 = b_chunk[3];

            // Row 0
            let c00 = a00.mul(b00).add(a01.mul(b10));
            let c01 = a00.mul(b01).add(a01.mul(b11));
            // Row 1
            let c10 = a10.mul(b00).add(a11.mul(b10));
            let c11 = a10.mul(b01).add(a11.mul(b11));

            data.extend_from_slice(&[c00, c01, c10, c11]);
        }
        GHRRVector { dim: self.dim, data }
    }

    /// Inverse: block-wise conjugate transpose.
    /// inverse(A)_j = A_j^*
    pub fn inverse(&self) -> GHRRVector {
        let mut data = Vec::with_capacity(self.dim * 4);
        for a_chunk in self.data.chunks_exact(4) {
            // [[a00, a01],
            //  [a10, a11]]
            // Conjugate transpose:
            // [[a00*, a10*],
            //  [a01*, a11*]]
            data.extend_from_slice(&[
                a_chunk[0].conj(),
                a_chunk[2].conj(),
                a_chunk[1].conj(),
                a_chunk[3].conj(),
            ]);
        }
        GHRRVector { dim: self.dim, data }
    }

    /// Cosine similarity: normalized real trace of matrix product.
    /// (1 / (2*D)) * sum_j Re(Tr(A_j * B_j^*))
    pub fn cosine_similarity(&self, other: &GHRRVector) -> f64 {
        assert_eq!(self.dim, other.dim);
        let mut trace_sum = 0.0;
        for (a_chunk, b_chunk) in self.data.chunks_exact(4).zip(other.data.chunks_exact(4)) {
            let a00 = a_chunk[0];
            let a01 = a_chunk[1];
            let a10 = a_chunk[2];
            let a11 = a_chunk[3];

            // B_j^* elements (conjugate transpose)
            let b00_conj = b_chunk[0].conj();
            let b01_conj = b_chunk[1].conj();
            let b10_conj = b_chunk[2].conj();
            let b11_conj = b_chunk[3].conj();

            // Tr(A_j * B_j^*)
            // = (a00 * b00_conj + a01 * b01_conj) + (a10 * b10_conj + a11 * b11_conj)
            let term1 = a00.mul(b00_conj).add(a01.mul(b01_conj));
            let term2 = a10.mul(b10_conj).add(a11.mul(b11_conj));
            trace_sum += term1.add(term2).re;
        }
        trace_sum / (2.0 * self.dim as f64)
    }

    /// Gram-Schmidt orthonormalization for a 2x2 matrix block
    fn orthonormalize_block(block: &[Complex]) -> [Complex; 4] {
        // Row 1: v1 = [block[0], block[1]]
        let n1_sq = block[0].re * block[0].re + block[0].im * block[0].im
                  + block[1].re * block[1].re + block[1].im * block[1].im;
        let n1 = n1_sq.sqrt();
        let u00 = if n1 > 1e-9 { Complex { re: block[0].re / n1, im: block[0].im / n1 } } else { Complex { re: 1.0, im: 0.0 } };
        let u01 = if n1 > 1e-9 { Complex { re: block[1].re / n1, im: block[1].im / n1 } } else { Complex { re: 0.0, im: 0.0 } };

        // Row 2: v2 = [block[2], block[3]]
        // Project v2 onto u1: dot = v2 . u1*
        let dot = block[2].mul(u00.conj()).add(block[3].mul(u01.conj()));

        // v2_orth = v2 - dot * u1
        let v2_0 = block[2].sub(dot.mul(u00));
        let v2_1 = block[3].sub(dot.mul(u01));

        // Normalize v2_orth
        let n2_sq = v2_0.re * v2_0.re + v2_0.im * v2_0.im
                  + v2_1.re * v2_1.re + v2_1.im * v2_1.im;
        let n2 = n2_sq.sqrt();
        let u10 = if n2 > 1e-9 { Complex { re: v2_0.re / n2, im: v2_0.im / n2 } } else { Complex { re: -u01.re, im: u01.im } };
        let u11 = if n2 > 1e-9 { Complex { re: v2_1.re / n2, im: v2_1.im / n2 } } else { Complex { re: u00.re, im: -u00.im } };

        [u00, u01, u10, u11]
    }

    /// Project this vector back onto the unitary group group (block-wise orthonormalization).
    pub fn project(&self) -> Self {
        let mut data = Vec::with_capacity(self.dim * 4);
        for chunk in self.data.chunks_exact(4) {
            let orth = Self::orthonormalize_block(chunk);
            data.extend_from_slice(&orth);
        }
        GHRRVector { dim: self.dim, data }
    }

    /// Bundle two GHRR vectors: block-wise matrix addition followed by unitary projection.
    pub fn bundle(&self, other: &GHRRVector) -> GHRRVector {
        assert_eq!(self.dim, other.dim);
        let mut data = Vec::with_capacity(self.dim * 4);
        for (a_chunk, b_chunk) in self.data.chunks_exact(4).zip(other.data.chunks_exact(4)) {
            // Sum elements
            let raw_block = [
                a_chunk[0].add(b_chunk[0]),
                a_chunk[1].add(b_chunk[1]),
                a_chunk[2].add(b_chunk[2]),
                a_chunk[3].add(b_chunk[3]),
            ];
            let orth = Self::orthonormalize_block(&raw_block);
            data.extend_from_slice(&orth);
        }
        GHRRVector { dim: self.dim, data }
    }

    /// Weighted bundling of multiple GHRR vectors.
    pub fn bundle_weighted(vectors: &[GHRRVector], weights: &[f64]) -> GHRRVector {
        if vectors.is_empty() {
            return GHRRVector::identity(0);
        }
        let dim = vectors[0].dim;
        let mut data = Vec::with_capacity(dim * 4);
        for j in 0..dim {
            let offset = j * 4;
            let mut sum00 = Complex { re: 0.0, im: 0.0 };
            let mut sum01 = Complex { re: 0.0, im: 0.0 };
            let mut sum10 = Complex { re: 0.0, im: 0.0 };
            let mut sum11 = Complex { re: 0.0, im: 0.0 };

            for (k, v) in vectors.iter().enumerate() {
                debug_assert_eq!(v.dim, dim);
                let w = weights[k];
                let w_complex = Complex { re: w, im: 0.0 };
                let chunk = &v.data[offset..offset + 4];
                sum00 = sum00.add(chunk[0].mul(w_complex));
                sum01 = sum01.add(chunk[1].mul(w_complex));
                sum10 = sum10.add(chunk[2].mul(w_complex));
                sum11 = sum11.add(chunk[3].mul(w_complex));
            }

            let raw_block = [sum00, sum01, sum10, sum11];
            let orth = Self::orthonormalize_block(&raw_block);
            data.extend_from_slice(&orth);
        }
        GHRRVector { dim, data }
    }

    /// Construct a GHRRVector from a slice of phase angles (FHRR compatible).
    /// Creates diagonal unitary matrices of form [[e^{iθ}, 0], [0, e^{-iθ}]] per block.
    pub fn from_phases(phases: &[f64]) -> Self {
        let dim = phases.len();
        let mut data = Vec::with_capacity(dim * 4);
        let zero = Complex { re: 0.0, im: 0.0 };
        for &p in phases {
            let cos_p = p.cos();
            let sin_p = p.sin();
            let val = Complex { re: cos_p, im: sin_p };
            let val_conj = val.conj();

            data.push(val);
            data.push(zero);
            data.push(zero);
            data.push(val_conj);
        }
        GHRRVector { dim, data }
    }
}

/// Deterministic GHRR vector generator based on base_seed + string hash.
///
/// **Backward compatibility re-export**: The canonical implementation now
/// lives in [`crate::seed::deterministic_ghrr_vector`].  This function
/// delegates to it so existing call sites continue to compile.
pub fn deterministic_ghrr_vector(base_seed: u64, key: &str, dim: usize) -> GHRRVector {
    crate::seed::deterministic_ghrr_vector(base_seed, key, dim)
}


