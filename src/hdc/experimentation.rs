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
use crate::hdc::vector::{BinaryHDVector, HDVector};

#[derive(Clone)]
pub struct SenojianHybrid {
    pub map: HDVector,
    pub bsc: BinaryHDVector,
}

impl SenojianHybrid {
    pub fn new(map: HDVector, bsc: BinaryHDVector) -> Self {
        assert_eq!(map.dim(), bsc.dim());
        Self { map, bsc }
    }

    pub fn random(dim: usize) -> Self {
        Self {
            map: HDVector::random(dim),
            bsc: BinaryHDVector::random(dim),
        }
    }

    pub fn zeros(dim: usize) -> Self {
        Self {
            map: HDVector::zeros(dim),
            bsc: BinaryHDVector::zeros(dim),
        }
    }

    pub fn dim(&self) -> usize {
        self.map.dim()
    }

    pub fn bind(&self, other: &Self) -> Self {
        Self {
            map: self.map.bind(&other.map),
            bsc: self.bsc.xor_bind(&other.bsc),
        }
    }

    pub fn similarity(&self, other: &Self) -> f64 {
        let sim_map = self.map.cosine_similarity(&other.map);
        let sim_bsc = self.bsc.hamming_similarity(&other.bsc);
        // Average the two similarities
        (sim_map + sim_bsc) / 2.0
    }
}

use crate::hdc::phase_fhrr::CartesianFhrrVector;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SenojianCross {
    pub map: HDVector,
    pub bsc: BinaryHDVector,
}

impl SenojianCross {
    pub fn new(map: HDVector, bsc: BinaryHDVector) -> Self {
        assert_eq!(map.dim(), bsc.dim());
        Self { map, bsc }
    }

    pub fn random(dim: usize) -> Self {
        Self {
            map: HDVector::random(dim),
            bsc: BinaryHDVector::random(dim),
        }
    }

    pub fn zeros(dim: usize) -> Self {
        Self {
            map: HDVector::zeros(dim),
            bsc: BinaryHDVector::zeros(dim),
        }
    }

    pub fn zero(dim: usize) -> Self {
        Self::zeros(dim)
    }

    pub fn from_key(seed: u64, key: &str, dim: usize) -> Self {
        Self {
            map: crate::seed::deterministic_hd_vector(seed, key, dim),
            bsc: crate::seed::deterministic_binary_hd_vector(seed, key, dim),
        }
    }

    pub fn bundle(&self, other: &Self) -> Self {
        Self {
            map: self.map.bundle(&other.map),
            bsc: self.bsc.majority_bundle(&other.bsc),
        }
    }

    pub fn bundle_all(vectors: &[Self]) -> Self {
        if vectors.is_empty() {
            return Self::zeros(0);
        }
        let maps: Vec<&HDVector> = vectors.iter().map(|v| &v.map).collect();
        let bscs: Vec<BinaryHDVector> = vectors.iter().map(|v| v.bsc.clone()).collect();

        let mut map_accum = vec![0.0; vectors[0].dim()];
        for m in maps {
            map_accum = map_accum
                .iter()
                .zip(m.data())
                .map(|(a, b)| *a + *b)
                .collect();
        }

        Self {
            map: HDVector::from_slice(&map_accum),
            bsc: BinaryHDVector::majority_bundle_all(&bscs),
        }
    }

    pub fn weighted_bundle(vectors: &[(&Self, f64)]) -> Self {
        if vectors.is_empty() {
            return Self::zeros(0);
        }
        let dim = vectors[0].0.dim();
        let mut map_accum = vec![0.0; dim];
        let mut bsc_accum = vec![0f64; dim.div_ceil(64)];

        for (v, w) in vectors {
            let map_data = v.map.data();
            let bsc_words = v.bsc.words();
            map_accum = map_accum
                .iter()
                .zip(map_data)
                .map(|(a, b)| *a + b * w)
                .collect();
            for (idx, word) in bsc_words.iter().enumerate() {
                let delta = if *word == 0 { -*w } else { *w };
                bsc_accum[idx] += delta;
            }
        }

        let mut bsc_words = vec![0u64; dim.div_ceil(64)];
        for (idx, word) in bsc_accum.iter().enumerate() {
            if *word > 0.0 {
                bsc_words[idx] = u64::MAX;
            }
        }

        Self {
            map: HDVector::from_slice(&map_accum),
            bsc: BinaryHDVector {
                dim,
                words: bsc_words,
            },
        }
    }

    pub fn unbind(&self, other: &Self) -> Self {
        Self {
            map: self.map.unbind(&other.map),
            bsc: self.bsc.xor_bind(&other.bsc),
        }
    }

    pub fn permute(&self, shift: usize) -> Self {
        Self {
            map: self.map.permute(shift),
            bsc: self.bsc.rotate(shift),
        }
    }

    pub fn dim(&self) -> usize {
        self.map.dim()
    }

    pub fn bind(&self, other: &Self) -> Self {
        let dim = self.dim();
        let n_words = dim.div_ceil(64);
        let map_a = self.map.data();
        let map_b = other.map.data();
        let bsc_a = self.bsc.words();
        let bsc_b = other.bsc.words();

        // MAP half: element-wise product, sign-flipped wherever b's BSC bit is 0.
        let new_map: Vec<f64> = map_a
            .iter()
            .zip(map_b.iter())
            .enumerate()
            .map(|(d, (&a, &b))| {
                let bit_b = (bsc_b[d / 64] >> (d % 64)) & 1;
                f64::from_bits((a * b).to_bits() ^ ((bit_b ^ 1) << 63))
            })
            .collect();

        // BSC half: a XOR b XOR the sign bits of b's MAP half.
        let new_bsc = sign_xor_words(bsc_a, bsc_b, map_b, n_words);

        Self {
            map: HDVector::from_slice(&new_map),
            bsc: BinaryHDVector {
                dim,
                words: new_bsc,
            },
        }
    }

    pub fn similarity(&self, other: &Self) -> f64 {
        let sim_map = self.map.cosine_similarity(&other.map);
        let sim_bsc = self.bsc.hamming_similarity(&other.bsc);
        (sim_map + sim_bsc) / 2.0
    }
}

#[derive(Clone)]
pub struct SenojianComplex {
    pub map: HDVector,
    pub fhrr: CartesianFhrrVector,
}

impl SenojianComplex {
    pub fn new(map: HDVector, fhrr: CartesianFhrrVector) -> Self {
        assert_eq!(map.dim(), fhrr.dim());
        Self { map, fhrr }
    }

    pub fn random(dim: usize) -> Self {
        Self {
            map: HDVector::random(dim),
            fhrr: CartesianFhrrVector::random_unit(dim, 42),
        }
    }

    pub fn random_seed(dim: usize, seed: u64) -> Self {
        Self {
            map: crate::seed::deterministic_hd_vector(seed, "complex_map", dim),
            fhrr: CartesianFhrrVector::random_unit(dim, seed),
        }
    }

    pub fn dim(&self) -> usize {
        self.map.dim()
    }

    pub fn bind(&self, other: &Self) -> Self {
        // MAP acts as amplitude polarity, FHRR acts as phase (Cartesian)
        Self {
            map: self.map.bind(&other.map),
            fhrr: self.fhrr.bind(&other.fhrr),
        }
    }

    pub fn similarity(&self, other: &Self) -> f64 {
        let dim = self.dim();
        let map_a = self.map.data();
        let map_b = other.map.data();
        let fhrr_a = &self.fhrr;
        let fhrr_b = &other.fhrr;

        let mut sim = 0.0;
        for d in 0..dim {
            // Complex inner product: Re(a * conj(b)) = re_a*re_b + im_a*im_b
            let re_a = fhrr_a.re[d];
            let im_a = fhrr_a.im[d];
            let re_b = fhrr_b.re[d];
            let im_b = fhrr_b.im[d];
            let phase_sim = (re_a * re_b + im_a * im_b) as f64; // cos(theta_a - theta_b)

            let r1 = map_a[d];
            let r2 = map_b[d];
            sim += r1 * r2 * phase_sim;
        }
        sim / (dim as f64)
    }
}

/// `bsc_a[w] ^ bsc_b[w] ^ signbits(map_b[64w .. 64w+64])` for each word `w`.
fn sign_xor_words(bsc_a: &[u64], bsc_b: &[u64], map_b: &[f64], n_words: usize) -> Vec<u64> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 support was just checked at runtime.
            return unsafe { sign_xor_words_avx2(bsc_a, bsc_b, map_b, n_words) };
        }
    }
    sign_xor_words_scalar(bsc_a, bsc_b, map_b, n_words)
}

fn sign_xor_words_scalar(bsc_a: &[u64], bsc_b: &[u64], map_b: &[f64], n_words: usize) -> Vec<u64> {
    let dim = map_b.len();
    bsc_a
        .iter()
        .zip(bsc_b.iter())
        .take(n_words)
        .enumerate()
        .map(|(w, (&a, &b))| {
            let base = w * 64;
            let mut x = a ^ b;
            for (offset, v) in map_b[base..dim.min(base + 64)].iter().enumerate() {
                x ^= (v.to_bits() >> 63) << offset;
            }
            x
        })
        .collect()
}

/// AVX2 version of [`sign_xor_words_scalar`]: packs four sign bits per load
/// with `movemask`, and finishes any partial group with scalar code so it
/// never reads past the end of `map_b`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sign_xor_words_avx2(
    bsc_a: &[u64],
    bsc_b: &[u64],
    map_b: &[f64],
    n_words: usize,
) -> Vec<u64> {
    use std::arch::x86_64::{_mm256_loadu_pd, _mm256_movemask_pd};
    let dim = map_b.len();
    let mut out = Vec::with_capacity(n_words);
    for (w, (&a, &b)) in bsc_a.iter().zip(bsc_b.iter()).take(n_words).enumerate() {
        let base = w * 64;
        let end = dim.min(base + 64);
        let mut x = a ^ b;
        let mut i = base;
        while i + 4 <= end {
            // SAFETY: i + 4 <= end <= map_b.len(), so the 4-lane load is in bounds.
            let lanes = _mm256_loadu_pd(map_b.as_ptr().add(i));
            x ^= (_mm256_movemask_pd(lanes) as u64) << (i - base);
            i += 4;
        }
        for (offset, v) in map_b[i..end].iter().enumerate() {
            x ^= (v.to_bits() >> 63) << (i - base + offset);
        }
        out.push(x);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_xor_words_matches_reference_at_awkward_dims() {
        // Lengths that are not multiples of 4 or 64 exercise the scalar tail.
        for &dim in &[1usize, 3, 63, 64, 65, 130, 1023, 4096] {
            let n_words = dim.div_ceil(64);
            let a = BinaryHDVector::random(dim);
            let b = BinaryHDVector::random(dim);
            let map_b = HDVector::random(dim);
            let mut expected = vec![0u64; n_words];
            for (w, e) in expected.iter_mut().enumerate() {
                *e = a.words()[w] ^ b.words()[w];
            }
            for (d, v) in map_b.data().iter().enumerate() {
                expected[d / 64] ^= (v.to_bits() >> 63) << (d % 64);
            }
            let scalar = sign_xor_words_scalar(a.words(), b.words(), map_b.data(), n_words);
            assert_eq!(scalar, expected, "scalar, dim={dim}");
            let dispatched = sign_xor_words(a.words(), b.words(), map_b.data(), n_words);
            assert_eq!(dispatched, expected, "dispatched, dim={dim}");
        }
    }

    #[test]
    fn test_senojian_cross_bind_properties() {
        let dim = 1024;
        let a = SenojianCross::random(dim);
        let b = SenojianCross::random(dim);

        let bound = a.bind(&b);

        assert_eq!(bound.dim(), dim);

        // Crossover alters standard properties, but binding should still have ~0 similarity
        // with the original randomly distributed vectors.
        // Note: MAP cosine sim is ~0.0, BSC hamming sim is ~0.5.
        // Therefore, (0.0 + 0.5) / 2.0 = ~0.25 for orthogonal SenojianCross vectors.
        let sim_a = bound.similarity(&a);
        let sim_b = bound.similarity(&b);

        assert!(
            (sim_a - 0.25).abs() < 0.1,
            "Bound vector should be roughly orthogonal to a (expected ~0.25, got {})",
            sim_a
        );
        assert!(
            (sim_b - 0.25).abs() < 0.1,
            "Bound vector should be roughly orthogonal to b (expected ~0.25, got {})",
            sim_b
        );
    }

    #[test]
    fn test_senojian_cross_bundle() {
        let dim = 1024;
        let a = SenojianCross::random(dim);
        let b = SenojianCross::random(dim);

        let bundled = a.bundle(&b);

        let sim_a = bundled.similarity(&a);
        let sim_b = bundled.similarity(&b);

        assert!(sim_a > 0.4, "Bundled vector should be similar to a");
        assert!(sim_b > 0.4, "Bundled vector should be similar to b");
    }
}
