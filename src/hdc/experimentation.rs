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
use crate::hdc::vector::{HDVector, BinaryHDVector};

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

use serde::{Serialize, Deserialize};
use crate::hdc::phase_fhrr::CartesianFhrrVector;

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
            map_accum = map_accum.iter().zip(m.data()).map(|(a, b)| *a + *b).collect();
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
        let mut bsc_accum = vec![0f64; (dim + 63) / 64];

        for (v, w) in vectors {
            let map_data = v.map.data();
            let bsc_words = v.bsc.words();
            map_accum = map_accum.iter().zip(map_data).map(|(a, b)| *a + b * w).collect();
            for (idx, word) in bsc_words.iter().enumerate() {
                let delta = if *word == 0 { -*w } else { *w };
                bsc_accum[idx] += delta;
            }
        }

        let mut bsc_words = vec![0u64; (dim + 63) / 64];
        for (idx, word) in bsc_accum.iter().enumerate() {
            if *word > 0.0 {
                bsc_words[idx] = u64::MAX;
            }
        }

        Self {
            map: HDVector::from_slice(&map_accum),
            bsc: BinaryHDVector { dim, words: bsc_words },
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
        let n_words = (dim + 63) / 64;
        let mut new_map = vec![0.0; dim];
        let mut new_bsc = vec![0u64; n_words];

        let map_a = self.map.data();
        let map_b = other.map.data();
        let bsc_a = self.bsc.words();
        let bsc_b = other.bsc.words();

        if dim >= 64 && (cfg!(target_arch = "x86_64") && is_x86_feature_detected!("avx2")) {
            unsafe {
                use core::arch::x86_64::*;
                for chunk in 0..n_words {
                    let mut xor_val = bsc_a[chunk] ^ bsc_b[chunk];
                    let base = chunk * 64;
                    if base < dim {
                        for i in (base..dim.min(base + 64)).step_by(4) {
                            let vec = _mm256_loadu_pd(map_b.as_ptr().add(i));
                            let mask = _mm256_movemask_pd(vec) as u64;
                            xor_val ^= mask << (i - base);
                        }
                    }
                    new_bsc[chunk] = xor_val;
                }
                for d in 0..dim {
                    let a = *map_a.get_unchecked(d);
                    let b = *map_b.get_unchecked(d);
                    let m_val = a * b;
                    let chunk_b = d / 64;
                    let offset_b = d % 64;
                    let bit_b = (*bsc_b.get_unchecked(chunk_b) >> offset_b) & 1;
                    let flip_mask = ((bit_b ^ 1) as u64) << 63;
                    *new_map.get_unchecked_mut(d) = f64::from_bits(m_val.to_bits() ^ flip_mask);
                }
            }
        } else {
            let mut xor_val = bsc_a[0] ^ bsc_b[0];
            for d in 0..dim {
                let m_val = map_a[d] * map_b[d];
                let bit_b = (bsc_b[d / 64] >> (d % 64)) & 1;
                let sign_bit = (map_b[d].to_bits() >> 63) & 1;
                xor_val ^= sign_bit << (d % 64);
                let flip_mask = ((bit_b ^ 1) as u64) << 63;
                new_map[d] = f64::from_bits(m_val.to_bits() ^ flip_mask);
                if d % 64 == 63 {
                    new_bsc[d / 64] = xor_val;
                    if (d / 64) + 1 < n_words {
                        xor_val = bsc_a[d / 64 + 1] ^ bsc_b[d / 64 + 1];
                    }
                }
            }
            if (dim - 1) % 64 != 63 {
                new_bsc[(dim - 1) / 64] = xor_val;
            }
        }

        Self {
            map: HDVector::from_slice(&new_map),
            bsc: BinaryHDVector { dim, words: new_bsc },
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

#[cfg(test)]
mod tests {
    use super::*;

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

        assert!((sim_a - 0.25).abs() < 0.1, "Bound vector should be roughly orthogonal to a (expected ~0.25, got {})", sim_a);
        assert!((sim_b - 0.25).abs() < 0.1, "Bound vector should be roughly orthogonal to b (expected ~0.25, got {})", sim_b);
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