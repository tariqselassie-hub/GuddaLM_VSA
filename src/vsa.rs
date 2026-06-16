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
use serde::{Serialize, Deserialize};
use crate::hdc::quantize::pack_bits;
use crate::hdc::vector::HDVector;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VsaEngine {
    pub perm_forward: Vec<usize>,
    pub perm_inverse: Vec<usize>,
    pub dim: usize,
}

impl VsaEngine {
    // Generate a fixed, highly chaotic shuffle mapping using a deterministic seed
    pub fn new(dim: usize) -> Self {
        // LCG (Linear Congruential Generator) to maintain 100% dependency-free offline builds
        let mut seed: u64 = 13374269;
        let mut lcg_rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed
        };

        let mut perm_forward: Vec<usize> = (0..dim).collect();
        // Fisher-Yates Shuffle using our offline LCG
        for i in (1..dim).rev() {
            let j = (lcg_rand() as usize) % (i + 1);
            perm_forward.swap(i, j);
        }
        
        // Generate the exact mathematical inverse mapping
        let mut perm_inverse = vec![0; dim];
        for (i, &target) in perm_forward.iter().enumerate() {
            perm_inverse[target] = i;
        }
        
        Self { perm_forward, perm_inverse, dim }
    }

    // ➡️ HIGH-ENTROPY FORWARD SHUFFLE (Used in train_rust.rs)
    pub fn permute(&self, v: &HDVector, steps: usize) -> HDVector {
        assert_eq!(self.dim, v.dim());
        if steps == 0 { return v.clone(); }
        let mut current = v.data().to_vec();
        let mut next = vec![0.0; self.dim];
        
        for _ in 0..steps {
            for i in 0..self.dim {
                next[self.perm_forward[i]] = current[i];
            }
            current.copy_from_slice(&next);
        }
        HDVector::from_slice_with_binary(&current, v.is_binary())
    }

    /// Permute raw f64 data in-place without allocating an HDVector.
    /// The slice must have length == self.dim.
    pub fn permute_data(&self, data: &mut [f64], steps: usize) {
        assert_eq!(self.dim, data.len());
        if steps == 0 { return; }
        let mut scratch = vec![0.0; self.dim];
        for _ in 0..steps {
            for i in 0..self.dim {
                scratch[self.perm_forward[i]] = data[i];
            }
            data.copy_from_slice(&scratch);
        }
    }

    // ↩️ EXACT INVERSE UNPERMUTE (Used in generate.rs)
    pub fn unpermute(&self, v: &HDVector, steps: usize) -> HDVector {
        assert_eq!(self.dim, v.dim());
        if steps == 0 { return v.clone(); }
        let mut current = v.data().to_vec();
        let mut next = vec![0.0; self.dim];
        
        for _ in 0..steps {
            for i in 0..self.dim {
                next[self.perm_inverse[i]] = current[i];
            }
            current.copy_from_slice(&next);
        }
        HDVector::from_slice_with_binary(&current, v.is_binary())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Codebook {
    pub weights: Vec<HDVector>,
    pub vocab_size: usize,
    pub dim: usize,
    pub engine: VsaEngine, // 🌟 Embedded engine ensures permanent structural alignment
    pub packed: Vec<Vec<u64>>, // Pre-packed bipolar bits for fast XOR-popcount similarity
}

impl Codebook {
    /// Create a zero-initialized codebook (weights must be loaded or trained).
    pub fn new(vocab_size: usize, dim: usize) -> Self {
        let weights = vec![HDVector::zeros(dim); vocab_size];
        let packed = vec![vec![0u64; (dim + 63) / 64]; vocab_size];
        let engine = VsaEngine::new(dim);
        Self { weights, vocab_size, dim, engine, packed }
    }

    /// Create a codebook with random bipolar (±1) prototype vectors.
    ///
    /// Each prototype is an independent random HD vector, and bit-packed
    /// signatures are pre-computed for fast XNOR-popcount similarity.
    pub fn random(vocab_size: usize, dim: usize) -> Self {
        let engine = VsaEngine::new(dim);
        let weights: Vec<HDVector> = (0..vocab_size).map(|_| HDVector::random(dim)).collect();
        let packed: Vec<Vec<u64>> = weights.iter().map(|w| pack_bits(w)).collect();
        Self { weights, vocab_size, dim, engine, packed }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vsa_engine_determinism() {
        let engine1 = VsaEngine::new(100);
        let engine2 = VsaEngine::new(100);
        assert_eq!(engine1.perm_forward, engine2.perm_forward);
        assert_eq!(engine1.perm_inverse, engine2.perm_inverse);
    }

    #[test]
    fn test_vsa_engine_invertibility() {
        let engine = VsaEngine::new(256);
        let original = HDVector::random(256);
        
        let permuted = engine.permute(&original, 3);
        let recovered = engine.unpermute(&permuted, 3);
        
        let sim = original.cosine_similarity(&recovered);
        assert!((sim - 1.0).abs() < 1e-9, "Unpermute must recover the exact original vector");
    }

    #[test]
    fn test_vsa_engine_permute_data_in_place() {
        let engine = VsaEngine::new(128);
        let mut data = vec![0.0; 128];
        for i in 0..128 {
            data[i] = i as f64;
        }
        let original_data = data.clone();
        
        engine.permute_data(&mut data, 2);
        assert_ne!(data, original_data, "Data must be permuted");
        
        // Recover using unpermute
        let vec_version = HDVector::from_slice_with_binary(&data, false);
        let recovered_vec = engine.unpermute(&vec_version, 2);
        let recovered_data = recovered_vec.data();
        
        for i in 0..128 {
            assert!((recovered_data[i] - i as f64).abs() < 1e-9);
        }
    }

    #[test]
    fn test_codebook_initialization() {
        let codebook = Codebook::new(50, 512);
        assert_eq!(codebook.vocab_size, 50);
        assert_eq!(codebook.dim, 512);
        assert_eq!(codebook.weights.len(), 50);
        assert_eq!(codebook.packed.len(), 50);
        assert_eq!(codebook.packed[0].len(), (512 + 63) / 64);
    }
}

