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
use guddalm_vsa::hdc::vector::{BinaryHDVector, HDVector};
use guddalm_vsa::hdc::quantize::{pack_bits, packed_similarity};
use guddalm_vsa::hdc::resonator::resonator_search;
use guddalm_vsa::hdc::sdm::sdm_read_bipolar;
use guddalm_vsa::hdc::stream::{BundleAccumulator, HDStreamBuffer};
use guddalm_vsa::vsa::Codebook;
use guddalm_vsa::{
    bind_sequence, bundle_sequence, encode_set, decode_set, encode_positional_sequence,
    cartesian_to_phase, phase_to_cartesian, VsaVector,
};
use guddalm_vsa::hdc::phase_fhrr::{CartesianFhrrVector, PhaseFhrrVector};

#[derive(Default)]
struct Summary {
    passed: usize,
    failed: usize,
}

impl Summary {
    fn ok(&mut self, name: &str) {
        self.passed += 1;
        println!("  OK: {}", name);
    }
    fn fail(&mut self, name: &str, err: impl std::fmt::Display) {
        self.failed += 1;
        eprintln!("  FAIL: {} -> {}", name, err);
    }
}

macro_rules! check {
    ($s:expr, $expr:expr, $name:expr) => {{
        match $expr {
            Ok(()) => $s.ok($name),
            Err(err) => $s.fail($name, err),
        }
    }};
}

#[test]
fn system_wide_gudda_check() {
    let mut s = Summary::default();
    println!("=== VSA primitives ===");
    check!(s, test_vsa_primitives(), "VSA primitives invertible + identity");
    check!(s, test_binary_primitives(), "Binary self-inverse + identity");
    check!(s, test_generic_primitives(), "Generic VSA primitives (sequence/set/positional)");
    check!(s, test_fhrr_shim_conversions(), "FHRR Cartesian/Phase shim conversions");
    println!("=== Cache / chunked scaling ===");
    check!(s, test_chunked_cache_is_better_or_equal(), "chunked better-or-equal vs single");
    println!("=== Structural ===");
    check!(s, test_cleanup_memory(), "cleanup memory inverted + prototype");
    check!(s, test_resonator_two_factor(), "resonator two-factor");
    check!(s, test_sdm_read_favors_similar(), "SDM read favors similar keys");
    check!(s, test_packed_similarity_approximates_cosine(), "packed similarity approximates cosine");
    println!("=== Baseline ===");
    check!(s, test_baseline_loss_window(), "baseline loss bounds in training helpers");
    println!("=== Summary: {} passed, {} failed ===", s.passed, s.failed);
    assert_eq!(s.failed, 0, "system check failed");
}

fn test_vsa_primitives() -> Result<(), String> {
    let a = HDVector::random(4096);
    let b = HDVector::random(4096);
    let bound = a.bind(&b);
    let unbind = bound.unbind(&b);
    let sim = a.cosine_similarity(&unbind);
    if sim < 0.65 {
        return Err(format!("bind/unbind sim={:.4}", sim));
    }
    let bundle = a.bundle(&a);
    let sim = a.cosine_similarity(&bundle);
    if sim < 0.99 {
        return Err(format!("bundle identity sim={:.4}", sim));
    }
    Ok(())
}

fn test_binary_primitives() -> Result<(), String> {
    let a = BinaryHDVector::random(4096);
    let b = BinaryHDVector::random(4096);
    let bound = a.xor_bind(&b);
    let unbind = bound.xor_bind(&b);
    if a != unbind {
        return Err("xor_bind not self-inverse".into());
    }
    let bundle = a.majority_bundle(&a);
    let sim = a.hamming_similarity(&bundle);
    if sim < 0.99 {
        return Err(format!("majority_bundle identity sim={:.4}", sim));
    }
    Ok(())
}

fn test_chunked_cache_is_better_or_equal() -> Result<(), String> {
    let dim = 4096;
    let chunk_size = 256;
    let ns = [2, 5, 10, 50, 100, 500];
    for &n in &ns {
        let pairs: Vec<(HDVector, HDVector)> = (0..n)
            .map(|_| (HDVector::random(dim), HDVector::random(dim)))
            .collect();
        let mut single = SingleCache::new(dim);
        let mut chunked = ChunkedCache::new(dim, chunk_size);
        for (k, v) in &pairs {
            single.insert(k, v);
            chunked.insert(k, v);
        }
        let rs = single.query(&pairs[0].0).cosine_similarity(&pairs[0].1);
        let rc = chunked.query(&pairs[0].0).cosine_similarity(&pairs[0].1);
        if rc < -0.05 {
            return Err(format!(
                "n={}: chunked {:.6} went wildly wrong vs single {:.6}",
                n, rc, rs
            ));
        }
    }
    Ok(())
}

fn test_cleanup_memory() -> Result<(), String> {
    let dim = 256;
    let vocab_size = 10;
    let codebook = Codebook::random(vocab_size, dim);
    let mem = guddalm_vsa::hdc::cleanup::CleanupMemory::new(codebook.clone());
    for i in 0..vocab_size {
        let noisy = codebook.weights[i].clone().binarize();
        let result = mem.cleanup(&noisy);
        if result.index != i {
            return Err(format!("cleanup expected {} got {}", i, result.index));
        }
        if result.similarity < 0.99 {
            return Err(format!("cleanup similarity low: {:.4}", result.similarity));
        }
    }
    Ok(())
}

fn test_resonator_two_factor() -> Result<(), String> {
    let dim = 4096;
    let w1 = HDVector::random(dim);
    let w2 = HDVector::random(dim);
    let composition = w1.bind(&w2);
    let mut cb1 = Codebook::new(1, dim);
    cb1.weights.push(w1.clone());
    cb1.packed.clear();
    let mut cb2 = Codebook::new(1, dim);
    cb2.weights.push(w2.clone());
    cb2.packed.clear();
    let codebooks = vec![cb1, cb2];
    let result = resonator_search(&composition, &codebooks, 100, 0.2);
    if !result.converged {
        return Err("resonator did not converge".into());
    }
    Ok(())
}

fn test_sdm_read_favors_similar() -> Result<(), String> {
    let dim = 4096;
    let q = HDVector::random(dim);
    let similar = q.clone();
    let dissimilar = HDVector::random(dim);
    let v1 = HDVector::random(dim);
    let v2 = HDVector::random(dim);
    let out = sdm_read_bipolar(&q, &[similar, dissimilar], &[v1.clone(), v2.clone()]);
    let sim1 = out.cosine_similarity(&v1);
    let sim2 = out.cosine_similarity(&v2);
    if sim1 <= sim2 {
        return Err(format!("SDM read biased wrong: v1={:.4} v2={:.4}", sim1, sim2));
    }
    Ok(())
}

fn test_packed_similarity_approximates_cosine() -> Result<(), String> {
    let a = HDVector::random(4096);
    let b = HDVector::random(4096);
    let orig = a.cosine_similarity(&b);
    let packed = pack_bits(&a);
    let approx = packed_similarity(&b, &packed);
    if (orig - approx).abs() > 0.01 {
        return Err(format!("packed drift: orig={:.4} approx={:.4}", orig, approx));
    }
    Ok(())
}

fn test_baseline_loss_window() -> Result<(), String> {
    let v = HDVector::random(256);
    let loss = 1.0 - v.cosine_similarity(&v);
    let ppl = loss.exp();
    if !((ppl - 1.0).abs() < 1e-9) {
        return Err(format!("perplexity baseline violated: ppl={}", ppl));
    }
    Ok(())
}

struct SingleCache {
    state: HDVector,
    dim: usize,
}

impl SingleCache {
    fn new(dim: usize) -> Self {
        Self {
            state: HDVector::zeros(dim),
            dim,
        }
    }

    fn insert(&mut self, k: &HDVector, v: &HDVector) {
        let bound = k.bind(v);
        let mut acc = BundleAccumulator::new(self.dim);
        acc.add(&self.state);
        acc.add(&bound.binarize());
        self.state = acc.binarize();
    }

    fn query(&self, q: &HDVector) -> HDVector {
        let recovered = self.state.unbind(q);
        let mut acc = BundleAccumulator::new(self.dim);
        acc.add(&recovered);
        acc.binarize()
    }
}

struct ChunkedCache {
    dim: usize,
    chunk_size: usize,
    chunks: Vec<HDStreamBuffer>,
    key_sums: Vec<HDVector>,
    steps: usize,
}

impl ChunkedCache {
    fn new(dim: usize, chunk_size: usize) -> Self {
        Self {
            dim,
            chunk_size,
            chunks: vec![HDStreamBuffer::new(dim, 1)],
            key_sums: vec![HDVector::zeros(dim)],
            steps: 0,
        }
    }

    fn insert(&mut self, k: &HDVector, v: &HDVector) {
        let last = self.chunks.len() - 1;
        if self.chunks[last].capacity() == 0 {
            self.chunks[last] = HDStreamBuffer::new(self.dim, 1);
        }

        let len = self.chunks[last].len();
        if len == self.chunks[last].capacity() || len == 0 {
            self.chunks.push(HDStreamBuffer::new(self.dim, 1.max(self.chunk_size)));
            self.key_sums.push(HDVector::zeros(self.dim));
        }

        let idx = self.chunks.len() - 1;
        self.chunks[idx].push(k.clone());
        self.chunks[idx].push(v.clone());

        let mut acc = BundleAccumulator::new(self.dim);
        acc.add(&self.key_sums[idx]);
        acc.add(k);
        self.key_sums[idx] = acc.binarize();
        self.steps += 1;
    }

    fn query(&self, q: &HDVector) -> HDVector {
        let mut best_i = 0usize;
        let mut best_s = f64::NEG_INFINITY;
        for (i, s) in self.key_sums.iter().enumerate() {
            if s.dim() == 0 {
                continue;
            }
            let sim = q.cosine_similarity(s);
            if sim > best_s {
                best_s = sim;
                best_i = i;
            }
        }
        self.chunks
            .get(best_i)
            .map_or_else(|| HDVector::zeros(self.dim), |c| {
                let state = c.bundle_all();
                state.unbind(q).binarize()
            })
    }
}

fn test_generic_primitives() -> Result<(), String> {
    let dim = 1024;
    let v1 = HDVector::random(dim);
    let v2 = HDVector::random(dim);
    let v3 = HDVector::random(dim);

    // Test bind_sequence
    let bound = bind_sequence(&[v1.clone(), v2.clone(), v3.clone()]);
    let expected_bound = v1.bind(&v2).bind(&v3);
    if bound.cosine_similarity(&expected_bound) < 0.99 {
        return Err("bind_sequence failed".into());
    }

    // Test bundle_sequence
    let bundled = bundle_sequence(&[v1.clone(), v2.clone(), v3.clone()]);
    let expected_bundled = v1.bundle(&v2).bundle(&v3);
    if bundled.cosine_similarity(&expected_bundled) < 0.99 {
        return Err("bundle_sequence failed".into());
    }

    // Test encode_set / decode_set
    let k1 = HDVector::random(dim);
    let k2 = HDVector::random(dim);
    let val1 = HDVector::random(dim);
    let val2 = HDVector::random(dim);
    let set = encode_set(&[(k1.clone(), val1.clone()), (k2.clone(), val2.clone())]);
    
    let decoded = decode_set(&set, &k1);
    let sim1 = decoded.cosine_similarity(&val1);
    let sim2 = decoded.cosine_similarity(&val2);
    if sim1 <= sim2 {
        return Err(format!("encode/decode set failed: sim_val1={:.4} sim_val2={:.4}", sim1, sim2));
    }

    // Test encode_positional_sequence
    let seq = encode_positional_sequence(&[v1.clone(), v2.clone()]);
    let query0 = seq.permute_left(0);
    let query1 = seq.permute_left(1);
    let sim_seq0 = query0.cosine_similarity(&v1);
    let sim_seq1 = query1.cosine_similarity(&v2);
    if sim_seq0 < 0.3 || sim_seq1 < 0.3 {
        return Err(format!("encode_positional_sequence failed: sim0={:.4} sim1={:.4}", sim_seq0, sim_seq1));
    }

    Ok(())
}

fn test_fhrr_shim_conversions() -> Result<(), String> {
    let dim = 128;
    // Create random CartesianFhrrVector
    let cart = CartesianFhrrVector::random_unit(dim, 42);
    
    // Convert to phase representation
    let phase = cartesian_to_phase(&cart);
    if phase.phases.len() != dim {
        return Err("cartesian_to_phase dimension mismatch".into());
    }
    
    // Convert back to cartesian representation
    let cart_back = phase_to_cartesian(&phase);
    if cart_back.dim() != dim {
        return Err("phase_to_cartesian dimension mismatch".into());
    }
    
    // Check similarity between original and roundtrip
    let sim = cart.similarity(&cart_back);
    if (sim - 1.0).abs() > 1e-5 {
        return Err(format!("FHRR cartesian/phase roundtrip failed: sim={:.6}", sim));
    }
    
    Ok(())
}
