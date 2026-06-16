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
use guddalm_vsa::hdc::stream::HDStreamBuffer;
use guddalm_vsa::map::MapSetup;
use guddalm_vsa::fhrr::FhrrSetup;
use guddalm_vsa::vsa::Codebook;

#[test]
fn demo_gudda_vsa_system() {
    let mut passed = 0usize;
    let mut failed = 0usize;

    let mut ok = |name: &str| {
        passed += 1;
        println!("  OK: {}", name);
    };
    let mut fail = |name: &str, err: String| {
        failed += 1;
        eprintln!("  FAIL: {} -> {}", name, err);
    };

    println!("=== VSA Map System ===");
    match demo_map_system() {
        Ok(()) => ok("map_system"),
        Err(err) => fail("map_system", err),
    }

    println!("=== Binary / BSC System ===");
    match demo_bsc_system() {
        Ok(()) => ok("bsc_system"),
        Err(err) => fail("bsc_system", err),
    }

    println!("=== FHRR System ===");
    match demo_fhrr_system() {
        Ok(()) => ok("fhrr_system"),
        Err(err) => fail("fhrr_system", err),
    }

    println!("=== Quantization and similarity ===");
    match demo_quantized_similarity() {
        Ok(()) => ok("quantized_similarity"),
        Err(err) => fail("quantized_similarity", err),
    }

    println!("=== Resonator search ===");
    match demo_resonator() {
        Ok(()) => ok("resonator"),
        Err(err) => fail("resonator", err),
    }

    println!("=== SDM read ===");
    match demo_sdm() {
        Ok(()) => ok("sdm"),
        Err(err) => fail("sdm", err),
    }

    println!("=== Streamed bundling ===");
    match demo_streamed_bundle() {
        Ok(()) => ok("streamed_bundle"),
        Err(err) => fail("streamed_bundle", err),
    }

    println!("=== Summary: {} passed, {} failed ===", passed, failed);
    assert!(failed == 0, "demo had failures");
}

fn demo_map_system() -> Result<(), String> {
    let a = MapSetup::random(4096usize);
    let b = MapSetup::random(4096usize);
    let bound = a.bind(&b);
    let recovered = bound.unbind(&b);
    let sim = a.cosine_similarity(&recovered);
    if sim < 0.65 {
        return Err(format!("MAP bind/unbind sim={:.4}", sim));
    }
    let bundle = a.bundle(&a);
    let id_sim = a.cosine_similarity(&bundle);
    if id_sim < 0.99 {
        return Err(format!("bundle identity sim={:.4}", id_sim));
    }
    Ok(())
}

fn demo_bsc_system() -> Result<(), String> {
    let a = BinaryHDVector::random(10000usize);
    let b = BinaryHDVector::random(10000usize);
    let bound = a.xor_bind(&b);
    let unbind = bound.xor_bind(&b);
    if a != unbind {
        return Err("XOR bind not self-inverse".into());
    }
    let bundle = a.majority_bundle(&a);
    let sim = a.hamming_similarity(&bundle);
    if sim < 0.99 {
        return Err(format!("majority_bundle identity sim={:.4}", sim));
    }
    Ok(())
}

fn demo_fhrr_system() -> Result<(), String> {
    let a = FhrrSetup::random(4096usize);
    let b = FhrrSetup::random(4096usize);
    let bound = a.bind(&b);
    let recovered = bound.bind(&b.inverse());
    let sim = a.cosine_similarity(&recovered);
    if sim < 0.65 {
        return Err(format!("FHRR bind/inverse sim={:.4}", sim));
    }
    Ok(())
}

fn demo_quantized_similarity() -> Result<(), String> {
    let a = HDVector::random(4096usize);
    let b = HDVector::random(4096usize);
    let orig = a.cosine_similarity(&b);
    let packed = pack_bits(&a);
    let approx = packed_similarity(&b, &packed);
    let drift = (orig - approx).abs();
    if drift > 0.01 {
        return Err(format!("packed drift={:.4}", drift));
    }
    Ok(())
}

fn demo_resonator() -> Result<(), String> {
    let dim = 4096usize;
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

fn demo_sdm() -> Result<(), String> {
    let dim = 4096usize;
    let q = HDVector::random(dim);
    let similar = q.clone();
    let dissimilar = HDVector::random(dim);
    let v1 = HDVector::random(dim);
    let v2 = HDVector::random(dim);
    let out = sdm_read_bipolar(&q, &[similar, dissimilar], &[v1.clone(), v2.clone()]);
    let sim1 = out.cosine_similarity(&v1);
    let sim2 = out.cosine_similarity(&v2);
    if sim1 <= sim2 {
        return Err(format!(
            "SDM read biased wrong: v1={:.4} v2={:.4}",
            sim1, sim2
        ));
    }
    Ok(())
}

fn demo_streamed_bundle() -> Result<(), String> {
    let dim = 4096usize;
    let mut buf = HDStreamBuffer::new(dim, 64usize);
    for _ in 0..32usize {
        buf.push(HDVector::random(dim));
    }
    let bundled = buf.bundle_all();
    if bundled.dim() != dim {
        return Err(format!("streamed bundle dim={}", bundled.dim()));
    }
    Ok(())
}
