//! Zenith ↔ VSA qudit bench harness
//! Run: cargo run --bin zenith_qudit_bench -p guddalm_vsa
//! Output: bench_results/zenith_qudit_report.json

use std::time::Instant;
use serde::Serialize;
use guddalm_vsa::{IndexVector, hdc::cleanup::BinaryCleanupMemory};
use guddalm_vsa::zenith::{HeptalBinaryMapper, QuditVsaAdapter, ZeroCopyQuditView};

#[derive(Serialize)]
struct Report { generated_at: String, seed: u64, suites: Vec<Suite> }
#[derive(Serialize)]
struct Suite { name: &'static str, axes: Vec<Axis> }
#[derive(Serialize)]
struct Axis { label: String, value: serde_json::Value, unit: &'static str, notes: String }

fn utc_now() -> String { chrono::Utc::now().to_rfc3339() }

fn bsc_roundtrip_exact() -> Vec<Axis> {
  let mut out = Vec::new();
  let adapter = QuditVsaAdapter::<HeptalBinaryMapper>::new(HeptalBinaryMapper, 4096);
  for n in [7usize, 64] {
    let view = ZeroCopyQuditView::heptal(n);
    let encoded = adapter.encode(&view);
    let back = adapter.decode(&encoded);
    let exact = view.strata.iter().zip(back.iter()).filter(|(a,b)| a==b).count();
    out.push(Axis {
      label: format!("roundtrip.bsc_{}", n),
      value: serde_json::json!({ "exact": exact, "total": view.strata.len() }),
      unit: "ratio",
      notes: format!("sign-preserving bsc roundtrip for {} qudits", n),
    });
  }
  out
}

fn bind_unbind_bench() -> Vec<Axis> {
  let mut out = Vec::new();
  let adapter = QuditVsaAdapter::<HeptalBinaryMapper>::new(HeptalBinaryMapper, 4096);
  let view = ZeroCopyQuditView::heptal(7);
  let encoded = adapter.encode(&view);
  let bound = encoded.bind(&encoded);
  let t0 = Instant::now();
  let recovered = bound.unbind(&encoded);
  let ms = t0.elapsed().as_secs_f64() * 1000.0;
  let sim = recovered.similarity_to(&encoded);
  out.push(Axis {
    label: "bind_unbind.bsc_7".into(),
    value: serde_json::json!({ "ms": ms, "similarity": sim }),
    unit: "ms",
    notes: "bsc 7-qudit bind/unbind latency".into(),
  });
  out
}

fn cleanup_repair_bench() -> Vec<Axis> {
  let mut out = Vec::new();
  let adapter = QuditVsaAdapter::<HeptalBinaryMapper>::new(HeptalBinaryMapper, 4096);
  for n in [10usize, 50, 100] {
    let mut keys: Vec<(IndexVector, IndexVector)> = Vec::new();
    for idx in 0..n {
      let k_arr: [i8; 7] = [idx as i8 % 7 - 3; 7];
      let v_arr: [i8; 7] = [(idx as i8 + 3) % 7 - 3; 7];
      let k = adapter.encode(&ZeroCopyQuditView::new(&k_arr));
      let v = adapter.encode(&ZeroCopyQuditView::new(&v_arr));
      keys.push((k, v));
    }
    let bound = keys[0].0.bind(&keys[0].1);
    let cand = bound.unbind(&keys[0].0);
    let codebook = keys.iter().map(|(_, v)| v.clone().into_inner()).collect::<Vec<_>>();
    let mem = BinaryCleanupMemory::new(codebook);
    let t0 = Instant::now();
    let (_, _, proto) = mem.cleanup(&cand.clone().into_inner());
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let repaired = IndexVector::new(proto);
    let sim = repaired.similarity_to(&keys[0].1);
    out.push(Axis {
      label: format!("cleanup.bsc_{}", n),
      value: serde_json::json!({ "ms": ms, "repaired_similarity": sim, "keys": n }),
      unit: "ms",
      notes: format!("bsc cleanup repair for {} keys", n),
    });
  }
  out
}

fn main() {
  let report = Report {
    generated_at: utc_now(),
    seed: 42,
    suites: vec![
      Suite { name: "roundtrip", axes: bsc_roundtrip_exact() },
      Suite { name: "bind_unbind", axes: bind_unbind_bench() },
      Suite { name: "cleanup_repair", axes: cleanup_repair_bench() },
    ],
  };
  let _ = std::fs::create_dir_all("bench_results");
  let path = "bench_results/zenith_qudit_report.json";
  let json = serde_json::to_string_pretty(&report).unwrap();
  std::fs::write(path, &json).unwrap();
  println!("{}", json);
  println!("\nSaved report to {}", path);
}
