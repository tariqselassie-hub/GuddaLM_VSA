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
//! GuddaLM VSA qudit bench harness
//! Run: cargo run --bin gudda_qudit_bench -p guddalm_vsa
//! Output: bench_results/gudda_qudit_report.json
//!
//! REGRESSION SHIELD: Every metric axis is compared against a persistent
//! baseline (`bench_results/gudda_qudit_baseline.json`).
//!   - Latency   : lower is better  → baseline updated only when new run is faster
//!   - Fidelity  : higher is better → baseline updated only when new run is higher
//!   - Ratio     : higher is better → baseline updated only when new run is higher
//! Regressions are printed to stderr with a [REGRESSION] prefix so CI can catch them.
//! New records are printed with a [NEW RECORD] prefix.

use std::time::Instant;
use serde::{Deserialize, Serialize};
use guddalm_vsa::{IndexVector, hdc::cleanup::BinaryCleanupMemory};
use guddalm_vsa::qudit::{HeptalBinaryMapper, QuditVsaAdapter, OwnedQuditView, ZeroCopyQuditView};

// ── Warmup / sampling config ──────────────────────────────────────────────────
const WARMUP_ITERS: usize = 5;
const SAMPLE_ITERS: usize = 50;

// ── Report schema ─────────────────────────────────────────────────────────────
#[derive(Serialize, Deserialize, Clone)]
struct Report { generated_at: String, seed: u64, suites: Vec<Suite> }

#[derive(Serialize, Deserialize, Clone)]
struct Suite { name: String, axes: Vec<Axis> }

#[derive(Serialize, Deserialize, Clone)]
struct Axis { label: String, value: serde_json::Value, unit: String, notes: String }

fn utc_now() -> String { chrono::Utc::now().to_rfc3339() }

// ── Regression Shield ─────────────────────────────────────────────────────────

enum Improvement { LowerIsBetter, HigherIsBetter }

/// Extract a scalar from a possibly-composite JSON value for comparison.
fn scalar_of(v: &serde_json::Value) -> Option<f64> {
    if let Some(f) = v.as_f64() { return Some(f); }
    for key in &["ms", "similarity", "repaired_similarity", "sign_match"] {
        if let Some(f) = v.get(key).and_then(|x| x.as_f64()) { return Some(f); }
    }
    None
}

/// Compare new vs baseline for a single axis.
/// Returns `true` when the new value is strictly better (should update baseline).
fn evaluate_axis(label: &str, new_val: &serde_json::Value, old_val: &serde_json::Value, imp: Improvement) -> bool {
    let (Some(nv), Some(ov)) = (scalar_of(new_val), scalar_of(old_val)) else { return true };
    let pct = (nv - ov) / ov.abs().max(1e-12) * 100.0;
    let new_is_better = match imp { Improvement::LowerIsBetter => nv < ov, Improvement::HigherIsBetter => nv > ov };
    let sign = if pct >= 0.0 { "+" } else { "" };
    if new_is_better {
        eprintln!("[NEW RECORD]  {:40} {:10.6} vs {:10.6} ({}{:.2}%)", label, nv, ov, sign, pct);
    } else if (nv - ov).abs() < 1e-9 {
        eprintln!("[UNCHANGED]   {:40} {:10.6}", label, nv);
    } else {
        eprintln!("[REGRESSION]  {:40} {:10.6} vs {:10.6} ({}{:.2}%) → Retaining baseline", label, nv, ov, sign, pct);
    }
    new_is_better
}

/// Merge current report into baseline: keep the best result per axis.
fn merge_into_baseline(current: &Report, baseline: &mut Report) {
    for suite in &current.suites {
        if let Some(bs) = baseline.suites.iter_mut().find(|s| s.name == suite.name) {
            for axis in &suite.axes {
                let imp = if axis.unit == "ms" { Improvement::LowerIsBetter } else { Improvement::HigherIsBetter };
                if let Some(ba) = bs.axes.iter_mut().find(|a| a.label == axis.label) {
                    if evaluate_axis(&axis.label, &axis.value, &ba.value, imp) {
                        ba.value = axis.value.clone();
                    }
                } else {
                    bs.axes.push(axis.clone());
                }
            }
        } else {
            baseline.suites.push(suite.clone());
        }
    }
}

// ── Timed helper: warmup + minimum of N samples ───────────────────────────────
/// Returns the minimum (best-case) elapsed milliseconds across SAMPLE_ITERS
/// executions after WARMUP_ITERS un-timed warmup calls.
fn timed_min_ms<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP_ITERS { f(); }
    let mut best = f64::MAX;
    for _ in 0..SAMPLE_ITERS {
        let t0 = Instant::now();
        f();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if ms < best { best = ms; }
    }
    best
}

// ── Suite 1: BSC roundtrip — sign-parity comparator ──────────────────────────
fn bsc_roundtrip_exact() -> Vec<Axis> {
    let mut out = Vec::new();
    let adapter = QuditVsaAdapter::<HeptalBinaryMapper>::new(HeptalBinaryMapper, 4096);
    for n in [7usize, 64] {
        // Use OwnedQuditView — no Box::leak, dropped at end of scope
        let owned = OwnedQuditView::heptal(n);
        let view  = owned.as_view();
        let encoded = adapter.encode(&view);
        let back = adapter.decode(&encoded);

        // Correct comparator: back ∈ {-1, 1} (bipolar), strata ∈ [-3..3] (7-level).
        // "Sign-preserving" means sign(strata) == sign(back) for non-zero entries.
        let sign_match = view.strata.iter().zip(back.iter())
            .filter(|(&a, &b)| {
                if a == 0 { b <= 0 } // 0 stratum maps to -1 in BSC (no positive bit)
                else { (a > 0) == (b > 0) }
            })
            .count();
        let total = view.strata.len();

        out.push(Axis {
            label: format!("roundtrip.bsc_{}", n),
            value: serde_json::json!({ "sign_match": sign_match, "total": total }),
            unit: "ratio".into(),
            notes: format!("sign-preserving bsc roundtrip for {} qudits", n),
        });
    }
    out
}

// ── Suite 2: bind/unbind latency (min-of-50) ─────────────────────────────────
fn bind_unbind_bench() -> Vec<Axis> {
    let adapter = QuditVsaAdapter::<HeptalBinaryMapper>::new(HeptalBinaryMapper, 4096);
    let owned   = OwnedQuditView::heptal(7);
    let view    = owned.as_view();
    let encoded = adapter.encode(&view);
    let bound   = encoded.bind(&encoded);

    let mut last_sim = 0.0;
    let ms = timed_min_ms(|| {
        let recovered = bound.unbind(&encoded);
        last_sim = recovered.similarity_to(&encoded);
    });

    vec![Axis {
        label: "bind_unbind.bsc_7".into(),
        value: serde_json::json!({ "ms": ms, "similarity": last_sim }),
        unit: "ms".into(),
        notes: format!("bsc 7-qudit bind/unbind — min of {} samples", SAMPLE_ITERS),
    }]
}

// ── Suite 3: cleanup/repair latency (min-of-50) ───────────────────────────────
fn cleanup_repair_bench() -> Vec<Axis> {
    let mut out = Vec::new();
    let adapter = QuditVsaAdapter::<HeptalBinaryMapper>::new(HeptalBinaryMapper, 4096);

    for n in [10usize, 50, 100] {
        // Non-degenerate codebook: each entry uses a distinct stratum pattern
        // derived from two different multipliers so no two rows are identical
        // for any n ≤ 100 (42*7=294 unique combos available).
        let mut keys: Vec<(IndexVector, IndexVector)> = Vec::new();
        for idx in 0..n {
            let k_arr: [i8; 7] = std::array::from_fn(|slot| {
                ((idx as i32 * 3 + slot as i32).rem_euclid(7) - 3) as i8
            });
            let v_arr: [i8; 7] = std::array::from_fn(|slot| {
                ((idx as i32 * 5 + slot as i32 + 1).rem_euclid(7) - 3) as i8
            });
            let k = adapter.encode(&ZeroCopyQuditView::new(&k_arr));
            let v = adapter.encode(&ZeroCopyQuditView::new(&v_arr));
            keys.push((k, v));
        }

        let bound     = keys[0].0.bind(&keys[0].1);
        let cand      = bound.unbind(&keys[0].0);
        let codebook: Vec<_> = keys.iter().map(|(_, v)| v.clone().into_inner()).collect();
        // BugFix: build cleanup memory ONCE outside the timed loop
        let mem        = BinaryCleanupMemory::new(codebook);
        let cand_inner = cand.clone().into_inner();

        let mut last_sim = 0.0;
        let ms = timed_min_ms(|| {
            let (_, _, proto) = mem.cleanup(&cand_inner);
            let repaired = IndexVector::new(proto);
            last_sim = repaired.similarity_to(&keys[0].1);
        });

        out.push(Axis {
            label: format!("cleanup.bsc_{}", n),
            value: serde_json::json!({
                "keys": n,
                "ms":   ms,
                "repaired_similarity": last_sim
            }),
            unit: "ms".into(),
            notes: format!("bsc cleanup repair for {} keys — min of {} samples", n, SAMPLE_ITERS),
        });
    }
    out
}

// ── Main ──────────────────────────────────────────────────────────────────────
fn main() {
    let current = Report {
        generated_at: utc_now(),
        seed: 42,
        suites: vec![
            Suite { name: "roundtrip".into(),      axes: bsc_roundtrip_exact()  },
            Suite { name: "bind_unbind".into(),    axes: bind_unbind_bench()    },
            Suite { name: "cleanup_repair".into(), axes: cleanup_repair_bench() },
        ],
    };

    let _ = std::fs::create_dir_all("bench_results");
    let current_path  = "bench_results/gudda_qudit_report.json";
    let baseline_path = "bench_results/gudda_qudit_baseline.json";

    // Write current run unconditionally
    let current_json = serde_json::to_string_pretty(&current).unwrap();
    std::fs::write(current_path, &current_json).unwrap();
    println!("{}", current_json);
    println!("\nSaved current run  → {}", current_path);

    // Load or bootstrap baseline
    let mut baseline: Report = std::fs::read_to_string(baseline_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| {
            eprintln!("[BASELINE] No prior baseline found — bootstrapping from current run.");
            current.clone()
        });

    // Merge: retain best per axis, emit regression / new-record notices
    eprintln!("\n── Regression Shield ───────────────────────────────────────────");
    merge_into_baseline(&current, &mut baseline);
    baseline.generated_at = utc_now();

    let baseline_json = serde_json::to_string_pretty(&baseline).unwrap();
    std::fs::write(baseline_path, &baseline_json).unwrap();
    println!("Saved baseline     → {}", baseline_path);
}
