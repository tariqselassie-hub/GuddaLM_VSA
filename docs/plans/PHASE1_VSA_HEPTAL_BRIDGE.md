# Phase 1 Plan: GuddaLM VSA ↔ Heptal Qudit Bridge

## Goal
Run VSA hyperdimensional operations on 7-strata qudit registers.

## Deliverables
1. `guddalm_quantum_core/src/qudit_vsa.rs` — VSA state bridge
2. `GuddaLM_VSA/src/guddalm/` — GuddaLM adapter crate
3. Joint runner: `guddalm_quantum_core/examples/guddalm_vsa_demo.rs`
4. Bench harness: `GuddaLM_VSA/benches/guddalm_qudit.rs`

## Steps
1. **Interface contract**
   - Map `HeptalStrata{-3,-2,-1,0,1,2,3}` ↔ `VSAVector` permutations
   - Decide polarity mapping: stratum sign → bit/mag axis
2. **Adapter implementations**
   - BSC: pack 7 strata into 7-bit fields within 4096-bit words
   - MAP: encode stratum as phase angle across 7 frequency bins
   - FHRR: same bin strategy but complex phase
3. **Zero-copy path**
   - Expose `guddalm_quantum_core::qudit_register` buffers as `&[u64]`
   - Avoid heap alloc in hot paths
4. **Joint demo**
   - Bind role/filler in VSA, read back as strata vector, run CPhase gate, unbind, compare fidelity
5. **Benchmarks**
   - Latency: VSA bind/unbind vs native CPhase on qudit array
   - Capacity: number of superpositioned pairs before cleanup fails
6. **Validation gate**
   - `cargo run -p guddalm_vsa --example guddalm_vsa_demo` exits 0
   - Benchmarks saved to `research_divizion/GuddaLM_VSA/bench_results/guddalm_qudit_report.json`

## Repo paths
- Plan copy 1: `research_divizion/research_workspace/plans/PHASE1_VSA_HEPTAL_BRIDGE.md`
- Plan copy 2: `research_divizion/GuddaLM_VSA/plans/PHASE1_VSA_HEPTAL_BRIDGE.md`

## Status
Draft only. No edits made yet.
