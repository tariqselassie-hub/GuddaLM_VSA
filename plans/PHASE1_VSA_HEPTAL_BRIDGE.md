# Phase 1 Plan: GuddaLM VSA ↔ Zenith Heptal Bridge

## Goal
Run VSA hyperdimensional operations on Zenith’s 7-strata qudit registers.

## Deliverables
1. `zenith_quantum_core/src/qudit_vsa.rs` — VSA state bridge
2. `GuddaLM_VSA/src/zenith/` — Zenith adapter crate
3. Joint runner: `zenith_quantum_core/examples/zenith_vsa_demo.rs`
4. Bench harness: `GuddaLM_VSA/benches/zenith_qudit.rs`

## Steps
1. **Interface contract**
   - Map `HeptalStrata{-3,-2,-1,0,1,2,3}` ↔ `VSAVector` permutations
   - Decide polarity mapping: stratum sign → bit/mag axis
2. **Adapter implementations**
   - BSC: pack 7 strata into 7-bit fields within 4096-bit words
   - MAP: encode stratum as phase angle across 7 frequency bins
   - FHRR: same bin strategy but complex phase
3. **Zero-copy path**
   - Expose `zenith_quantum_core::qudit_register` buffers as `&[u64]`
   - Avoid heap alloc in hot paths
4. **Joint demo**
   - Bind role/filler in VSA, read back as strata vector, run CPhase gate, unbind, compare fidelity
5. **Benchmarks**
   - Latency: VSA bind/unbind vs native CPhase on qudit array
   - Capacity: number of superpositioned pairs before cleanup fails
6. **Validation gate**
   - `cargo run -p guddalm_vsa --example zenith_vsa_demo` exits 0
   - Benchmarks saved to `zenith_research_divizion/GuddaLM_VSA/bench_results/zenith_qudit_report.json`

## Repo paths
- Plan copy 1: `zenith_research_divizion/zenith_research_workspace/plans/PHASE1_VSA_HEPTAL_BRIDGE.md`
- Plan copy 2: `zenith_research_divizion/GuddaLM_VSA/plans/PHASE1_VSA_HEPTAL_BRIDGE.md`

## Status
Draft only. No edits made yet.
