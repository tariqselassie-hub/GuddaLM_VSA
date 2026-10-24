# Changelog

All notable changes to GuddaLM_VSA will be documented here.
Project scope is GuddaLM_VSA only.

Format based on Keep a Changelog.

## [0.1.2] - 2026-10-22
### Changed — GuddaLM Identity Adoption
- **Workspace rebranded** from `zenith_workspace_divizion` → `GuddaLM_workspace`. No more Zenith Research Division branding.
- **`src/zenith.rs` → `src/qudit.rs`**: The qudit adapter module is now `guddalm_vsa::qudit`. All public types (`ZeroCopyQuditView`, `OwnedQuditView`, `HeptalBinaryMapper`, `QuditVsaAdapter`, etc.) are unchanged — only the module path changed.
- **`zenith_qudit_bench` → `gudda_qudit_bench`**: Replaced with a fully rewritten bench harness.
- **`zenith_bench` → `gudda_bench`**: Binary alias renamed in `Cargo.toml`.

### Added
- **`OwnedQuditView`**: Owned variant of `ZeroCopyQuditView` that carries its own backing buffer. Eliminates the `Box::leak` memory leak that permanently grew the heap on every `ZeroCopyQuditView::heptal()` call.
- **Regression Shield** (`bench_results/gudda_qudit_baseline.json`): Persistent per-axis baseline tracker. Prints `[REGRESSION]`, `[NEW RECORD]`, or `[UNCHANGED]` to stderr on every bench run. Baselines are only updated when new results are strictly better — regressions are flagged but the prior best is retained.
- **Warmup + min-of-50 latency sampling**: Replaces the single cold one-shot `Instant::now()` measurement. Eliminates Windows OS scheduler and cache-miss noise from sub-millisecond timings.

### Fixed
- **`Box::leak` memory leak** in `ZeroCopyQuditView::heptal` — heap was permanently growing on every call. Fixed via `OwnedQuditView`.
- **Flawed sign-parity comparator** in `bsc_roundtrip_exact`: was comparing 7-level qudit values against bipolar `{-1,1}` with `==`, artificially capping scores at ~28.6%. Now uses correct sign-parity semantics.
- **Degenerate cleanup codebook** in `cleanup_repair_bench`: was generating only 7 unique vectors for any `n`. Fixed with per-index distinct stratum patterns using coprime multipliers.
- **`BinaryCleanupMemory` inner-loop allocation**: was being re-constructed on every retrieval iteration. Fixed — built once outside the timed block.
- **Cargo manifest lint**: Removed duplicate `license-file = "LICENSE"` that caused `warning: only one of 'license' or 'license-file' is necessary` on every build.

## [0.1.1] - 2026-09-27
### Added
- `prelude` module for turnkey project onboarding (`use guddalm_vsa::prelude::*;`).
- `ItemMemory<V>` associative symbol table for deterministic token-to-hypervector generation, nearest-neighbor cleanup, top-$k$ recall, and record encoding.
- `ScalarEncoder` for continuous numerical telemetry and level hypervector encoding via monotonic thermometer walks and fractional phase rotations.
- 4 runnable starter examples (`01_quickstart_key_value`, `02_analogical_reasoning`, `03_sensor_level_encoding`, `04_sequence_and_text_ngrams`).

### Changed
- Fixed multi-factor Resonator network convergence by initializing factor estimates with codebook superpositions (Frady et al.).
- Hardened `.gitignore` to prevent accidental tracking of distillation datasets (`*.jsonl`), python virtualenv caches, and model weight blobs.
- Re-architected `ItemMemory` and `generate_rc_codebook` with expression blocks to pass Karnak AST static analysis.
- Upgraded `README.md` with complete architecture comparison matrix, quickstart, and examples guide.

### Removed
- Deprecated empty `src/board.rs` stub and empty directories.

## [0.1.0] - 2026-06-16
### Added
- Initial release candidate scaffolding.
- AGPLv3 license and project-scoped attribution.
- Release lint and compliance generation scripts.
- Third-party notices and SBOM output.
