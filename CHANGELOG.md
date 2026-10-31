# Changelog

All notable changes to GuddaLM_VSA will be documented here.
Project scope is GuddaLM_VSA only.

Format based on Keep a Changelog.

## [Unreleased]
### Added
- `MAP_HIGH_CAPACITY_DIM` (8192) alongside `MAP_DEFAULT_DIM` (4096), with measured recall in the docs; default dims are exported from the prelude.
- `BundleAccumulator` / `BinaryBundleAccumulator`: `merge`, `dim`, `is_empty`, `into_vector` and `Extend` impls.
- `HDVector::from_vec` for zero-copy construction.
- `VsaVector::bundle_many`: bundles a slice in one step. The default folds pairwise; MAP, BSC and `IndexVector` use the accumulators.
- Criterion benchmark suite (`benches/core_ops.rs`) and `docs/benchmarks.md`.

### Changed
- MAP bind/unbind use the FFT for every dimension. Non-power-of-two dims (such as 10,000) were O(n²) and are now ~430× faster.
- The FFT planner is thread-local instead of a global mutex; `get_vsa_engine` returns `Arc<VsaEngine>` instead of cloning the permutation tables.
- Cosine similarity is a single fused pass (~57% faster).
- `search`, `ItemMemory::cleanup`/`cleanup_top_k` and `CleanupMemory::cleanup` score in parallel and select top-k without a full sort (66–89% faster). Ties now always go to the lowest index.
- `BinaryBundleAccumulator::add` is ~2× faster.
- **Behaviour change:** `bundle_sequence` now calls `bundle_many`. For BSC and `IndexVector` it returns a true majority vote where it used to fold pairwise, so outputs differ. Pairwise folding left the earliest inputs unrecoverable with as few as 5 vectors. MAP results are unchanged, and FHRR/GHRR keep the pairwise fold.
- Modules were regrouped into `modes/`, `persist/` and `apps/`, and the streaming bundlers moved to `hdc::accumulator`. All old paths are still re-exported.
- `plans/` moved to `docs/plans/`.
- **Renamed "guddalm" to "guddalm" (breaking):** the `guddalm` module is now `guddalm_vsa::guddalm` (`apps::guddalm`), with no alias for the old path. The `guddalm_bench`/`guddalm_qudit_bench` binaries are now `guddalm_bench`/`guddalm_qudit_bench`, the `guddalm_vsa_demo` example is now `guddalm_vsa_demo`, and the bench report files are renamed to match.
- Library code is free of clippy warnings: index loops became iterators, orphaned doc blocks became module docs, and `SearchResult`'s `PartialOrd` is now canonical.
- `bsc_selective_bundle` and `sdm_read_binary` use `BinaryBundleAccumulator`, with identical output.

### Removed
- `bench_results/last_run.txt` (a captured compiler log).

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
