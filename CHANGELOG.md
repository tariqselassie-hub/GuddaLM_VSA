# Changelog

All notable changes to GuddaLM_VSA will be documented here.
Project scope is GuddaLM_VSA only.

Format based on Keep a Changelog.

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
