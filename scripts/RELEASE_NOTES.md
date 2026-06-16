# Release Compliance Notes
## Project scope
- `GuddaLM_VSA` includes: `LICENSE`, `Cargo.toml`, `src/**`, `tests/**`
- `GuddaLM` is distinct and should not receive these release artifacts.

## Release artifacts
- `LICENSE`
- `Cargo.toml` with `license = "AGPL-3.0-only"`
- `README.md`
- `NOTICE` and `THIRD_PARTY_NOTICES.md` generated from `Cargo.lock`
- `sbom.spdx.json` and `sbom.cdx.json` generated from `Cargo.lock`
- `RELEASES.sha256`

## Authoritative attribution
- Copyright holder / code signatory: **Terrence A. Jones Sr.**
  - Applies only to `GuddaLM_VSA` and this codebase.
  - Kept local; do not copy this name to the `GuddaLM` project.

## Notes
- Third-party contents are originals from `Cargo.lock`.
