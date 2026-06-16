# Contributing to GuddaLM_VSA

This project is for the community.
Any contributions you make are contributed to the community, not to an individual maintainer.
This is intended to remain an open source project into the near future. {No Promises}

Contact: Zoddjr@gmail.com
C/O Mr Terrence A. Jones Sr.
That personal work does not interfere with or substitute for this main system.

Project scope: GuddaLM_VSA only.
GuddaLM is a distinct project; do not copy changes across by default.

## Getting started
- Install Rust (stable toolchain).
- Clone and check that `cargo check` passes.
- Run `cargo test` before opening a PR.

## Branching
- Use feature branches from `main`.
- Name branches by topic, e.g. `feat/...`, `fix/...`, `docs/...`.

## Commits and PRs
- Write clear commit messages.
- Keep changes minimal and focused.
- Include rationale in the PR description and link related issues.
- Update `CHANGELOG.md` for user-visible changes.

## Code rules
- All Rust files must include the AGPL header with `SPDX-License-Identifier: AGPL-3.0`.
- License metadata must remain in `Cargo.toml`.
- Do not add release artifacts for the `GuddaLM` project.

## Release lint
- Run `uv run scripts/lint_release.py` before pushing.
- Update `RELEASES.sha256` after generated artifacts change.
