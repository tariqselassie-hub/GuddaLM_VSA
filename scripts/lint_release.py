#!/usr/bin/env python3
"""release lint for GuddaLM_VSA"""
from pathlib import Path
import hashlib
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "LICENSE",
    "README.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "CHANGELOG.md",
    "SECURITY.md",
    "RELEASES.sha256",
    "Cargo.toml",
    "Cargo.lock",
]
REQUIRED_GENERATED = [
    "scripts/generated/NOTICE",
    "scripts/generated/THIRD_PARTY_NOTICES.md",
    "scripts/generated/sbom.spdx.json",
    "scripts/generated/sbom.cdx.json",
]
HEADER_RE = re.compile(r"SPDX-License-Identifier: AGPL-3\.0")

errors = []

def check(msg, cond, detail=""):
    if not cond:
        txt = f"FAIL: {msg}"
        if detail:
            txt += f" -> {detail}"
        errors.append(txt)

# 1) required docs/files
for rel in REQUIRED_FILES + REQUIRED_GENERATED:
    p = ROOT / rel
    check(f"required file exists: {rel}", p.exists(), str(p))

# 2) Cargo.toml license
cargo = ROOT / "Cargo.toml"
if cargo.exists():
    text = cargo.read_text(encoding="utf-8")
    check("Cargo.toml license is set", 'license = "AGPL-3.0-only"' in text)
    check("Cargo.toml license-file is set", 'license-file = "LICENSE"' in text)

# 3) license header in every Rust project source/test file
for path in sorted((ROOT / "src").rglob("*.rs")) + sorted((ROOT / "tests").rglob("*.rs")):
    text = path.read_text(encoding="utf-8")
    check(f"AGPL header in {path}", HEADER_RE.search(text) is not None)

# 4) checksum validation where possible
sha = ROOT / "RELEASES.sha256"
if sha.exists():
    lookup = {}
    for line in sha.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if " *" in line:
            digest, target = line.split(" *", 1)
            lookup[target] = digest
    for rel, expected in lookup.items():
        target = ROOT / rel
        if target.exists():
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            check(f"sha256 matches: {rel}", actual == expected, f"expected={expected} actual={actual}")

if errors:
    print("release lint FAILED")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print("release lint OK")
