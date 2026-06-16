from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

root = Path(r"C:\Users\zoddj\GuddaLM_VSA")
gen = root / "scripts" / "generated"
lockfile = root / "Cargo.lock"

gen.mkdir(exist_ok=True)

# Parse Cargo.lock for dependencies
section = []
in_dep = False
name = version = None
deps = []

for line in lockfile.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line.startswith("[[package]]"):
        in_dep = True
        name = version = None
        continue
    if not in_dep:
        continue
    if not line:
        in_dep = False
        if name and name != "guddalm_vsa":
            deps.append((name, version, "unknown"))
        continue
    if line.startswith("name = "):
        name = line.split("=", 1)[1].strip().strip('"')
    elif line.startswith("version = "):
        version = line.split("=", 1)[1].strip().strip('"')

notice_lines = [
    "Dependency name,Version,License(as declared by upstream)",
    "guddalm_vsa,0.1.0,AGPL-3.0",
]
for dep_name, dep_version, _ in deps:
    notice_lines.append(f"{dep_name},{dep_version},(see package metadata)")

(gen / "NOTICE").write_text("\n".join(notice_lines) + "\n", encoding="utf-8")

notice_md = [
    "# Third Party Notices",
    "",
    f"_Generated {datetime.now(timezone.utc).date().isoformat()} from `Cargo.lock`._",
    "",
    "| Dependency | Version | License | Source |",
    "| --- | --- | --- | --- |",
    "| guddalm_vsa | 0.1.0 | AGPL-3.0 | local |",
]
for dep_name, dep_version, _ in deps:
    crates_io = f"https://crates.io/crates/{dep_name}"
    notice_md.append(f"| {dep_name} | {dep_version} | (see crates.io metadata) | {crates_io} |")
(gen / "THIRD_PARTY_NOTICES.md").write_text("\n".join(notice_md) + "\n\n", encoding="utf-8")

# Minimal SPDX 2.3 JSON
packages = []
for dep_name, dep_version, _ in deps:
    packages.append({
        "name": dep_name,
        "SPDXID": f"SPDXRef-Package-{dep_name}",
        "versionInfo": dep_version,
        "supplier": "NOASSERTION",
        "downloadLocation": f"https://crates.io/crates/{dep_name}",
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": "NOASSERTION",
        "filesAnalyzed": False,
    })

spdx = {
    "spdxVersion": "SPDX-2.3",
    "dataLicense": "CC0-1.0",
    "SPDXID": "SPDXRef-DOCUMENT",
    "name": root.name,
    "documentNamespace": f"https://spdx.guddalm_vsa/{root.name}-{datetime.now(timezone.utc).date().isoformat()}",
    "creationInfo": {
        "created": datetime.now(timezone.utc).isoformat(),
        "creators": ["Tool: guddalm_vsa-release-scripts"]
    },
    "packages": packages,
}
(gen / "sbom.spdx.json").write_text(json.dumps(spdx, indent=2) + "\n", encoding="utf-8")

# Minimal CycloneDX JSON
cdx = {
    "bomFormat": "CycloneDX",
    "specVersion": "1.4",
    "version": 1,
    "metadata": {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": {
            "name": root.name,
            "type": "application",
            "licenses": [{"license": {"id": "AGPL-3.0"}}],
        },
    },
    "components": [],
}
for dep_name, dep_version, _ in deps:
    cdx["components"].append({
        "type": "library",
        "name": dep_name,
        "version": dep_version,
        "purl": f"pkg:cargo/{dep_name}",
        "licenses": [{"expression": "NOASSERTION"}],
        "externalReferences": [
            {
                "type": "distribution",
                "url": f"https://crates.io/crates/{dep_name}",
            }
        ],
    })
(gen / "sbom.cdx.json").write_text(json.dumps(cdx, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

# SHA256 of source files and compliance docs
files = [
    root / "LICENSE",
    root / "Cargo.toml",
    root / "Cargo.lock",
    gen / "NOTICE",
    gen / "THIRD_PARTY_NOTICES.md",
    gen / "sbom.spdx.json",
    gen / "sbom.cdx.json",
]
files.extend(sorted((root / "src").rglob("*.rs")))
files.extend(sorted((root / "tests").rglob("*.rs")))

lines = ["# Release checksums", ""]
for path in files:
    rel = path.relative_to(root)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest} *{rel}")
(root / "RELEASES.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
