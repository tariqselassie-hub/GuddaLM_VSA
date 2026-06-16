from pathlib import Path
root = Path(r"C:\Users\zoddj\GuddaLM_VSA")
header = """// Copyright (C) 2025 guddalm_vsa contributors.
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
"""
files = [
    root / "src" / "lib.rs",
    root / "src" / "board.rs",
    root / "src" / "bsc.rs",
    root / "src" / "error.rs",
    root / "src" / "fhrr.rs",
    root / "src" / "map.rs",
    root / "src" / "primitives.rs",
    root / "src" / "primitives_shim.rs",
    root / "src" / "seed.rs",
    root / "src" / "serialize_vsa.rs",
    root / "src" / "setup.rs",
    root / "src" / "vsa.rs",
    root / "src" / "search" / "mod.rs",
    root / "src" / "dnvs" / "mod.rs",
    root / "src" / "dnvs" / "classifier.rs",
    root / "src" / "dnvs" / "config.rs",
    root / "src" / "dnvs" / "encoder.rs",
    root / "src" / "dnvs" / "retrain.rs",
    root / "src" / "hdc" / "mod.rs",
    root / "src" / "hdc" / "accumulator.rs",
    root / "src" / "hdc" / "attention.rs",
    root / "src" / "hdc" / "autograd.rs",
    root / "src" / "hdc" / "bind.rs",
    root / "src" / "hdc" / "bundle.rs",
    root / "src" / "hdc" / "cleanup.rs",
    root / "src" / "hdc" / "experimentation.rs",
    root / "src" / "hdc" / "fhrr.rs",
    root / "src" / "hdc" / "ghrr.rs",
    root / "src" / "hdc" / "graph.rs",
    root / "src" / "hdc" / "permute.rs",
    root / "src" / "hdc" / "phase_fhrr.rs",
    root / "src" / "hdc" / "quantize.rs",
    root / "src" / "hdc" / "resonator.rs",
    root / "src" / "hdc" / "rff.rs",
    root / "src" / "hdc" / "sdm.rs",
    root / "src" / "hdc" / "sequence.rs",
    root / "src" / "hdc" / "stream.rs",
    root / "src" / "hdc" / "tensor.rs",
    root / "src" / "hdc" / "transformer.rs",
    root / "src" / "hdc" / "vector.rs",
    root / "src" / "hdc" / "vsa_trait.rs",
    root / "tests" / "experiments.rs",
    root / "tests" / "gudda_system_check.rs",
]

for path in files:
    original = path.read_text(encoding="utf-8")
    if "SPDX-License-Identifier: AGPL-3.0" in original:
        continue
    path.write_text(header + original, encoding="utf-8")
