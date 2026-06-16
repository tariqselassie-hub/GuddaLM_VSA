// Copyright (C) 2025 guddalm_vsa contributors.
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
use crate::hdc::quantize::packed_similarity;
use crate::hdc::vector::HDVector;

/// BSC / binary spatter coding setup and surface.
///
/// This module makes the BSC representation a first-class subsystem:
/// `guddalm_vsa::bsc::BinaryHDVector`, `BscSetup`, and helpers.
pub use crate::hdc::vector::BinaryHDVector;

/// BSC setup configurator.
#[derive(Debug, Clone, Copy)]
pub struct BscSetup;

impl BscSetup {
    pub const DEFAULT_DIM: usize = 4096;

    /// Convert a MAP bipolar vector into BSC bit-packed form.
    #[inline(always)]
    pub fn from_hd(vector: &HDVector) -> BinaryHDVector {
        BinaryHDVector::from_bipolar(vector)
    }

    /// Fast popcount-based similarity between a bipolar vector and BSC signature.
    #[inline(always)]
    pub fn packed_similarity(vector: &HDVector, packed: &[u64]) -> f64 {
        packed_similarity(vector, packed)
    }

    /// Human-readable similarity in dB-like units: `10 * log10((1+sim)/(1-sim))`.
    #[inline(always)]
    pub fn similarity_db(vector: &HDVector, packed: &[u64]) -> f64 {
        let sim = Self::packed_similarity(vector, packed).clamp(-1.0, 1.0);
        let denom = 1.0 - sim;
        if denom <= 0.0 {
            f64::INFINITY
        } else {
            10.0 * ((1.0 + sim) / denom).log10()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bsc_setup_round_trip() {
        let a = HDVector::random(4096);
        let _b = BscSetup::from_hd(&a);
        let packed = crate::hdc::quantize::pack_bits(&a);
        assert!((BscSetup::packed_similarity(&a, &packed) - 1.0).abs() < 1e-6);
    }
}
