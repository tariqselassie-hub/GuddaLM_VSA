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
/// FHRR / Fourier holographic reduced representation setup and surface.
///
/// This module makes FHRR a first-class subsystem:
/// `guddalm_vsa::fhrr::FHRRVector`, `FhrrSetup`, and helpers.
pub use crate::hdc::fhrr::FHRRVector;

/// FHRR setup configurator.
#[derive(Debug, Clone, Copy)]
pub struct FhrrSetup;

impl FhrrSetup {
    pub const DEFAULT_DIM: usize = 1024;

    #[inline(always)]
    pub fn random(dim: usize) -> FHRRVector {
        FHRRVector::random(dim)
    }

    #[inline(always)]
    pub fn zeros(dim: usize) -> FHRRVector {
        FHRRVector::zeros(dim)
    }

    #[inline(always)]
    pub fn from_phases(phases: &[f64]) -> FHRRVector {
        FHRRVector::from_phases(phases)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fhrr_setup_defaults() {
        let v = FhrrSetup::random(FhrrSetup::DEFAULT_DIM);
        assert_eq!(v.dim(), FhrrSetup::DEFAULT_DIM);
    }
}
