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
/// MAP / bipolar VSA setup and surface.
///
/// This module exists to make the MAP representation a first-class,
/// independently addressable subsystem under `guddalm_vsa::map`.
///
/// It re-exports `HDVector` and map-specific helpers so callers can
/// write:
/// ```ignore
/// use guddalm_vsa::map::{HDVector, MapSetup};
/// ```
pub use crate::hdc::vector::HDVector;

/// MAP setup configurator.
///
/// Use this when you want an explicit "this is MAP mode" entrypoint
/// rather than reaching for the generic `HDVector` type directly.
#[derive(Debug, Clone, Copy)]
pub struct MapSetup;

impl MapSetup {
    pub const DEFAULT_DIM: usize = 4096;

    #[inline(always)]
    pub fn random(dim: usize) -> HDVector {
        HDVector::random(dim)
    }

    #[inline(always)]
    pub fn zeros(dim: usize) -> HDVector {
        HDVector::zeros(dim)
    }

    #[inline(always)]
    pub fn from_slice(slice: &[f64]) -> HDVector {
        HDVector::from_slice(slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_setup_defaults() {
        let v = MapSetup::random(MapSetup::DEFAULT_DIM);
        assert_eq!(v.dim(), MapSetup::DEFAULT_DIM);
    }
}
