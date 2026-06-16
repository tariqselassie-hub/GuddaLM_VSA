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
use crate::error::GuddaResult;

use crate::map::{MapSetup, HDVector as MapVector};
use crate::bsc::{BscSetup, BinaryHDVector as BscVector};
use crate::fhrr::{FhrrSetup, FHRRVector as FhrrVector};

/// Unified representation selector for GuddaLM VSA systems.
///
/// This enum lets generic setup code choose one of the three
/// canonical VSA modes without hard-coding type-specific branches.
#[derive(Debug, Clone, Copy)]
pub enum VsaMode {
    Map,
    Bsc,
    Fhrr,
}

impl VsaMode {
    pub fn default_dim(self) -> usize {
        match self {
            VsaMode::Map => MapSetup::DEFAULT_DIM,
            VsaMode::Bsc => BscSetup::DEFAULT_DIM,
            VsaMode::Fhrr => FhrrSetup::DEFAULT_DIM,
        }
    }
}

/// Common system-level options for any VSA representation.
#[derive(Debug, Clone, Copy)]
pub struct VsaSystemOptions {
    pub mode: VsaMode,
    pub dim: usize,
}

impl Default for VsaSystemOptions {
    fn default() -> Self {
        Self {
            mode: VsaMode::Map,
            dim: MapSetup::DEFAULT_DIM,
        }
    }
}

impl VsaSystemOptions {
    pub fn new(mode: VsaMode, dim: usize) -> Self {
        Self { mode, dim }
    }

    pub fn map(dim: usize) -> Self {
        Self::new(VsaMode::Map, dim)
    }

    pub fn bsc(dim: usize) -> Self {
        Self::new(VsaMode::Bsc, dim)
    }

    pub fn fhrr(dim: usize) -> Self {
        Self::new(VsaMode::Fhrr, dim)
    }
}

/// High-level VSA configuration builder.
///
/// Use this when you want a single system configurator that clearly
/// distinguishes MAP, BSC, and FHRR setups under the same API.
pub struct VsaSystem {
    pub mode: VsaMode,
    pub dim: usize,
}

impl VsaSystem {
    pub fn new(options: VsaSystemOptions) -> Self {
        Self {
            mode: options.mode,
            dim: options.dim,
        }
    }

    pub fn mode(&self) -> VsaMode {
        self.mode
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn random(&self) -> GuddaResult<SystemVector> {
        self.new_vector(|mode, dim| match mode {
            VsaMode::Map => MapSetup::random(dim).into(),
            VsaMode::Bsc => BscSetup::from_hd(&MapSetup::random(dim)).into(),
            VsaMode::Fhrr => FhrrSetup::random(dim).into(),
        })
    }

    pub fn zeros(&self) -> GuddaResult<SystemVector> {
        self.new_vector(|mode, dim| match mode {
            VsaMode::Map => MapSetup::zeros(dim).into(),
            VsaMode::Bsc => BscSetup::from_hd(&MapSetup::zeros(dim)).into(),
            VsaMode::Fhrr => FhrrSetup::zeros(dim).into(),
        })
    }

    #[inline(always)]
    fn new_vector<F>(&self, f: F) -> GuddaResult<SystemVector>
    where
        F: FnOnce(VsaMode, usize) -> SystemVector,
    {
        Ok(f(self.mode, self.dim))
    }
}

/// Opaque vector handle for system-level code that needs to stay
/// representation-agnostic.
///
/// Internally this is just a tagged wrapper around one of the three
/// concrete VSA vector types. Downcast back to concrete types if you
/// need representation-specific APIs.
#[derive(Debug, Clone)]
pub enum SystemVector {
    Map(MapVector),
    Bsc(BscVector),
    Fhrr(FhrrVector),
}

impl SystemVector {
    pub fn dim(&self) -> usize {
        match self {
            SystemVector::Map(v) => v.dim(),
            SystemVector::Bsc(v) => v.dim(),
            SystemVector::Fhrr(v) => v.dim(),
        }
    }
}

impl From<MapVector> for SystemVector {
    fn from(value: MapVector) -> Self {
        SystemVector::Map(value)
    }
}

impl From<BscVector> for SystemVector {
    fn from(value: BscVector) -> Self {
        SystemVector::Bsc(value)
    }
}

impl From<FhrrVector> for SystemVector {
    fn from(value: FhrrVector) -> Self {
        SystemVector::Fhrr(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_setup_creates_vectors() {
        for mode in [VsaMode::Map, VsaMode::Bsc, VsaMode::Fhrr] {
            let options = VsaSystemOptions::new(mode, 1024);
            let system = VsaSystem::new(options);
            let v = system.random().unwrap();
            assert_eq!(v.dim(), 1024);
        }
    }
}
