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
//! Default hypervector dimensions used across GuddaLM_VSA representations.
//!
//! | Representation | Constant | Default | Rationale |
//! |---|---|---|---|
//! | MAP (bipolar) | [`MAP_DEFAULT_DIM`] | 4096 | FFT bind sweet spot; matches cleanup packed4096 |
//! | BSC (binary) | [`BSC_DEFAULT_DIM`] | 4096 | Same capacity target as MAP |
//! | FHRR (phase) | [`FHRR_DEFAULT_DIM`] | 1024 | Higher bundling capacity per dim; cheaper complex ops |
//! | DNVS (spatial) | [`DNVS_DEFAULT_DIM`] | 10000 | One position vector per pixel index in MNIST-scale inputs |

/// Default dimension for MAP / bipolar hypervectors.
pub const MAP_DEFAULT_DIM: usize = 4096;

/// Default dimension for binary spatter coding (BSC).
pub const BSC_DEFAULT_DIM: usize = 4096;

/// Default dimension for Fourier HRR (FHRR).
pub const FHRR_DEFAULT_DIM: usize = 1024;

/// Default dimension for DNVS spatial encoding (position vector count).
pub const DNVS_DEFAULT_DIM: usize = 10000;
