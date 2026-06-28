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

use crate::hdc::phase_fhrr::{CartesianFhrrVector, PhaseFhrrVector};

/// A shim/wrapper around FHRR complex cartesian and phase representation vectors.
/// Provides convenient conversions and utility operations.

/// Convert a `CartesianFhrrVector` into a `PhaseFhrrVector` by calculating the phase angles ($\theta = \text{atan2}(im, re)$).
pub fn cartesian_to_phase(v: &CartesianFhrrVector) -> PhaseFhrrVector {
    let phases = ndarray::Zip::from(&v.re)
        .and(&v.im)
        .map_collect(|&re, &im| im.atan2(re));
    PhaseFhrrVector::new(phases)
}

/// Convert a `PhaseFhrrVector` into a `CartesianFhrrVector` by mapping phase angle $\theta$ to complex unit circle ($e^{i\theta} = \cos(\theta) + i\sin(\theta)$).
pub fn phase_to_cartesian(v: &PhaseFhrrVector) -> CartesianFhrrVector {
    let re = v.phases.mapv(|p| p.cos());
    let im = v.phases.mapv(|p| p.sin());
    CartesianFhrrVector::new(re, im)
}
