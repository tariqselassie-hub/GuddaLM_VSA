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
use crate::hdc::phase_fhrr::CartesianFhrrVector;
use ndarray::Array1;

#[inline(always)]
pub fn bind(a: &CartesianFhrrVector, b: &CartesianFhrrVector) -> CartesianFhrrVector {
    a.bind(b)
}

#[inline(always)]
pub fn unbind(a: &CartesianFhrrVector, b: &CartesianFhrrVector) -> CartesianFhrrVector {
    let b_conj = CartesianFhrrVector::new(b.re.clone(), -&b.im);
    a.bind(&b_conj)
}

#[inline(always)]
pub fn bundle(a: &CartesianFhrrVector, b: &CartesianFhrrVector) -> CartesianFhrrVector {
    a.bundle(b)
}

#[inline(always)]
pub fn bundle_accumulate(acc: &mut CartesianFhrrVector, b: &CartesianFhrrVector) {
    acc.bundle_accumulate(b)
}

#[inline(always)]
pub fn similarity(a: &CartesianFhrrVector, b: &CartesianFhrrVector) -> f32 {
    a.similarity(b)
}

#[inline(always)]
pub fn permute(v: &CartesianFhrrVector, shift: isize) -> CartesianFhrrVector {
    v.permute(shift)
}

#[inline(always)]
pub fn zero(dim: usize) -> CartesianFhrrVector {
    CartesianFhrrVector::new(Array1::zeros(dim), Array1::zeros(dim))
}
