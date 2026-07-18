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

use crate::hdc::vsa_trait::VsaVector;

/// Generic algebraic primitives for Vector Symbolic Architectures (VSA).
/// These primitives operate on any type implementing the `VsaVector` trait.

/// Bind a series of vectors sequentially: v0 ⊗ v1 ⊗ v2 ⊗ ...
pub fn bind_sequence<V: VsaVector>(vectors: &[V]) -> V {
    if vectors.is_empty() {
        panic!("bind_sequence: empty vector slice");
    }
    let mut result = vectors[0].clone();
    for v in &vectors[1..] {
        result = result.bind(v);
    }
    result
}

/// Bundle a series of vectors together using pairwise superposition: v0 ⊕ v1 ⊕ v2 ⊕ ...
pub fn bundle_sequence<V: VsaVector>(vectors: &[V]) -> V {
    if vectors.is_empty() {
        panic!("bundle_sequence: empty vector slice");
    }
    let mut result = vectors[0].clone();
    for v in &vectors[1..] {
        result = result.bundle(v);
    }
    result
}

/// Encode a set of key-value pairs into a single composite representation: ⊕_i (key_i ⊗ value_i)
pub fn encode_set<V: VsaVector>(pairs: &[(V, V)]) -> V {
    if pairs.is_empty() {
        return V::zero(0);
    }
    let dim = pairs[0].0.dim();
    let mut result = V::zero(dim);
    for (k, v) in pairs {
        let bound = k.bind(v);
        result = result.bundle(&bound);
    }
    result
}

/// Decode a value from a composite representation given its key: unbind(set, key)
pub fn decode_set<V: VsaVector>(set: &V, key: &V) -> V {
    set.unbind(key)
}

/// Encode an ordered sequence of vectors using positional permutation: ⊕_i permute(v_i, i)
pub fn encode_positional_sequence<V: VsaVector>(vectors: &[V]) -> V {
    if vectors.is_empty() {
        return V::zero(0);
    }
    let dim = vectors[0].dim();
    let mut result = V::zero(dim);
    for (i, v) in vectors.iter().enumerate() {
        let permuted = v.permute(i);
        result = result.bundle(&permuted);
    }
    result
}
