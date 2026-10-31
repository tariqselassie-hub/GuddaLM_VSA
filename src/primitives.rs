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
//! Generic algebraic primitives for Vector Symbolic Architectures (VSA).
//! These primitives operate on any type implementing the `VsaVector` trait.

use crate::hdc::vsa_trait::VsaVector;

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

/// Bundle a series of vectors into one superposition: v0 ⊕ v1 ⊕ v2 ⊕ ...
///
/// Every input carries equal weight (see [`VsaVector::bundle_many`]). For
/// BSC this is a true per-bit majority vote, so the result differs from
/// folding `bundle` pairwise, which lets the last inputs dominate.
pub fn bundle_sequence<V: VsaVector>(vectors: &[V]) -> V {
    if vectors.is_empty() {
        panic!("bundle_sequence: empty vector slice");
    }
    V::bundle_many(vectors)
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

/// Encode a sequence of vectors using a sliding N-gram window:
/// ⊕_i ( v_i ⊗ permute(v_{i+1}, 1) ⊗ permute(v_{i+2}, 2) ⊗ ... ⊗ permute(v_{i+n-1}, n-1) )
pub fn encode_ngram_sequence<V: VsaVector>(vectors: &[V], n: usize) -> V {
    if vectors.is_empty() || n == 0 {
        return V::zero(0);
    }
    if vectors.len() < n {
        // Fallback to binding whatever we have
        return bind_sequence(vectors);
    }

    // First window
    let mut first_window = vectors[0].clone();
    for (j, v) in vectors.iter().enumerate().take(n).skip(1) {
        let permuted = v.permute(j);
        first_window = first_window.bind(&permuted);
    }

    let mut result = first_window;
    for i in 1..=(vectors.len() - n) {
        let mut window_bound = vectors[i].clone();
        for j in 1..n {
            let permuted = vectors[i + j].permute(j);
            window_bound = window_bound.bind(&permuted);
        }
        result = result.bundle(&window_bound);
    }
    result
}

/// Encode a set of directed graph edges using role-filler binding: ⊕_(u, v) ( source_marker ⊗ u ⊕ sink_marker ⊗ v )
/// This is the classic commutative VSA representation for graphs.
pub fn encode_graph_edges<V: VsaVector>(edges: &[(V, V)], source_marker: &V, sink_marker: &V) -> V {
    if edges.is_empty() {
        return V::zero(0);
    }
    let mut result = edges[0]
        .0
        .bind(source_marker)
        .bundle(&edges[0].1.bind(sink_marker));
    for (u, v) in &edges[1..] {
        let edge = u.bind(source_marker).bundle(&v.bind(sink_marker));
        result = result.bundle(&edge);
    }
    result
}

/// Encode a set of directed edges using non-commutative binding (such as GHRR): ⊕_(u, v) ( u ⊗ v )
pub fn encode_non_commutative_edges<V: VsaVector>(edges: &[(V, V)]) -> V {
    if edges.is_empty() {
        return V::zero(0);
    }
    let mut result = edges[0].0.bind(&edges[0].1);
    for (u, v) in &edges[1..] {
        let edge = u.bind(v);
        result = result.bundle(&edge);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::fhrr::FHRRVector;
    use crate::hdc::vector::{BinaryHDVector, HDVector};
    use crate::hdc::vsa_trait::IndexVector;

    #[test]
    fn bundle_sequence_map_equals_pairwise_sum() {
        let vs: Vec<HDVector> = (0..9).map(|_| HDVector::random(1000)).collect();
        let pairwise = vs[1..].iter().fold(vs[0].clone(), |a, b| a.bundle(b));
        let bundled = bundle_sequence(&vs);
        for (x, y) in bundled.data().iter().zip(pairwise.data().iter()) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn bundle_sequence_bsc_is_exact_majority() {
        let vs: Vec<BinaryHDVector> = (0..7).map(|_| BinaryHDVector::random(1000)).collect();
        assert_eq!(
            bundle_sequence(&vs),
            BinaryHDVector::majority_bundle_all(&vs)
        );
    }

    #[test]
    fn bundle_sequence_bsc_keeps_every_member() {
        let vs: Vec<BinaryHDVector> = (0..15).map(|_| BinaryHDVector::random(8192)).collect();
        let bundled = bundle_sequence(&vs);
        for (i, v) in vs.iter().enumerate() {
            assert!(bundled.bipolar_similarity(v) > 0.1, "member {i} lost");
        }
    }

    #[test]
    fn bundle_sequence_index_vector_matches_bsc() {
        let raw: Vec<BinaryHDVector> = (0..5).map(|_| BinaryHDVector::random(512)).collect();
        let wrapped: Vec<IndexVector> = raw.iter().cloned().map(IndexVector::new).collect();
        assert_eq!(
            bundle_sequence(&wrapped).into_inner(),
            bundle_sequence(&raw)
        );
    }

    #[test]
    fn bundle_sequence_single_input_and_default_fold() {
        let a = BinaryHDVector::random(256);
        assert_eq!(bundle_sequence(std::slice::from_ref(&a)), a);
        // FHRR keeps the trait's default pairwise fold.
        let fs: Vec<FHRRVector> = (0..3).map(|_| FHRRVector::random(64)).collect();
        let expected = fs[0].bundle(&fs[1]).bundle(&fs[2]);
        assert_eq!(bundle_sequence(&fs).phases(), expected.phases());
    }

    #[test]
    #[should_panic(expected = "empty vector slice")]
    fn bundle_sequence_empty_panics() {
        let empty: Vec<HDVector> = Vec::new();
        bundle_sequence(&empty);
    }
}
