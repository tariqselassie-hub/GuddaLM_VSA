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
    let bound: Vec<V> = pairs.iter().map(|(k, v)| k.bind(v)).collect();
    V::bundle_many(&bound)
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
    let permuted: Vec<V> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| v.permute(i))
        .collect();
    V::bundle_many(&permuted)
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

    let windows: Vec<V> = vectors
        .windows(n)
        .map(|w| {
            w.iter()
                .enumerate()
                .skip(1)
                .fold(w[0].clone(), |acc, (j, v)| acc.bind(&v.permute(j)))
        })
        .collect();
    V::bundle_many(&windows)
}

/// Encode a set of directed graph edges using role-filler binding: ⊕_(u, v) ( source_marker ⊗ u ⊕ sink_marker ⊗ v )
/// This is the classic commutative VSA representation for graphs.
pub fn encode_graph_edges<V: VsaVector>(edges: &[(V, V)], source_marker: &V, sink_marker: &V) -> V {
    if edges.is_empty() {
        return V::zero(0);
    }
    // Every role-filler term goes into one superposition, rather than
    // bundling each edge's pair first and then folding edges together.
    let terms: Vec<V> = edges
        .iter()
        .flat_map(|(u, v)| [u.bind(source_marker), v.bind(sink_marker)])
        .collect();
    V::bundle_many(&terms)
}

/// Encode a set of directed edges using non-commutative binding (such as GHRR): ⊕_(u, v) ( u ⊗ v )
pub fn encode_non_commutative_edges<V: VsaVector>(edges: &[(V, V)]) -> V {
    if edges.is_empty() {
        return V::zero(0);
    }
    let bound: Vec<V> = edges.iter().map(|(u, v)| u.bind(v)).collect();
    V::bundle_many(&bound)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::fhrr::FHRRVector;
    use crate::hdc::ghrr::GHRRVector;
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
    fn bundle_sequence_single_input_and_phase_types() {
        let a = BinaryHDVector::random(256);
        assert_eq!(bundle_sequence(std::slice::from_ref(&a)), a);
        // FHRR and GHRR sum everything and normalize once.
        let fs: Vec<FHRRVector> = (0..3).map(|_| FHRRVector::random(64)).collect();
        assert_eq!(
            bundle_sequence(&fs).phases(),
            FHRRVector::bundle_all(&fs).phases()
        );
        let gs: Vec<GHRRVector> = (0..3).map(|_| GHRRVector::random(16)).collect();
        let (got, want) = (bundle_sequence(&gs), GHRRVector::bundle_all(&gs));
        assert!(got
            .data
            .iter()
            .zip(want.data.iter())
            .all(|(a, b)| a.re == b.re && a.im == b.im));
    }

    #[test]
    fn encoders_match_old_sum_for_map() {
        // MAP bundling is a plain sum, so the one-shot encoders must equal
        // the old fold that started from a zero vector.
        let dim = 512;
        let ks: Vec<HDVector> = (0..6).map(|_| HDVector::random(dim)).collect();
        let vs: Vec<HDVector> = (0..6).map(|_| HDVector::random(dim)).collect();
        let pairs: Vec<(HDVector, HDVector)> = ks.iter().cloned().zip(vs.iter().cloned()).collect();
        let close = |a: &HDVector, b: &HDVector| {
            a.data()
                .iter()
                .zip(b.data().iter())
                .all(|(x, y)| (x - y).abs() < 1e-9)
        };

        let old_set = pairs
            .iter()
            .fold(HDVector::zeros(dim), |acc, (k, v)| acc.bundle(&k.bind(v)));
        assert!(close(&encode_set(&pairs), &old_set));

        let old_seq = ks
            .iter()
            .enumerate()
            .fold(HDVector::zeros(dim), |acc, (i, v)| {
                acc.bundle(&v.permute(i))
            });
        assert!(close(&encode_positional_sequence(&ks), &old_seq));

        let (src, snk) = (HDVector::random(dim), HDVector::random(dim));
        let old_edges = pairs.iter().fold(HDVector::zeros(dim), |acc, (u, v)| {
            acc.bundle(&u.bind(&src).bundle(&v.bind(&snk)))
        });
        assert!(close(&encode_graph_edges(&pairs, &src, &snk), &old_edges));

        let old_nc = pairs
            .iter()
            .fold(HDVector::zeros(dim), |acc, (u, v)| acc.bundle(&u.bind(v)));
        assert!(close(&encode_non_commutative_edges(&pairs), &old_nc));
    }

    #[test]
    fn encoders_bundle_in_one_step() {
        let dim = 1000;
        let ks: Vec<BinaryHDVector> = (0..7).map(|_| BinaryHDVector::random(dim)).collect();
        let vs: Vec<BinaryHDVector> = (0..7).map(|_| BinaryHDVector::random(dim)).collect();
        let pairs: Vec<(BinaryHDVector, BinaryHDVector)> =
            ks.iter().cloned().zip(vs.iter().cloned()).collect();

        let bound: Vec<BinaryHDVector> = pairs.iter().map(|(k, v)| k.bind(v)).collect();
        assert_eq!(
            encode_set(&pairs),
            BinaryHDVector::majority_bundle_all(&bound)
        );

        let permuted: Vec<BinaryHDVector> =
            ks.iter().enumerate().map(|(i, v)| v.permute(i)).collect();
        assert_eq!(
            encode_positional_sequence(&ks),
            BinaryHDVector::majority_bundle_all(&permuted)
        );

        let fs: Vec<FHRRVector> = (0..5).map(|_| FHRRVector::random(128)).collect();
        let fpairs: Vec<(FHRRVector, FHRRVector)> =
            fs.iter().cloned().zip(fs.iter().rev().cloned()).collect();
        let fbound: Vec<FHRRVector> = fpairs.iter().map(|(k, v)| k.bind(v)).collect();
        assert_eq!(
            encode_set(&fpairs).phases(),
            FHRRVector::bundle_all(&fbound).phases()
        );
    }

    #[test]
    fn ngram_sequence_bundles_every_window() {
        let dim = 512;
        let vs: Vec<HDVector> = (0..6).map(|_| HDVector::random(dim)).collect();
        let window = |i: usize| {
            vs[i]
                .bind(&vs[i + 1].permute(1))
                .bind(&vs[i + 2].permute(2))
        };
        let expected = (1..4).fold(window(0), |acc, i| acc.bundle(&window(i)));
        let got = encode_ngram_sequence(&vs, 3);
        for (x, y) in got.data().iter().zip(expected.data().iter()) {
            assert!((x - y).abs() < 1e-9);
        }
        // Fewer vectors than n falls back to binding them all.
        assert_eq!(encode_ngram_sequence(&vs[..2], 3), bind_sequence(&vs[..2]));
    }

    #[test]
    fn empty_inputs_return_empty_vectors() {
        let none: Vec<(HDVector, HDVector)> = Vec::new();
        assert_eq!(encode_set(&none).dim(), 0);
        assert_eq!(encode_positional_sequence::<HDVector>(&[]).dim(), 0);
        assert_eq!(encode_ngram_sequence::<HDVector>(&[], 3).dim(), 0);
        let m = HDVector::random(8);
        assert_eq!(encode_graph_edges(&none, &m, &m).dim(), 0);
        assert_eq!(encode_non_commutative_edges(&none).dim(), 0);
    }

    #[test]
    #[should_panic(expected = "empty vector slice")]
    fn bundle_sequence_empty_panics() {
        let empty: Vec<HDVector> = Vec::new();
        bundle_sequence(&empty);
    }
}
