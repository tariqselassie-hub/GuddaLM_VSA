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
use crate::hdc::vector::HDVector;

/// Encode an n-gram sequence by binding each token at its position.
///
/// Given tokens [t₀, t₁, ..., tₙ], produces:
///   result = t₀ ⊛ permute(t₁, 1) ⊛ permute(t₂, 2) ⊛ ... ⊛ permute(tₙ, n)
///
/// Each token is permuted by its index before binding. This creates a
/// position-dependent composite that is invertible: given the result and
/// n-1 of the tokens, the remaining token can be recovered by unbinding.
pub fn ngram_encode(tokens: &[HDVector]) -> HDVector {
    if tokens.is_empty() {
        return HDVector::zeros(0);
    }
    let mut combined = tokens[0].clone();
    for (i, token) in tokens.iter().enumerate().skip(1) {
        let permuted = token.permute(i);
        combined = combined.bind(&permuted);
    }
    combined
}

/// Encode a single token at a given position using cyclic shift.
///
/// The position is encoded by permuting the token vector via cyclic shift
/// by `position` elements. This is invertible (shift back by dim - position).
/// Unlike sinusoidal positional encoding, this is the HD-native approach:
/// permutation distributes position information uniformly across all
/// dimensions without introducing learned position vectors.
pub fn positional_encode(token: &HDVector, position: usize) -> HDVector {
    let shift = position % token.dim();
    token.permute(shift)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_encode_empty() {
        let result = ngram_encode(&[]);
        assert_eq!(result.dim(), 0, "empty -> dim 0");
    }

    #[test]
    fn test_ngram_encode_single() {
        let v = HDVector::random(256);
        let result = ngram_encode(&[v.clone()]);
        assert_eq!(result, v, "single token -> equals itself");
    }

    #[test]
    fn test_ngram_encode_order_matters() {
        let a = HDVector::random(256);
        let b = HDVector::random(256);
        let ab = ngram_encode(&[a.clone(), b.clone()]);
        let ba = ngram_encode(&[b, a]);
        let sim = ab.cosine_similarity(&ba);
        assert!(sim < 0.9, "order must matter (sim={sim})");
    }

    #[test]
    fn test_ngram_encode_dimension() {
        let a = HDVector::random(128);
        let b = HDVector::random(128);
        let result = ngram_encode(&[a, b]);
        assert_eq!(result.dim(), 128);
    }

    #[test]
    fn test_positional_encode_identity() {
        let v = HDVector::random(256);
        let p0 = positional_encode(&v, 0);
        assert_eq!(p0, v, "position 0 -> identity");
    }

    #[test]
    fn test_positional_encode_different_positions() {
        let v = HDVector::random(256);
        let p1 = positional_encode(&v, 1);
        let p2 = positional_encode(&v, 2);
        let sim = p1.cosine_similarity(&p2);
        assert!(sim < 0.9, "different positions should produce different vectors (sim={sim})");
    }

    #[test]
    fn test_positional_encode_mod_dim() {
        let v = HDVector::random(256);
        let p0 = positional_encode(&v, 0);
        let p256 = positional_encode(&v, 256);
        assert_eq!(p0, p256, "position % dim -> same result");
        let p1 = positional_encode(&v, 1);
        let p257 = positional_encode(&v, 257);
        assert_eq!(p1, p257, "position % dim -> same result after wrap");
    }
}
