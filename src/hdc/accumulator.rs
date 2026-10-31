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
//! Streaming bundlers that accumulate many hypervectors before thresholding.
//!
//! Folding `bundle` pairwise allocates a new vector per input, and for BSC it
//! is lossy: each pairwise majority halves the weight of everything bundled
//! so far, so the result is dominated by the last few inputs. These
//! accumulators keep running per-component totals and threshold once, so
//! every input contributes equally and memory stays at one vector's worth.
//!
//! ```
//! use guddalm_vsa::prelude::*;
//!
//! let items: Vec<BinaryHDVector> = (0..9).map(|_| BinaryHDVector::random(4096)).collect();
//! let mut acc = BinaryBundleAccumulator::new(4096);
//! acc.extend(&items);
//! let bundle = acc.binarize();
//! // Every input stays recognisable in the bundle.
//! assert!(items.iter().all(|v| bundle.bipolar_similarity(v) > 0.15));
//! ```
use crate::hdc::vector::{BinaryHDVector, HDVector, majority_from_sums};

/// Streaming accumulator for MAP (bipolar) bundling.
///
/// Accumulates element-wise weighted sums incrementally and produces
/// a binarized HDVector on demand. Avoids repeated intermediate
/// allocations when bundling a stream of vectors.
#[derive(Clone, Debug, PartialEq)]
pub struct BundleAccumulator {
    dim: usize,
    sum: Vec<f64>,
    count: usize,
}

impl BundleAccumulator {
    pub fn new(dim: usize) -> Self {
        BundleAccumulator {
            dim,
            sum: vec![0.0; dim],
            count: 0,
        }
    }

    /// Add a vector with unit weight.
    #[inline(always)]
    pub fn add(&mut self, v: &HDVector) {
        assert_eq!(self.dim, v.dim());
        for (s, d) in self.sum.iter_mut().zip(v.data().iter()) {
            *s += d;
        }
        self.count += 1;
    }

    /// Add a vector with a custom weight.
    #[inline(always)]
    pub fn add_weighted(&mut self, v: &HDVector, weight: f64) {
        assert_eq!(self.dim, v.dim());
        for (s, d) in self.sum.iter_mut().zip(v.data().iter()) {
            *s += weight * d;
        }
        self.count += 1;
    }

    /// Fold another accumulator into this one, e.g. to combine partial
    /// sums built on separate threads.
    pub fn merge(&mut self, other: &BundleAccumulator) {
        assert_eq!(self.dim, other.dim);
        for (s, d) in self.sum.iter_mut().zip(other.sum.iter()) {
            *s += d;
        }
        self.count += other.count;
    }

    /// Current accumulated sum (not binarized).
    pub fn get(&self) -> HDVector {
        HDVector::from_slice(&self.sum)
    }

    /// Consume the accumulator and return the sum without copying it.
    pub fn into_vector(self) -> HDVector {
        HDVector::from_vec(self.sum)
    }

    /// Binarize the accumulated sum to a bipolar vector.
    pub fn binarize(&self) -> HDVector {
        if self.count == 0 {
            return HDVector::zeros(self.dim);
        }
        HDVector::from_slice(&self.sum).binarize()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of vectors accumulated so far.
    pub fn count(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Reset the accumulator (zero out sum and count).
    pub fn reset(&mut self) {
        self.sum.fill(0.0);
        self.count = 0;
    }
}

impl<'a> Extend<&'a HDVector> for BundleAccumulator {
    fn extend<I: IntoIterator<Item = &'a HDVector>>(&mut self, iter: I) {
        for v in iter {
            self.add(v);
        }
    }
}

/// Streaming accumulator for BSC (binary) bundling.
///
/// Accumulates per-dimension votes as signed counters and produces
/// a BinaryHDVector via majority rule on demand.
#[derive(Clone, Debug, PartialEq)]
pub struct BinaryBundleAccumulator {
    dim: usize,
    sums: Vec<i64>,
    count: usize,
}

impl BinaryBundleAccumulator {
    pub fn new(dim: usize) -> Self {
        BinaryBundleAccumulator {
            dim,
            sums: vec![0i64; dim],
            count: 0,
        }
    }

    /// Add a binary vector: +1 for 1-bits, -1 for 0-bits.
    #[inline(always)]
    pub fn add(&mut self, v: &BinaryHDVector) {
        self.add_votes(v, 1);
    }

    /// Add a binary vector with a custom weight.
    #[inline(always)]
    pub fn add_weighted(&mut self, v: &BinaryHDVector, weight: f64) {
        self.add_votes(v, weight as i64);
    }

    /// Add ±`w` per bit, one packed byte at a time: a lookup table turns each
    /// byte into eight ±1 votes, so the inner loop is a fixed-width add the
    /// compiler can vectorize.
    #[inline(always)]
    fn add_votes(&mut self, v: &BinaryHDVector, w: i64) {
        assert_eq!(self.dim, v.dim());
        let lut = vote_lut();
        let bytes = v.words().iter().flat_map(|word| word.to_le_bytes());
        for (chunk, byte) in self.sums.chunks_mut(8).zip(bytes) {
            let votes = &lut[byte as usize];
            if let Ok(lanes) = <&mut [i64; 8]>::try_from(&mut *chunk) {
                for i in 0..8 {
                    lanes[i] += votes[i] as i64 * w;
                }
            } else {
                // Trailing partial byte when `dim` is not a multiple of 8.
                for (s, &vote) in chunk.iter_mut().zip(votes.iter()) {
                    *s += vote as i64 * w;
                }
            }
        }
        self.count += 1;
    }

    /// Fold another accumulator into this one, e.g. to combine partial
    /// counts built on separate threads.
    pub fn merge(&mut self, other: &BinaryBundleAccumulator) {
        assert_eq!(self.dim, other.dim);
        for (s, d) in self.sums.iter_mut().zip(other.sums.iter()) {
            *s += d;
        }
        self.count += other.count;
    }

    /// Produce the majority-rule binary vector from accumulated votes.
    pub fn binarize(&self) -> BinaryHDVector {
        majority_from_sums(&self.sums, self.dim)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of vectors accumulated.
    pub fn count(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Reset the accumulator.
    pub fn reset(&mut self) {
        self.sums.fill(0);
        self.count = 0;
    }
}

impl<'a> Extend<&'a BinaryHDVector> for BinaryBundleAccumulator {
    fn extend<I: IntoIterator<Item = &'a BinaryHDVector>>(&mut self, iter: I) {
        for v in iter {
            self.add(v);
        }
    }
}

/// `VOTE_LUT[b][i]` is +1 if bit `i` of byte `b` is set, else −1.
fn vote_lut() -> &'static [[i8; 8]; 256] {
    static LUT: std::sync::OnceLock<[[i8; 8]; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [[0i8; 8]; 256];
        for (b, row) in lut.iter_mut().enumerate() {
            for (i, vote) in row.iter_mut().enumerate() {
                *vote = if (b >> i) & 1 == 1 { 1 } else { -1 };
            }
        }
        lut
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundle_accumulator() {
        let dim = 1000;
        let a = HDVector::random(dim);
        let b = HDVector::random(dim);

        let mut acc = BundleAccumulator::new(dim);
        assert_eq!(acc.count(), 0);

        acc.add(&a);
        acc.add(&b);
        assert_eq!(acc.count(), 2);

        let bundled = acc.binarize();
        let sim_a = bundled.cosine_similarity(&a);
        let sim_b = bundled.cosine_similarity(&b);
        // Both should have reasonable similarity
        assert!(sim_a > 0.0);
        assert!(sim_b > 0.0);

        acc.reset();
        assert_eq!(acc.count(), 0);
    }

    #[test]
    fn test_bundle_accumulator_weighted() {
        let dim = 1000;
        let a = HDVector::random(dim);
        let b = HDVector::random(dim);

        let mut acc = BundleAccumulator::new(dim);
        acc.add_weighted(&a, 2.0);
        acc.add_weighted(&b, 0.5);

        let bundled = acc.binarize();
        // Heavily weighted a should dominate
        let sim_a = bundled.cosine_similarity(&a);
        let sim_b = bundled.cosine_similarity(&b);
        assert!(sim_a > sim_b);
    }

    #[test]
    fn test_binary_bundle_accumulator() {
        let dim = 1000;
        let a = BinaryHDVector::random(dim);
        let b = BinaryHDVector::random(dim);

        let mut acc = BinaryBundleAccumulator::new(dim);
        assert_eq!(acc.count(), 0);

        acc.add(&a);
        acc.add(&b);
        assert_eq!(acc.count(), 2);

        let bundled = acc.binarize();
        let sim_a = bundled.hamming_similarity(&a);
        let sim_b = bundled.hamming_similarity(&b);
        assert!(sim_a > 0.0);
        assert!(sim_b > 0.0);
    }

    #[test]
    fn map_sum_matches_pairwise_bundle() {
        let vs: Vec<HDVector> = (0..7).map(|_| HDVector::random(1000)).collect();
        let mut acc = BundleAccumulator::new(1000);
        acc.extend(&vs);
        let pairwise = vs[1..].iter().fold(vs[0].clone(), |a, b| a.bundle(b));
        assert_eq!(acc.count(), 7);
        assert_eq!(acc.clone().into_vector(), acc.get());
        for (x, y) in acc.get().data().iter().zip(pairwise.data().iter()) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn map_merge_equals_single_pass() {
        let vs: Vec<HDVector> = (0..6).map(|_| HDVector::random(512)).collect();
        let mut whole = BundleAccumulator::new(512);
        whole.extend(&vs);
        let mut a = BundleAccumulator::new(512);
        a.extend(&vs[..2]);
        let mut b = BundleAccumulator::new(512);
        b.extend(&vs[2..]);
        a.merge(&b);
        assert_eq!(a.count(), 6);
        for (x, y) in a.get().data().iter().zip(whole.get().data().iter()) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn binary_majority_is_exact_for_odd_counts() {
        let dim = 130; // spans a partial last word
        let vs: Vec<BinaryHDVector> = (0..5).map(|_| BinaryHDVector::random(dim)).collect();
        let mut acc = BinaryBundleAccumulator::new(dim);
        acc.extend(&vs);
        let out = acc.binarize().to_bits();
        let bits: Vec<Vec<u8>> = vs.iter().map(|v| v.to_bits()).collect();
        for i in 0..dim {
            let ones = bits.iter().filter(|b| b[i] == 1).count();
            assert_eq!(out[i] == 1, ones >= 3, "bit {i}");
        }
        assert_eq!(acc.binarize(), BinaryHDVector::majority_bundle_all(&vs));
    }

    #[test]
    fn binary_inputs_weigh_equally_unlike_pairwise_fold() {
        let dim = 8192;
        let vs: Vec<BinaryHDVector> = (0..15).map(|_| BinaryHDVector::random(dim)).collect();
        let mut acc = BinaryBundleAccumulator::new(dim);
        acc.extend(&vs);
        let bundle = acc.binarize();
        let pairwise = vs[1..].iter().fold(vs[0].clone(), |a, b| a.majority_bundle(b));
        // The first input survives the accumulator but is washed out pairwise.
        assert!(bundle.bipolar_similarity(&vs[0]) > 0.1);
        assert!(pairwise.bipolar_similarity(&vs[0]).abs() < 0.05);
    }

    #[test]
    fn binary_merge_equals_single_pass() {
        let vs: Vec<BinaryHDVector> = (0..6).map(|_| BinaryHDVector::random(300)).collect();
        let mut whole = BinaryBundleAccumulator::new(300);
        whole.extend(&vs);
        let mut a = BinaryBundleAccumulator::new(300);
        a.extend(&vs[..3]);
        let mut b = BinaryBundleAccumulator::new(300);
        b.extend(&vs[3..]);
        a.merge(&b);
        assert_eq!(a, whole);
        assert_eq!(a.binarize(), whole.binarize());
    }
}
