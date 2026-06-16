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
use std::collections::VecDeque;
use crate::hdc::vector::{BinaryHDVector, HDVector, majority_from_sums};

/// Streaming accumulator for MAP (bipolar) bundling.
///
/// Accumulates element-wise weighted sums incrementally and produces
/// a binarized HDVector on demand. Avoids repeated intermediate
/// allocations when bundling a stream of vectors.
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

    /// Current accumulated sum (not binarized).
    pub fn get(&self) -> HDVector {
        HDVector::from_slice(&self.sum)
    }

    /// Binarize the accumulated sum to a bipolar vector.
    pub fn binarize(&self) -> HDVector {
        if self.count == 0 {
            return HDVector::zeros(self.dim);
        }
        HDVector::from_slice(&self.sum).binarize()
    }

    /// Number of vectors accumulated so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset the accumulator (zero out sum and count).
    pub fn reset(&mut self) {
        self.sum.fill(0.0);
        self.count = 0;
    }
}

/// Streaming accumulator for BSC (binary) bundling.
///
/// Accumulates per-dimension votes as signed counters and produces
/// a BinaryHDVector via majority rule on demand.
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
        assert_eq!(self.dim, v.dim());
        for i in 0..self.dim {
            let bit = (v.words()[i / 64] >> (i % 64)) & 1;
            self.sums[i] += if bit == 1 { 1 } else { -1 };
        }
        self.count += 1;
    }

    /// Add a binary vector with a custom weight.
    #[inline(always)]
    pub fn add_weighted(&mut self, v: &BinaryHDVector, weight: f64) {
        assert_eq!(self.dim, v.dim());
        let w = weight as i64;
        for i in 0..self.dim {
            let bit = (v.words()[i / 64] >> (i % 64)) & 1;
            self.sums[i] += if bit == 1 { w } else { -w };
        }
        self.count += 1;
    }

    /// Produce the majority-rule binary vector from accumulated votes.
    pub fn binarize(&self) -> BinaryHDVector {
        majority_from_sums(&self.sums, self.dim)
    }

    /// Number of vectors accumulated.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset the accumulator.
    pub fn reset(&mut self) {
        self.sums.fill(0);
        self.count = 0;
    }
}

/// Tracks running similarity statistics between a reference vector
/// and a sliding window of streaming vectors.
pub struct RunningSimilarity {
    reference: HDVector,
    window: VecDeque<f64>,
    window_size: usize,
    sum: f64,
}

impl RunningSimilarity {
    pub fn new(reference: HDVector, window_size: usize) -> Self {
        RunningSimilarity {
            reference,
            window: VecDeque::with_capacity(window_size + 1),
            window_size,
            sum: 0.0,
        }
    }

    /// Push a new vector, compute its similarity to the reference,
    /// and update the sliding window. Returns the instantaneous similarity.
    pub fn push(&mut self, v: &HDVector) -> f64 {
        let sim = self.reference.cosine_similarity(v);
        self.window.push_back(sim);
        self.sum += sim;
        if self.window.len() > self.window_size {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old;
            }
        }
        sim
    }

    /// Mean similarity over the current window.
    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        self.sum / self.window.len() as f64
    }

    /// Standard deviation of similarity over the current window.
    pub fn std(&self) -> f64 {
        let n = self.window.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance: f64 = self.window.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        variance.sqrt()
    }

    /// Z-score of the most recent similarity value.
    pub fn zscore(&self) -> f64 {
        let n = self.window.len();
        if n < 2 {
            return 0.0;
        }
        let last = self.window.back().copied().unwrap_or(0.0);
        let std = self.std();
        if std == 0.0 {
            return 0.0;
        }
        (last - self.mean()) / std
    }

    /// Replace the reference vector.
    pub fn set_reference(&mut self, reference: HDVector) {
        self.reference = reference;
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }

    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// Clear the window but keep the reference.
    pub fn reset_window(&mut self) {
        self.window.clear();
        self.sum = 0.0;
    }
}

/// Ring buffer of HDVectors for windowed streaming operations.
pub struct HDStreamBuffer {
    dim: usize,
    capacity: usize,
    buffer: Vec<Option<HDVector>>,
    cursor: usize,
    count: usize,
}

impl HDStreamBuffer {
    pub fn new(dim: usize, capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(None);
        }
        HDStreamBuffer {
            dim,
            capacity,
            buffer,
            cursor: 0,
            count: 0,
        }
    }

    /// Push a vector into the buffer, overwriting the oldest if full.
    pub fn push(&mut self, v: HDVector) {
        assert_eq!(self.dim, v.dim());
        self.buffer[self.cursor] = Some(v);
        self.cursor = (self.cursor + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Bundle all vectors currently in the buffer.
    pub fn bundle_all(&self) -> HDVector {
        let mut acc = BundleAccumulator::new(self.dim);
        for i in 0..self.count {
            let idx = (self.cursor + self.capacity - self.count + i) % self.capacity;
            if let Some(ref v) = self.buffer[idx] {
                acc.add(v);
            }
        }
        acc.binarize()
    }

    /// Weighted bundle using per-position weights.
    /// weights.len() must equal the number of occupied slots.
    pub fn weighted_bundle(&self, weights: &[f64]) -> HDVector {
        assert_eq!(weights.len(), self.count);
        let mut acc = BundleAccumulator::new(self.dim);
        for i in 0..self.count {
            let idx = (self.cursor + self.capacity - self.count + i) % self.capacity;
            if let Some(ref v) = self.buffer[idx] {
                acc.add_weighted(v, weights[i]);
            }
        }
        acc.binarize()
    }

    /// Access a vector by its logical index (0 = oldest, count-1 = newest).
    pub fn get(&self, index: usize) -> Option<&HDVector> {
        if index >= self.count {
            return None;
        }
        let idx = (self.cursor + self.capacity - self.count + index) % self.capacity;
        self.buffer[idx].as_ref()
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Reset the buffer to empty.
    pub fn clear(&mut self) {
        for slot in self.buffer.iter_mut() {
            *slot = None;
        }
        self.cursor = 0;
        self.count = 0;
    }
}

/// Binary ring buffer for streaming BSC bundling.
pub struct BinaryHDStreamBuffer {
    dim: usize,
    capacity: usize,
    buffer: Vec<Option<BinaryHDVector>>,
    cursor: usize,
    count: usize,
}

impl BinaryHDStreamBuffer {
    pub fn new(dim: usize, capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(None);
        }
        BinaryHDStreamBuffer {
            dim,
            capacity,
            buffer,
            cursor: 0,
            count: 0,
        }
    }

    pub fn push(&mut self, v: BinaryHDVector) {
        assert_eq!(self.dim, v.dim());
        self.buffer[self.cursor] = Some(v);
        self.cursor = (self.cursor + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Majority-rule bundle of all vectors in the buffer.
    pub fn majority_bundle(&self) -> BinaryHDVector {
        let mut acc = BinaryBundleAccumulator::new(self.dim);
        for i in 0..self.count {
            let idx = (self.cursor + self.capacity - self.count + i) % self.capacity;
            if let Some(ref v) = self.buffer[idx] {
                acc.add(v);
            }
        }
        acc.binarize()
    }

    pub fn get(&self, index: usize) -> Option<&BinaryHDVector> {
        if index >= self.count {
            return None;
        }
        let idx = (self.cursor + self.capacity - self.count + index) % self.capacity;
        self.buffer[idx].as_ref()
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn clear(&mut self) {
        for slot in self.buffer.iter_mut() {
            *slot = None;
        }
        self.cursor = 0;
        self.count = 0;
    }
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
    fn test_running_similarity() {
        let dim = 1000;
        let ref_v = HDVector::random(dim);

        let mut rs = RunningSimilarity::new(ref_v.clone(), 5);
        assert_eq!(rs.window_len(), 0);
        assert_eq!(rs.mean(), 0.0);

        // Push identical vector → similarity should be 1.0
        let sim = rs.push(&ref_v);
        assert!((sim - 1.0).abs() < 1e-6);
        assert_eq!(rs.window_len(), 1);
        assert!((rs.mean() - 1.0).abs() < 1e-6);

        // Push random vectors
        for _ in 0..10 {
            let v = HDVector::random(dim);
            rs.push(&v);
        }
        assert_eq!(rs.window_len(), 5); // window capped at 5
        assert!(rs.mean() > -1.0 && rs.mean() < 1.0);
    }

    #[test]
    fn test_hd_stream_buffer() {
        let dim = 1000;
        let capacity = 4;

        let mut buf = HDStreamBuffer::new(dim, capacity);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);

        let v = HDVector::random(dim);
        buf.push(v.clone());
        assert_eq!(buf.len(), 1);

        let bundled = buf.bundle_all();
        let sim = bundled.cosine_similarity(&v);
        assert!(sim > 0.99); // single vector, should be near-identical

        // Fill buffer
        for _ in 0..capacity - 1 {
            buf.push(HDVector::random(dim));
        }
        assert_eq!(buf.len(), capacity);

        // Weighted bundle
        let weights = vec![0.25; capacity];
        let w_bundled = buf.weighted_bundle(&weights);
        assert_eq!(w_bundled.dim(), dim);
    }

    #[test]
    fn test_binary_hd_stream_buffer() {
        let dim = 1000;
        let capacity = 3;

        let mut buf = BinaryHDStreamBuffer::new(dim, capacity);
        assert!(buf.is_empty());

        let v = BinaryHDVector::random(dim);
        buf.push(v.clone());
        assert_eq!(buf.len(), 1);

        let bundled = buf.majority_bundle();
        let sim = bundled.hamming_similarity(&v);
        assert!(sim > 0.99);
    }

    #[test]
    fn test_stream_buffer_overwrite() {
        let dim = 100;
        let mut buf = HDStreamBuffer::new(dim, 3);

        let oldest = HDVector::random(dim);
        buf.push(oldest.clone());
        buf.push(HDVector::random(dim));
        buf.push(HDVector::random(dim));
        buf.push(HDVector::random(dim)); // overwrites oldest

        assert_eq!(buf.len(), 3);
        // Oldest should no longer be in buffer
        assert!(buf.get(0).map_or(true, |v| v != &oldest));
    }
}
