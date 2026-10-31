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
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde::de::{self, Visitor};
use serde::ser::SerializeStruct;
use std::fmt;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::Mutex;
use rayon::prelude::*;
static VSA_ENGINES: OnceLock<Mutex<Vec<Arc<crate::vsa::VsaEngine>>>> = OnceLock::new();

/// Shared permutation engine for `dim`. Engines are built once per dimension
/// and handed out as `Arc`s, so callers never copy the permutation tables.
pub fn get_vsa_engine(dim: usize) -> Arc<crate::vsa::VsaEngine> {
    let mutex = VSA_ENGINES.get_or_init(|| Mutex::new(Vec::new()));
    let mut engines = mutex.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(engine) = engines.iter().find(|e| e.dim == dim) {
        return Arc::clone(engine);
    }
    let engine = Arc::new(crate::vsa::VsaEngine::new(dim));
    engines.push(Arc::clone(&engine));
    engine
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[inline(always)]
    pub(crate) fn add(self, other: Self) -> Self {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
    
    #[inline(always)]
    pub(crate) fn sub(self, other: Self) -> Self {
        Complex {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
    
    #[inline(always)]
    pub(crate) fn mul(self, other: Self) -> Self {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
    
    #[inline(always)]
    pub(crate) fn conj(self) -> Self {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }
}

thread_local! {
    // One planner per thread: rustfft caches plans inside the planner, so
    // repeated transforms of the same length reuse the plan without locking.
    static FFT_PLANNER: std::cell::RefCell<rustfft::FftPlanner<f64>> =
        std::cell::RefCell::new(rustfft::FftPlanner::new());
}

/// In-place FFT of any length (rustfft picks mixed-radix / Bluestein plans
/// for non-power-of-two sizes). The inverse transform is scaled by `1/n`.
pub(crate) fn fft(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    if n == 0 {
        return;
    }
    let fft_obj = FFT_PLANNER.with(|planner| {
        let mut planner = planner.borrow_mut();
        if inverse {
            planner.plan_fft_inverse(n)
        } else {
            planner.plan_fft_forward(n)
        }
    });

    // SAFETY: `Complex` is `#[repr(C)]` with two `f64` fields, the same
    // layout as `num_complex::Complex64`.
    let rustfft_data: &mut [rustfft::num_complex::Complex64] = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut rustfft::num_complex::Complex64, n)
    };

    fft_obj.process(rustfft_data);

    if inverse {
        let scale = 1.0 / n as f64;
        for x in data.iter_mut() {
            x.re *= scale;
            x.im *= scale;
        }
    }
}

/// Un-normalized circular convolution (`conj_b = false`) or correlation
/// (`conj_b = true`) of two real slices via FFT, O(n log n) for any length.
pub(crate) fn fft_circular(a: &[f64], b: &[f64], conj_b: bool) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    let mut a_c: Vec<Complex> = a.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
    let mut b_c: Vec<Complex> = b.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
    fft(&mut a_c, false);
    fft(&mut b_c, false);
    for (x, y) in a_c.iter_mut().zip(b_c.iter()) {
        *x = if conj_b { x.mul(y.conj()) } else { x.mul(*y) };
    }
    fft(&mut a_c, true);
    a_c.into_iter().map(|x| x.re).collect()
}

fn l2_normalize(mut v: Vec<f64>) -> Vec<f64> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    v
}

/// Bipolar HD vector (+1, -1) using MAP (Multiply-Add-Permute) architecture.
/// Binding = element-wise multiplication (self-inverse).
/// Bundling = element-wise addition.
/// Permute = cyclic shift.
#[derive(Clone, Debug)]
pub struct HDVector {
    dim: usize,
    data: Arc<Vec<f64>>,
    is_binary: bool,
}

impl Serialize for HDVector {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("HDVector", 2)?;
        state.serialize_field("dim", &self.dim)?;
        state.serialize_field("data", &*self.data)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for HDVector {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field { Dim, Data }

        struct HDVectorVisitor;
        impl<'de> Visitor<'de> for HDVectorVisitor {
            type Value = HDVector;
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct HDVector")
            }
            fn visit_seq<V: de::SeqAccess<'de>>(self, mut seq: V) -> Result<HDVector, V::Error> {
                let dim = seq.next_element::<usize>()?.ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let data = seq.next_element::<Vec<f64>>()?.ok_or_else(|| de::Error::invalid_length(1, &self))?;
                Ok(HDVector { dim, data: Arc::new(data), is_binary: false })
            }
            fn visit_map<M: de::MapAccess<'de>>(self, mut map: M) -> Result<HDVector, M::Error> {
                let mut dim = None;
                let mut data = None;
                while let Some(key) = map.next_key::<Field>()? {
                    match key {
                        Field::Dim => { dim = Some(map.next_value()?); }
                        Field::Data => { data = Some(map.next_value()?); }
                    }
                }
                let dim = dim.ok_or_else(|| de::Error::missing_field("dim"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;
                Ok(HDVector { dim, data: Arc::new(data), is_binary: false })
            }
        }
        deserializer.deserialize_struct("HDVector", &["dim", "data"], HDVectorVisitor)
    }
}

impl HDVector {
    pub fn random(dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..dim)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
            .collect();
        HDVector {
            dim,
            data: Arc::new(data),
            is_binary: true,
        }
    }

    pub fn zeros(dim: usize) -> Self {
        HDVector {
            dim,
            data: Arc::new(vec![0.0; dim]),
            is_binary: false,
        }
    }

    pub fn from_slice(slice: &[f64]) -> Self {
        HDVector {
            dim: slice.len(),
            data: Arc::new(slice.to_vec()),
            is_binary: false,
        }
    }

    pub fn from_slice_with_binary(slice: &[f64], is_binary: bool) -> Self {
        HDVector {
            dim: slice.len(),
            data: Arc::new(slice.to_vec()),
            is_binary,
        }
    }

    /// Take ownership of `data` without copying it.
    pub fn from_vec(data: Vec<f64>) -> Self {
        HDVector {
            dim: data.len(),
            data: Arc::new(data),
            is_binary: false,
        }
    }

    #[inline(always)]
    pub fn is_binary(&self) -> bool {
        self.is_binary
    }

    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline(always)]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Mutable access to the underlying data, with copy-on-write.
    /// Clones the backing Vec only if other references exist.
    #[inline(always)]
    pub fn data_mut(&mut self) -> &mut [f64] {
        if Arc::strong_count(&self.data) > 1 {
            self.data = Arc::new((*self.data).clone());
        }
        Arc::get_mut(&mut self.data)
            .expect("data_mut: unique after COW check")
            .as_mut_slice()
    }

    #[inline(always)]
    pub fn scale(&self, scalar: f64) -> HDVector {
        let new_data: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
        HDVector {
            dim: self.dim,
            data: Arc::new(new_data),
            is_binary: self.is_binary,
        }
    }

    #[inline(always)]
    pub fn bind(&self, other: &HDVector) -> HDVector {
        assert_eq!(self.dim, other.dim);
        let result = convolve_slices(&self.data, &other.data);
        HDVector {
            dim: self.dim,
            data: Arc::new(result),
            is_binary: false,
        }
    }

    #[inline(always)]
    pub fn unbind(&self, other: &HDVector) -> HDVector {
        assert_eq!(self.dim, other.dim);
        let result = correlate_slices(&self.data, &other.data);
        HDVector {
            dim: self.dim,
            data: Arc::new(result),
            is_binary: false,
        }
    }

    #[inline(always)]
    pub fn bundle(&self, other: &HDVector) -> HDVector {
        assert_eq!(self.dim, other.dim);
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        HDVector {
            dim: self.dim,
            data: Arc::new(data),
            is_binary: false,
        }
    }

    pub fn permute(&self, steps: usize) -> HDVector {
        let engine = get_vsa_engine(self.dim);
        engine.permute(self, steps)
    }

    pub fn permute_left(&self, steps: usize) -> HDVector {
        let engine = get_vsa_engine(self.dim);
        engine.unpermute(self, steps)
    }

    pub fn binarize(&self) -> HDVector {
        // Deterministic tie-break for zero values using a hash of the
        // dimension index.  This ensures reproducibility while maintaining
        // the ~50/50 split that random tie-breaking provides.
        let tie_dash = |i: usize| -> f64 {
            let h = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if (h >> 63) & 1 == 0 { 1.0 } else { -1.0 }
        };
        let data: Vec<f64> = self
            .data
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    tie_dash(i)
                }
            })
            .collect();
        HDVector {
            dim: self.dim,
            data: Arc::new(data),
            is_binary: true,
        }
    }

    #[inline(always)]
    pub fn cosine_similarity(&self, other: &HDVector) -> f64 {
        assert_eq!(self.dim, other.dim);
        cosine_similarity_slice(&self.data, &other.data)
    }
    /// Sub-sample this vector to a lower dimension by truncation.
    ///
    /// Because HD vectors distribute information i.i.d. across all dimensions,
    /// truncation preserves approximate structure. The first `target_dim`
    /// dimensions are kept; the rest are discarded.
    ///
    /// Research on Binary Hyperdimensional Transformers shows that dim can
    /// be reduced by up to 64% (e.g. 10000 → 3600) with <10% accuracy loss
    /// but ~50% speedup and memory savings.
    pub fn subsample(&self, target_dim: usize) -> HDVector {
        if target_dim >= self.dim {
            return self.clone();
        }
        HDVector::from_slice(&self.data[..target_dim])
    }

    /// Resample to any target dimension: truncate or zero-pad.
    pub fn resample(&self, target_dim: usize) -> HDVector {
        if target_dim == self.dim {
            return self.clone();
        }
        if target_dim < self.dim {
            return self.subsample(target_dim);
        }
        // Zero-pad to larger dimension
        let mut data = self.data.to_vec();
        data.resize(target_dim, 0.0);
        HDVector::from_slice(&data)
    }

    pub fn hamming_similarity(&self, other: &HDVector) -> f64 {
        assert_eq!(self.dim, other.dim);
        let a_bin = self.binarize();
        let b_bin = other.binarize();
        let matches: usize = a_bin
            .data
            .iter()
            .zip(b_bin.data.iter())
            .filter(|(a, b)| a == b)
            .count();
        matches as f64 / self.dim as f64
    }
}

impl PartialEq for HDVector {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && *self.data == *other.data
    }
}

impl Eq for HDVector {}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_product_slice_avx2(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    
    let len = a.len().min(b.len());
    let chunks = len / 4;
    let mut sum_vec = _mm256_setzero_pd();
    
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    
    for i in 0..chunks {
        let va = _mm256_loadu_pd(a_ptr.add(i * 4));
        let vb = _mm256_loadu_pd(b_ptr.add(i * 4));
        sum_vec = _mm256_fmadd_pd(va, vb, sum_vec);
    }
    
    let mut temp = [0.0; 4];
    _mm256_storeu_pd(temp.as_mut_ptr(), sum_vec);
    let mut dot = temp[0] + temp[1] + temp[2] + temp[3];
    
    for i in (chunks * 4)..len {
        dot += a[i] * b[i];
    }
    dot
}

#[inline(always)]
pub fn dot_product_slice(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_slice_avx2(a, b) };
        }
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Dot product and both squared norms in a single pass over `a` and `b`.
///
/// Four independent accumulator lanes let the compiler vectorize the loop
/// without needing to reassociate floating-point sums.
#[inline]
fn dot_and_sq_norms(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    const LANES: usize = 4;
    let mut dot = [0.0f64; LANES];
    let mut na = [0.0f64; LANES];
    let mut nb = [0.0f64; LANES];
    let chunks_a = a.chunks_exact(LANES);
    let chunks_b = b.chunks_exact(LANES);
    let (rem_a, rem_b) = (chunks_a.remainder(), chunks_b.remainder());
    for (ca, cb) in chunks_a.zip(chunks_b) {
        for l in 0..LANES {
            dot[l] += ca[l] * cb[l];
            na[l] += ca[l] * ca[l];
            nb[l] += cb[l] * cb[l];
        }
    }
    let mut d: f64 = dot.iter().sum();
    let mut sa: f64 = na.iter().sum();
    let mut sb: f64 = nb.iter().sum();
    for (x, y) in rem_a.iter().zip(rem_b) {
        d += x * y;
        sa += x * x;
        sb += y * y;
    }
    (d, sa, sb)
}

pub fn cosine_similarity_slice(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (dot, sq_a, sq_b) = dot_and_sq_norms(a, b);
    if sq_a == 0.0 || sq_b == 0.0 {
        return 0.0;
    }
    (dot / (sq_a * sq_b).sqrt()).clamp(-1.0, 1.0)
}

/// Optimized cosine similarity for bipolar vectors (all elements ±1).
///
/// For bipolar vectors, ‖v‖ = √dim, so the full cosine similarity
/// `dot(a,b) / (‖a‖ · ‖b‖)` simplifies to `dot(a,b) / dim`.
/// This eliminates two norm passes (each O(dim)) and two sqrt calls.
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
#[inline(always)]
pub fn bipolar_cosine_similarity_slice(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dim = a.len() as f64;
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    (dot / dim).clamp(-1.0, 1.0)
}

/// Optimized cosine similarity for bipolar HDVectors (all elements ±1).
///
/// Same optimization as `bipolar_cosine_similarity_slice`: skips the
/// norm computation since ‖v‖ = √dim for any bipolar vector.
#[inline(always)]
pub fn bipolar_cosine_similarity(a: &HDVector, b: &HDVector) -> f64 {
    assert_eq!(a.dim(), b.dim());
    let dim = a.dim() as f64;
    let dot = a.data().iter().zip(b.data().iter()).map(|(x, y)| x * y).sum::<f64>();
    (dot / dim).clamp(-1.0, 1.0)
}

/// Circular convolution of two f64 slices with L2 normalization.
///
/// Avoids constructing HDVector wrappers when operating on sub-slices
/// (e.g., per-head sub-vectors in attention).  This is the same operation
/// as `HDVector::bind` but operates directly on `&[f64]` to eliminate
/// intermediate `Arc<Vec<f64>>` allocations.
pub fn convolve_slices(a: &[f64], b: &[f64]) -> Vec<f64> {
    l2_normalize(fft_circular(a, b, false))
}

/// Circular correlation of two f64 slices (inverse of convolution) with L2 normalization.
///
/// This is the same operation as `HDVector::unbind` but operates directly
/// on `&[f64]` to eliminate intermediate `Arc<Vec<f64>>` allocations.
pub fn correlate_slices(a: &[f64], b: &[f64]) -> Vec<f64> {
    l2_normalize(fft_circular(a, b, true))
}

/// Binary HD vector (0, 1) using BSC (Binary Spatter Coding) architecture.
/// Binding = bitwise XOR (self-inverse).
/// Bundling = majority rule (sum > threshold → 1).
/// Permute = cyclic bit rotation.
/// Storage is bit-packed into u64 words for maximum efficiency.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BinaryHDVector {
    pub dim: usize,
    pub words: Vec<u64>,
}

impl BinaryHDVector {
    pub fn random(dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let n_words = dim.div_ceil(64);
        let words: Vec<u64> = (0..n_words).map(|_| rng.gen()).collect();
        let mut v = BinaryHDVector { dim, words };
        v.clear_phantom_bits();
        v
    }

    pub fn zeros(dim: usize) -> Self {
        let n_words = dim.div_ceil(64);
        BinaryHDVector {
            dim,
            words: vec![0u64; n_words],
        }
    }

    pub fn from_bipolar(bipolar: &HDVector) -> Self {
        let dim = bipolar.dim();
        let n_words = dim.div_ceil(64);
        let mut words = vec![0u64; n_words];
        let mut word_idx = 0;
        let mut bit_idx = 0;
        for &val in bipolar.data().iter() {
            if val > 0.0 {
                words[word_idx] |= 1u64 << bit_idx;
            }
            bit_idx += 1;
            if bit_idx == 64 {
                bit_idx = 0;
                word_idx += 1;
            }
        }
        let mut v = BinaryHDVector {
            dim,
            words,
        };
        v.clear_phantom_bits();
        v
    }

    pub fn from_bits(bits: &[u8]) -> Self {
        let dim = bits.len();
        let n_words = dim.div_ceil(64);
        let mut words = vec![0u64; n_words];
        let mut word_idx = 0;
        let mut bit_idx = 0;
        for &bit in bits.iter() {
            if bit != 0 {
                words[word_idx] |= 1u64 << bit_idx;
            }
            bit_idx += 1;
            if bit_idx == 64 {
                bit_idx = 0;
                word_idx += 1;
            }
        }
        let mut v = BinaryHDVector { dim, words };
        v.clear_phantom_bits();
        v
    }

    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline(always)]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Return the packed words as a fixed-size `[u64; 64]` reference.
    /// Only succeeds when `dim == 4096` (the canonical hyperdimension).
    /// The fixed-size type forces LLVM to perfectly unroll SIMD loops
    /// under LTO, achieving ~4.5× speedup over dynamic-slice iteration.
    pub fn as_array64(&self) -> Option<&crate::hdc::quantize::PackedArray64> {
        if self.dim == 4096 && self.words.len() == 64 {
            self.words.as_slice().try_into().ok()
        } else {
            None
        }
    }

    pub fn to_bits(&self) -> Vec<u8> {
        let mut bits = vec![0u8; self.dim];
        let mut word_idx = 0;
        let mut bit_idx = 0;
        for i in 0..self.dim {
            bits[i] = ((self.words[word_idx] >> bit_idx) & 1) as u8;
            bit_idx += 1;
            if bit_idx == 64 {
                bit_idx = 0;
                word_idx += 1;
            }
        }
        bits
    }

    /// Sub-sample to a lower dimension by keeping the first target_dim bits.
    pub fn subsample(&self, target_dim: usize) -> BinaryHDVector {
        if target_dim >= self.dim {
            return self.clone();
        }
        let n_words = target_dim.div_ceil(64);
        let mut words = self.words[..n_words].to_vec();
        // Clear phantom bits in the new last word
        let last_bit = target_dim % 64;
        if last_bit != 0 {
            let mask = (1u64 << last_bit) - 1;
            if let Some(last) = words.last_mut() {
                *last &= mask;
            }
        }
        BinaryHDVector {
            dim: target_dim,
            words,
        }
    }

    #[inline(always)]
    pub fn xor_bind(&self, other: &BinaryHDVector) -> BinaryHDVector {
        assert_eq!(self.dim, other.dim);
        let words: Vec<u64> = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        BinaryHDVector {
            dim: self.dim,
            words,
        }
    }

    /// Word-level majority bundling: O(n_words) instead of O(dim).
    ///
    /// For each word, computes:
    ///   agreed_ones = a & b          (both are 1)
    ///   contested  = a ^ b           (bits differ → tie → random)
    ///   result     = agreed_ones | (contested & random_mask)
    ///
    /// Random tie-breaking uses a uniform random u64 per word.
    /// Phantom bits are cleared at the end.
    #[inline(always)]
    pub fn majority_bundle(&self, other: &BinaryHDVector) -> BinaryHDVector {
        assert_eq!(self.dim, other.dim);
        let words: Vec<u64> = self
            .words
            .iter()
            .zip(other.words.iter())
            .enumerate()
            .map(|(wi, (a, b))| {
                let agreed = a & b;
                let contested = a ^ b;
                // Deterministic unbiased tie-breaking: use a hash of both
                // words so the outcome does not systematically favor `a` or `b`.
                let random_bits = a.wrapping_mul(0x9E3779B97F4A7C15)
                    ^ b.rotate_left(37)
                    ^ (wi as u64).wrapping_mul(0xBF58476D1CE4E5B9);
                // Ensure at least some bit flips by mixing
                let random_bits = random_bits ^ random_bits.rotate_right(17);
                agreed | (contested & random_bits)
            })
            .collect();
        let mut result = BinaryHDVector {
            dim: self.dim,
            words,
        };
        result.clear_phantom_bits();
        result
    }

    /// Multi-vector word-level majority bundling: O(n_words * n_vectors).
    ///
    /// For each word position, accumulates popcounts across all vectors
    /// using only word-level operations, then thresholds per 64-bit chunk.
    /// Uses the first input vector as deterministic tie-breaker.
    pub fn majority_bundle_all(vectors: &[BinaryHDVector]) -> BinaryHDVector {
        if vectors.is_empty() {
            return BinaryHDVector::zeros(0);
        }
        let dim = vectors[0].dim;
        let n_words = vectors[0].words.len();
        let tie_breaker = &vectors[0];
        let mut words = vec![0u64; n_words];

        for w_idx in 0..n_words {
            let start_bit = w_idx * 64;
            let end_bit = (start_bit + 64).min(dim);
            
            let mut counts = [0i32; 64];
            for v in vectors {
                assert_eq!(v.dim, dim);
                let word = v.words[w_idx];
                for bit_idx in 0..(end_bit - start_bit) {
                    let bit = (word >> bit_idx) & 1;
                    counts[bit_idx] += if bit == 1 { 1 } else { -1 };
                }
            }
            
            let tie_word = tie_breaker.words[w_idx];
            let mut final_word = 0u64;
            for bit_idx in 0..(end_bit - start_bit) {
                let c = counts[bit_idx];
                let bit = if c > 0 {
                    1
                } else if c < 0 {
                    0
                } else {
                    (tie_word >> bit_idx) & 1
                };
                final_word |= bit << bit_idx;
            }
            words[w_idx] = final_word;
        }

        let mut result = BinaryHDVector { dim, words };
        result.clear_phantom_bits();
        result
    }

    /// BSC permutation: cyclic rotation of bits.
    pub fn rotate(&self, shift: usize) -> BinaryHDVector {
        if self.dim == 0 {
            return self.clone();
        }
        let bit_shift = shift % self.dim;
        if bit_shift == 0 {
            return self.clone();
        }

        let n_words = self.words.len();
        
        // Fast path for dimensions that are multiples of 64
        if self.dim.is_multiple_of(64) {
            let word_shift = bit_shift / 64;
            let intra_shift = bit_shift % 64;
            let mut new_words = vec![0u64; n_words];
            if intra_shift == 0 {
                for i in 0..n_words {
                    new_words[(i + word_shift) % n_words] = self.words[i];
                }
            } else {
                for i in 0..n_words {
                    let val = self.words[i];
                    let target_1 = (i + word_shift) % n_words;
                    let target_2 = (i + word_shift + 1) % n_words;
                    new_words[target_1] |= val << intra_shift;
                    new_words[target_2] |= val >> (64 - intra_shift);
                }
            }
            let mut result = BinaryHDVector {
                dim: self.dim,
                words: new_words,
            };
            result.clear_phantom_bits();
            return result;
        }

        // Highly optimized fallback path with zero division/modulo inside the loop
        let mut new_words = vec![0u64; n_words];
        let mut src_word_idx = 0;
        let mut src_bit_idx = 0;
        let mut dst = bit_shift;
        let mut dst_word_idx = dst / 64;
        let mut dst_bit_idx = dst % 64;
        
        for _ in 0..self.dim {
            let src_bit = (self.words[src_word_idx] >> src_bit_idx) & 1;
            new_words[dst_word_idx] |= src_bit << dst_bit_idx;
            
            src_bit_idx += 1;
            if src_bit_idx == 64 {
                src_bit_idx = 0;
                src_word_idx += 1;
            }
            
            dst += 1;
            if dst == self.dim {
                dst = 0;
                dst_word_idx = 0;
                dst_bit_idx = 0;
            } else {
                dst_bit_idx += 1;
                if dst_bit_idx == 64 {
                    dst_bit_idx = 0;
                    dst_word_idx += 1;
                }
            }
        }

        let mut result = BinaryHDVector {
            dim: self.dim,
            words: new_words,
        };
        result.clear_phantom_bits();
        result
    }

    /// Zero out bits beyond dim in the last word to ensure clean comparisons.
    fn clear_phantom_bits(&mut self) {
        let last_bit = self.dim % 64;
        if last_bit != 0 {
            let mask = (1u64 << last_bit) - 1;
            if let Some(last) = self.words.last_mut() {
                *last &= mask;
            }
        }
    }

    /// Hamming similarity via XOR + popcount on full words.
    /// Range [0, 1] where 1 = identical.
    ///
    /// Automatically uses the fixed-size `[u64; 64]` SIMD fast path
    /// when both vectors have dim == 4096.
    #[inline(always)]
    pub fn hamming_similarity(&self, other: &BinaryHDVector) -> f64 {
        assert_eq!(self.dim, other.dim);
        if let (Some(a64), Some(b64)) = (self.as_array64(), other.as_array64()) {
            let sim = crate::hdc::quantize::wordwise_xnor_similarity_array64(a64, b64);
            return (sim + 1.0) / 2.0;
        }
        let diff_bits: u64 = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| (a ^ b).count_ones() as u64)
            .sum();
        1.0 - (diff_bits as f64 / self.dim as f64)
    }

    /// Bipolar similarity via XNOR + popcount on full words.
    /// Range [-1, 1] where 1 = identical, -1 = completely opposite.
    ///
    /// Automatically uses the fixed-size `[u64; 64]` SIMD fast path
    /// when both vectors have dim == 4096.
    #[inline(always)]
    pub fn bipolar_similarity(&self, other: &BinaryHDVector) -> f64 {
        assert_eq!(self.dim, other.dim);
        if let (Some(a64), Some(b64)) = (self.as_array64(), other.as_array64()) {
            return crate::hdc::quantize::wordwise_xnor_similarity_array64(a64, b64);
        }
        let diff_bits: u64 = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| (a ^ b).count_ones() as u64)
            .sum();
        let same_bits = self.dim as u64 - diff_bits;
        (same_bits as f64 - diff_bits as f64) / self.dim as f64
    }
}

pub fn majority_from_sums(sums: &[i64], dim: usize) -> BinaryHDVector {
    let n_words = dim.div_ceil(64);
    let mut words = vec![0u64; n_words];
    let mut word_idx = 0;
    let mut bit_idx = 0;
    for i in 0..dim {
        let bit = if sums[i] > 0 {
            1u64
        } else if sums[i] < 0 {
            0u64
        } else {
            // Deterministic tie-breaking using a hash of the bit position
            let h = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            (h >> 63) & 1
        };
        words[word_idx] |= bit << bit_idx;
        bit_idx += 1;
        if bit_idx == 64 {
            bit_idx = 0;
            word_idx += 1;
        }
    }
    BinaryHDVector { dim, words }
}

impl PartialEq for BinaryHDVector {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.words == other.words
    }
}

impl Eq for BinaryHDVector {}

/// Parallel batch convolution: convolve a single vector against multiple
/// others using rayon.
pub fn par_convolve_batch(a: &[f64], batch: &[&[f64]]) -> Vec<Vec<f64>> {
    batch.par_iter()
        .map(|b| convolve_slices(a, b))
        .collect()
}

/// Parallel batch correlation: correlate a single vector against multiple
/// others using rayon.
pub fn par_correlate_batch(a: &[f64], batch: &[&[f64]]) -> Vec<Vec<f64>> {
    batch.par_iter()
        .map(|b| correlate_slices(a, b))
        .collect()
}

/// Parallel cosine similarity: compute similarities between pairs of slices.
pub fn par_cosine_similarity_slice(a: &[f64], batch: &[&[f64]]) -> Vec<f64> {
    batch.par_iter()
        .map(|b| cosine_similarity_slice(a, b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_slice_avx2_equivalence() {
        let a = vec![1.5, -2.0, 3.25, 4.0, -0.5, 0.25, 9.75, 11.1];
        let b = vec![0.5, 4.0, -2.0, 1.5, 8.0, -6.0, 1.25, 0.1];
        
        let expected = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
        let actual = dot_product_slice(&a, &b);
        
        assert!((expected - actual).abs() < 1e-10);
    }

    /// Reference O(n²) circular convolution / correlation from the definitions.
    fn direct_circular(a: &[f64], b: &[f64], correlate: bool) -> Vec<f64> {
        let n = a.len();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let k = if correlate { (j + n - i) % n } else { (i + n - j) % n };
                        a[j] * b[k]
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn test_fft_circular_matches_definition_any_length() {
        // Powers of two, odd, prime and mixed-radix lengths.
        for &n in &[8usize, 64, 100, 127, 1000, 1024] {
            let a = HDVector::random(n);
            let b = HDVector::random(n);
            for &correlate in &[false, true] {
                let expected = direct_circular(a.data(), b.data(), correlate);
                let actual = fft_circular(a.data(), b.data(), correlate);
                for (e, x) in expected.iter().zip(actual.iter()) {
                    assert!((e - x).abs() < 1e-8, "n={n} correlate={correlate}: {e} vs {x}");
                }
            }
        }
    }

    #[test]
    fn test_bind_unbind_round_trip_non_power_of_two() {
        let a = HDVector::random(10_000);
        let b = HDVector::random(10_000);
        let recovered = a.bind(&b).unbind(&b);
        assert!(a.cosine_similarity(&recovered) > 0.65);
    }
}
