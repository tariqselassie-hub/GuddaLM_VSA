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
static VSA_ENGINES: OnceLock<Mutex<Vec<crate::vsa::VsaEngine>>> = OnceLock::new();
pub fn get_vsa_engine(dim: usize) -> crate::vsa::VsaEngine {
    let mutex = VSA_ENGINES.get_or_init(|| Mutex::new(Vec::new()));
    let mut engines = mutex.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(engine) = engines.iter().find(|e| e.dim == dim) {
        return engine.clone();
    }
    let engine = crate::vsa::VsaEngine::new(dim);
    engines.push(engine.clone());
    engine
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Complex { re: 0.0, im: 0.0 }
    }
    
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

struct TwiddleCache {
    forward: Vec<Complex>,
    inverse: Vec<Complex>,
}

static TWIDDLES: OnceLock<Mutex<std::collections::HashMap<usize, Arc<TwiddleCache>>>> = OnceLock::new();

fn get_twiddles(n: usize) -> Arc<TwiddleCache> {
    let map_mutex = TWIDDLES.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut map = map_mutex.lock().unwrap();
    if let Some(cache) = map.get(&n) {
        return cache.clone();
    }
    
    let mut forward = Vec::with_capacity(n / 2);
    let mut inverse = Vec::with_capacity(n / 2);
    for i in 0..(n / 2) {
        let angle = -2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
        forward.push(Complex { re: angle.cos(), im: angle.sin() });
        inverse.push(Complex { re: (-angle).cos(), im: (-angle).sin() });
    }
    let cache = Arc::new(TwiddleCache { forward, inverse });
    map.insert(n, cache.clone());
    cache
}

pub(crate) fn fft(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    assert!(n.is_power_of_two());
    
    let mut j = 0;
    for i in 0..n {
        if i < j {
            data.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    
    let twiddles = get_twiddles(n);
    let twiddle_array = if inverse { &twiddles.inverse } else { &twiddles.forward };
    
    let mut len = 2;
    while len <= n {
        let half = len >> 1;
        let step = n / len;
        for i in (0..n).step_by(len) {
            for k in 0..half {
                let w = twiddle_array[k * step];
                let u = data[i + k];
                let v = data[i + k + half].mul(w);
                data[i + k] = u.add(v);
                data[i + k + half] = u.sub(v);
            }
        }
        len <<= 1;
    }
    
    if inverse {
        let scale = n as f64;
        for x in data.iter_mut() {
            x.re /= scale;
            x.im /= scale;
        }
    }
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
        let dim = self.dim;
        
        let c = if dim.is_power_of_two() {
            let mut a_complex: Vec<Complex> = self.data.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
            let mut b_complex: Vec<Complex> = other.data.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
            
            fft(&mut a_complex, false);
            fft(&mut b_complex, false);
            
            let mut c_complex = vec![Complex::zero(); dim];
            for i in 0..dim {
                c_complex[i] = a_complex[i].mul(b_complex[i]);
            }
            
            fft(&mut c_complex, true);
            c_complex.iter().map(|x| x.re).collect()
        } else {
            let mut c_direct = vec![0.0; dim];
            let a = &self.data;
            let b = &other.data;
            for i in 0..dim {
                let mut sum = 0.0;
                for j in 0..dim {
                    let k = if i >= j { i - j } else { dim + i - j };
                    sum += a[j] * b[k];
                }
                c_direct[i] = sum;
            }
            c_direct
        };
        
        let mut c_normalized = c;
        let norm = c_normalized.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in c_normalized.iter_mut() {
                *x /= norm;
            }
        }
        
        HDVector {
            dim,
            data: Arc::new(c_normalized),
            is_binary: false,
        }
    }

    #[inline(always)]
    pub fn unbind(&self, other: &HDVector) -> HDVector {
        assert_eq!(self.dim, other.dim);
        let dim = self.dim;
        
        let a = if dim.is_power_of_two() {
            let mut context_complex: Vec<Complex> = self.data.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
            let mut b_complex: Vec<Complex> = other.data.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
            
            fft(&mut context_complex, false);
            fft(&mut b_complex, false);
            
            let mut d_complex = vec![Complex::zero(); dim];
            for i in 0..dim {
                d_complex[i] = context_complex[i].mul(b_complex[i].conj());
            }
            
            fft(&mut d_complex, true);
            d_complex.iter().map(|x| x.re).collect()
        } else {
            let mut a_direct = vec![0.0; dim];
            let context = &self.data;
            let b = &other.data;
            for i in 0..dim {
                let mut sum = 0.0;
                for j in 0..dim {
                    let k = if j >= i { j - i } else { dim + j - i };
                    sum += context[j] * b[k];
                }
                a_direct[i] = sum;
            }
            a_direct
        };
        
        let mut a_normalized = a;
        let norm = a_normalized.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in a_normalized.iter_mut() {
                *x /= norm;
            }
        }
        
        HDVector {
            dim,
            data: Arc::new(a_normalized),
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
        let dot: f64 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f64 = self.data.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = other.data.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
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

pub fn cosine_similarity_slice(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot = dot_product_slice(a, b);
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
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
    let dim = a.len();
    assert_eq!(dim, b.len());

    let c = if dim.is_power_of_two() {
        let mut a_complex: Vec<Complex> = a.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
        let mut b_complex: Vec<Complex> = b.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();

        fft(&mut a_complex, false);
        fft(&mut b_complex, false);

        let mut c_complex = vec![Complex::zero(); dim];
        for i in 0..dim {
            c_complex[i] = a_complex[i].mul(b_complex[i]);
        }

        fft(&mut c_complex, true);
        c_complex.iter().map(|x| x.re).collect()
    } else {
        let mut c_direct = vec![0.0; dim];
        for i in 0..dim {
            let mut sum = 0.0;
            for j in 0..dim {
                let k = if i >= j { i - j } else { dim + i - j };
                sum += a[j] * b[k];
            }
            c_direct[i] = sum;
        }
        c_direct
    };

    let mut c_normalized = c;
    let norm = c_normalized.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in c_normalized.iter_mut() {
            *x /= norm;
        }
    }
    c_normalized
}

/// Circular correlation of two f64 slices (inverse of convolution) with L2 normalization.
///
/// This is the same operation as `HDVector::unbind` but operates directly
/// on `&[f64]` to eliminate intermediate `Arc<Vec<f64>>` allocations.
pub fn correlate_slices(a: &[f64], b: &[f64]) -> Vec<f64> {
    let dim = a.len();
    assert_eq!(dim, b.len());

    let d = if dim.is_power_of_two() {
        let mut a_complex: Vec<Complex> = a.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
        let mut b_complex: Vec<Complex> = b.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();

        fft(&mut a_complex, false);
        fft(&mut b_complex, false);

        let mut d_complex = vec![Complex::zero(); dim];
        for i in 0..dim {
            d_complex[i] = a_complex[i].mul(b_complex[i].conj());
        }

        fft(&mut d_complex, true);
        d_complex.iter().map(|x| x.re).collect()
    } else {
        let mut a_direct = vec![0.0; dim];
        for i in 0..dim {
            let mut sum = 0.0;
            for j in 0..dim {
                let k = if j >= i { j - i } else { dim + j - i };
                sum += a[j] * b[k];
            }
            a_direct[i] = sum;
        }
        a_direct
    };

    let mut d_normalized = d;
    let norm = d_normalized.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in d_normalized.iter_mut() {
            *x /= norm;
        }
    }
    d_normalized
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
        let n_words = (dim + 63) / 64;
        let words: Vec<u64> = (0..n_words).map(|_| rng.gen()).collect();
        let mut v = BinaryHDVector { dim, words };
        v.clear_phantom_bits();
        v
    }

    pub fn zeros(dim: usize) -> Self {
        let n_words = (dim + 63) / 64;
        BinaryHDVector {
            dim,
            words: vec![0u64; n_words],
        }
    }

    pub fn from_bipolar(bipolar: &HDVector) -> Self {
        let n_words = (bipolar.dim() + 63) / 64;
        let mut words = vec![0u64; n_words];
        for (i, &val) in bipolar.data().iter().enumerate() {
            if val > 0.0 {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        let mut v = BinaryHDVector {
            dim: bipolar.dim(),
            words,
        };
        v.clear_phantom_bits();
        v
    }

    pub fn from_bits(bits: &[u8]) -> Self {
        let dim = bits.len();
        let n_words = (dim + 63) / 64;
        let mut words = vec![0u64; n_words];
        for (i, &bit) in bits.iter().enumerate() {
            if bit != 0 {
                words[i / 64] |= 1u64 << (i % 64);
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
        for i in 0..self.dim {
            bits[i] = ((self.words[i / 64] >> (i % 64)) & 1) as u8;
        }
        bits
    }

    /// Sub-sample to a lower dimension by keeping the first target_dim bits.
    pub fn subsample(&self, target_dim: usize) -> BinaryHDVector {
        if target_dim >= self.dim {
            return self.clone();
        }
        let n_words = (target_dim + 63) / 64;
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

        let mut bit_counts = vec![0i32; dim];
        for v in vectors {
            assert_eq!(v.dim, dim);
            for i in 0..dim {
                let bit = (v.words[i / 64] >> (i % 64)) & 1;
                bit_counts[i] += if bit == 1 { 1 } else { -1 };
            }
        }

        let tie_breaker = &vectors[0];
        let mut words = vec![0u64; n_words];
        for i in 0..dim {
            let bit = if bit_counts[i] > 0 {
                1u64
            } else if bit_counts[i] < 0 {
                0u64
            } else {
                // Deterministic tie-breaking: use the first input vector's bit
                (tie_breaker.words[i / 64] >> (i % 64)) & 1
            };
            words[i / 64] |= bit << (i % 64);
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
        let mut new_words = vec![0u64; n_words];
        for i in 0..self.dim {
            let src_bit = (self.words[i / 64] >> (i % 64)) & 1;
            let dst = (i + bit_shift) % self.dim;
            new_words[dst / 64] |= src_bit << (dst % 64);
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
    let n_words = (dim + 63) / 64;
    let mut words = vec![0u64; n_words];
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
        words[i / 64] |= bit << (i % 64);
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
}
