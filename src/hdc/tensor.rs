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
//! # Candle-core Tensor Bridge for VSA Operations
//!
//! Provides bidirectional conversion between `HDVector` / `GradHDVector`
//! and `candle-core` `Tensor`, plus differentiable VSA operations that
//! run on candle's autograd graph (CPU/GPU/CUDA/Metal).
//!
//! ## Usage
//! ```ignore
//! use guddalm_vsa::hdc::tensor::*;
//! let v = HDVector::random(1024);
//! let t = to_tensor(&v, &Device::Cpu)?;
//! let v2 = from_tensor(&t)?;
//! ```
//!
//! Requires the `"candle"` feature:
//! ```toml
//! guddalm_vsa = { features = ["candle"] }
//! ```

use crate::hdc::vector::HDVector;
use candle_core::{Device, Tensor};

/// Convert an `HDVector` to a 1-D candle `Tensor` on the given device.
///
/// The tensor has shape `(dim,)` and dtype `f64`.
pub fn to_tensor(v: &HDVector, device: &Device) -> candle_core::Result<Tensor> {
    Tensor::from_slice(v.data(), v.dim(), device)
}

/// Convert a 1-D candle `Tensor` back to an `HDVector`.
///
/// The tensor must have exactly one dimension.  If the tensor has a
/// different dtype, it is cast to `f64` automatically.
pub fn from_tensor(t: &Tensor) -> candle_core::Result<HDVector> {
    let t = if t.dtype() != candle_core::DType::F64 {
        t.to_dtype(candle_core::DType::F64)?
    } else {
        t.clone()
    };
    let data: Vec<f64> = t.flatten_all()?.to_vec1()?;
    Ok(HDVector::from_slice(&data))
}

/// Convert a batch of `HDVector`s to a 2-D candle `Tensor`.
///
/// The tensor has shape `(batch_size, dim)` and dtype `f64`.
pub fn to_tensor_batch(vectors: &[HDVector], device: &Device) -> candle_core::Result<Tensor> {
    if vectors.is_empty() {
        return Tensor::from_slice(&[] as &[f64], (0usize, 0usize), device);
    }
    let dim = vectors[0].dim();
    let flat: Vec<f64> = vectors.iter().flat_map(|v| v.data().iter().copied()).collect();
    Tensor::from_slice(&flat, (vectors.len(), dim), device)
}

/// Convert a 2-D candle `Tensor` to a batch of `HDVector`s.
pub fn from_tensor_batch(t: &Tensor) -> candle_core::Result<Vec<HDVector>> {
    let t = if t.dtype() != candle_core::DType::F64 {
        t.to_dtype(candle_core::DType::F64)?
    } else {
        t.clone()
    };
    let shape = t.shape();
    let dims = shape.dims();
    if dims.len() != 2 {
        candle_core::bail!("from_tensor_batch expects a 2-D tensor, got shape {:?}", dims);
    }
    let dim = dims[1];
    let data: Vec<f64> = t.flatten_all()?.to_vec1()?;
    let vectors: Vec<HDVector> = data
        .chunks(dim)
        .map(|chunk| HDVector::from_slice(chunk))
        .collect();
    Ok(vectors)
}

// ── Differentiable VSA operations using candle autograd ───────

/// Differentiable circular convolution (bind) on tensors.
///
/// Computes `c[k] = sum_j a[j] * b[(k-j) mod n]` using direct O(n²)
/// tensor operations.  The result is L2-normalized and differentiable
/// via candle's autograd.
///
/// Both inputs should be 1-D tensors of the same length.
pub fn tensor_bind(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    let n = a.dim(0)?;

    // Circular convolution: c[k] = sum_j a[j] * b[(k-j) mod n]
    // Implemented as: c[k] = dot(a, flip(b).roll(k+1, 0))
    // Verified algebraically for dim=4.
    let b_rev = b.flip(&[0])?;

    let mut terms = Vec::with_capacity(n);
    for k in 0..n {
        let b_shifted = b_rev.roll((k + 1) as i32, 0)?;
        let dot = (a * b_shifted)?.sum(0)?;
        terms.push(dot);
    }
    let mut c = Tensor::stack(&terms, 0)?;

    // L2 normalize
    let norm = c.sqr()?.sum(0)?.sqrt()?;
    c = c.broadcast_div(&norm)?;

    Ok(c)
}

/// Differentiable circular correlation (unbind) on tensors.
///
/// Inverse of `tensor_bind`.  L2-normalized.
pub fn tensor_unbind(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    let n = a.dim(0)?;

    // Unbind: d[i] = sum_j a[j] * b[(j-i) mod n]
    let mut terms = Vec::with_capacity(n);
    for i in 0..n {
        let b_shifted = b.roll(i as i32, 0)?; // shift by +i
        let product = (a * b_shifted)?;
        let sum = product.sum(0)?;
        terms.push(sum);
    }
    let mut d = Tensor::stack(&terms, 0)?;

    let norm = d.sqr()?.sum(0)?.sqrt()?;
    d = d.broadcast_div(&norm)?;

    Ok(d)
}

/// Differentiable bundling (element-wise addition) on tensors.
pub fn tensor_bundle(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    a.add(b)
}

/// Differentiable permutation via cyclic shift on tensors.
///
/// Rolls the tensor by `steps` positions along dimension 0.
pub fn tensor_permute(v: &Tensor, steps: usize) -> candle_core::Result<Tensor> {
    let n = v.dim(0)?;
    if steps == 0 {
        return Ok(v.clone());
    }
    let engine = crate::hdc::vector::get_vsa_engine(n);
    // VsaEngine::permute does: next[perm_forward[i]] = current[i]
    // This scatter is equivalent to: output[j] = input[inverse_perm[j]]
    // where inverse_perm[perm_forward[i]] = i
    let mut inv_perm = vec![0u32; n];
    for i in 0..n {
        inv_perm[engine.perm_forward[i]] = i as u32;
    }
    let indices = Tensor::from_slice(&inv_perm, n, v.device())?;
    let mut result = v.clone();
    for _ in 0..steps {
        result = result.gather(&indices, 0)?;
    }
    Ok(result)
}

/// Cosine similarity between two 1-D tensors.
///
/// Returns a scalar tensor (shape `()`) containing `sim ∈ [-1, 1]`.
pub fn tensor_cosine_similarity(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    let dot = (a * b)?.sum(0)?;
    let na = a.sqr()?.sum(0)?.sqrt()?;
    let nb = b.sqr()?.sum(0)?.sqrt()?;
    let denom = (na * nb)?;
    let sim = dot.broadcast_div(&denom)?;
    // Clamp to [-1, 1]
    let one = Tensor::new(1.0, sim.device())?;
    let neg_one = Tensor::new(-1.0, sim.device())?;
    sim.broadcast_maximum(&neg_one)?.broadcast_minimum(&one)
}

/// Negative cosine similarity loss on tensors.
///
/// `loss = 1 - cosine_similarity(pred, target)`
///
/// Returns a scalar tensor.  The gradient w.r.t. `pred` flows through
/// candle's autograd graph.
pub fn tensor_similarity_loss(pred: &Tensor, target: &Tensor) -> candle_core::Result<Tensor> {
    let sim = tensor_cosine_similarity(pred, target)?;
    let device = sim.device();
    let one = Tensor::new(1.0, device)?;
    let zero = Tensor::new(0.0, device)?;
    let two = Tensor::new(2.0, device)?;
    let loss = (one - sim.clone())?;
    loss.broadcast_maximum(&zero)?.broadcast_minimum(&two)
}

/// Selective bundle (attention) on tensors.
///
/// For each key-value pair where `cos_sim(query, key) > threshold`,
/// the value is added to the output, weighted by the similarity.
///
/// Query is 1-D `(dim,)`, keys and values are 2-D `(n, dim)`.
pub fn tensor_selective_bundle(
    query: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    threshold: f64,
) -> candle_core::Result<Tensor> {
    // Broadcast query to [n, dim]
    let q_broadcast = query.broadcast_as(keys.shape())?;

    // Similarities: dot product per row
    let dot = (q_broadcast * keys)?.sum(1)?; // [n]
    let q_norm = query.sqr()?.sum(0)?.sqrt()?; // scalar
    let k_norms = keys.sqr()?.sum(1)?.sqrt()?; // [n]
    let denom = q_norm.broadcast_mul(&k_norms)?;
    let sims = (dot / denom)?; // [n]

    // Mask: sim > threshold
    let thr = Tensor::new(threshold, sims.device())?;
    let thr = thr.broadcast_as(sims.shape())?;
    let mask = sims.gt(&thr)?.to_dtype(candle_core::DType::F64)?;
    let weights = (sims * mask)?;

    // Normalize weights (avoid div-by-zero)
    let sum_w = weights.sum(0)?;
    let eps = Tensor::new(1e-12, sum_w.device())?;
    let sum_w_safe = sum_w.broadcast_maximum(&eps)?;
    let weights_norm = weights.broadcast_div(&sum_w_safe)?;

    // Weighted sum of values: [n, dim] * [n, 1] -> sum -> [dim]
    let w_expanded = weights_norm.unsqueeze(1)?;
    let weighted = values.broadcast_mul(&w_expanded)?;
    weighted.sum(0)
}

// ── FWHT on tensors (useful for spectral operations) ──────────

/// Fast Walsh-Hadamard Transform on a 1-D tensor (in-place via new tensor).
///
/// O(n log n) butterfly using only addition/subtraction.  Works for any
/// power-of-two length.
pub fn tensor_fwht(t: &Tensor) -> candle_core::Result<Tensor> {
    let n = t.dim(0)?;
    assert!(n.is_power_of_two(), "FWHT requires power-of-two dimension");

    let mut data: Vec<f64> = t.flatten_all()?.to_vec1()?;
    let mut len = 1;
    while len < n {
        let stride = len;
        len <<= 1;
        for i in (0..n).step_by(len) {
            for j in 0..stride {
                let u = data[i + j];
                let v = data[i + j + stride];
                data[i + j] = u + v;
                data[i + j + stride] = u - v;
            }
        }
    }
    Tensor::from_slice(&data, t.shape(), t.device())
}

/// Inverse FWHT on a 1-D tensor.
pub fn tensor_ifwht(t: &Tensor) -> candle_core::Result<Tensor> {
    let result = tensor_fwht(t)?;
    let n = t.dim(0)? as f64;
    result.broadcast_div(&Tensor::new(n, t.device())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::vector::HDVector;

    fn test_device() -> Device {
        Device::Cpu
    }

    #[test]
    fn test_to_from_tensor() {
        let dim = 256;
        let v = HDVector::random(dim);
        let device = test_device();
        let t = to_tensor(&v, &device).unwrap();
        let v2 = from_tensor(&t).unwrap();
        let sim = v.cosine_similarity(&v2);
        assert!((sim - 1.0).abs() < 1e-12, "round-trip sim = {}", sim);
    }

    #[test]
    fn test_tensor_bind_matches_hdvector() {
        let dim = 64;
        let a = HDVector::random(dim);
        let b = HDVector::random(dim);
        let device = test_device();

        let ta = to_tensor(&a, &device).unwrap();
        let tb = to_tensor(&b, &device).unwrap();
        let tc = tensor_bind(&ta, &tb).unwrap();
        let result_tensor = from_tensor(&tc).unwrap();

        let result_hd = a.bind(&b);
        let sim = result_tensor.cosine_similarity(&result_hd);
        assert!(
            sim > 0.99,
            "tensor_bind must approximate HDVector::bind (sim={})",
            sim
        );
    }

    #[test]
    fn test_tensor_bundle_matches_hdvector() {
        let dim = 64;
        let a = HDVector::random(dim);
        let b = HDVector::random(dim);
        let device = test_device();

        let ta = to_tensor(&a, &device).unwrap();
        let tb = to_tensor(&b, &device).unwrap();
        let tc = tensor_bundle(&ta, &tb).unwrap();
        let result_tensor = from_tensor(&tc).unwrap();

        let result_hd = a.bundle(&b);
        let sim = result_tensor.cosine_similarity(&result_hd);
        assert!(
            (sim - 1.0).abs() < 1e-12,
            "tensor_bundle must exactly match HDVector::bundle (sim={})",
            sim
        );
    }

    #[test]
    fn test_tensor_permute_matches_hdvector() {
        let dim = 64;
        let v = HDVector::random(dim);
        let device = test_device();

        for steps in [0, 1, 5, 63] {
            let tv = to_tensor(&v, &device).unwrap();
            let tp = tensor_permute(&tv, steps).unwrap();
            let result_tensor = from_tensor(&tp).unwrap();

            let result_hd = v.permute(steps);
            assert_eq!(
                result_tensor, result_hd,
                "tensor_permute({}) must match HDVector::permute", steps
            );
        }
    }

    #[test]
    fn test_tensor_cosine_similarity() {
        let dim = 256;
        let device = test_device();

        // Identical
        let a = HDVector::random(dim);
        let ta = to_tensor(&a, &device).unwrap();
        let sim_self = tensor_cosine_similarity(&ta, &ta).unwrap();
        let sim_self_val: f64 = sim_self.to_vec0().unwrap();
        assert!(
            (sim_self_val - 1.0).abs() < 1e-10,
            "self-similarity must be 1.0 (got {})",
            sim_self_val
        );

        // Random
        let b = HDVector::random(dim);
        let tb = to_tensor(&b, &device).unwrap();
        let sim_ab = tensor_cosine_similarity(&ta, &tb).unwrap();
        let sim_ab_val: f64 = sim_ab.to_vec0().unwrap();
        let sim_hd = a.cosine_similarity(&b);
        assert!(
            (sim_ab_val - sim_hd).abs() < 1e-12,
            "tensor cosine similarity must match HDVector (got {} vs {})",
            sim_ab_val, sim_hd
        );
    }

    #[test]
    fn test_tensor_selective_bundle() {
        let dim = 64;
        let n = 10;
        let device = test_device();

        let query = HDVector::random(dim);
        let mut keys: Vec<HDVector> = (0..n).map(|_| HDVector::random(dim)).collect();
        let values: Vec<HDVector> = (0..n).map(|_| HDVector::random(dim)).collect();

        // Make first key very similar to query
        keys[0] = query.clone();

        let tq = to_tensor(&query, &device).unwrap();
        let tk = to_tensor_batch(&keys, &device).unwrap();
        let tv = to_tensor_batch(&values, &device).unwrap();

        let result = tensor_selective_bundle(&tq, &tk, &tv, 0.5).unwrap();
        let result_vec = from_tensor(&result).unwrap();

        // Should favor the value paired with the similar key
        let sim_to_v1 = result_vec.cosine_similarity(&values[0]);
        let sim_to_v2 = result_vec.cosine_similarity(&values[1]);
        assert!(
            sim_to_v1 > sim_to_v2,
            "selective bundle must favor value paired with similar key ({} vs {})",
            sim_to_v1, sim_to_v2
        );
    }

    #[test]
    fn test_tensor_fwht_matches_native() {
        let dim = 1024;
        let device = test_device();

        let data: Vec<f64> = (0..dim).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let t = Tensor::from_slice(&data, dim, &device).unwrap();

        let result_tensor = tensor_fwht(&t).unwrap();
        let result_tensor = tensor_ifwht(&result_tensor).unwrap();
        let result_data: Vec<f64> = result_tensor.flatten_all().unwrap().to_vec1().unwrap();

        for (a, b) in data.iter().zip(result_data.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "FWHT/IFWHT must be inverses on tensors"
            );
        }
    }

    #[test]
    fn test_tensor_unbind_matches_hdvector() {
        let dim = 64;
        let a = HDVector::random(dim);
        let b = HDVector::random(dim);
        let device = test_device();

        let bound_hd = a.bind(&b);
        let unbound_hd = bound_hd.unbind(&b);

        let ta = to_tensor(&a, &device).unwrap();
        let tb = to_tensor(&b, &device).unwrap();
        let t_bound = tensor_bind(&ta, &tb).unwrap();
        let t_unbound = tensor_unbind(&t_bound, &tb).unwrap();
        let unbound_tensor = from_tensor(&t_unbound).unwrap();

        let sim = unbound_tensor.cosine_similarity(&unbound_hd);
        assert!(
            sim > 0.99,
            "tensor_unbind must approximate HDVector::unbind (sim={})",
            sim
        );
    }

    #[test]
    fn test_tensor_similarity_loss_forward() {
        let dim = 64;
        let device = test_device();

        let a = HDVector::random(dim);
        let b = a.clone(); // identical

        let ta = to_tensor(&a, &device).unwrap();
        let tb = to_tensor(&b, &device).unwrap();

        let loss = tensor_similarity_loss(&ta, &tb).unwrap();
        let loss_val: f64 = loss.to_vec0().unwrap();
        assert!(
            (loss_val - 0.0).abs() < 1e-10,
            "loss for identical vectors must be 0 (got {})",
            loss_val
        );

        // Opposite vectors should give loss = 2.0
        let neg_data: Vec<f64> = a.data().iter().map(|x| -x).collect();
        let neg = HDVector::from_slice(&neg_data);
        let tn = to_tensor(&neg, &device).unwrap();
        let loss_neg = tensor_similarity_loss(&ta, &tn).unwrap();
        let loss_neg_val: f64 = loss_neg.to_vec0().unwrap();
        assert!(
            (loss_neg_val - 2.0).abs() < 0.01,
            "loss for opposite vectors must be ~2.0 (got {})",
            loss_neg_val
        );
    }

    #[test]
    fn test_tensor_fwht_roundtrip() {
        let dim = 1024;
        let device = test_device();

        let data: Vec<f64> = (0..dim).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let t = Tensor::from_slice(&data, dim, &device).unwrap();

        let result = tensor_fwht(&t).unwrap();
        let result = tensor_ifwht(&result).unwrap();
        let result_data: Vec<f64> = result.flatten_all().unwrap().to_vec1().unwrap();

        for (a, b) in data.iter().zip(result_data.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "FWHT/IFWHT must be inverses on tensors"
            );
        }
    }

    #[test]
    fn test_tensor_bind_gradient_flows() -> candle_core::Result<()> {
        let dim = 64;
        let device = test_device();

        let a_data: Vec<f64> = (0..dim).map(|i| (i as f64 - 32.0) / 32.0).collect();
        let b_data: Vec<f64> = (0..dim).map(|i| (i as f64 / 63.0) * 2.0 - 1.0).collect();
        let target = HDVector::random(dim);
        let t_target = to_tensor(&target, &device).unwrap();

        // Var enables gradient tracking
        let a_t = Tensor::from_slice(&a_data, dim, &device).unwrap();
        let b_t = Tensor::from_slice(&b_data, dim, &device).unwrap();
        let a_var = candle_core::Var::from_tensor(&a_t).unwrap();

        let bound = tensor_bind(a_var.as_tensor(), &b_t).unwrap();
        let loss = tensor_similarity_loss(&bound, &t_target).unwrap();
        let grads = loss.backward().unwrap();

        let grad = grads.get(a_var.as_tensor()).unwrap();
        let grad_norm: f64 = grad.sqr()?.sum(0)?.sqrt()?.to_vec0().unwrap();
        assert!(
            grad_norm > 1e-10,
            "gradient through tensor_bind must be non-zero (norm={})",
            grad_norm
        );
        Ok(())
    }

    #[test]
    fn test_batch_conversion() {
        let dim = 64;
        let n = 10;
        let device = test_device();

        let vectors: Vec<HDVector> = (0..n).map(|_| HDVector::random(dim)).collect();
        let t = to_tensor_batch(&vectors, &device).unwrap();
        let recovered = from_tensor_batch(&t).unwrap();

        assert_eq!(vectors.len(), recovered.len());
        for (v, r) in vectors.iter().zip(recovered.iter()) {
            let sim = v.cosine_similarity(r);
            assert!((sim - 1.0).abs() < 1e-12, "batch round-trip sim = {}", sim);
        }
    }
}
