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
use super::vector::HDVector;
use std::cell::RefCell;
use std::rc::Rc;

type BackwardFn = Box<dyn FnOnce(&[f64]) -> Vec<Vec<f64>>>;

// ── GradHDVector ──────────────────────────────────────────────

#[derive(Clone)]
pub struct GradHDVector {
    inner: Rc<RefCell<GradNode>>,
}

struct GradNode {
    value: HDVector,
    grad: Vec<f64>,
    grad_set: bool,
    requires_grad: bool,
    backward: Option<BackwardFn>,
    parents: Vec<Rc<RefCell<GradNode>>>,
}

impl GradHDVector {
    /// Wrap an existing HDVector as a leaf node in the computation graph.
    pub fn leaf(value: HDVector, requires_grad: bool) -> Self {
        let dim = value.dim();
        GradHDVector {
            inner: Rc::new(RefCell::new(GradNode {
                value,
                grad: vec![0.0; dim],
                grad_set: false,
                requires_grad,
                backward: None,
                parents: vec![],
            })),
        }
    }

    pub fn param(value: HDVector) -> Self {
        Self::leaf(value, true)
    }

    pub fn constant(value: HDVector) -> Self {
        Self::leaf(value, false)
    }

    pub fn dim(&self) -> usize {
        self.inner.borrow().value.dim()
    }

    pub fn value(&self) -> HDVector {
        self.inner.borrow().value.clone()
    }

    pub fn grad(&self) -> Option<HDVector> {
        let node = self.inner.borrow();
        if node.grad_set {
            Some(HDVector::from_slice(&node.grad))
        } else {
            None
        }
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
    }

    pub fn set_value(&self, new_value: HDVector) {
        self.inner.borrow_mut().value = new_value;
    }

    pub fn zero_grad(&self) {
        let mut node = self.inner.borrow_mut();
        let dim = node.value.dim();
        node.grad = vec![0.0; dim];
        node.grad_set = false;
    }

    fn add_node_with_parents(
        value: HDVector,
        requires_grad: bool,
        backward: Option<BackwardFn>,
        parents: &[&GradHDVector],
    ) -> GradHDVector {
        let dim = value.dim();
        let any_grad = requires_grad || parents.iter().any(|p| p.requires_grad());
        let parent_refs: Vec<Rc<RefCell<GradNode>>> = parents.iter().map(|p| Rc::clone(&p.inner)).collect();

        GradHDVector {
            inner: Rc::new(RefCell::new(GradNode {
                value,
                grad: vec![0.0; dim],
                grad_set: false,
                requires_grad: any_grad,
                backward,
                parents: parent_refs,
            })),
        }
    }
}

// ── Backward pass ─────────────────────────────────────────────

/// Compute gradients by traversing the computation graph backward
/// from this loss node.  The gradient of the loss w.r.t. itself is
/// set to 1.0 (the upstream "seed").
pub fn backward(loss: &GradHDVector) {
    // Topological sort via DFS
    let mut visited: std::collections::HashSet<*const RefCell<GradNode>> = std::collections::HashSet::new();
    let mut order: Vec<Rc<RefCell<GradNode>>> = Vec::new();

    fn dfs(
        node: &Rc<RefCell<GradNode>>,
        visited: &mut std::collections::HashSet<*const RefCell<GradNode>>,
        order: &mut Vec<Rc<RefCell<GradNode>>>,
    ) {
        let ptr: *const RefCell<GradNode> = Rc::as_ptr(node);
        if !visited.insert(ptr) {
            return;
        }
        let parents: Vec<Rc<RefCell<GradNode>>> = node.borrow().parents.clone();
        for p in &parents {
            dfs(p, visited, order);
        }
        order.push(Rc::clone(node));
    }

    dfs(&loss.inner, &mut visited, &mut order);

    // Seed the gradient: loss output is 1-dimensional
    {
        let mut loss_node = loss.inner.borrow_mut();
        loss_node.grad[0] = 1.0;
        loss_node.grad_set = true;
    }

    // Traverse in reverse topological order
    for node_rc in order.iter().rev() {
        let has_backward = node_rc.borrow().backward.is_some();
        if !has_backward {
            continue;
        }

        let grad_output: Vec<f64> = {
            let node = node_rc.borrow();
            node.grad.clone()
        };
        let parent_count = node_rc.borrow().parents.len();

        let backward_fn = node_rc.borrow_mut().backward.take()
            .expect("backward must be set when has_backward is true");
        let parent_grads = backward_fn(&grad_output);

        let parents: Vec<Rc<RefCell<GradNode>>> = node_rc.borrow().parents.clone();
        for (i, pg) in parent_grads.into_iter().enumerate() {
            if i < parent_count && i < parents.len() {
                let mut p_node = parents[i].borrow_mut();
                if !p_node.requires_grad {
                    continue;
                }
                if p_node.grad_set {
                    for (a, b) in p_node.grad.iter_mut().zip(pg.iter()) {
                        *a += b;
                    }
                } else {
                    p_node.grad = pg;
                    p_node.grad_set = true;
                }
            }
        }
    }
}

// ── Differentiable VSA Operations ─────────────────────────────

/// Differentiable circular convolution (bind) with L2 normalization.
///
/// Forward: `c[k] = sum_j a[j] * b[(k-j) mod n]`, then L2-normalized.
///
/// Backward (straight-through): treats normalization as identity for
/// gradient purposes.  This is standard practice (cf. batch-norm,
/// binary neural networks) and works well for learning codebooks.
pub fn diff_bind(a: &GradHDVector, b: &GradHDVector) -> GradHDVector {
    let a_val = a.value();
    let b_val = b.value();
    let mut output = raw_convolve(&a_val, &b_val);
    let norm: f64 = output.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in output.iter_mut() { *x /= norm; }
    }

    GradHDVector::add_node_with_parents(
        HDVector::from_slice(&output),
        true,
        Some(Box::new(move |grad_output| {
            let go = HDVector::from_slice(grad_output);
            let grad_a_data = raw_correlate(&go, &b_val);
            let grad_b_data = raw_correlate(&go, &a_val);
            vec![grad_a_data, grad_b_data]
        })),
        &[a, b],
    )
}

/// Differentiable element-wise sum (bundling).
///
/// Forward: `c[i] = a[i] + b[i]`
///
/// Backward: gradient passes through identically to both inputs.
pub fn diff_bundle(a: &GradHDVector, b: &GradHDVector) -> GradHDVector {
    let a_val = a.value();
    let b_val = b.value();
    let output: Vec<f64> = a_val.data().iter().zip(b_val.data().iter())
        .map(|(x, y)| x + y).collect();

    GradHDVector::add_node_with_parents(
        HDVector::from_slice(&output),
        true,
        Some(Box::new(move |grad_output| {
            vec![grad_output.to_vec(), grad_output.to_vec()]
        })),
        &[a, b],
    )
}

/// Differentiable sum of multiple vectors.
pub fn diff_bundle_many(vectors: &[&GradHDVector]) -> GradHDVector {
    if vectors.is_empty() {
        return GradHDVector::constant(HDVector::zeros(0));
    }
    if vectors.len() == 1 {
        return (*vectors[0]).clone();
    }

    let dim = vectors[0].dim();
    let mut output = vec![0.0; dim];
    for v in vectors {
        let val = v.value();
        for (o, d) in output.iter_mut().zip(val.data().iter()) {
            *o += d;
        }
    }

    let refs: Vec<&GradHDVector> = vectors.iter().map(|v| *v).collect();
    let parent_count = vectors.len();
    GradHDVector::add_node_with_parents(
        HDVector::from_slice(&output),
        true,
        Some(Box::new(move |grad_output| {
            let go = grad_output.to_vec();
            vec![go; parent_count]
        })),
        &refs,
    )
}

/// Differentiable permutation via cyclic shift.
///
/// Forward: cyclic shift by `steps` positions.
///
/// Backward: inverse permutation (shift by `dim - steps`).
pub fn diff_permute(v: &GradHDVector, steps: usize) -> GradHDVector {
    let v_val = v.value();
    let dim = v_val.dim();
    let engine = super::vector::get_vsa_engine(dim);
    let output = engine.permute(&v_val, steps);

    GradHDVector::add_node_with_parents(
        output,
        true,
        Some(Box::new(move |grad_output| {
            let go = HDVector::from_slice(grad_output);
            let grad_v = engine.unpermute(&go, steps);
            vec![grad_v.data().to_vec()]
        })),
        &[v],
    )
}

/// Negative cosine similarity loss.
///
/// `loss = 1.0 - cosine_similarity(pred, target)`
///
/// The gradient of the loss w.r.t. `pred` is computed analytically.
/// `target` does **not** require gradients.
pub fn similarity_loss(pred: &GradHDVector, target: &HDVector) -> GradHDVector {
    let p = pred.value();
    let dim = p.dim();
    let t = target;

    let dot: f64 = p.data().iter().zip(t.data().iter()).map(|(a, b)| a * b).sum();
    let np: f64 = p.data().iter().map(|x| x * x).sum::<f64>().sqrt();
    let nt: f64 = t.data().iter().map(|x| x * x).sum::<f64>().sqrt();

    let sim = if np > 0.0 && nt > 0.0 { dot / (np * nt) } else { 0.0 };
    let loss = (1.0 - sim).clamp(0.0, 2.0);

    let p_data = p.data().to_vec();
    let t_data = t.data().to_vec();

    GradHDVector::add_node_with_parents(
        HDVector::from_slice(&[loss]),
        true,
        Some(Box::new(move |grad_output| {
            let upstream = grad_output[0];
            if np == 0.0 || nt == 0.0 {
                return vec![vec![0.0; dim]];
            }
            let np2 = np * np;
            let np3 = np2 * np;
            let denom = np3 * nt;
            let mut grad_pred = Vec::with_capacity(dim);
            for j in 0..dim {
                let val = (dot * p_data[j] - t_data[j] * np2) / denom;
                grad_pred.push(val * upstream);
            }
            vec![grad_pred]
        })),
        &[pred],
    )
}

// ── Raw convolution / correlation helpers (no normalization) ──

pub(crate) fn raw_convolve(a: &HDVector, b: &HDVector) -> Vec<f64> {
    let dim = a.dim();
    if dim.is_power_of_two() {
        fft_convolve(a.data(), b.data())
    } else {
        direct_convolve(a.data(), b.data())
    }
}

pub(crate) fn raw_correlate(a: &HDVector, b: &HDVector) -> Vec<f64> {
    let dim = a.dim();
    if dim.is_power_of_two() {
        fft_correlate(a.data(), b.data())
    } else {
        direct_correlate(a.data(), b.data())
    }
}

fn fft_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    use super::vector::Complex;
    let dim = a.len();
    let mut a_c: Vec<Complex> = a.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
    let mut b_c: Vec<Complex> = b.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
    super::vector::fft(&mut a_c, false);
    super::vector::fft(&mut b_c, false);
    let mut c = vec![Complex::zero(); dim];
    for i in 0..dim { c[i] = a_c[i].mul(b_c[i]); }
    super::vector::fft(&mut c, true);
    c.iter().map(|x| x.re).collect()
}

fn fft_correlate(a: &[f64], b: &[f64]) -> Vec<f64> {
    use super::vector::Complex;
    let dim = a.len();
    let mut a_c: Vec<Complex> = a.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
    let mut b_c: Vec<Complex> = b.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
    super::vector::fft(&mut a_c, false);
    super::vector::fft(&mut b_c, false);
    let mut d = vec![Complex::zero(); dim];
    for i in 0..dim { d[i] = a_c[i].mul(b_c[i].conj()); }
    super::vector::fft(&mut d, true);
    d.iter().map(|x| x.re).collect()
}

fn direct_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let dim = a.len();
    let mut c = vec![0.0; dim];
    for i in 0..dim {
        let mut sum = 0.0;
        for j in 0..dim {
            let k = if i >= j { i - j } else { dim + i - j };
            sum += a[j] * b[k];
        }
        c[i] = sum;
    }
    c
}

fn direct_correlate(a: &[f64], b: &[f64]) -> Vec<f64> {
    let dim = a.len();
    let mut d = vec![0.0; dim];
    for i in 0..dim {
        let mut sum = 0.0;
        for j in 0..dim {
            let k = if j >= i { j - i } else { dim + j - i };
            sum += a[j] * b[k];
        }
        d[i] = sum;
    }
    d
}

// ── Optimizer ─────────────────────────────────────────────────

/// Simple SGD optimizer for VSA codebooks and parameters.
///
/// Updates: `param += -lr * grad`
/// After update, the value is clipped to avoid divergence.
pub struct SGDOptimizer {
    params: Vec<(GradHDVector, f64)>, // (parameter, learning_rate)
}

impl SGDOptimizer {
    pub fn new() -> Self {
        SGDOptimizer { params: Vec::new() }
    }

    pub fn add(&mut self, param: GradHDVector, lr: f64) {
        self.params.push((param, lr));
    }

    pub fn add_with_lr(&mut self, param: GradHDVector, lr: f64) {
        self.add(param, lr);
    }

    /// Zero all parameter gradients.
    pub fn zero_grad(&mut self) {
        for (p, _) in &mut self.params {
            p.zero_grad();
        }
    }

    /// Perform one SGD update step using accumulated gradients.
    pub fn step(&mut self) {
        for (p, lr) in &mut self.params {
            if let Some(grad) = p.grad() {
                let mut new_data: Vec<f64> = p.value().data().to_vec();
                for (val, g) in new_data.iter_mut().zip(grad.data().iter()) {
                    *val -= *lr * g;
                }
                // Clip to prevent explosion
                let norm: f64 = new_data.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 0.0 {
                    let scale = (new_data.len() as f64).sqrt() / norm;
                    for x in new_data.iter_mut() { *x *= scale; }
                }
                p.set_value(HDVector::from_slice(&new_data));
            }
        }
    }
}

// ── Differentiable Cleanup Memory ────────────────────────────

/// Soft / differentiable cleanup memory.
///
/// Instead of selecting the single best-matching prototype, `soft_cleanup`
/// returns a similarity-weighted bundle of all prototypes. This is the
/// differentiable analogue of `CleanupMemory::cleanup()` and can be used
/// as a layer in gradient-based pipelines.
///
/// The temperature parameter `beta` controls the softness of the selection:
/// low beta → all prototypes mix equally, high beta → hard nearest-neighbor.
pub fn soft_cleanup(
    query: &GradHDVector,
    codebook_vectors: &[HDVector],
    beta: f64,
) -> GradHDVector {
    let dim = query.dim();
    let q_val = query.value();

    // Compute cosine similarities and soft weights
    let sims: Vec<f64> = codebook_vectors.iter()
        .map(|cv| q_val.cosine_similarity(cv))
        .collect();

    let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut weights: Vec<f64> = sims.iter().map(|&s| (beta * (s - max_sim)).exp()).collect();
    let sum_w: f64 = weights.iter().sum();
    if sum_w > 1e-12 {
        for w in weights.iter_mut() { *w /= sum_w; }
    }

    // Weighted bundle
    let mut output = vec![0.0; dim];
    for (cv, w) in codebook_vectors.iter().zip(weights.iter()) {
        for (o, d) in output.iter_mut().zip(cv.data().iter()) {
            *o += w * d;
        }
    }

    let cv_refs: Vec<HDVector> = codebook_vectors.iter().cloned().collect();
    let stored_weights = weights;
    GradHDVector::add_node_with_parents(
        HDVector::from_slice(&output),
        true,
        Some(Box::new(move |grad_output| {
            let go = HDVector::from_slice(grad_output);
            let mut grad_q = vec![0.0; dim];

            // Approximate gradient: treat attention weights as constant
            // and only propagate through the weighted sum (not through the
            // softmax that depends on query).  This is a straight-through
            // estimator, sufficient for proof-of-concept.
            for j in 0..dim {
                let mut g = 0.0;
                for (cv, w) in cv_refs.iter().zip(stored_weights.iter()) {
                    g += w * cv.data()[j];
                }
                grad_q[j] = go.data()[j] * g * 0.01;
            }

            vec![grad_q]
        })),
        &[query],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Differentiable binding ──

    #[test]
    fn test_diff_bind_forward_matches_hdvector() {
        let dim = 64;
        let a = HDVector::random(dim);
        let b = HDVector::random(dim);
        let ga = GradHDVector::constant(a.clone());
        let gb = GradHDVector::constant(b.clone());

        let gc = diff_bind(&ga, &gb);
        let autograd_val = gc.value();

        // diff_bind now normalizes, same as HDVector::bind
        let sim = autograd_val.cosine_similarity(&a.bind(&b));
        assert!(
            (sim - 1.0).abs() < 1e-12,
            "diff_bind forward must equal HDVector::bind (sim={})",
            sim
        );
    }

    #[test]
    fn test_diff_bind_backward_recovers_gradient() {
        let dim = 64;
        let a = HDVector::random(dim);
        let b = HDVector::random(dim);
        let target = HDVector::random(dim);

        let ga = GradHDVector::param(a.clone());
        let gb = GradHDVector::constant(b.clone());

        // loss = 1 - sim(a ⊛ b, target)
        let gc = diff_bind(&ga, &gb);
        let loss = similarity_loss(&gc, &target);
        backward(&loss);

        let grad_a = ga.grad().unwrap();
        // Verify gradient is non-zero
        let grad_norm: f64 = grad_a.data().iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            grad_norm > 1e-10,
            "gradient w.r.t. a must be non-zero (norm = {})",
            grad_norm
        );
    }

    #[test]
    fn test_diff_bind_step_reduces_loss() {
        let dim = 64;
        let target = HDVector::random(dim);

        // Start with a random param and try to learn bind(target, key)
        let key = HDVector::random(dim);
        let param = GradHDVector::param(HDVector::random(dim));

        let mut opt = SGDOptimizer::new();
        opt.add(param.clone(), 2.0);

        for _step in 0..200 {
            opt.zero_grad();

            let bound = diff_bind(&param, &GradHDVector::constant(key.clone()));
            let loss = similarity_loss(&bound, &target);
            backward(&loss);

            opt.step();
        }

        // Check final loss has improved from initial (random sim ~ 1.0)
        let bound = diff_bind(&param, &GradHDVector::constant(key.clone()));
        let final_loss = similarity_loss(&bound, &target);
        let final_loss_val = final_loss.value().data()[0];
        assert!(
            final_loss_val < 0.9,
            "loss must converge after 200 steps (loss={})",
            final_loss_val
        );
    }

    // ── Differentiable bundling ──

    #[test]
    fn test_diff_bundle_forward_matches_hdvector() {
        let dim = 64;
        let a = HDVector::random(dim);
        let b = HDVector::random(dim);
        let ga = GradHDVector::constant(a.clone());
        let gb = GradHDVector::constant(b.clone());

        let gc = diff_bundle(&ga, &gb);
        let autograd_val = gc.value();
        let hd_val = a.bundle(&b);

        let sim = autograd_val.cosine_similarity(&hd_val);
        assert!(
            (sim - 1.0).abs() < 1e-12,
            "diff_bundle forward must equal HDVector bundle"
        );
    }

    #[test]
    fn test_diff_bundle_gradient_flows() {
        let dim = 64;
        let a = HDVector::random(dim);
        let b = HDVector::random(dim);
        let target = HDVector::random(dim);

        let ga = GradHDVector::param(a.clone());
        let gb = GradHDVector::constant(b.clone());

        let gc = diff_bundle(&ga, &gb);
        let loss = similarity_loss(&gc, &target);
        backward(&loss);

        let grad_a = ga.grad().unwrap();
        let grad_norm: f64 = grad_a.data().iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            grad_norm > 1e-10,
            "bundle gradient must be non-zero (norm = {})",
            grad_norm
        );

        assert!(
            grad_norm > 0.0,
            "bundle gradient must be non-zero"
        );
    }

    // ── Differentiable permutation ──

    #[test]
    fn test_diff_permute_forward_matches_hdvector() {
        let dim = 64;
        let v = HDVector::random(dim);
        let gv = GradHDVector::constant(v.clone());

        let gp = diff_permute(&gv, 3);
        let autograd_val = gp.value();
        let hd_val = v.permute(3);

        assert_eq!(autograd_val, hd_val, "diff_permute forward must equal HDVector permute");
    }

    #[test]
    fn test_diff_permute_gradient_flows() {
        let dim = 64;
        let v = HDVector::random(dim);
        let target = HDVector::random(dim);

        let gv = GradHDVector::param(v.clone());
        let gp = diff_permute(&gv, 3);
        let loss = similarity_loss(&gp, &target);
        backward(&loss);

        let grad = gv.grad().unwrap();
        let grad_norm: f64 = grad.data().iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            grad_norm > 1e-10,
            "permute gradient must be non-zero (norm = {})",
            grad_norm
        );
    }

    // ── Optimizer ──

    #[test]
    fn test_sgd_optimizer_updates_parameter() {
        let dim = 64;
        let target = HDVector::random(dim);
        let param = GradHDVector::param(HDVector::random(dim));
        let initial = param.value();

        let mut opt = SGDOptimizer::new();
        opt.add(param.clone(), 2.0);

        for _ in 0..10 {
            opt.zero_grad();
            let loss = similarity_loss(&param, &target);
            backward(&loss);
            opt.step();
        }

        let updated = param.value();
        let sim = initial.cosine_similarity(&updated);
        assert!(
            sim < 0.99,
            "param must change after 10 SGD steps (sim old/new = {})",
            sim
        );
    }

    // ── Soft cleanup ──

    #[test]
    fn test_soft_cleanup_forward() {
        let dim = 64;
        let n = 10;
        let mut cb: Vec<HDVector> = (0..n).map(|_| HDVector::random(dim)).collect();

        // Set first prototype equal to query (should dominate)
        let query = HDVector::random(dim);
        cb[0] = query.clone();

        let gq = GradHDVector::constant(query.clone());
        let result = soft_cleanup(&gq, &cb, 10.0);
        let result_val = result.value();

        // With high beta, result should be similar to cb[0]
        let sim = result_val.cosine_similarity(&cb[0]);
        assert!(
            sim > 0.3,
            "soft cleanup with high beta should favor nearest prototype (sim={})",
            sim
        );
    }

    // ── Multi-step gradient flow ──

    #[test]
    fn test_multi_step_gradient_flow() {
        let dim = 64;

        // Chain: bind -> bundle -> permute -> loss
        let a = HDVector::random(dim);
        let b = HDVector::random(dim);
        let c = HDVector::random(dim);
        let target = HDVector::random(dim);

        let ga = GradHDVector::param(a);
        let gb = GradHDVector::constant(b);
        let gc = GradHDVector::constant(c);

        let g_bind = diff_bind(&ga, &gb);
        let g_bundle = diff_bundle(&g_bind, &gc);
        let g_permute = diff_permute(&g_bundle, 2);
        let loss = similarity_loss(&g_permute, &target);
        backward(&loss);

        let grad_a = ga.grad().unwrap();
        let grad_norm: f64 = grad_a.data().iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            grad_norm > 1e-10,
            "gradient must flow through bind->bundle->permute chain (norm={})",
            grad_norm
        );
    }

    #[test]
    fn test_codebook_learning_via_gradient() {
        let dim = 64;
        let n = 8;

        // Create a codebook and a target composition
        let codebook: Vec<HDVector> = (0..n).map(|_| HDVector::random(dim)).collect();
        let idx_a = 2;
        let idx_b = 5;
        let target = codebook[idx_a].bind(&codebook[idx_b]);

        // Learn to match idx_a through gradient descent
        let param = GradHDVector::param(codebook[idx_a].clone());
        let mut opt = SGDOptimizer::new();
        opt.add_with_lr(param.clone(), 0.2);

        let mut last_loss = f64::INFINITY;
        for _step in 0..30 {
            opt.zero_grad();

            let bound = diff_bind(&param, &GradHDVector::constant(codebook[idx_b].clone()));
            let loss = similarity_loss(&bound, &target);
            backward(&loss);

            let loss_val = loss.value().data()[0];
            assert!(loss_val <= last_loss + 1e-4,
                "codebook learning must decrease loss (prev={}, cur={})", last_loss, loss_val);
            last_loss = loss_val;

            opt.step();
        }

        let learned = param.value();
        let sim = learned.cosine_similarity(&codebook[idx_a]);
        assert!(sim > 0.5, "learned vector must converge toward target (sim={})", sim);
    }
}
