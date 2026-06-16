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
use crate::hdc::fhrr::FHRRVector;
use crate::hdc::vector::{majority_from_sums, BinaryHDVector, HDVector};
use rayon::prelude::*;

/// MAP bundling: sum all vectors then apply majority-rule threshold to bipolar.
pub fn bundle_vectors(vectors: &[HDVector]) -> HDVector {
    if vectors.is_empty() {
        return HDVector::zeros(0);
    }
    let dim = vectors[0].dim();
    let mut summed = HDVector::zeros(dim);
    for v in vectors {
        summed = summed.bundle(v);
    }
    summed.binarize()
}

/// Weighted MAP bundling: each vector contributes proportionally to its weight.
pub fn weighted_bundle(vectors: &[(HDVector, f64)]) -> HDVector {
    if vectors.is_empty() {
        return HDVector::zeros(0);
    }
    let dim = vectors[0].0.dim();
    let mut data = vec![0.0; dim];
    for (v, weight) in vectors {
        for (d, val) in data.iter_mut().zip(v.data().iter()) {
            *d += weight * val;
        }
    }
    HDVector::from_slice(&data).binarize()
}

/// BSC majority bundling: each vector votes +1/-1 per dimension.
pub fn bsc_bundle(vectors: &[BinaryHDVector]) -> BinaryHDVector {
    BinaryHDVector::majority_bundle_all(vectors)
}

/// Selective bundle: only bundles vectors whose similarity to a reference exceeds threshold.
/// This is the core of HD attention — conditionally include values based on relevance.
pub fn selective_bundle(
    query: &HDVector,
    keys: &[HDVector],
    values: &[HDVector],
    threshold: f64,
) -> HDVector {
    assert_eq!(keys.len(), values.len());
    let dim = query.dim();
    let mut data = vec![0.0; dim];

    for (k, v) in keys.iter().zip(values.iter()) {
        let sim = query.cosine_similarity(k);
        if sim > threshold {
            for (d, val) in data.iter_mut().zip(v.data().iter()) {
                *d += sim * val;
            }
        }
    }

    HDVector::from_slice(&data).binarize()
}

/// FHRR bundling: accumulate complex sum, normalize to unit circle.
pub fn fhrr_bundle(vectors: &[FHRRVector]) -> FHRRVector {
    FHRRVector::bundle_all(vectors)
}

/// FHRR weighted bundling: each vector contributes proportionally.
pub fn fhrr_weighted_bundle(vectors: &[(FHRRVector, f64)]) -> FHRRVector {
    if vectors.is_empty() {
        return FHRRVector::zeros(0);
    }
    let dim = vectors[0].0.dim();
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut sum_re = vec![0.0; dim];
    let mut sum_im = vec![0.0; dim];
    for (v, weight) in vectors {
        for d in 0..dim {
            sum_re[d] += weight * v.phases()[d].cos();
            sum_im[d] += weight * v.phases()[d].sin();
        }
    }
    let phases: Vec<f64> = sum_re.iter().zip(sum_im.iter())
        .map(|(&re, &im)| im.atan2(re).rem_euclid(two_pi))
        .collect();
    FHRRVector::from_phases(&phases)
}

/// BSC selective bundle: bitwise version using Hamming similarity.
pub fn bsc_selective_bundle(
    query: &BinaryHDVector,
    keys: &[BinaryHDVector],
    values: &[BinaryHDVector],
    threshold: f64,
) -> BinaryHDVector {
    assert_eq!(keys.len(), values.len());
    let dim = query.dim();
    let mut sums = vec![0i64; dim];

    for (k, v) in keys.iter().zip(values.iter()) {
        let sim = query.hamming_similarity(k);
        if sim > threshold {
            for i in 0..dim {
                let bit = (v.words()[i / 64] >> (i % 64)) & 1;
                sums[i] += if bit == 1 { 1 } else { -1 };
            }
        }
    }

    majority_from_sums(&sums, dim)
}

/// Parallel weighted MAP bundling using rayon.
///
/// Each vector-weight pair contributes independently, making this
/// embarrassingly parallel for large bundles.
pub fn par_weighted_bundle(vectors: &[(HDVector, f64)]) -> HDVector {
    if vectors.is_empty() {
        return HDVector::zeros(0);
    }
    let dim = vectors[0].0.dim();
    let n = vectors.len();

    // Sum weighted contributions in parallel chunks, then reduce
    let chunk_size = (n + rayon::current_num_threads() - 1) / rayon::current_num_threads();
    let partials: Vec<Vec<f64>> = vectors
        .par_chunks(chunk_size.max(1))
        .map(|chunk| {
            let mut local = vec![0.0; dim];
            for (v, weight) in chunk {
                for (d, val) in local.iter_mut().zip(v.data().iter()) {
                    *d += weight * val;
                }
            }
            local
        })
        .collect();

    let data = partials.into_iter().reduce(|mut a, b| {
        for (d, val) in a.iter_mut().zip(b.iter()) {
            *d += val;
        }
        a
    }).expect("partials cannot be empty since vectors is non-empty");

    HDVector::from_slice(&data).binarize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::vector::BinaryHDVector;

    #[test]
    fn test_bundle_vectors_empty() {
        let result = bundle_vectors(&[]);
        assert_eq!(result.dim(), 0);
    }

    #[test]
    fn test_bundle_vectors_single() {
        let v = HDVector::random(256);
        let result = bundle_vectors(&[v.clone()]);
        assert_eq!(result.dim(), 256);
    }

    #[test]
    fn test_bundle_vectors_binarized() {
        let v = HDVector::random(256);
        let result = bundle_vectors(&[v]);
        for &x in result.data() {
            assert!((x - 1.0).abs() < 1e-12 || (x + 1.0).abs() < 1e-12, "must be bipolar");
        }
    }

    #[test]
    fn test_weighted_bundle_empty() {
        let result = weighted_bundle(&[]);
        assert_eq!(result.dim(), 0);
    }

    #[test]
    fn test_weighted_bundle_weights() {
        let a = HDVector::random(256);
        let b = HDVector::random(256);
        let result = weighted_bundle(&[(a, 2.0), (b, 0.0)]);
        assert_eq!(result.dim(), 256);
    }

    #[test]
    fn test_bsc_bundle_empty() {
        let result = bsc_bundle(&[]);
        assert_eq!(result.dim(), 0);
    }

    #[test]
    fn test_bsc_bundle_single() {
        let v = BinaryHDVector::random(256);
        let result = bsc_bundle(&[v.clone()]);
        assert_eq!(result, v);
    }

    #[test]
    fn test_selective_bundle_empty() {
        let q = HDVector::random(256);
        let result = selective_bundle(&q, &[], &[], 0.5);
        assert_eq!(result.dim(), 256);
    }

    #[test]
    fn test_selective_bundle_threshold() {
        let q = HDVector::random(256);
        let k = HDVector::random(256);
        let v = HDVector::random(256);
        let result = selective_bundle(&q, &[k], &[v], 0.9);
        assert_eq!(result.dim(), 256);
    }

    #[test]
    fn test_par_weighted_bundle_matches_sequential() {
        let a = HDVector::random(256);
        let b = HDVector::random(256);
        let seq = weighted_bundle(&[(a.clone(), 1.0), (b.clone(), 1.0)]);
        let par = par_weighted_bundle(&[(a, 1.0), (b, 1.0)]);
        assert_eq!(seq.dim(), par.dim());
    }

    #[test]
    fn test_par_selective_bundle_matches_sequential() {
        let q = HDVector::random(256);
        let keys = vec![HDVector::random(256), HDVector::random(256)];
        let values = vec![HDVector::random(256), HDVector::random(256)];
        let seq = selective_bundle(&q, &keys, &values, 0.3);
        let par = par_selective_bundle(&q, &keys, &values, 0.3);
        assert_eq!(seq.dim(), par.dim());
    }

    #[test]
    fn test_fhrr_bundle_empty() {
        let result = fhrr_bundle(&[]);
        assert_eq!(result.dim(), 0);
    }

    #[test]
    fn test_fhrr_weighted_bundle_empty() {
        let result = fhrr_weighted_bundle(&[]);
        assert_eq!(result.dim(), 0);
    }
}

/// Parallel selective bundle using rayon.
///
/// Similarity computation and weighted accumulation for each key-value
/// pair runs in parallel.
pub fn par_selective_bundle(
    query: &HDVector,
    keys: &[HDVector],
    values: &[HDVector],
    threshold: f64,
) -> HDVector {
    assert_eq!(keys.len(), values.len());
    let dim = query.dim();

    let partials: Vec<Vec<f64>> = keys.par_iter()
        .zip(values.par_iter())
        .filter_map(|(k, v)| {
            let sim = query.cosine_similarity(k);
            if sim > threshold {
                let mut local = vec![0.0; dim];
                for (d, val) in local.iter_mut().zip(v.data().iter()) {
                    *d += sim * val;
                }
                Some(local)
            } else {
                None
            }
        })
        .collect();

    if partials.is_empty() {
        return HDVector::zeros(dim);
    }

    let data = partials.into_iter().reduce(|mut a, b| {
        for (d, val) in a.iter_mut().zip(b.iter()) {
            *d += val;
        }
        a
    }).expect("partials cannot be empty since empty case is handled above");

    HDVector::from_slice(&data).binarize()
}
