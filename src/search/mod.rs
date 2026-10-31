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
use crate::hdc::vector::{BinaryHDVector, HDVector};
use crate::hdc::vsa_trait::VsaVector;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    pub index: usize,
    pub score: f64,
}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for SearchResult {}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl SearchResult {
    #[inline(always)]
    pub fn new(index: usize, score: f64) -> Self {
        Self { index, score }
    }
}

pub fn search_map(query: &HDVector, candidates: &[HDVector], top_k: usize) -> Vec<SearchResult> {
    ranked_scores(query, candidates, top_k)
}

pub fn search_bsc(
    query: &BinaryHDVector,
    candidates: &[BinaryHDVector],
    top_k: usize,
) -> Vec<SearchResult> {
    ranked_scores(query, candidates, top_k)
}

pub fn search_fhrr(
    query: &FHRRVector,
    candidates: &[FHRRVector],
    top_k: usize,
) -> Vec<SearchResult> {
    ranked_scores(query, candidates, top_k)
}

#[derive(Debug, Clone, Default)]
pub struct SuperSearch;

impl SuperSearch {
    pub fn query<V: VsaVector + Sync>(
        query: &V,
        candidates: &[V],
        top_k: usize,
    ) -> Vec<SearchResult> {
        ranked_scores(query, candidates, top_k)
    }

    /// Weighted score fusion across several views of the same items.
    ///
    /// Each entry in `sets` is one view (for example text and image
    /// embeddings) with a weight, and index `i` must be the same item in
    /// every view. Item `i` scores `Σ weight × similarity(query, view[i])`,
    /// and the `top_k` best items come back best first, with ties going to
    /// the lowest index.
    ///
    /// Panics if the views have different lengths. Returns nothing when
    /// `sets` is empty or `top_k` is 0.
    ///
    /// ```
    /// use guddalm_vsa::prelude::*;
    /// use guddalm_vsa::search::SuperSearch;
    ///
    /// let text: Vec<HDVector> = (0..4).map(|_| HDVector::random(1024)).collect();
    /// let image: Vec<HDVector> = (0..4).map(|_| HDVector::random(1024)).collect();
    /// // A query that matches item 2's text view.
    /// let hits = SuperSearch::query_weighted(&text[2], &[(&text[..], 0.7), (&image[..], 0.3)], 1);
    /// assert_eq!(hits[0].index, 2);
    /// ```
    pub fn query_weighted<V: VsaVector + Sync>(
        query: &V,
        sets: &[(&[V], f64)],
        top_k: usize,
    ) -> Vec<SearchResult> {
        let Some(((first, _), rest)) = sets.split_first() else {
            return Vec::new();
        };
        let n = first.len();
        for (i, (view, _)) in rest.iter().enumerate() {
            assert_eq!(
                view.len(),
                n,
                "query_weighted: view {} has {} items but view 0 has {}",
                i + 1,
                view.len(),
                n
            );
        }
        top_k_by(n, top_k, |i| {
            sets.iter()
                .map(|(view, weight)| weight * query.cosine_similarity(&view[i]))
                .sum()
        })
        .into_iter()
        .map(|(i, score)| SearchResult::new(i, score))
        .collect()
    }
}

#[inline]
fn ranked_scores<V: VsaVector + Sync>(
    query: &V,
    candidates: &[V],
    top_k: usize,
) -> Vec<SearchResult> {
    top_k_by(candidates.len(), top_k, |i| {
        query.cosine_similarity(&candidates[i])
    })
    .into_iter()
    .map(|(i, score)| SearchResult::new(i, score))
    .collect()
}

/// Candidate count at which scoring fans out across the rayon pool.
/// Below this, thread hand-off costs more than the scoring itself.
const PAR_THRESHOLD: usize = 256;

/// Descending score order; ties go to the lower index so results are
/// deterministic regardless of how rayon splits the work.
#[inline]
fn rank_order(a: &(usize, f64), b: &(usize, f64)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
}

/// Index and score of the highest-scoring item among `0..n`, or `None` if `n == 0`.
pub(crate) fn best_by<F>(n: usize, score: F) -> Option<(usize, f64)>
where
    F: Fn(usize) -> f64 + Sync,
{
    let pick = |a: (usize, f64), b: (usize, f64)| {
        if rank_order(&a, &b) == std::cmp::Ordering::Greater {
            b
        } else {
            a
        }
    };
    if n >= PAR_THRESHOLD {
        (0..n)
            .into_par_iter()
            .map(|i| (i, score(i)))
            .reduce_with(pick)
    } else {
        (0..n).map(|i| (i, score(i))).reduce(pick)
    }
}

/// The `k` highest-scoring items among `0..n`, sorted best first.
///
/// Scores in parallel for large `n`, then uses an O(n) partial selection
/// instead of sorting every candidate.
pub(crate) fn top_k_by<F>(n: usize, k: usize, score: F) -> Vec<(usize, f64)>
where
    F: Fn(usize) -> f64 + Sync,
{
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let mut scores: Vec<(usize, f64)> = if n >= PAR_THRESHOLD {
        (0..n).into_par_iter().map(|i| (i, score(i))).collect()
    } else {
        (0..n).map(|i| (i, score(i))).collect()
    };
    if k < scores.len() {
        scores.select_nth_unstable_by(k - 1, rank_order);
        scores.truncate(k);
    }
    scores.sort_unstable_by(rank_order);
    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsc::BscSetup;
    use crate::fhrr::FhrrSetup;
    use crate::map::MapSetup;

    #[test]
    fn map_search_finds_self() {
        let q = MapSetup::random(1024);
        let cands = [q.clone(), MapSetup::random(1024)];
        let top = search_map(&q, &cands, 2);
        assert_eq!(top[0].index, 0);
        assert!(top[0].score > 0.99);
    }

    #[test]
    fn bsc_search_scores_self_highest() {
        let q = HDVector::random(1024);
        let b = BscSetup::from_hd(&q);
        let cands = [b.clone(), BinaryHDVector::random(1024)];
        let top = search_bsc(&b, &cands, 2);
        assert_eq!(top[0].index, 0);
    }

    #[test]
    fn fhrr_search_finds_self() {
        let q = FhrrSetup::random(1024);
        let cands = [q.clone(), FhrrSetup::random(1024)];
        let top = search_fhrr(&q, &cands, 2);
        assert_eq!(top[0].index, 0);
        assert!(top[0].score > 0.99);
    }

    #[test]
    fn top_k_by_matches_full_sort_serial_and_parallel() {
        for &n in &[10usize, PAR_THRESHOLD - 1, PAR_THRESHOLD, 5_000] {
            let scores: Vec<f64> = (0..n)
                .map(|i| ((i * 7919) % 1013) as f64 / 1013.0)
                .collect();
            let mut expected: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
            expected.sort_by(rank_order);
            for &k in &[1usize, 5, n, n + 3] {
                let got = top_k_by(n, k, |i| scores[i]);
                assert_eq!(got, expected[..k.min(n)].to_vec(), "n={n} k={k}");
            }
            assert_eq!(best_by(n, |i| scores[i]), Some(expected[0]));
        }
    }

    #[test]
    fn ties_resolve_to_lowest_index() {
        let n = 1_000;
        assert_eq!(best_by(n, |_| 0.5), Some((0, 0.5)));
        let top: Vec<usize> = top_k_by(n, 3, |_| 0.5)
            .into_iter()
            .map(|(i, _)| i)
            .collect();
        assert_eq!(top, vec![0, 1, 2]);
        assert!(top_k_by(0, 3, |_| 0.0).is_empty());
        assert_eq!(best_by(0, |_| 0.0), None);
    }

    #[test]
    fn query_weighted_fuses_views_by_weight() {
        let dim = 1024;
        let text: Vec<HDVector> = (0..300).map(|_| HDVector::random(dim)).collect();
        let image: Vec<HDVector> = (0..300).map(|_| HDVector::random(dim)).collect();
        let query = text[7].clone();
        let sets: [(&[HDVector], f64); 2] = [(&text, 0.7), (&image, 0.3)];

        let got = SuperSearch::query_weighted(&query, &sets, 5);
        assert_eq!(got.len(), 5);
        assert_eq!(got[0].index, 7);

        // Every returned score is exactly the weighted sum for that item.
        for r in &got {
            let expected = 0.7 * query.cosine_similarity(&text[r.index])
                + 0.3 * query.cosine_similarity(&image[r.index]);
            assert!((r.score - expected).abs() < 1e-12);
        }
        // And they come back best first.
        assert!(got.windows(2).all(|w| w[0].score >= w[1].score));
    }

    #[test]
    fn query_weighted_weights_change_the_winner() {
        let dim = 1024;
        let a: Vec<HDVector> = (0..4).map(|_| HDVector::random(dim)).collect();
        let b: Vec<HDVector> = (0..4).map(|_| HDVector::random(dim)).collect();
        // The query matches item 1 in view a and item 3 in view b.
        let query = a[1].bundle(&b[3]);
        let favour_a = SuperSearch::query_weighted(&query, &[(&a[..], 1.0), (&b[..], 0.0)], 1);
        let favour_b = SuperSearch::query_weighted(&query, &[(&a[..], 0.0), (&b[..], 1.0)], 1);
        assert_eq!(favour_a[0].index, 1);
        assert_eq!(favour_b[0].index, 3);
    }

    #[test]
    fn query_weighted_single_view_matches_query() {
        let cands: Vec<HDVector> = (0..50).map(|_| HDVector::random(512)).collect();
        let q = HDVector::random(512);
        let fused = SuperSearch::query_weighted(&q, &[(&cands[..], 1.0)], 10);
        assert_eq!(fused, SuperSearch::query(&q, &cands, 10));
    }

    #[test]
    fn query_weighted_empty_inputs() {
        let q = HDVector::random(64);
        let cands: Vec<HDVector> = (0..3).map(|_| HDVector::random(64)).collect();
        assert!(SuperSearch::query_weighted::<HDVector>(&q, &[], 5).is_empty());
        assert!(SuperSearch::query_weighted(&q, &[(&cands[..], 1.0)], 0).is_empty());
        let none: Vec<HDVector> = Vec::new();
        assert!(SuperSearch::query_weighted(&q, &[(&none[..], 1.0)], 5).is_empty());
    }

    #[test]
    #[should_panic(expected = "view 1 has 2 items but view 0 has 3")]
    fn query_weighted_rejects_mismatched_views() {
        let q = HDVector::random(64);
        let a: Vec<HDVector> = (0..3).map(|_| HDVector::random(64)).collect();
        let b: Vec<HDVector> = (0..2).map(|_| HDVector::random(64)).collect();
        SuperSearch::query_weighted(&q, &[(&a[..], 1.0), (&b[..], 1.0)], 5);
    }
}
