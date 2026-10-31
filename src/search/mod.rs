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
use rayon::prelude::*;
use crate::hdc::vsa_trait::VsaVector;
use crate::hdc::vector::{BinaryHDVector, HDVector};

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
        self.score.partial_cmp(&other.score).unwrap_or(std::cmp::Ordering::Equal)
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
    pub fn query<V: VsaVector + Sync>(query: &V, candidates: &[V], top_k: usize) -> Vec<SearchResult> {
        ranked_scores(query, candidates, top_k)
    }

    pub fn query_weighted<V: VsaVector>(
        query: &V,
        sets: &[(&[V], f64)],
        top_k: usize,
    ) -> Vec<SearchResult> {
        let fused: Vec<Option<f64>> = sets.iter().fold(
            vec![None; sets.iter().map(|s| s.0.len()).sum()],
            |mut acc, (candidates, weight)| {
                let local: Vec<(usize, f64)> = candidates
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, query.cosine_similarity(c) * weight))
                    .collect();
                let mut sorted = local;
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                for (idx, score) in sorted {
                    acc[idx] = Some(acc[idx].unwrap_or(0.0) + score);

                }
                acc
            },
        );

        let mut out: Vec<SearchResult> = fused
            .into_iter()
            .enumerate()
            .filter_map(|(i, s)| s.map(|score| SearchResult::new(i, score)))
            .collect();
        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(top_k);
        out
    }
}

#[inline]
fn ranked_scores<V: VsaVector + Sync>(query: &V, candidates: &[V], top_k: usize) -> Vec<SearchResult> {
    top_k_by(candidates.len(), top_k, |i| query.cosine_similarity(&candidates[i]))
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
        if rank_order(&a, &b) == std::cmp::Ordering::Greater { b } else { a }
    };
    if n >= PAR_THRESHOLD {
        (0..n).into_par_iter().map(|i| (i, score(i))).reduce_with(pick)
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
    use crate::map::MapSetup;
    use crate::bsc::BscSetup;
    use crate::fhrr::FhrrSetup;

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
            let scores: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1013) as f64 / 1013.0).collect();
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
        let top: Vec<usize> = top_k_by(n, 3, |_| 0.5).into_iter().map(|(i, _)| i).collect();
        assert_eq!(top, vec![0, 1, 2]);
        assert!(top_k_by(0, 3, |_| 0.0).is_empty());
        assert_eq!(best_by(0, |_| 0.0), None);
    }
}
