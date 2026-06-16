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
use crate::hdc::vsa_trait::VsaVector;
use crate::hdc::vector::{BinaryHDVector, HDVector};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    pub index: usize,
    pub score: f64,
}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Eq for SearchResult {}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
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
    pub fn query<V: VsaVector>(query: &V, candidates: &[V], top_k: usize) -> Vec<SearchResult> {
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
fn ranked_scores<V: VsaVector>(query: &V, candidates: &[V], top_k: usize) -> Vec<SearchResult> {
    let mut scores: Vec<SearchResult> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| SearchResult::new(i, query.cosine_similarity(c)))
        .collect();
    scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_k);
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
}
