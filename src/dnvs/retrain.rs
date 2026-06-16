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
//! DNVS Retrainer — Dynamic retraining with adaptive margins (HDVector)

use crate::hdc::vector::HDVector;
use crate::dnvs::config::DnvsConfig;

/// Retrainer for DNVS classifiers (HDVector-specific)
pub struct DnvsRetrainer {
    config: DnvsConfig,
    /// Prototype accumulators (pre-binarization)
    prototype_sums: Vec<Vec<f64>>,
    /// Current prototypes
    prototypes: Vec<HDVector>,
}

impl DnvsRetrainer {
    /// Create a new retrainer with initial prototypes
    pub fn new(config: DnvsConfig, prototypes: Vec<HDVector>) -> Self {
        let dim = config.dim;
        let n_classes = prototypes.len();
        Self {
            config,
            prototype_sums: vec![vec![0.0; dim]; n_classes],
            prototypes,
        }
    }

    /// Create from encoder and initial training data
    pub fn from_encoder<E>(
        config: DnvsConfig,
        encoder: &E,
        train_data: &[&[f32]],
        train_labels: &[usize],
        n_classes: usize,
    ) -> Self
    where
        E: Fn(&[f32]) -> HDVector,
    {
        let mut sums = vec![vec![0.0; config.dim]; n_classes];

        for (signal, &label) in train_data.iter().zip(train_labels.iter()) {
            let encoded = encoder(signal);
            let data = encoded.data();
            for d in 0..config.dim {
                sums[label][d] += data[d];
            }
        }

        let prototypes: Vec<HDVector> = sums
            .iter()
            .map(|sum| HDVector::from_slice(sum))
            .collect();

        Self {
            config,
            prototype_sums: sums,
            prototypes,
        }
    }

    /// Run one retraining round
    pub fn retrain_round<E>(&mut self, train_data: &[&[f32]], train_labels: &[usize], encode_fn: E) -> usize
    where
        E: Fn(&[f32]) -> HDVector,
    {
        let mut errors = 0;

        for (signal, &true_label) in train_data.iter().zip(train_labels.iter()) {
            let encoded = encode_fn(signal);

            // Find best matching prototype
            let mut best = 0;
            let mut best_sim = -1.0;
            let mut sims = vec![0.0; self.prototypes.len()];

            for (c, proto) in self.prototypes.iter().enumerate() {
                let s = encoded.cosine_similarity(proto);
                sims[c] = s;
                if s > best_sim {
                    best_sim = s;
                    best = c;
                }
            }

            if best != true_label {
                let diff = sims[best] - sims[true_label];
                let margin = self.compute_margin(diff);

                let hd = encoded.data();
                for d in 0..self.config.dim {
                    self.prototype_sums[best][d] -= hd[d] * margin;
                    self.prototype_sums[true_label][d] += hd[d] * margin;
                }
                errors += 1;
            }
        }

        // Re-binarize prototypes
        for c in 0..self.prototypes.len() {
            self.prototypes[c] = HDVector::from_slice(&self.prototype_sums[c]).binarize();
        }

        errors
    }

    /// Compute adaptive margin based on config
    fn compute_margin(&self, diff: f64) -> f64 {
        if self.config.retrain_weight == 0.0 {
            1.0
        } else if self.config.retrain_weight < 0.0 {
            // Dynamic: larger correction for near-boundary samples
            (1.0 + (1.0 - diff) * self.config.retrain_weight.abs())
                .clamp(1.0, 1.0 + self.config.retrain_weight.abs())
        } else {
            // Fixed proportional
            (diff * self.config.retrain_weight).clamp(0.0, 1.0)
        }
    }

    /// Run multiple retraining rounds
    pub fn retrain<E>(&mut self, train_data: &[&[f32]], train_labels: &[usize], encode_fn: E) -> Vec<usize>
    where
        E: Fn(&[f32]) -> HDVector,
    {
        let mut errors_per_round = Vec::new();
        for round in 0..self.config.retrain_rounds {
            let errors = self.retrain_round(train_data, train_labels, &encode_fn);
            errors_per_round.push(errors);
            eprintln!(
                "DNVS retrain round {}/{}: {} errors",
                round + 1,
                self.config.retrain_rounds,
                errors
            );
        }
        errors_per_round
    }

    /// Get current prototypes
    pub fn prototypes(&self) -> &[HDVector] {
        &self.prototypes
    }

    /// Get mutable prototypes
    pub fn prototypes_mut(&mut self) -> &mut [HDVector] {
        &mut self.prototypes
    }
}