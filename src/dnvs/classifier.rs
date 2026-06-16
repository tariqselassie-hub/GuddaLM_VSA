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
//! DNVS Classifier — Complete pipeline for VSA-based classification (HDVector)

use crate::hdc::vector::HDVector;
use crate::dnvs::config::DnvsConfig;
use crate::dnvs::encoder::DnvsEncoder;
use crate::dnvs::retrain::DnvsRetrainer;

/// Complete DNVS classifier (HDVector-specific)
pub struct DnvsClassifier {
    config: DnvsConfig,
    encoder: DnvsEncoder,
    retrainer: DnvsRetrainer,
}

impl DnvsClassifier {
    /// Create a new classifier from config, encoder, and retrainer
    pub fn new(config: DnvsConfig, encoder: DnvsEncoder, retrainer: DnvsRetrainer) -> Self {
        Self { config, encoder, retrainer }
    }

    /// Create from config with vector generators
    pub fn from_config<F>(config: DnvsConfig, mut gen_position: F, mut gen_level: F) -> Self
    where
        F: FnMut() -> HDVector,
    {
        let encoder = DnvsEncoder::from_config(config.clone(), &mut gen_position, &mut gen_level);
        let prototypes = vec![HDVector::zeros(config.dim); config.n_classes()];
        let retrainer = DnvsRetrainer::new(config.clone(), prototypes);
        Self { config, encoder, retrainer }
    }

    /// Train on labeled data
    pub fn train<E>(&mut self, train_data: &[&[f32]], train_labels: &[usize])
    where
        E: Fn(&[f32]) -> HDVector,
    {
        // First pass: build initial prototypes
        let n_classes = self.config.n_classes();
        let mut sums = vec![vec![0.0; self.config.dim]; n_classes];

        for (signal, &label) in train_data.iter().zip(train_labels.iter()) {
            let encoded = self.encoder.encode(signal);
            let data = encoded.data();
            for d in 0..self.config.dim {
                sums[label][d] += data[d];
            }
        }

        // Build initial prototypes
        let prototypes: Vec<HDVector> = sums
            .iter()
            .map(|sum| HDVector::from_slice(sum).binarize())
            .collect();

        // Create retrainer with these prototypes
        self.retrainer = DnvsRetrainer::new(self.config.clone(), prototypes);

        // Run retraining
        self.retrainer
            .retrain(train_data, train_labels, |s| self.encoder.encode(s));
    }

    /// Predict class for a single signal
    pub fn predict(&self, signal: &[f32]) -> (usize, f64) {
        let encoded = self.encoder.encode(signal);
        let prototypes = self.retrainer.prototypes();

        let mut best = 0;
        let mut best_sim = -1.0;
        for (c, proto) in prototypes.iter().enumerate() {
            let s = encoded.cosine_similarity(proto);
            if s > best_sim {
                best_sim = s;
                best = c;
            }
        }
        (best, best_sim)
    }

    /// Evaluate on test data
    pub fn evaluate(&self, test_data: &[&[f32]], test_labels: &[usize]) -> (f64, Vec<f64>) {
        let n_classes = self.config.n_classes();
        let mut correct = 0;
        let mut per_class_correct = vec![0; n_classes];
        let mut per_class_total = vec![0; n_classes];

        for (signal, &label) in test_data.iter().zip(test_labels.iter()) {
            let (pred, _) = self.predict(signal);
            per_class_total[label] += 1;
            if pred == label {
                correct += 1;
                per_class_correct[label] += 1;
            }
        }

        let acc = correct as f64 / test_data.len() as f64;
        let per_class_acc: Vec<f64> = per_class_correct
            .iter()
            .zip(per_class_total.iter())
            .map(|(c, t)| *c as f64 / (*t).max(1) as f64)
            .collect();

        (acc, per_class_acc)
    }

    /// Get config
    pub fn config(&self) -> &DnvsConfig {
        &self.config
    }

    /// Get encoder
    pub fn encoder(&self) -> &DnvsEncoder {
        &self.encoder
    }

    /// Get prototypes
    pub fn prototypes(&self) -> &[HDVector] {
        self.retrainer.prototypes()
    }
}

impl DnvsConfig {
    /// Number of classes for default MNIST (10)
    pub fn n_classes(&self) -> usize {
        10
    }
}