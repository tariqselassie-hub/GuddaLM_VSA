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
//! Dynamic Negative Vector Search (DNVS) — HDVector (MAP) Implementation
//!
//! Implements the DNVS algorithm as a reusable system for HDVector (MAP representation).
//!
//! **Core idea**: Encode only the "negative space" (background/voids) of an
//! input signal. In image classification, this means encoding only pixels
//! below a threshold (e.g., `val < 0.01`), forming the image vector exclusively
//! from background regions.
//!
//! The "Dynamic" component applies iterative retraining with adaptive margins
//! to push misclassified samples away from wrong prototypes and toward correct ones.

pub mod config;
pub mod encoder;
pub mod retrain;
pub mod classifier;

pub use config::{DnvsConfig, DnvsMode};
pub use encoder::DnvsEncoder;
pub use retrain::DnvsRetrainer;
pub use classifier::DnvsClassifier;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed::deterministic_hd_vector;

    fn toy_config(n_classes: usize) -> DnvsConfig {
        DnvsConfig {
            dim: 128,
            n_classes,
            n_levels: 4,
            mode: DnvsMode::Positive,
            threshold: 0.5,
            gamma: 1.0,
            retrain_rounds: 2,
            retrain_weight: -1.0,
            skip_empty: false,
        }
    }

    fn pattern_signal(dim: usize, class: usize, n_classes: usize) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                if i % n_classes == class {
                    0.9
                } else {
                    0.0
                }
            })
            .collect()
    }

    #[test]
    fn config_n_classes_is_configurable() {
        let cfg = DnvsConfig {
            n_classes: 5,
            ..Default::default()
        };
        assert_eq!(cfg.n_classes(), 5);
        assert_eq!(DnvsConfig::mnist_negative(256).n_classes(), 10);
    }

    #[test]
    fn encoder_respects_signal_length_bound() {
        let config = toy_config(2);
        let mut pos = 0usize;
        let encoder = DnvsEncoder::from_config(
            config.clone(),
            || {
                pos += 1;
                deterministic_hd_vector(7, &format!("pos:{pos}"), config.dim)
            },
            || deterministic_hd_vector(7, "level", config.dim),
        );
        let long_signal = vec![0.9_f32; config.dim + 50];
        let encoded = encoder.encode(&long_signal);
        assert_eq!(encoded.dim(), config.dim);
    }

    #[test]
    fn classifier_trains_and_predicts_toy_data() {
        let config = toy_config(3);
        let dim = config.dim;
        let mut pos = 0usize;
        let mut lvl = 0usize;
        let mut classifier = DnvsClassifier::from_config(
            config.clone(),
            || {
                pos += 1;
                deterministic_hd_vector(11, &format!("pos:{pos}"), dim)
            },
            || {
                lvl += 1;
                deterministic_hd_vector(11, &format!("lvl:{lvl}"), dim)
            },
        );

        let s0 = pattern_signal(dim, 0, 3);
        let s1 = pattern_signal(dim, 1, 3);
        let s2 = pattern_signal(dim, 2, 3);
        let train = [&s0[..], &s1[..], &s2[..], &s0[..], &s1[..], &s2[..]];
        let labels = [0, 1, 2, 0, 1, 2];

        classifier.train(&train, &labels);

        for (signal, &label) in train.iter().zip(labels.iter()) {
            let (pred, _) = classifier.predict(signal);
            assert_eq!(pred, label, "training sample should classify correctly");
        }

        let (acc, _) = classifier.evaluate(&train, &labels);
        assert!(acc >= 1.0 - 1e-9);
        assert_eq!(classifier.prototypes().len(), 3);
    }
}
