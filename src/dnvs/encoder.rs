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
//! DNVS Encoder — HDVector (MAP) implementation

use crate::hdc::vector::HDVector;
use crate::dnvs::config::{DnvsConfig, DnvsMode};

/// DNVS encoder for HDVector (MAP representation)
pub struct DnvsEncoder {
    config: DnvsConfig,
    /// Precomputed position vectors for spatial encoding
    position_vectors: Vec<HDVector>,
    /// Intensity level vectors
    level_vectors: Vec<HDVector>,
}

impl DnvsEncoder {
    /// Create a new encoder with the given config and precomputed vectors
    pub fn new(config: DnvsConfig, position_vectors: Vec<HDVector>, level_vectors: Vec<HDVector>) -> Self {
        assert_eq!(position_vectors.len(), config.dim, "position vectors must match dimension");
        assert_eq!(level_vectors.len(), config.n_levels);
        Self { config, position_vectors, level_vectors }
    }

    /// Build encoder from a config using a closure to generate vectors
    pub fn from_config<F>(config: DnvsConfig, mut gen_position: F, mut gen_level: F) -> Self
    where
        F: FnMut() -> HDVector,
    {
        let position_vectors = (0..config.dim).map(|_| gen_position()).collect();
        let level_vectors = (0..config.n_levels).map(|_| gen_level()).collect();
        Self::new(config, position_vectors, level_vectors)
    }

    /// Encode a signal (e.g., flattened image pixels) into a VSA vector
    pub fn encode(&self, signal: &[f32]) -> HDVector {
        let mut accum = HDVector::zeros(self.config.dim);

        for (idx, &val) in signal.iter().enumerate() {
            // Apply mode filtering
            let include = match self.config.mode {
                DnvsMode::Negative => val < self.config.threshold,
                DnvsMode::Positive => val >= self.config.threshold,
                DnvsMode::All => true,
            };

            if !include {
                if self.config.skip_empty && val < self.config.threshold {
                    continue;
                }
                if self.config.mode == DnvsMode::All && val < self.config.threshold {
                    continue;
                }
            }

            let adjusted = val.powf(self.config.gamma as f32);
            let level = (adjusted * (self.config.n_levels - 1) as f32) as usize;
            let level = level.min(self.config.n_levels - 1);

            // Bind position and intensity: pos ⊙ level
            let pos = &self.position_vectors[idx];
            let level_vec = &self.level_vectors[level];
            let bound = pos.bind(level_vec);

            accum = accum.bundle(&bound);
        }

        accum
    }

    /// Encode to raw accumulator (pre-binarization)
    pub fn encode_to_accum(&self, signal: &[f32]) -> Vec<f64> {
        let mut accum = vec![0.0_f64; self.config.dim];

        for (idx, &val) in signal.iter().enumerate() {
            let include = match self.config.mode {
                DnvsMode::Negative => val < self.config.threshold,
                DnvsMode::Positive => val >= self.config.threshold,
                DnvsMode::All => true,
            };

            if !include {
                continue;
            }

            let adjusted = val.powf(self.config.gamma as f32);
            let level = (adjusted * (self.config.n_levels - 1) as f32) as usize;
            let level = level.min(self.config.n_levels - 1);

            let pos = self.position_vectors[idx].data();
            let lev_data = self.level_vectors[level].data();
            for d in 0..self.config.dim {
                accum[d] += pos[d] * lev_data[d];
            }
        }

        accum
    }

    /// Get the config
    pub fn config(&self) -> &DnvsConfig {
        &self.config
    }

    /// Get position vectors
    pub fn position_vectors(&self) -> &[HDVector] {
        &self.position_vectors
    }

    /// Get level vectors
    pub fn level_vectors(&self) -> &[HDVector] {
        &self.level_vectors
    }
}