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
//! Configuration types for DNVS

/// DNVS operating mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DnvsMode {
    /// Encode only negative space (val < threshold)
    Negative,
    /// Encode only positive space (val >= threshold)
    Positive,
    /// Encode all values
    All,
}

/// Configuration for DNVS encoding and retraining
#[derive(Debug, Clone)]
pub struct DnvsConfig {
    /// Vector dimension
    pub dim: usize,
    /// Number of intensity levels
    pub n_levels: usize,
    /// Encoding mode
    pub mode: DnvsMode,
    /// Intensity threshold for negative/positive split
    pub threshold: f32,
    /// Gamma correction for intensity
    pub gamma: f64,
    /// Number of retraining rounds
    pub retrain_rounds: usize,
    /// Retraining weight (negative = dynamic margin, positive = fixed margin)
    pub retrain_weight: f64,
    /// Skip empty/zero values
    pub skip_empty: bool,
}

impl Default for DnvsConfig {
    fn default() -> Self {
        Self {
            dim: 10000,
            n_levels: 32,
            mode: DnvsMode::Negative,
            threshold: 0.01,
            gamma: 1.0,
            retrain_rounds: 3,
            retrain_weight: -1.0,
            skip_empty: false,
        }
    }
}

impl DnvsConfig {
    /// Create config for classic MNIST DNVS (negative encoding)
    pub fn mnist_negative(dim: usize) -> Self {
        Self {
            dim,
            mode: DnvsMode::Negative,
            ..Default::default()
        }
    }

    /// Create config for positive encoding
    pub fn positive(dim: usize) -> Self {
        Self {
            dim,
            mode: DnvsMode::Positive,
            ..Default::default()
        }
    }
}