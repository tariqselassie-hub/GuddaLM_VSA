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