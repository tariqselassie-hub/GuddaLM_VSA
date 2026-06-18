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
use crate::hdc::vector::HDVector;
use crate::vsa::{Codebook, VsaEngine};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use dirs;

pub const BIN_FILENAME: &str = "codebook_episodic.bin";
pub const BIN_DIRNAME: &str = "vsa_model_bins";

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct VsaPersistentBundle {
    pub schema_version: u32,
    pub saved_at: String,
    pub dim: usize,
    pub engine: VsaEngine,
    #[serde(default)]
    pub weights: Vec<HDVector>,
    #[serde(default)]
    pub packed: Vec<Vec<u64>>,
    #[serde(default)]
    pub symbol_registry: HashMap<String, usize>,
}

impl VsaPersistentBundle {
    pub fn from_codebook(codebook: &Codebook) -> Self {
        let saved_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true).to_string();

        Self {
            schema_version: 1,
            saved_at,
            dim: codebook.dim,
            engine: codebook.engine.clone(),
            weights: codebook.weights.clone(),
            packed: codebook.packed.clone(),
            symbol_registry: HashMap::new(),
        }
    }

    pub fn try_into_codebook(self) -> Codebook {
        let vocab_size = self.weights.len();
        Codebook {
            weights: self.weights,
            vocab_size,
            dim: self.dim,
            engine: self.engine,
            packed: self.packed,
        }
    }

    pub fn attach_packed(mut self) -> Self {
        if self.packed.is_empty() && !self.weights.is_empty() {
            self.packed = self
                .weights
                .iter()
                .map(|w| crate::hdc::quantize::pack_bits(w))
                .collect();
        }
        self
    }
}

impl Default for VsaPersistentBundle {
    fn default() -> Self {
        Self {
            schema_version: 0,
            saved_at: String::new(),
            dim: 0,
            engine: VsaEngine::new(0),
            weights: Vec::new(),
            packed: Vec::new(),
            symbol_registry: HashMap::new(),
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum VsaPersistenceError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialize(#[from] bincode::Error),
    #[error("invalid schema version: {0}")]
    SchemaVersion(u32),
}

pub type VsaPersistenceResult<T> = Result<T, VsaPersistenceError>;

pub fn default_store_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("guddalm")
        .join(BIN_DIRNAME)
}

pub fn save_bundle(
    path: impl AsRef<Path>,
    bundle: &VsaPersistentBundle,
) -> VsaPersistenceResult<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, bundle)?;
    Ok(())
}

pub fn load_bundle(path: impl AsRef<Path>) -> VsaPersistenceResult<VsaPersistentBundle> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(VsaPersistentBundle::default());
    }
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let bundle: VsaPersistentBundle = bincode::deserialize_from(reader)?;
    if bundle.schema_version != 1 {
        return Err(VsaPersistenceError::SchemaVersion(bundle.schema_version));
    }
    Ok(bundle)
}

pub fn model_bin_path() -> PathBuf {
    default_store_dir().join(BIN_FILENAME)
}

pub fn model_bin_path_env(env_key: &str) -> Option<PathBuf> {
    std::env::var_os(env_key).map(PathBuf::from)
}
