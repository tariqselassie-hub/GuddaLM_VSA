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
//! # Unified Error Types for GuddaLM
//!
//! Provides a single error enum and result alias used across the workspace.
//! Individual crates can wrap their domain-specific errors into these variants.

use std::fmt;

/// Unified error type for all GuddaLM crate operations.
#[derive(Debug)]
pub enum GuddaError {
    /// I/O errors (file read/write, network).
    Io(std::io::Error),

    /// Serialization/deserialization errors (bincode, JSON).
    Serialize(String),

    /// Dimension mismatch between vectors.
    DimMismatch { expected: usize, got: usize },

    /// Invalid configuration or parameter.
    Config(String),

    /// Parse errors (AST, tokenizer, GGUF).
    Parse(String),

    /// Generic catch-all for errors that don't fit other categories.
    Other(String),
}

impl fmt::Display for GuddaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GuddaError::Io(e) => write!(f, "I/O error: {}", e),
            GuddaError::Serialize(msg) => write!(f, "serialization error: {}", msg),
            GuddaError::DimMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
            GuddaError::Config(msg) => write!(f, "configuration error: {}", msg),
            GuddaError::Parse(msg) => write!(f, "parse error: {}", msg),
            GuddaError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for GuddaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GuddaError::Io(e) => Some(e),
            _ => None,
        }
    }
}

// ── From impls for ergonomic `?` usage ───────────────────────

impl From<std::io::Error> for GuddaError {
    fn from(e: std::io::Error) -> Self {
        GuddaError::Io(e)
    }
}

impl From<Box<bincode::ErrorKind>> for GuddaError {
    fn from(e: Box<bincode::ErrorKind>) -> Self {
        GuddaError::Serialize(format!("bincode: {}", e))
    }
}

impl From<serde_json::Error> for GuddaError {
    fn from(e: serde_json::Error) -> Self {
        GuddaError::Serialize(format!("json: {}", e))
    }
}

impl From<String> for GuddaError {
    fn from(s: String) -> Self {
        GuddaError::Other(s)
    }
}

impl From<&str> for GuddaError {
    fn from(s: &str) -> Self {
        GuddaError::Other(s.to_string())
    }
}

/// Alias for `Result<T, GuddaError>`.
pub type GuddaResult<T> = Result<T, GuddaError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = GuddaError::DimMismatch {
            expected: 1024,
            got: 512,
        };
        assert!(e.to_string().contains("1024"));
        assert!(e.to_string().contains("512"));
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let gudda_err: GuddaError = io_err.into();
        assert!(gudda_err.to_string().contains("file missing"));
    }

    #[test]
    fn test_from_string() {
        let err: GuddaError = "something went wrong".into();
        assert!(err.to_string().contains("something went wrong"));
    }
}
