// Copyright (C) 2025-2026 guddalm_vsa contributors.
// SPDX-License-Identifier: AGPL-3.0
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! The `guddalm_vsa` prelude.
//!
//! Convenient re-exports of core primitives, vector types, and traits needed
//! to immediately start building a Vector Symbolic Architecture application.
//!
//! # Usage
//!
//! ```
//! use guddalm_vsa::prelude::*;
//!
//! let mut mem: ItemMemory<HDVector> = ItemMemory::new(10000);
//! let token_a = mem.get_or_create("apple");
//! let token_b = mem.get_or_create("fruit");
//! let bound = token_a.bind(&token_b);
//! let recovered = bound.unbind(&token_a);
//!
//! let (symbol, sim) = mem.cleanup(&recovered).unwrap();
//! assert_eq!(symbol, "fruit");
//! ```

pub use crate::error::{GuddaError, GuddaResult};

// Vector representations
pub use crate::hdc::vector::{BinaryHDVector, Complex, HDVector};
pub use crate::hdc::fhrr::FHRRVector;
pub use crate::hdc::ghrr::GHRRVector;
pub use crate::hdc::vsa_trait::{IndexVector, VsaVector, VsaVectorRaw};

// Associative symbol & item memories
pub use crate::hdc::item_memory::{DeterministicSymbol, ItemMemory};
pub use crate::vsa::{Codebook, VsaEngine};
pub use crate::hdc::cleanup::{BinaryCleanupMemory, CleanupMemory, CleanupResult, FhrrCleanupMemory};

// Resonator factorization
pub use crate::hdc::resonator::{
    generate_rc_codebook, resonator_search, resonator_search_auto,
    resonator_search_auto_acf, ResonatorResult,
};

// Continuous & spatial encoders
pub use crate::hdc::level::ScalarEncoder;
pub use crate::hdc::rff::ContinuousSpaceEncoder;

// Algebraic primitives
pub use crate::primitives::{
    bind_sequence, bundle_sequence, decode_set, encode_graph_edges,
    encode_ngram_sequence, encode_non_commutative_edges,
    encode_positional_sequence, encode_set,
};

// Deterministic seed generation
pub use crate::seed::{
    deterministic_binary_hd_vector, deterministic_fhrr_vector,
    deterministic_ghrr_vector as deterministic_ghrr_vector_seed,
    deterministic_hd_vector, deterministic_seed,
};
