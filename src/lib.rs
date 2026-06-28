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
pub mod error;
pub mod hdc;
pub mod map;
pub mod bsc;
pub mod fhrr;
pub mod search;
pub mod dnvs;
pub mod seed;
pub mod serialize_vsa;
pub mod setup;
pub mod vsa;
pub mod vsa_persist;
pub mod primitives;
pub mod primitives_shim;

pub use hdc::vector::{HDVector, BinaryHDVector, Complex};
pub use hdc::fhrr::FHRRVector;
pub use hdc::ghrr::GHRRVector;
pub use hdc::vector::{
    dot_product_slice, cosine_similarity_slice,
    convolve_slices, correlate_slices,
    par_convolve_batch, par_correlate_batch, par_cosine_similarity_slice,
    majority_from_sums,
};
pub use vsa::{Codebook, VsaEngine};
pub use hdc::bundle::{weighted_bundle, selective_bundle};
pub use hdc::sdm::{sdm_snr_threshold_bipolar, optimal_snr_hamming_radius, sdm_read_bipolar};
pub use hdc::quantize::{pack_bits, pack_bits_array64, PackedArray64};
pub use hdc::rff::ContinuousSpaceEncoder;
pub use hdc::cleanup::CleanupMemory;
pub use hdc::stream::HDStreamBuffer;
pub use hdc::resonator::{resonator_search, resonator_search_auto, resonator_search_auto_acf, generate_rc_codebook, ResonatorResult};
pub use hdc::attention::MultiHeadAttention;
pub use hdc::sequence::SequenceLearner;
pub use hdc::graph::{GraphEncoder, GhrrGraphEncoder};
pub use seed::{
    deterministic_seed,
    deterministic_hd_vector,
    deterministic_binary_hd_vector,
    deterministic_fhrr_vector,
    deterministic_fhrr_continuous,
    deterministic_ghrr_vector as deterministic_ghrr_vector_seed,
};
pub use setup::{VsaSystem, VsaMode, VsaSystemOptions, SystemVector};
pub use vsa_persist::{
    default_store_dir,
    load_bundle,
    model_bin_path,
    model_bin_path_env,
    save_bundle,
    VsaPersistenceError,
    VsaPersistentBundle,
    VsaPersistenceResult,
    BIN_FILENAME,
    BIN_DIRNAME,
};
pub use hdc::vsa_trait::{VsaVectorRaw, VsaVector};
pub use error::{GuddaError, GuddaResult};
pub use primitives::{bind_sequence, bundle_sequence, encode_set, decode_set, encode_positional_sequence};
pub use primitives_shim::{cartesian_to_phase, phase_to_cartesian};
pub use hdc::autograd::{GradHDVector, backward, diff_bind, diff_bundle, diff_bundle_many, diff_permute, similarity_loss, SGDOptimizer, soft_cleanup};
#[cfg(feature = "candle")]
pub use hdc::tensor::{to_tensor, from_tensor, to_tensor_batch, from_tensor_batch, tensor_bind, tensor_unbind, tensor_bundle, tensor_permute, tensor_cosine_similarity, tensor_similarity_loss, tensor_selective_bundle, tensor_fwht, tensor_ifwht};
#[cfg(feature = "candle")]
pub use hdc::transformer::DiffVSAEncoderLayer;

#[cfg(test)]
mod tests {
    use crate::hdc::bundle::{selective_bundle, weighted_bundle};
    use crate::hdc::quantize;
    use crate::hdc::vector::{BinaryHDVector, HDVector};

    #[test]
    fn test_binding_self_inverse() {
        let a = HDVector::random(8192);
        let b = HDVector::random(8192);
        let bound = a.bind(&b);
        let unbind = bound.unbind(&b);
        let sim = a.cosine_similarity(&unbind);
        assert!(
            sim > 0.65,
            "MAP binding must be invertible via unbind (got sim={})", sim
        );
    }

    #[test]
    fn test_bundling_similarity() {
        let a = HDVector::random(8192);
        let bundle = a.bundle(&a);
        let sim = a.cosine_similarity(&bundle);
        assert!(
            sim > 0.99,
            "bundle(a,a) must be nearly identical to a (got {})", sim
        );
    }

    #[test]
    fn test_permutation_orthogonality() {
        let a = HDVector::random(8192);
        let shifted = a.permute(1);
        let sim = a.cosine_similarity(&shifted);
        assert!(
            sim.abs() < 0.1,
            "permuted vectors should be nearly orthogonal (got {})", sim
        );
    }

    #[test]
    fn test_permutation_inverse() {
        let a = HDVector::random(8192);
        let shifted = a.permute(1);
        let unshifted = shifted.permute_left(1);
        assert_eq!(a, unshifted, "cyclic shift must be invertible");
    }

    #[test]
    fn test_binary_xor_self_inverse() {
        let a = BinaryHDVector::random(10000);
        let b = BinaryHDVector::random(10000);
        let bound = a.xor_bind(&b);
        let unbind = bound.xor_bind(&b);
        assert_eq!(a, unbind, "XOR binding must be self-inverse");
    }

    #[test]
    fn test_binary_majority_bundle() {
        let a = BinaryHDVector::random(10000);
        let bundle = a.majority_bundle(&a);
        let sim = a.hamming_similarity(&bundle);
        assert!(
            sim > 0.99,
            "majority bundle of identical vectors must be nearly identical"
        );
    }

    #[test]
    fn test_binary_rotation() {
        let a = BinaryHDVector::random(10000);
        let rotated = a.rotate(1);
        let unrotated = rotated.rotate(9999);
        assert_eq!(a, unrotated, "bit rotation must be invertible");
    }

    #[test]
    fn test_conversion_bipolar_to_binary() {
        let bipolar = HDVector::random(1000);
        let binary = BinaryHDVector::from_bipolar(&bipolar);
        let back = quantize::unpack_bits(binary.words(), bipolar.dim());
        let sim = bipolar.cosine_similarity(&back);
        assert!(
            (sim - 1.0).abs() < 0.01,
            "bipolar ↔ binary round-trip must preserve content"
        );
    }

    #[test]
    fn test_packed_similarity() {
        let a = HDVector::random(10000);
        let b = HDVector::random(10000);
        let packed = quantize::pack_bits(&b);
        let sim_original = a.cosine_similarity(&b);
        let sim_packed = quantize::packed_similarity(&a, &packed);
        let diff = (sim_original - sim_packed).abs();
        assert!(
            diff < 0.01,
            "packed similarity must approximate cosine similarity (diff = {})", diff
        );
    }

    #[test]
    fn test_weighted_bundle_favors_similar() {
        let query = HDVector::random(10000);
        let similar = query.clone();
        let dissimilar = HDVector::random(10000);

        let result = weighted_bundle(&[(similar, 1.0), (dissimilar, 0.1)]);

        let sim_to_query = result.cosine_similarity(&query);
        assert!(
            sim_to_query > 0.3,
            "weighted bundle must favor more similar inputs (sim = {})", sim_to_query
        );
    }

    #[test]
    fn test_selective_bundle() {
        let query = HDVector::random(10000);
        let k1 = query.clone();
        let k2 = HDVector::random(10000);
        let v1 = HDVector::random(10000);
        let v2 = HDVector::random(10000);

        let result = selective_bundle(&query, &[k1, k2], &[v1.clone(), v2.clone()], 0.5);
        let sim_to_v1 = result.cosine_similarity(&v1);
        let sim_to_v2 = result.cosine_similarity(&v2);
        assert!(sim_to_v1 > sim_to_v2, "selective bundle must favor value paired with similar key");
    }
}
