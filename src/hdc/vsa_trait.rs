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
/// # VsaVector Trait — Unified VSA Vector Interface
///
/// This trait provides a common API across all three VSA representations
/// (MAP bipolar, BSC binary, and FHRR complex phase), enabling generic
/// code that works with any representation.
///
/// ## Representations
///
/// | Type | Struct | Storage | Bind | Bundle | Sim |
/// |---|---|---|---|---|
/// | MAP | `HDVector` | `f64` ±1 | FFT convolution | Add | Cosine |
/// | BSC | `BinaryHDVector` | Bit-packed `u64` | XOR | Majority | Popcount |
/// | FHRR | `FHRRVector` | Phase `f64` | Phase add | Complex sum→norm | Avg cos(Δθ) |
///
/// ## Usage
///
/// ```ignore
/// use guddalm_vsa::VsaVector;
///
/// fn compute<V: VsaVector>(a: &V, b: &V) -> f64 {
///     let bound = a.bind(b);
///     let bundled = bound.bundle(a);
///     bundled.cosine_similarity(b)
/// }
/// ```
use crate::hdc::vector::{BinaryHDVector, HDVector};
use crate::hdc::fhrr::FHRRVector;

/// Unified trait for all VSA vector representations.
///
/// Implementors: [`HDVector`] (MAP), [`BinaryHDVector`] (BSC),
/// [`FHRRVector`] (FHRR).
pub trait VsaVector: Clone + Sized {
    /// Return the dimensionality of this vector.
    fn dim(&self) -> usize;

    /// Bind two vectors.
    ///
    /// - MAP: circular convolution (self-inverse, FFT-based)
    /// - BSC: XOR (self-inverse)
    /// - FHRR: phase addition (not self-inverse; inverse = conjugate)
    fn bind(&self, other: &Self) -> Self;

    /// Unbind two vectors (inverse of bind).
    ///
    /// - MAP: circular correlation
    /// - BSC: XOR (same as bind, self-inverse)
    /// - FHRR: bind with inverse (conjugate)
    fn unbind(&self, other: &Self) -> Self;

    /// Bundle two vectors (superposition).
    ///
    /// - MAP: element-wise addition
    /// - BSC: majority rule
    /// - FHRR: complex sum → normalize to unit circle
    fn bundle(&self, other: &Self) -> Self;

    /// Permute (rotate/shuffle) the vector by `shift` positions.
    ///
    /// - MAP / FHRR: cyclic shift of elements
    /// - BSC: cyclic bit rotation
    fn permute(&self, shift: usize) -> Self;

    /// Cosine similarity in [0, 1] or [-1, 1].
    ///
    /// - MAP: dot / (norm_a * norm_b)
    /// - BSC: (same - diff) / dim (bipolar)
    /// - FHRR: (1/D) Σ cos(Δθ)
    fn cosine_similarity(&self, other: &Self) -> f64;

    /// Convert to quantized binary representation (BSC).
    ///
    /// - MAP: sign-threshold → 0/1 bits
    /// - BSC: clone (already binary)
    /// - FHRR: cos(phase) > 0 → 1, else 0
    fn binarize(&self) -> BinaryHDVector;

    /// Create a zero vector of the given dimension.
    fn zero(dim: usize) -> Self;

    /// Create a random vector of the given dimension.
    fn random(dim: usize) -> Self;
}

/// Extension trait for VSA vectors that support raw slice access.
///
/// Only [`HDVector`] (MAP) implements this; BSC and FHRR use different
/// internal representations and do not support direct slice access.
pub trait VsaVectorRaw: VsaVector {
    /// Create a vector from a slice of f64 values.
    fn from_slice(slice: &[f64]) -> Self;

    /// Get the raw data slice.
    fn data(&self) -> &[f64];

    /// Get mutable raw data slice.
    fn data_mut(&mut self) -> &mut [f64];
}

// ── Implementations ───────────────────────────────────────────

impl VsaVector for HDVector {
    #[inline(always)]
    fn dim(&self) -> usize { self.dim() }

    #[inline(always)]
    fn bind(&self, other: &Self) -> Self { HDVector::bind(self, other) }

    #[inline(always)]
    fn unbind(&self, other: &Self) -> Self { HDVector::unbind(self, other) }

    #[inline(always)]
    fn bundle(&self, other: &Self) -> Self { HDVector::bundle(self, other) }

    #[inline(always)]
    fn permute(&self, shift: usize) -> Self { HDVector::permute(self, shift) }

    #[inline(always)]
    fn cosine_similarity(&self, other: &Self) -> f64 {
        if self.is_binary() && other.is_binary() {
            crate::hdc::vector::bipolar_cosine_similarity(self, other)
        } else {
            HDVector::cosine_similarity(self, other)
        }
    }

    fn binarize(&self) -> BinaryHDVector {
        BinaryHDVector::from_bipolar(self)
    }

    fn zero(dim: usize) -> Self { HDVector::zeros(dim) }

    fn random(dim: usize) -> Self { HDVector::random(dim) }
}

impl VsaVectorRaw for HDVector {
    #[inline(always)]
    fn from_slice(slice: &[f64]) -> Self { HDVector::from_slice(slice) }

    #[inline(always)]
    fn data(&self) -> &[f64] { self.data() }

    #[inline(always)]
    fn data_mut(&mut self) -> &mut [f64] { self.data_mut() }
}

impl VsaVector for BinaryHDVector {
    #[inline(always)]
    fn dim(&self) -> usize { self.dim() }

    #[inline(always)]
    fn bind(&self, other: &Self) -> Self { self.xor_bind(other) }

    #[inline(always)]
    fn unbind(&self, other: &Self) -> Self { self.xor_bind(other) }

    #[inline(always)]
    fn bundle(&self, other: &Self) -> Self { self.majority_bundle(other) }

    #[inline(always)]
    fn permute(&self, shift: usize) -> Self { self.rotate(shift) }

    #[inline(always)]
    fn cosine_similarity(&self, other: &Self) -> f64 { self.bipolar_similarity(other) }

    fn binarize(&self) -> BinaryHDVector { self.clone() }

    fn zero(dim: usize) -> Self { BinaryHDVector::zeros(dim) }

    fn random(dim: usize) -> Self { BinaryHDVector::random(dim) }
}

impl VsaVector for FHRRVector {
    #[inline(always)]
    fn dim(&self) -> usize { self.dim() }

    #[inline(always)]
    fn bind(&self, other: &Self) -> Self { FHRRVector::bind(self, other) }

    #[inline(always)]
    fn unbind(&self, other: &Self) -> Self { self.bind(&other.inverse()) }

    #[inline(always)]
    fn bundle(&self, other: &Self) -> Self { FHRRVector::bundle(self, other) }

    #[inline(always)]
    fn permute(&self, shift: usize) -> Self { FHRRVector::permute(self, shift) }

    #[inline(always)]
    fn cosine_similarity(&self, other: &Self) -> f64 { FHRRVector::cosine_similarity(self, other) }

    fn binarize(&self) -> BinaryHDVector {
        let bits: Vec<u8> = self.phases()
            .iter()
            .map(|&p| if p.cos() > 0.0 { 1u8 } else { 0u8 })
            .collect();
        BinaryHDVector::from_bits(&bits)
    }

    fn zero(dim: usize) -> Self { FHRRVector::zeros(dim) }

    fn random(dim: usize) -> Self { FHRRVector::random(dim) }
}

// ── IndexVector ───────────────────────────────────────────────

/// The canonical index vector for GuddaLM.
///
/// `IndexVector` is a newtype around [`BinaryHDVector`] (BSC / Binary
/// Spatter Coding), chosen for:
///
/// - **Speed**: XOR bind, majority bundle, popcount sim — ~100× faster
///   than MAP's FFT-based convolution
/// - **Compactness**: bit-packed `u64` storage, ~64× denser than `f64`
/// - **Hardware-ready**: maps directly to FPGA / CiM XOR-popcount cells
///
/// It adds deterministic string-keyed generation for reproducible
/// indexing (e.g. AST nodes, token codebooks) and convenience methods
/// for role-filler encoding.
///
/// ## Example
///
/// ```ignore
/// use guddalm_vsa::IndexVector;
///
/// let role  = IndexVector::from_key(42, "role:subject", 4096);
/// let filler = IndexVector::from_key(42, "filler:cat", 4096);
/// let encoded = IndexVector::encode_role_filler(&role, &filler);
/// let decoded = encoded.unbind(&role);
/// let sim = decoded.similarity_to(&filler);
/// assert!(sim > 0.3, "role-filler should recover with >0.3 sim");
/// ```
#[derive(Clone, Debug)]
pub struct IndexVector(pub BinaryHDVector);

impl IndexVector {
    /// Wrap a `BinaryHDVector` into an `IndexVector`.
    #[inline(always)]
    pub fn new(inner: BinaryHDVector) -> Self {
        IndexVector(inner)
    }

    /// Unwrap into the underlying `BinaryHDVector`.
    #[inline(always)]
    pub fn into_inner(self) -> BinaryHDVector {
        self.0
    }

    /// Borrow the underlying `BinaryHDVector`.
    #[inline(always)]
    pub fn inner(&self) -> &BinaryHDVector {
        &self.0
    }

    /// Dimensionality.
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.0.dim()
    }

    /// Create a random `IndexVector` of the given dimension.
    #[inline(always)]
    pub fn random(dim: usize) -> Self {
        IndexVector(BinaryHDVector::random(dim))
    }

    /// Create a zero `IndexVector` of the given dimension.
    #[inline(always)]
    pub fn zero(dim: usize) -> Self {
        IndexVector(BinaryHDVector::zeros(dim))
    }

    /// XOR bind (self-inverse).
    #[inline(always)]
    pub fn bind(&self, other: &IndexVector) -> IndexVector {
        IndexVector(self.0.xor_bind(&other.0))
    }

    /// XOR unbind (same as bind, XOR is self-inverse).
    #[inline(always)]
    pub fn unbind(&self, other: &IndexVector) -> IndexVector {
        IndexVector(self.0.xor_bind(&other.0))
    }

    /// Majority bundle.
    #[inline(always)]
    pub fn bundle(&self, other: &IndexVector) -> IndexVector {
        IndexVector(self.0.majority_bundle(&other.0))
    }

    /// Cyclic bit rotation.
    #[inline(always)]
    pub fn permute(&self, shift: usize) -> IndexVector {
        IndexVector(self.0.rotate(shift))
    }

    /// Bipolar similarity in [-1, 1] (1 = identical).
    #[inline(always)]
    pub fn similarity_to(&self, other: &IndexVector) -> f64 {
        self.0.bipolar_similarity(&other.0)
    }

    /// Hamming similarity in [0, 1] (1 = identical).
    #[inline(always)]
    pub fn hamming_to(&self, other: &IndexVector) -> f64 {
        self.0.hamming_similarity(&other.0)
    }

    /// Generate a deterministic `IndexVector` from a string key.
    ///
    /// Uses the same multiplicative hash as the rest of GuddaLM to
    /// guarantee cross-crate reproducibility:
    ///
    /// ```text
    /// seed = base_seed + key.bytes().fold(0, |acc, b| acc * 31 + b)
    /// ```
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let v1 = IndexVector::from_key(42, "token:hello", 4096);
    /// let v2 = IndexVector::from_key(42, "token:hello", 4096);
    /// assert!((v1.similarity_to(&v2) - 1.0).abs() < 1e-9);
    /// ```
    pub fn from_key(base_seed: u64, key: &str, dim: usize) -> Self {
        IndexVector(crate::seed::deterministic_binary_hd_vector(base_seed, key, dim))
    }

    /// Role-filler encoding via XOR binding.
    ///
    /// Returns `bind(role, filler)` — a bound composite that can be
    /// decoded by unbinding with the same role.
    ///
    /// This is the BSC (fast XOR) equivalent of MAP's circular
    /// convolution role-filler binding.
    #[inline(always)]
    pub fn encode_role_filler(role: &IndexVector, filler: &IndexVector) -> IndexVector {
        role.bind(filler)
    }

    /// Decode a role-filler composite by unbinding with the role.
    ///
    /// Returns `unbind(composite, role)` which should be similar to
    /// the original filler.
    #[inline(always)]
    pub fn decode_role_filler(composite: &IndexVector, role: &IndexVector) -> IndexVector {
        composite.unbind(role)
    }

    /// Bundle multiple `IndexVector`s into one.
    ///
    /// Uses multi-vector majority bundling for improved capacity.
    pub fn bundle_all(vectors: &[IndexVector]) -> IndexVector {
        if vectors.is_empty() {
            return IndexVector::zero(0);
        }
        let bvs: Vec<BinaryHDVector> = vectors.iter().map(|v| v.0.clone()).collect();
        IndexVector(BinaryHDVector::majority_bundle_all(&bvs))
    }
}

impl VsaVector for IndexVector {
    #[inline(always)]
    fn dim(&self) -> usize { self.0.dim() }

    #[inline(always)]
    fn bind(&self, other: &Self) -> Self { self.bind(other) }

    #[inline(always)]
    fn unbind(&self, other: &Self) -> Self { self.unbind(other) }

    #[inline(always)]
    fn bundle(&self, other: &Self) -> Self { self.bundle(other) }

    #[inline(always)]
    fn permute(&self, shift: usize) -> Self { self.permute(shift) }

    #[inline(always)]
    fn cosine_similarity(&self, other: &Self) -> f64 { self.similarity_to(other) }

    fn binarize(&self) -> BinaryHDVector { self.0.clone() }

    fn zero(dim: usize) -> Self { IndexVector::zero(dim) }

    fn random(dim: usize) -> Self { IndexVector::random(dim) }
}

impl serde::Serialize for IndexVector {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for IndexVector {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let inner = BinaryHDVector::deserialize(deserializer)?;
        Ok(IndexVector(inner))
    }
}

impl std::ops::Deref for IndexVector {
    type Target = BinaryHDVector;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for IndexVector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl PartialEq for IndexVector {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for IndexVector {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_vector_bind_unbind_roundtrip() {
        let a = IndexVector::random(1024);
        let b = IndexVector::random(1024);
        let bound = a.bind(&b);
        let roundtrip = bound.unbind(&b);
        let sim = roundtrip.similarity_to(&a);
        assert!((sim - 1.0).abs() < 1e-9, "XOR bind+unbind must recover exactly (sim={})", sim);
    }

    #[test]
    fn test_index_vector_bundle_similarity() {
        let a = IndexVector::random(1024);
        let bundle = a.bundle(&a);
        let sim = a.similarity_to(&bundle);
        assert!(sim > 0.99, "bundle(a,a) must be nearly identical (sim={})", sim);
    }

    #[test]
    fn test_index_vector_permute_cycle() {
        let v = IndexVector::random(1024);
        let rotated = v.permute(1);
        let unrotated = rotated.permute(1023);
        assert_eq!(v, unrotated, "full-cycle rotation must recover original");
    }

    #[test]
    fn test_index_vector_deterministic() {
        let v1 = IndexVector::from_key(42, "test:key", 4096);
        let v2 = IndexVector::from_key(42, "test:key", 4096);
        assert_eq!(v1, v2, "deterministic vectors must be identical");

        let v3 = IndexVector::from_key(42, "test:other", 4096);
        assert_ne!(v1, v3, "different keys must produce different vectors");
    }

    #[test]
    fn test_index_vector_role_filler() {
        let role = IndexVector::from_key(42, "role:subject", 4096);
        let filler = IndexVector::from_key(42, "filler:cat", 4096);
        let composite = IndexVector::encode_role_filler(&role, &filler);
        let decoded = IndexVector::decode_role_filler(&composite, &role);
        let sim = decoded.similarity_to(&filler);
        assert!((sim - 1.0).abs() < 1e-9, "role-filler must recover exactly (sim={})", sim);
    }

    #[test]
    fn test_index_vector_bundle_all() {
        let vecs: Vec<IndexVector> = (0..5).map(|_| IndexVector::random(1024)).collect();
        let bundled = IndexVector::bundle_all(&vecs);
        let sim = vecs[0].similarity_to(&bundled);
        assert!(sim > 0.0, "bundled must have non-zero sim to component (sim={})", sim);
    }

    #[test]
    fn test_index_vector_serde_roundtrip() {
        let v = IndexVector::random(1024);
        let bytes = bincode::serialize(&v).unwrap();
        let loaded: IndexVector = bincode::deserialize(&bytes).unwrap();
        assert_eq!(v, loaded, "bincode roundtrip must preserve IndexVector");
    }

    #[test]
    fn test_vsa_trait_polymorphic() {
        fn test_vec<V: VsaVector>(v: &V) {
            let dim = v.dim();
            let w = V::random(dim);
            let bound = v.bind(&w);
            let sim = bound.cosine_similarity(&w);
            assert!(sim.abs() < 1.0 || (sim - 1.0).abs() < 1e-9, "sim must be in [-1,1]");
            let binarized = v.binarize();
            assert_eq!(binarized.dim(), dim, "binarized must preserve dim");
        }

        let dim = 256;
        test_vec(&HDVector::random(dim));
        test_vec(&BinaryHDVector::random(dim));
        test_vec(&FHRRVector::random(dim));
        test_vec(&IndexVector::random(dim));
    }

    #[test]
    fn test_vsa_trait_bind_roundtrip_all() {
        fn test_roundtrip<V: VsaVector>(dim: usize) {
            let a = V::random(dim);
            let b = V::random(dim);
            let bound = a.bind(&b);
            let rebound = bound.unbind(&b);
            let sim = rebound.cosine_similarity(&a);
            assert!(sim > 0.4,
                "bind+unbind roundtrip should recover (sim={}) for {}, dim={}",
                sim, std::any::type_name::<V>(), dim);
        }

        for &dim in &[64, 128, 256] {
            test_roundtrip::<HDVector>(dim);
            test_roundtrip::<BinaryHDVector>(dim);
            test_roundtrip::<FHRRVector>(dim);
        }
    }

    #[test]
    fn test_vsa_trait_bundle_then_cleanup() {
        fn bundle_cleanup<V: VsaVector>(dim: usize) {
            let a = V::random(dim);
            let b = V::random(dim);
            let bundled = a.bundle(&b);
            let sim_a = bundled.cosine_similarity(&a);
            let sim_b = bundled.cosine_similarity(&b);
            assert!(sim_a > 0.2 && sim_b > 0.2,
                "bundle must retain similarity to both components (a={}, b={})", sim_a, sim_b);
        }

        bundle_cleanup::<HDVector>(128);
        bundle_cleanup::<BinaryHDVector>(128);
        bundle_cleanup::<FHRRVector>(128);
    }
}
