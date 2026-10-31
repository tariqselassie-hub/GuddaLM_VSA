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
use std::io::{BufReader, BufWriter, Read, Write};
use crate::hdc::vector::{BinaryHDVector, HDVector};
use crate::vsa::Codebook;

// ── JSON serialization (human-readable, slow) ──

pub fn save_hdvector_json(path: &str, vector: &HDVector) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(vector)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn load_hdvector_json(path: &str) -> Result<HDVector, Box<dyn std::error::Error>> {
    let json = std::fs::read_to_string(path)?;
    let vector: HDVector = serde_json::from_str(&json)?;
    Ok(vector)
}

pub fn save_binary_hdvector_json(
    path: &str,
    vector: &BinaryHDVector,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(vector)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn load_binary_hdvector_json(path: &str) -> Result<BinaryHDVector, Box<dyn std::error::Error>> {
    let json = std::fs::read_to_string(path)?;
    let vector: BinaryHDVector = serde_json::from_str(&json)?;
    Ok(vector)
}

// ── Bincode serialization (binary, fast, compact) ──

pub fn save_hdvector(path: &str, vector: &HDVector) -> Result<(), Box<dyn std::error::Error>> {
    let encoded = bincode::serialize(vector)?;
    std::fs::write(path, encoded)?;
    Ok(())
}

pub fn load_hdvector(path: &str) -> Result<HDVector, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let vector: HDVector = bincode::deserialize(&bytes)?;
    Ok(vector)
}

pub fn save_binary_hdvector(path: &str, vector: &BinaryHDVector) -> Result<(), Box<dyn std::error::Error>> {
    let encoded = bincode::serialize(vector)?;
    std::fs::write(path, encoded)?;
    Ok(())
}

pub fn load_binary_hdvector(path: &str) -> Result<BinaryHDVector, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let vector: BinaryHDVector = bincode::deserialize(&bytes)?;
    Ok(vector)
}

// ── Batch I/O (bincode) ──

pub fn save_vectors(path: &str, vectors: &[HDVector]) -> Result<(), Box<dyn std::error::Error>> {
    let encoded = bincode::serialize(vectors)?;
    std::fs::write(path, encoded)?;
    Ok(())
}

pub fn load_vectors(path: &str) -> Result<Vec<HDVector>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let vectors: Vec<HDVector> = bincode::deserialize(&bytes)?;
    Ok(vectors)
}

pub fn save_binary_vectors(path: &str, vectors: &[BinaryHDVector]) -> Result<(), Box<dyn std::error::Error>> {
    let encoded = bincode::serialize(vectors)?;
    std::fs::write(path, encoded)?;
    Ok(())
}

pub fn load_binary_vectors(path: &str) -> Result<Vec<BinaryHDVector>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let vectors: Vec<BinaryHDVector> = bincode::deserialize(&bytes)?;
    Ok(vectors)
}

// ── Codebook I/O (bincode) ──

pub fn save_codebook(path: &str, codebook: &Codebook) -> Result<(), Box<dyn std::error::Error>> {
    let encoded = bincode::serialize(codebook)?;
    std::fs::write(path, encoded)?;
    Ok(())
}

pub fn load_codebook(path: &str) -> Result<Codebook, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let codebook: Codebook = bincode::deserialize(&bytes)?;
    Ok(codebook)
}

// ── Stream I/O (bincode, memory-efficient for large batches) ──

pub fn write_vectors_stream<W: Write>(
    writer: W,
    vectors: &[HDVector],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut buf = BufWriter::new(writer);
    // Write count first
    let count = vectors.len() as u64;
    let count_bytes = bincode::serialize(&count)?;
    buf.write_all(&count_bytes)?;
    for v in vectors {
        let encoded = bincode::serialize(v)?;
        let len = encoded.len() as u64;
        let len_bytes = bincode::serialize(&len)?;
        buf.write_all(&len_bytes)?;
        buf.write_all(&encoded)?;
    }
    buf.flush()?;
    Ok(())
}

pub fn read_vectors_stream<R: Read>(
    reader: R,
) -> Result<Vec<HDVector>, Box<dyn std::error::Error>> {
    let mut buf = BufReader::new(reader);
    let count: u64 = bincode::deserialize_from(&mut buf)?;
    let mut vectors = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let len: u64 = bincode::deserialize_from(&mut buf)?;
        let mut bytes = vec![0u8; len as usize];
        buf.read_exact(&mut bytes)?;
        let v: HDVector = bincode::deserialize(&bytes)?;
        vectors.push(v);
    }
    Ok(vectors)
}

pub fn write_binary_vectors_stream<W: Write>(
    writer: W,
    vectors: &[BinaryHDVector],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut buf = BufWriter::new(writer);
    let count = vectors.len() as u64;
    let count_bytes = bincode::serialize(&count)?;
    buf.write_all(&count_bytes)?;
    for v in vectors {
        let encoded = bincode::serialize(v)?;
        let len = encoded.len() as u64;
        let len_bytes = bincode::serialize(&len)?;
        buf.write_all(&len_bytes)?;
        buf.write_all(&encoded)?;
    }
    buf.flush()?;
    Ok(())
}

pub fn read_binary_vectors_stream<R: Read>(
    reader: R,
) -> Result<Vec<BinaryHDVector>, Box<dyn std::error::Error>> {
    let mut buf = BufReader::new(reader);
    let count: u64 = bincode::deserialize_from(&mut buf)?;
    let mut vectors = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let len: u64 = bincode::deserialize_from(&mut buf)?;
        let mut bytes = vec![0u8; len as usize];
        buf.read_exact(&mut bytes)?;
        let v: BinaryHDVector = bincode::deserialize(&bytes)?;
        vectors.push(v);
    }
    Ok(vectors)
}

/// Legacy alias: use bincode-based `save_hdvector` by default.
/// Kept for backward compatibility with existing callers.
pub fn save_hdvector_legacy(path: &str, vector: &HDVector) -> Result<(), Box<dyn std::error::Error>> {
    save_hdvector(path, vector)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::vector::HDVector;
    use crate::vsa::Codebook;

    #[test]
    fn test_save_load_hdvector_bincode_roundtrip() {
        let v = HDVector::random(1000);
        let bytes = bincode::serialize(&v).unwrap();
        let loaded: HDVector = bincode::deserialize(&bytes).unwrap();
        assert_eq!(v.dim(), loaded.dim());
        let sim = v.cosine_similarity(&loaded);
        assert!((sim - 1.0).abs() < 0.001, "bincode roundtrip must preserve vector (sim={})", sim);
    }

    #[test]
    fn test_save_load_binary_hdvector_bincode_roundtrip() {
        let v = BinaryHDVector::random(1024);
        let bytes = bincode::serialize(&v).unwrap();
        let loaded: BinaryHDVector = bincode::deserialize(&bytes).unwrap();
        assert_eq!(v.dim(), loaded.dim());
        let sim = v.hamming_similarity(&loaded);
        assert!((sim - 1.0).abs() < 0.001, "bincode roundtrip must preserve vector (sim={})", sim);
    }

    #[test]
    fn test_save_load_vectors_roundtrip() {
        let vecs: Vec<HDVector> = (0..10).map(|_| HDVector::random(512)).collect();
        let bytes = bincode::serialize(&vecs).unwrap();
        let loaded: Vec<HDVector> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(vecs.len(), loaded.len());
        for (a, b) in vecs.iter().zip(loaded.iter()) {
            let sim = a.cosine_similarity(b);
            assert!((sim - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_save_load_codebook_roundtrip() {
        let cb = Codebook::new(50, 512);
        let bytes = bincode::serialize(&cb).unwrap();
        let loaded: Codebook = bincode::deserialize(&bytes).unwrap();
        assert_eq!(cb.vocab_size, loaded.vocab_size);
        assert_eq!(cb.dim, loaded.dim);
        assert_eq!(cb.weights.len(), loaded.weights.len());
        assert_eq!(cb.packed.len(), loaded.packed.len());
    }

    #[test]
    fn test_hdvector_file_roundtrip() {
        let v = HDVector::random(512);
        let v2 = v.clone();
        let path = "target/test_guddalm_vsa_vector.bin";
        let _ = std::fs::create_dir_all("target");
        save_hdvector(path, &v).unwrap();
        let loaded = load_hdvector(path).unwrap();
        let sim = v2.cosine_similarity(&loaded);
        assert!((sim - 1.0).abs() < 0.001, "file save/load must preserve vector");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_codebook_file_roundtrip() {
        let cb = Codebook::new(50, 512);
        let path = "target/test_guddalm_vsa_codebook.bin";
        let _ = std::fs::create_dir_all("target");
        save_codebook(path, &cb).unwrap();
        let loaded = load_codebook(path).unwrap();
        assert_eq!(cb.vocab_size, loaded.vocab_size);
        assert_eq!(cb.dim, loaded.dim);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_stream_write_read_vectors() {
        let vecs: Vec<HDVector> = (0..5).map(|_| HDVector::random(256)).collect();
        let mut buf = Vec::new();
        write_vectors_stream(&mut buf, &vecs).unwrap();
        let loaded = read_vectors_stream(&buf[..]).unwrap();
        assert_eq!(vecs.len(), loaded.len());
        for (a, b) in vecs.iter().zip(loaded.iter()) {
            let sim = a.cosine_similarity(b);
            assert!((sim - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_json_backward_compatibility() {
        let v = HDVector::random(100);
        let json = serde_json::to_string(&v).unwrap();
        let loaded: HDVector = serde_json::from_str(&json).unwrap();
        let sim = v.cosine_similarity(&loaded);
        assert!((sim - 1.0).abs() < 0.001, "JSON roundtrip must preserve vector");
    }
}
