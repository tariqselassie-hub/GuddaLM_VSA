# guddalm_vsa

High-performance, pure-Rust engine for **Vector Symbolic Architectures (VSA)** and **Hyperdimensional Computing (HDC)**.

`guddalm_vsa` provides the foundational algebraic primitives, vector spaces, and associative memory structures needed to build symbolic AI, cognitive architectures, sensor fusion pipelines, and analogical reasoning systems from scratch.

---

## ⚡ Quickstart

Add `guddalm_vsa` to your `Cargo.toml`:

```toml
[dependencies]
guddalm_vsa = "0.1.0"
```

Encode and query a key-value record in 10 lines:

```rust
use guddalm_vsa::prelude::*;

fn main() {
    // 1. Initialize ItemMemory for symbolic tokens (10,000 dimensions)
    let mut mem: ItemMemory<HDVector> = ItemMemory::new(10_000);

    let country = mem.get_or_create("country");
    let usa = mem.get_or_create("USA");
    let capital = mem.get_or_create("capital");
    let dc = mem.get_or_create("Washington_DC");

    // 2. Bind and bundle into a single composite hypervector: (country ⊗ USA) ⊕ (capital ⊗ Washington_DC)
    let record = country.bind(&usa).bundle(&capital.bind(&dc));

    // 3. Query the record: unbind the key to retrieve the value
    let query = record.unbind(&capital);
    let (match_symbol, similarity) = mem.cleanup(&query).unwrap();

    println!("Capital: {} (similarity: {:.4})", match_symbol, similarity);
    assert_eq!(match_symbol, "Washington_DC");
}
```

---

## 🏛️ Supported VSA Architectures

| Architecture | Type | Vector Space | Binding ($\otimes$) | Bundling ($\oplus$) | Similarity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MAP** | `HDVector` | Real/Bipolar $\{-1, +1\}^D$ | Circular Convolution (`rustfft`) | Element-wise Addition | Cosine Similarity |
| **BSC** | `BinaryHDVector` | Bit-packed $\{0, 1\}^D$ | Bitwise XOR | Bitwise Majority Vote | Hamming / XNOR Popcount |
| **FHRR** | `FHRRVector` | Complex Phase $[-\pi, \pi)^D$ | Phase Addition | Complex Unit Normalization | Mean Phase Cosine |
| **GHRR** | `GHRRVector` | Matrix Blocks $U(2)^K$ | Non-Commutative Product | Unitary Projection | Trace Inner Product |

All vector representations implement the unified [`VsaVector`](crate::hdc::vsa_trait::VsaVector) trait, allowing generic algorithms to operate across any architecture.

---

## 📦 Core Primitives

- **`ItemMemory<V>`**: Symbol table associating string tokens with deterministic, quasi-orthogonal hypervectors. Includes nearest-neighbor auto-associative cleanup and top-$k$ retrieval.
- **`ScalarEncoder`**: Projects continuous real numbers ($x \in [\text{min}, \text{max}]$) to hypervectors preserving metric distance ($\text{sim}(x_1, x_2) \propto 1 - |x_1 - x_2|$).
- **`ContinuousSpaceEncoder`**: Random Fourier Features (RFF) for continuous multi-dimensional coordinate spaces and RBF kernel approximation.
- **`CleanupMemory`**: Fast nearest-neighbor associative recall with bit-packed XNOR acceleration.
- **`Resonator`**: Iterative factorization networks for decomposing bound composite vectors into constituent factors (Frady et al.).
- **Structural Encoders**: Turnkey functions for sequences (`encode_positional_sequence`, `encode_ngram_sequence`), sets (`encode_set`, `decode_set`), and graphs (`encode_graph_edges`).

---

## 🚀 Starter Examples

Explore the runnable examples in the [`examples/`](examples) directory:

```bash
# 1. Basic key-value record encoding and unbinding
cargo run --example 01_quickstart_key_value

# 2. Solving Kanerva's proportional analogy (Dollar:USA :: Peso:Mexico)
cargo run --example 02_analogical_reasoning

# 3. Continuous sensor telemetry and level hypervector encoding
cargo run --example 03_sensor_level_encoding

# 4. Text sequence and N-gram representation
cargo run --example 04_sequence_and_text_ngrams
```

---

## 🧪 Build & Test

```bash
# Check code
cargo check --all-targets

# Run the test suite (180+ tests)
cargo test

# Build documentation
cargo doc --no-deps --open
```

---

## 📜 Compliance & License

- **License**: AGPL-3.0-only (`LICENSE`)
- **Attribution**: Terrence A. Jones Sr.
- **Contact**: Zoddjr@gmail.com
- **Project**: Zenith Research Division Detroit Project
