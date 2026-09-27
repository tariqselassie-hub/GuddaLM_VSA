//! Quickstart Key-Value Record Encoding with VSA
//!
//! Demonstrates the foundational VSA workflow:
//! 1. Initialize ItemMemory for symbolic tokens
//! 2. Bind keys to values (e.g. `field * value`)
//! 3. Bundle fields into a unified record representation
//! 4. Unbind a key to query the value and clean up with nearest-neighbor search
//!
//! Run with:
//!   `cargo run --example 01_quickstart_key_value`

use guddalm_vsa::prelude::*;

fn main() {
    println!("=== GuddaLM VSA: Quickstart Key-Value Record Demo ===\n");

    const DIM: usize = 10_000;
    let mut mem: ItemMemory<HDVector> = ItemMemory::new(DIM);

    // 1. Define fields and values
    let record_data = [
        ("first_name", "Ada"),
        ("last_name", "Lovelace"),
        ("field", "Computing"),
        ("era", "Victorian"),
        ("nationality", "British"),
    ];

    println!("Encoding record for Ada Lovelace into a single {DIM}-dimensional vector...");
    let mut record = HDVector::zeros(DIM);

    for &(k_str, v_str) in &record_data {
        let k = mem.get_or_create(k_str);
        let v = mem.get_or_create(v_str);
        // Bind key to value (role-filler binding)
        let bound = k.bind(&v);
        // Bundle into composite record (superposition)
        record = record.bundle(&bound);
    }

    println!("Record successfully encoded into {} hypervector.\n", record.dim());

    // 2. Querying fields by unbinding
    println!("Querying record with keys:");
    for &query_key in &["field", "nationality", "era", "first_name"] {
        let key_vec = mem.get_or_create(query_key);

        // Unbind: value_estimate = record ⊘ key
        let value_estimate = record.unbind(&key_vec);

        // Auto-associative cleanup: find the closest symbol in item memory
        if let Some((recovered_sym, sim)) = mem.cleanup(&value_estimate) {
            println!("  Key: {:<12} -> Recovered Value: {:<12} (similarity: {:.4})",
                query_key, recovered_sym, sim);
        }
    }

    println!("\nQuickstart demo completed successfully!");
}
