//! Analogical Reasoning with Vector Symbolic Architectures (Kanerva & Plate)
//!
//! Solves the classic VSA proportional analogy:
//! "Dollar of USA is to Peso of Mexico as Washington is to ... ?"
//!
//! Architecture:
//! - USA = (Country * USA) + (Capital * Washington) + (Currency * Dollar)
//! - Mexico = (Country * Mexico) + (Capital * MexicoCity) + (Currency * Peso)
//!
//! We compute the mapping between USA and Mexico, and apply it to Washington
//! to recover MexicoCity purely through hyperdimensional vector algebra!
//!
//! Run with:
//!   `cargo run --example 02_analogical_reasoning`

use guddalm_vsa::prelude::*;

fn main() {
    println!("=== GuddaLM VSA: Analogical Reasoning Demo ===\n");

    const DIM: usize = 10_000;
    let mut mem: ItemMemory<HDVector> = ItemMemory::new(DIM);

    // Roles
    let role_capital = mem.get_or_create("role:capital");
    let role_currency = mem.get_or_create("role:currency");

    // Entities
    let washington = mem.get_or_create("entity:Washington");
    let dollar = mem.get_or_create("entity:Dollar");
    let mexico_city = mem.get_or_create("entity:MexicoCity");
    let peso = mem.get_or_create("entity:Peso");

    // Distractor entities in memory
    let _london = mem.get_or_create("entity:London");
    let _pound = mem.get_or_create("entity:Pound");
    let _tokyo = mem.get_or_create("entity:Tokyo");
    let _yen = mem.get_or_create("entity:Yen");

    // Build holistic country representations:
    // USA = (capital * Washington) + (currency * Dollar)
    let usa = role_capital.bind(&washington).bundle(&role_currency.bind(&dollar));

    // Mexico = (capital * MexicoCity) + (currency * Peso)
    let mexico = role_capital.bind(&mexico_city).bundle(&role_currency.bind(&peso));

    println!("Constructed holistic USA and Mexico country representations.");

    // Analogy question:
    // "What is the Dollar of Mexico?"
    // Answer: unbind currency role from Mexico -> recovers Peso
    let mexico_currency = mexico.unbind(&role_currency);
    let (match_curr, sim_curr) = match mem.cleanup(&mexico_currency) {
        Some(res) => res,
        None => panic!("Failed to clean up Mexico currency"),
    };
    println!("1. Currency of Mexico: {} (sim: {:.4})", match_curr, sim_curr);
    assert_eq!(match_curr, "entity:Peso");

    // Proportional Analogy Question:
    // "Dollar is to USA as Peso is to [TargetCountry]?"
    // Translation operator T = Dollar ⊘ USA (or Mexico ⊗ inv(USA))
    // Direct substitution: Target = (USA ⊘ Dollar) ⊗ Peso ...
    // Relational transformation operator: M = Mexico ⊘ USA (correlate)
    // Applying transformation to Washington: Query = M ⊗ Washington
    let mapping = mexico.unbind(&usa);
    let analogy_query = mapping.bind(&washington);
    let top_matches = mem.cleanup_top_k(&analogy_query, 3);

    println!("\n2. Analogy query: Washington of USA is to Mexico as [?] is to Mexico:");
    for (i, (sym, sim)) in top_matches.iter().enumerate() {
        println!("   Top #{}: {:<20} (sim: {:.4})", i + 1, sym, sim);
    }
    assert_eq!(top_matches[0].0, "entity:MexicoCity");

    println!("\nAnalogical reasoning demo completed successfully!");
}
