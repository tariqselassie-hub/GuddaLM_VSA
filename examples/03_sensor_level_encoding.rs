//! Continuous Sensor & Scalar Level Encoding
//!
//! Demonstrates encoding continuous numerical readings (e.g. temperature, pressure, telemetry)
//! into hypervectors while preserving smooth metric distance.
//!
//! Run with:
//!   `cargo run --example 03_sensor_level_encoding`

use guddalm_vsa::prelude::*;

fn main() {
    println!("=== GuddaLM VSA: Continuous Scalar Encoding Demo ===\n");

    const DIM: usize = 4096;
    const NUM_LEVELS: usize = 64;

    // Define sensor ranges
    let temp_encoder = ScalarEncoder::new(-20.0, 50.0, DIM, NUM_LEVELS); // Temperature in °C
    let humidity_encoder = ScalarEncoder::new(0.0, 100.0, DIM, NUM_LEVELS); // Relative humidity %

    let mut mem: ItemMemory<HDVector> = ItemMemory::new(DIM);
    let key_temp = mem.get_or_create("sensor:temp");
    let key_hum = mem.get_or_create("sensor:humidity");

    // Sample reading 1: Cold & dry winter day (-5°C, 25% humidity)
    let reading_winter = key_temp.bind(&temp_encoder.encode(-5.0))
        .bundle(&key_hum.bind(&humidity_encoder.encode(25.0)));

    // Sample reading 2: Mild winter day (-3°C, 28% humidity)
    let reading_winter_mild = key_temp.bind(&temp_encoder.encode(-3.0))
        .bundle(&key_hum.bind(&humidity_encoder.encode(28.0)));

    // Sample reading 3: Hot summer day (38°C, 85% humidity)
    let reading_summer = key_temp.bind(&temp_encoder.encode(38.0))
        .bundle(&key_hum.bind(&humidity_encoder.encode(85.0)));

    let sim_similar = reading_winter.cosine_similarity(&reading_winter_mild);
    let sim_different = reading_winter.cosine_similarity(&reading_summer);

    println!("Cosine similarity between (-5°C, 25% RH) and (-3°C, 28% RH): {:.4}", sim_similar);
    println!("Cosine similarity between (-5°C, 25% RH) and ( 38°C, 85% RH): {:.4}", sim_different);

    assert!(sim_similar > sim_different + 0.3, "Close sensor readings must have significantly higher similarity!");

    // Query temperature from winter reading
    let query_temp = reading_winter.unbind(&key_temp);
    let decoded_temp = temp_encoder.decode_approx(&query_temp);
    println!("\nDecoding temperature from winter hypervector:");
    println!("  Actual: -5.0°C | Decoded: {:.1}°C", decoded_temp);
    assert!((decoded_temp - (-5.0)).abs() <= 2.5);

    println!("\nContinuous scalar encoding demo completed successfully!");
}
