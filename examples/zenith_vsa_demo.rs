//! Joint VSA ↔ Heptal Bridge Demo — validation gate
//!
//! Exercises:
//! - encode/decode roundtrip for BSC/MAP/FHRR
//! - bind/unbind role-filler recovery
//! - chunk bundling + cleanup repair
//!
//! Run:
//!   `cargo run --example zenith_vsa_demo -p guddalm_vsa`

use guddalm_vsa::{
  hdc::cleanup::BinaryCleanupMemory,
  IndexVector,
};
use guddalm_vsa::zenith::{
  QuditVsaAdapter, ZeroCopyQuditView,
};

fn encode_decode_roundtrip() {
  let view = ZeroCopyQuditView::new(&[-3, -2, -1, 0, 1, 2, 3]);

  // BSC
  {
    let adapter = QuditVsaAdapter::bsc();
    let iv = adapter.encode(&view);
    let back = adapter.decode(&iv);
    let exact = view.strata.iter().zip(back.iter()).filter(|(a,b)| a==b).count();
    println!("bsc roundtrip: {}/{} exact", exact, view.strata.len());
  }

  // MAP
  {
    let adapter = QuditVsaAdapter::map_rep();
    let v = adapter.encode(&view);
    let back = adapter.decode(&v);
    let exact = view.strata.iter().zip(back.iter()).filter(|(a,b)| a==b).count();
    println!("map roundtrip: {}/{} exact", exact, view.strata.len());
  }

  // FHRR
  {
    let adapter = QuditVsaAdapter::fhrr();
    let v = adapter.encode(&view);
    let back = adapter.decode(&v);
    let exact = view.strata.iter().zip(back.iter()).filter(|(a,b)| a==b).count();
    println!("fhrr roundtrip: {}/{} exact", exact, view.strata.len());
  }
}

fn bind_unbind_role_filler() {
  let adapter = QuditVsaAdapter::bsc();

  // Encode a "role" and "filler" from two views
  let role_view = ZeroCopyQuditView::new(&[1, 2, 3, 0, -1, -2, -3]);
  let filler_view = ZeroCopyQuditView::new(&[-3, -2, -1, 0, 1, 2, 3]);

  let role = adapter.encode(&role_view);
  let filler = adapter.encode(&filler_view);

  let bound = role.bind(&filler);
  let recovered = bound.unbind(&role);
  let decoded = adapter.decode(&recovered);

  let exact = filler_view.strata.iter().zip(decoded.iter()).filter(|(a,b)| a==b).count();
  println!("bind/unbind recovery: {}/{} exact", exact, filler_view.strata.len());
}

fn cleanup_repair_probe() {
  let adapter = QuditVsaAdapter::bsc();

  // Build 10 key-value pairs from strata-derived keys/values
  let mut keys: Vec<(IndexVector, IndexVector)> = Vec::new();
  for idx in 0..10 {
    let k_arr: [i8; 7] = [idx as i8 % 7 - 3; 7];
    let v_arr: [i8; 7] = [(idx as i8 + 3) % 7 - 3; 7];
    let k_view = ZeroCopyQuditView::new(&k_arr);
    let v_view = ZeroCopyQuditView::new(&v_arr);
    let k = adapter.encode(&k_view);
    let v = adapter.encode(&v_view);
    keys.push((k, v));
  }

  // Bundle first chunk (key 0 bound to value 0), then unbind with key 0
  let bound = keys[0].0.bind(&keys[0].1);
  let cand = bound.unbind(&keys[0].0);

  // Cleanup repair against all value vectors
  let codebook = keys.iter().map(|(_, v)| v.clone().into_inner()).collect::<Vec<_>>();
  let mem = BinaryCleanupMemory::new(codebook);
  let (_, _, proto) = mem.cleanup(&cand.clone().into_inner());
  let repaired = IndexVector::new(proto);

  let sim_original = cand.similarity_to(&keys[0].1);
  let sim_repaired = repaired.similarity_to(&keys[0].1);
  println!("cleanup repair: raw_sim={:.6} repaired_sim={:.6}", sim_original, sim_repaired);
}

fn main() {
  encode_decode_roundtrip();
  bind_unbind_role_filler();
  cleanup_repair_probe();
}
