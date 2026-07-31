use crate::{
  dimensions::{BSC_DEFAULT_DIM, FHRR_DEFAULT_DIM, MAP_DEFAULT_DIM},
  hdc::{
    fhrr::FHRRVector,
    vector::{BinaryHDVector, HDVector},
  },
  IndexVector,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum QuditViewError {
  DimMismatch,
  Overflow,
}

impl std::fmt::Display for QuditViewError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self { Self::DimMismatch => write!(f, "dimension mismatch"), Self::Overflow => write!(f, "index overflow") }
  }
}

#[derive(Clone, Debug)]
pub struct ZeroCopyQuditView<'a> {
  pub num_qudits: usize,
  pub strata: &'a [i8],
}

impl<'a> ZeroCopyQuditView<'a> {
  pub fn new(strata: &'a [i8]) -> Self { Self { num_qudits: strata.len(), strata } }
  pub fn heptal(num_qudits: usize) -> Self {
    let buf: Vec<i8> = vec![-3, -2, -1, 0, 1, 2, 3];
    let repeated = buf.repeat(num_qudits);
    let leaked: &'a [i8] = Box::leak(repeated.into_boxed_slice());
    Self { num_qudits, strata: leaked }
  }
  pub fn strata(&self) -> &'a [i8] { self.strata }
}

#[derive(Clone, Debug)]
pub struct HeptalBinaryMapper;

impl HeptalBinaryMapper {
  pub fn encode_words(&self, view: &ZeroCopyQuditView<'_>, dim: usize) -> Vec<u64> {
    let n_words = (dim + 63) / 64;
    let mut words = vec![0u64; n_words];
    for (slot, &val) in view.strata.iter().enumerate() {
      let bit = if val > 0 { 1u64 } else { 0u64 };
      let bitpos = (slot % 7) as u32;
      words[(slot / 7) % n_words] |= bit << bitpos;
    }
    words
  }

  pub fn decode_signed(&self, words: &[u64], num_qudits: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(num_qudits);
    for slot in 0..num_qudits {
      let wi = (slot / 7) % words.len();
      let bitpos = (slot % 7) as u32;
      let bit = (words[wi] >> bitpos) & 1;
      out.push(if bit == 1 { 1 } else { -1 });
    }
    out
  }
}

#[derive(Clone, Debug)]
pub struct HeptalBipolarMapper;

impl HeptalBipolarMapper {
  #[inline(always)]
  pub fn strata_to_unit(&self, val: i8) -> f64 {
    match val {
      -3 => -1.0,
      -2 => -0.666_666_666_666_666_6,
      -1 => -0.333_333_333_333_333_3,
       0 =>  0.0,
       1 =>  0.333_333_333_333_333_3,
       2 =>  0.666_666_666_666_666_6,
       3 =>  1.0,
       _ => 0.0,
    }
  }

  #[inline(always)]
  pub fn unit_to_stratum(&self, x: f64) -> i8 {
    let x = x.clamp(-1.0, 1.0);
    match (x * 3.0).round() as i8 {
      v if v > 3 => 3,
      v if v < -3 => -3,
      v => v,
    }
  }

  pub fn view_to_vec(&self, view: &ZeroCopyQuditView<'_>, dim: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(dim.min(view.strata.len()));
    for &s in view.strata.iter().take(dim) {
      out.push(self.strata_to_unit(s));
    }
    out.resize(dim, 0.0);
    out
  }

  pub fn vec_to_view(&self, data: &[f64], num_qudits: usize) -> Vec<i8> {
    data.iter().take(num_qudits).map(|&x| self.unit_to_stratum(x)).collect()
  }
}

#[derive(Clone, Debug)]
pub struct HeptalFhrrMapper;

impl HeptalFhrrMapper {
  #[inline(always)]
  pub fn stratum_phase(&self, val: i8) -> f64 {
    let frac = (val as f64) / 3.0;
    frac * std::f64::consts::PI
  }

  #[inline(always)]
  pub fn phase_to_stratum(&self, x: f64) -> i8 {
    let mut ang = x;
    if ang > std::f64::consts::PI { ang -= 2.0 * std::f64::consts::PI; }
    if ang < -std::f64::consts::PI { ang += 2.0 * std::f64::consts::PI; }
    (ang * 3.0 / std::f64::consts::PI).round().clamp(-3.0, 3.0) as i8
  }
}

#[derive(Clone, Debug)]
pub struct QuditVsaAdapter<M> {
  pub mapper: M,
  pub dim: usize,
}

impl<M> QuditVsaAdapter<M> {
  pub fn new(mapper: M, dim: usize) -> Self { Self { mapper, dim } }
}

impl QuditVsaAdapter<HeptalBinaryMapper> {
  pub fn bsc() -> Self { Self::new(HeptalBinaryMapper, BSC_DEFAULT_DIM) }

  pub fn encode(&self, view: &ZeroCopyQuditView<'_>) -> IndexVector {
    let words = self.mapper.encode_words(view, self.dim);
    IndexVector(BinaryHDVector { dim: self.dim, words })
  }

  pub fn decode(&self, vec: &IndexVector) -> Vec<i8> {
    self.mapper.decode_signed(vec.words(), vec.dim())
  }
}

impl QuditVsaAdapter<HeptalBipolarMapper> {
  pub fn map_rep() -> Self { Self::new(HeptalBipolarMapper, MAP_DEFAULT_DIM) }

  pub fn encode(&self, view: &ZeroCopyQuditView<'_>) -> HDVector {
    let data = self.mapper.view_to_vec(view, self.dim);
    let mut out = HDVector::zeros(self.dim);
    let dst = out.data_mut();
    for (i, v) in data.iter().enumerate().take(self.dim) {
      dst[i] = *v;
    }
    out
  }

  pub fn decode(&self, vec: &HDVector) -> Vec<i8> {
    self.mapper.vec_to_view(vec.data(), self.dim)
  }
}

impl QuditVsaAdapter<HeptalFhrrMapper> {
  pub fn fhrr() -> Self { Self::new(HeptalFhrrMapper, FHRR_DEFAULT_DIM) }

  pub fn encode(&self, view: &ZeroCopyQuditView<'_>) -> FHRRVector {
    let mut phases = Vec::with_capacity(self.dim);
    for (&s, _) in view.strata.iter().take(self.dim).zip(0..self.dim) {
      phases.push(self.mapper.stratum_phase(s));
    }
    phases.resize(self.dim, 0.0);
    FHRRVector::from_phases(&phases)
  }

  pub fn decode(&self, vec: &FHRRVector) -> Vec<i8> {
    vec.phases().iter().take(self.dim).map(|z| self.mapper.phase_to_stratum(*z)).collect()
  }
}

pub trait QuditMemoryProbe {
  fn probe_chunked(&self, target_idx: usize, chunk_size: usize, keys: &[(IndexVector, HDVector)]) -> (f64, f64, usize);
}

#[derive(Clone, Debug, Default)]
pub struct MemoryProbeResult {
  pub chunked_fidelity: f64,
  pub repaired_fidelity: f64,
  pub chunk_index: usize,
  pub chunk_count: usize,
}

#[derive(Clone, Debug, Default)]
pub struct ChunkMemoryStats {
  pub median: f64,
  pub p95: f64,
  pub p05: f64,
  pub count: usize,
}

#[derive(Clone)]
pub struct VsaMemoryHarness<M, A> {
  pub mapper: M,
  pub adapter: A,
  pub dim: usize,
  pub chunk_size: usize,
}

impl VsaMemoryHarness<HeptalBinaryMapper, QuditVsaAdapter<HeptalBinaryMapper>> {
  pub fn bsc(dim: usize, chunk_size: usize) -> Self {
    Self { mapper: HeptalBinaryMapper, adapter: QuditVsaAdapter::bsc(), dim, chunk_size }
  }
}

impl VsaMemoryHarness<HeptalBipolarMapper, QuditVsaAdapter<HeptalBipolarMapper>> {
  pub fn map_rep(dim: usize, chunk_size: usize) -> Self {
    Self { mapper: HeptalBipolarMapper, adapter: QuditVsaAdapter::map_rep(), dim, chunk_size }
  }
}

impl VsaMemoryHarness<HeptalFhrrMapper, QuditVsaAdapter<HeptalFhrrMapper>> {
  pub fn fhrr(dim: usize, chunk_size: usize) -> Self {
    Self { mapper: HeptalFhrrMapper, adapter: QuditVsaAdapter::fhrr(), dim, chunk_size }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn bsc_roundtrip() {
    let view = ZeroCopyQuditView::new(&[-3, -2, -1, 0, 1, 2, 3]);
    let adapter = QuditVsaAdapter::bsc();
    let iv = adapter.encode(&view);
    let back = HeptalBinaryMapper.decode_signed(iv.words(), 7);
    assert_eq!(back, vec![-1, -1, -1, -1, 1, 1, 1]);
  }

  #[test]
  fn map_spread() {
    let m = HeptalBipolarMapper;
    let data = m.view_to_vec(&ZeroCopyQuditView::new(&[-3, 0, 3]), 3);
    assert!((data[0] + 1.0).abs() < 1e-9, "{}", data[0]);
    assert!(data[1].abs() < 1e-9, "{}", data[1]);
    assert!((data[2] - 1.0).abs() < 1e-9, "{}", data[2]);
  }

  #[test]
  fn fhrr_roundtrip() {
    let view = ZeroCopyQuditView::new(&[-3, -2, -1, 0, 1, 2, 3]);
    let adapter = QuditVsaAdapter::fhrr();
    let vec = adapter.encode(&view);
    let back = HeptalFhrrMapper.phase_to_stratum(*vec.phases().first().unwrap_or(&0.0));
    assert!(back <= 3 && back >= -3, "phase decoded to {}", back);
  }
}
