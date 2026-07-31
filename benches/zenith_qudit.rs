use guddalm_vsa::{
  hdc::vsa_trait::VsaVector,
  IndexVector,
};
use guddalm_vsa::zenith::{
  HeptalBinaryMapper, HeptalBipolarMapper, HeptalFhrrMapper,
  QuditVsaAdapter, ZeroCopyQuditView,
};

fn bench_scaffold() {
  let view = ZeroCopyQuditView::heptal(7);
  let bsc: QuditVsaAdapter<HeptalBinaryMapper> = QuditVsaAdapter::bsc();
  let _ = bsc.encode(&view);
}
