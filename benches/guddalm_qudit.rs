use guddalm_vsa::guddalm::{
  HeptalBinaryMapper, QuditVsaAdapter, ZeroCopyQuditView,
};

#[allow(dead_code)]
fn bench_scaffold() {
  let view = ZeroCopyQuditView::heptal(7);
  let bsc: QuditVsaAdapter<HeptalBinaryMapper> = QuditVsaAdapter::bsc();
  let _ = bsc.encode(&view);
}
