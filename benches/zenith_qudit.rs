use guddalm_vsa::qudit::{
  HeptalBinaryMapper, QuditVsaAdapter, OwnedQuditView,
};

#[allow(dead_code)]
fn bench_scaffold() {
  let owned = OwnedQuditView::heptal(7);
  let view  = owned.as_view();
  let bsc: QuditVsaAdapter<HeptalBinaryMapper> = QuditVsaAdapter::bsc();
  let _ = bsc.encode(&view);
}
