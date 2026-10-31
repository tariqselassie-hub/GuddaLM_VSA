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
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use guddalm_vsa::prelude::*;
use guddalm_vsa::search::search_map;

fn bench_bind(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_bind");
    for dim in [1024usize, 4096, 8192, 10_000] {
        let a = HDVector::random(dim);
        let b = HDVector::random(dim);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| black_box(a.bind(&b)))
        });
    }
    group.finish();
}

fn bench_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_cosine");
    for dim in [4096usize, 8192, 10_000] {
        let a = HDVector::random(dim);
        let b = HDVector::random(dim);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| black_box(a.cosine_similarity(&b)))
        });
    }
    group.finish();
}

fn bench_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("bsc");
    for dim in [4096usize, 10_000] {
        let a = BinaryHDVector::random(dim);
        let b = BinaryHDVector::random(dim);
        group.bench_with_input(BenchmarkId::new("xor_bind", dim), &dim, |bench, _| {
            bench.iter(|| black_box(a.xor_bind(&b)))
        });
        group.bench_with_input(BenchmarkId::new("hamming", dim), &dim, |bench, _| {
            bench.iter(|| black_box(a.hamming_similarity(&b)))
        });
    }
    group.finish();
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_map_top5");
    group.sample_size(20);
    let dim = 4096;
    let query = HDVector::random(dim);
    for n in [1_000usize, 20_000] {
        let candidates: Vec<HDVector> = (0..n).map(|_| HDVector::random(dim)).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| black_box(search_map(&query, &candidates, 5)))
        });
    }
    group.finish();
}

fn bench_item_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("item_memory_cleanup");
    group.sample_size(20);
    for dim in [4096usize, 8192] {
        let mut mem: ItemMemory<HDVector> = ItemMemory::new(dim);
        for i in 0..1_000 {
            mem.get_or_create(&format!("sym{i}"));
        }
        let q = mem.get_or_create("sym500");
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| black_box(mem.cleanup(&q)))
        });
    }
    group.finish();
}

fn bench_bundle_100(c: &mut Criterion) {
    let mut group = c.benchmark_group("bundle_100");
    let dim = 4096;
    let map: Vec<HDVector> = (0..100).map(|_| HDVector::random(dim)).collect();
    let bsc: Vec<BinaryHDVector> = (0..100).map(|_| BinaryHDVector::random(dim)).collect();
    group.bench_function("map_pairwise", |b| b.iter(|| black_box(bundle_sequence(&map))));
    group.bench_function("map_accumulator", |b| {
        b.iter(|| {
            let mut acc = BundleAccumulator::new(dim);
            acc.extend(&map);
            black_box(acc.into_vector())
        })
    });
    group.bench_function("bsc_pairwise", |b| b.iter(|| black_box(bundle_sequence(&bsc))));
    group.bench_function("bsc_majority_bundle_all", |b| {
        b.iter(|| black_box(BinaryHDVector::majority_bundle_all(&bsc)))
    });
    group.bench_function("bsc_accumulator", |b| {
        b.iter(|| {
            let mut acc = BinaryBundleAccumulator::new(dim);
            acc.extend(&bsc);
            black_box(acc.binarize())
        })
    });
    group.finish();
}

criterion_group!(benches, bench_bundle_100, bench_bind, bench_similarity, bench_binary, bench_search, bench_item_memory);
criterion_main!(benches);
