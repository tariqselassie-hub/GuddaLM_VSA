# Benchmarks

Numbers come from `cargo bench --bench core_ops` (criterion, release
profile) on a 4-core Linux container. Your hardware will give different
absolute times, but the ratios should hold.

To reproduce a before/after comparison:

```bash
git checkout <old-commit> -- src && cargo bench --bench core_ops -- --save-baseline before
git checkout HEAD -- src && cargo bench --bench core_ops -- --baseline-lenient before
```

## Core operations (2026-10-31)

"Before" is `c105aec` (v0.1.1). "After" includes the FFT, search and
accumulator changes. Times are criterion medians.

| Benchmark | Before | After | Change |
|---|---|---|---|
| `map_bind/1024` | 11.7 µs | 10.2 µs | −28% |
| `map_bind/4096` | 67.7 µs | 53.2 µs | −31% |
| `map_bind/8192` | 130.9 µs | 118.2 µs | −10% |
| `map_bind/10000` | 84.2 ms | 195 µs | **−99.8% (≈430×)** |
| `map_cosine/4096` | 8.47 µs | 3.29 µs | −57% |
| `map_cosine/8192` | 16.2 µs | 6.46 µs | −59% |
| `map_cosine/10000` | 18.8 µs | 8.23 µs | −56% |
| `search_map_top5/1000` (dim 4096) | 2.75 ms | 0.91 ms | −66% |
| `search_map_top5/20000` (dim 4096) | 114.7 ms | 30.1 ms | −74% |
| `item_memory_cleanup/4096` (1000 symbols) | 8.49 ms | 1.25 ms | −85% |
| `item_memory_cleanup/8192` (1000 symbols) | 21.7 ms | 2.37 ms | −89% |
| `bsc/xor_bind`, `bsc/hamming` | 20–125 ns | 20–125 ns | unchanged (code not touched; ±10% noise) |

What changed:

- **bind/unbind:** every length uses the FFT, so non-power-of-two dims no
  longer fall back to O(n²). The FFT planner is thread-local instead of
  behind a global mutex.
- **cosine:** a single fused pass with four accumulator lanes, instead of
  three passes over the data.
- **search / cleanup:** scoring runs on rayon above 256 candidates, and
  top-k uses partial selection instead of a full sort.

## Bundling 100 vectors (dim 4096)

| Method | Time | Exact majority? |
|---|---|---|
| MAP `bundle_sequence` (pairwise fold) | 189 µs | n/a (sum) |
| MAP `BundleAccumulator` | 164 µs | n/a (sum, identical result) |
| BSC `bundle_sequence` (pairwise majority) | 12 µs | no |
| BSC `majority_bundle_all` | 271 µs | yes |
| BSC `BinaryBundleAccumulator` | 172 µs | yes |

The pairwise BSC fold is fast but lossy: each step halves the weight of
everything bundled so far. Weakest-member similarity, averaged over 10
trials (0 means unrecoverable):

| Vectors bundled | Pairwise | Accumulator |
|---|---|---|
| 5 | 0.054 | 0.357 |
| 15 | −0.020 | 0.180 |
| 50 | −0.032 | 0.079 |
| 100 | −0.037 | 0.040 |

## Choosing a MAP dimension

Recall when unbinding every key from one bundled record, with a
cleanup memory holding all keys and values (5 trials each):

| Pairs in record | 2048 | 4096 | 8192 |
|---|---|---|---|
| 50 | 99.6% | 100% | 100% |
| 100 | 94.8% | 100% | 100% |
| 150 | 76.0% | 98.9% | 100% |
| 200 | 57.7% | 91.6% | 99.8% |

`MAP_DEFAULT_DIM` (4096) is the default. Use `MAP_HIGH_CAPACITY_DIM` (8192)
for records with more than about 100 pairs. It costs about 2× the time and
memory per operation.
