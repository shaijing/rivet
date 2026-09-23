# Runtime hot path P0 results

Branch: `perf/runtime-hotpath`

Date: 2026-09-23

This change applies the first runtime optimizations from `tmp/perf.md`: profiling is opt-in, CPU-only execution skips queue byte accounting, panic isolation is at the fused batch or worker chunk boundary, and workers receive contiguous sample chunks. Chunk results are written into the existing per-sample slots with one coordinator lookup per chunk. Profiling-disabled persistent queues also skip wait timing while retaining their item and byte limits.

## CIFAR decoded benchmark

Command, run five times with the release Python extension:

```bash
.venv/bin/python bench/compare_cifar10_torchvision_rivet_decoded.py \
  --torch-root /data/datasets/pytorch \
  --rivet-root /data/datasets/rivet/cifar10 \
  --batch 128 --workers 4 --epochs 3
```

| Measurement | Mean | Range |
|---|---:|---:|
| Before this branch's changes | 703,958 img/s | 702,726–705,340 |
| After these changes | 728,296 img/s | 721,215–733,506 |
| AGENTS.md reference | 731,303 img/s | 725,681–737,451 |

The updated result is 3.5% above the pre-change branch measurement and within 0.5% of the repository reference mean. The reference is an environment comparison, not a strict before/after control.

## Rust pipeline measurements

The focused regression example was run five times. Before the changes, its recorded result was p50 0.108 ms, p95 0.113 ms, and 1,114,749 img/s. After the changes, the five-run means were p50 0.105 ms, p95 0.116 ms, and 1,093,957 img/s. This microbenchmark has a very short steady-state window; throughput is 1.9% lower while p50 latency is 2.9% lower, so the throughput difference needs longer runs before attributing it to the code.

The full matrix was extended to 2,048 samples and 64 batches:

```bash
RIVET_PIPELINE_BENCH_SAMPLES=2048 \
RIVET_PIPELINE_BENCH_BATCHES=64 \
cargo run -j 12 -p rivet-vision --release --no-default-features \
  --example pipeline_baseline_bench
```

For `full_cifar`, the measured throughput was 89.2k img/s inline, 259.3k with 4 workers and prefetch 2, and 630.2k with 24 workers and prefetch 2. The high-worker case remains near the guide's approximately 608k reference. A single worker remained slower for light CPU transforms; submitting one full batch per task did not remove that gap.

## Validation

Targeted library checks passed for `rivet-exec`, `rivet-vision`, and `rivet-python`. The release extension was rebuilt with `maturin develop --release -j 12`. No unit-test suite was run.
