# Runtime worker chunk sizing

Branch: `perf/runtime-hotpath`

Date: 2026-09-23

## Change

The runtime keeps its general chunk heuristic:

```text
ceil(batch_size / (workers * 2)).clamp(1, 8)
```

Adapters can now provide a chunk-size hint for a measured lightweight path.
Vision uses a chunk size of 16 only when the source is dense decoded data and
all sample operations are `RandomCrop`, `RandomHorizontalFlip`, or
`NormalizeSample`. Other plans, including encoded decode and ImageNet-style
transforms, keep the general heuristic.

## CIFAR decoded results

Release extension benchmark, five runs per configuration:

```bash
.venv/bin/python bench/compare_cifar10_torchvision_rivet_decoded.py \
  --torch-root /data/datasets/pytorch \
  --rivet-root /data/datasets/rivet/cifar10 \
  --batch 128 --workers 24 --epochs 3
```

| Workers | Chunk path | Mean img/s | Range | Change |
|---:|---|---:|---:|---:|
| 24 | General heuristic, chunk 3 | 950,101 | 936,313–965,843 | baseline |
| 24 | Dense decoded hint, chunk 16 | 1,284,999 | 1,274,409–1,297,293 | +35.2% |

At the repository's common 4-worker setting, the five-run mean was 748,131
img/s (743,191–752,761), compared with the recorded P0 mean of 728,296 img/s:
+2.7%.

## Heavy workload check

The encoded ImageNet-style pipeline does not qualify for the dense decoded
hint and therefore uses the general heuristic. A default-size pipeline matrix
run measured 2,414 img/s at 4 workers/prefetch 2 and 8,655 img/s at 24
workers/prefetch 2. The previous recorded representative values were 2,471
and 8,304 img/s respectively; this small matrix run shows no material
regression, but it is not a repeated statistical comparison.

## Validation

- `cargo check -j 12 -p rivet-exec --lib` passed.
- `cargo check -j 12 -p rivet-vision --lib` passed.
- `maturin develop --release -j 12` passed.
- The release CIFAR decoded benchmark was run five times at 4 and 24 workers.
- `pipeline_baseline_bench` was run at its default 256 samples / 32 batch /
  8 batches configuration to check the encoded ImageNet-style workload.

No unit-test suite was run.
