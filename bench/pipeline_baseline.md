# Pipeline Phase-0 baseline

Recorded on 2026-09-21 in the local release environment.

The complete matrix is emitted by:

```bash
cargo run -j 12 -p rivet-vision --release --no-default-features \
  --example pipeline_baseline_bench
```

The benchmark covers decode-only, random crop, crop+flip, crop+normalize,
crop+normalize+CHW, full CIFAR, and ImageNet-style
`RandomResizedCrop + Flip + ColorJitter + Normalize + CHW`. Each workload is
run with workers `0/1/4/physical-core-count` and prefetch `0/1/2`, and prints
`first_ms`, steady-state `steady_ms`, and `img_s` for every combination.

Reference run configuration:

```text
samples=256 batch=32 batches=8 physical_cores=24
```

Representative steady-state results from that run:

| workload | workers | prefetch | first ms | steady ms | img/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode | 0 | 0 | 0.330 | 0.265 | 120818 |
| decode | 4 | 2 | 0.114 | 0.091 | 350228 |
| random_crop | 4 | 2 | 0.048 | 0.027 | 293384 |
| random_crop_normalize_chw | 4 | 2 | 0.041 | 0.028 | 283191 |
| full_cifar | 4 | 2 | 0.047 | 0.033 | 242940 |
| imagenet_style | 4 | 2 | 27.387 | 12.952 | 2471 |
| imagenet_style | 24 | 2 | 16.730 | 3.853 | 8304 |

The executable output, rather than these representative rows, is the
authoritative baseline for future Phase comparisons.
