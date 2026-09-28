# Semantic IR rewrite validation

Baseline: `main` at `ca82e5b` before `feat/semantic-ir-rewrite`.

Scope: preserve declared IR order, perform guarded selection pushdown, maintain
stable stochastic identities, compose source sampling in declared order, and
guard batch lifting without changing tensor/vision hot kernels.

Differential tests compare full, unoptimized image-transform output against
selected optimized output by original source identity, across inline and pooled
workers. They cover padded random crop, normalize, skipped samples, source read
counts, graph round trips, random-node deletion and shared branch barriers.
Source sampler tests distinguish shuffle-before-take from take-before-shuffle,
exercise repeated shuffles and saturating large bounds. Physical executor tests
ensure stage classification alone, intervening sample operations and shared
nodes do not grant permission to lift a kernel across stacking.

The compiler continues to reject residual branch-local/post-barrier indexes,
multiple vision sources and device execution explicitly. This change does not
add a stream-global execution engine for indexes that cannot reach the sampler.

## Release CIFAR benchmark

Command (three complete runs per implementation):

```sh
.venv/bin/python bench/compare_cifar10_torchvision_rivet_decoded.py \
  --torch-root /home/lingyu/.data/pytorch \
  --rivet-root /home/lingyu/.cache/rivet/datasets/cifar10 \
  --batch 128 --workers 4 --epochs 3
```

Installed release extension measurements:

| Implementation | Three runs (img/s) | Mean (img/s) |
| --- | --- | --- |
| Baseline | 754459, 739801, 751253 | 748504 |
| Semantic rewrites | 757576, 766434, 760562 | 761524 |

The updated mean is 1.74% higher. These runs show no throughput regression;
the small difference should not be interpreted as a guaranteed speedup from
selection rewrites in this full-dataset benchmark. Typed HWC/NHWC normalization
and NHWC-to-NCHW kernels remain unchanged.

## Targeted checks

- `cargo test -j 12 -p rivet-plan -p rivet-exec -p rivet-data --lib
  --no-default-features`: 13 plan, 29 execution and 22 data tests passed.
- `cargo test -j 12 -p rivet-vision --lib`: 187 tests passed, including Lance.
- `.venv/bin/pytest -q tests/test_arrow_pipeline.py`: 37 tests passed with the
  rebuilt release extension.
- Scoped library Clippy completed; repository warnings remain in existing
  code. This is not a clean `-D warnings` result.
- `git diff --check` passed.
