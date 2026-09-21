## Build scope

Do not run unscoped Cargo commands from the workspace root for a full build.
Prefer targeting the crate and target needed for the task, for example:

```bash
cargo check -j 12 -p rivet-core --lib
cargo test -j 12 -p rivet-data --lib
```

Avoid `--all-targets` and `--examples` unless the task specifically requires
building or testing examples.

## PyO3 / Python extension

For the PyO3 project, prefer using maturin rather than invoking Cargo directly.
The Python extension is configured in `pyproject.toml`; typical commands are:

```bash
maturin develop -j 12
maturin build -j 12
```

```bash
# Lance is enabled by default. Use --no-default-features to disable it.
# Debug editable install
maturin develop -j 12

# Release editable install (use this for benchmarks)
maturin develop --release -j 12
```

## Tensor / vision hot-kernel performance

- Preserve specialized fast paths for common tensor layouts, dtypes, and
  channel counts. In particular, keep the HWC/NHWC three-channel normalize
  path and the dedicated NHWC-to-NCHW nested-loop path; do not replace them
  with a generic per-element iterator unless the specialized path remains
  available for the common case.
- Avoid introducing per-element dynamic dispatch, callback-based writes, or
  avoidable division/modulo in hot image kernels. Direct-output refactors must
  keep the typed/static writer path for common shapes and should write into the
  final aligned allocation without reintroducing an intermediate `Vec`.
- Before merging changes to tensor construction or vision transforms, run the
  relevant targeted tests and the CIFAR benchmark in release mode. Compare
  steady throughput against the previous implementation; a refactor that
  removes a copy but regresses the common benchmark must be investigated and
  must retain or restore the specialized kernel.

### CIFAR-10 decoded benchmark reference

Reference command:

```bash
.venv/bin/python bench/compare_cifar10_torchvision_rivet_decoded.py \
  --torch-root /data/datasets/pytorch \
  --rivet-root /data/datasets/rivet/cifar10 \
  --batch 128 --workers 4 --epochs 3
```

With the release Python extension and the current decoded CIFAR-10 dataset,
five complete runs produced speedups of `14.81x`, `14.94x`, `15.47x`,
`14.61x`, and `13.48x`: arithmetic mean `14.66x`, range `13.48x-15.47x`,
standard deviation `0.66x`. Treat this as a same-environment reference
baseline; rerun multiple times and investigate meaningful regressions rather
than comparing against a single noisy run.


# References
candle tensor: /home/ling/ws/rustWS/candle/candle-core/src/tensor.rs
