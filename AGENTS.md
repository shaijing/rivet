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
  --offline \
  --batch 128 --workers 4 --epochs 3
```

On 2026-11-26, five offline runs at commit `b323e8f` with the Python extension
built using `maturin develop --release -j 12` and Rust `1.99.1` produced Rivet
decoded steady throughputs of `1,119,801`, `1,128,255`, `1,120,164`, `1,127,857`, and
`1,124,615` img/s. The arithmetic mean was `1,124,138` img/s, the median was
`1,124,615` img/s, and the range was `1,119,801–1,128,255` img/s. The
population standard deviation was `3,622` img/s.

Steady throughput uses epochs 2 and 3 of each run. Treat these measurements as
a same-environment baseline. Rerun multiple times on the same machine and
investigate meaningful regressions before merging.


# References
candle tensor: /home/lingyu/ws/rustWS/candle/candle-core/src/tensor.rs
polars ir: /home/lingyu/ws/rustWS/polars/crates/polars-plan/src/plans/ir/mod.rs
tokenizers: /home/lingyu/ws/rustWS/tokenizers
huggingface_hub: /home/lingyu/ws/pyWS/huggingface_hub
datasets: /home/lingyu/ws/pyWS/datasets