## Build scope

Do not run unscoped Cargo commands from the workspace root for a full build.
Prefer targeting the crate and target needed for the task, for example:

```bash
cargo check -j 12 -p rivet-core --lib
cargo test -j 12 -p rivet-dataset --lib
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

Use the following commands when the Lance backend is needed:

```bash
# Debug editable install
maturin develop -j 12 --features lance

# Release editable install (use this for benchmarks)
maturin develop --release -j 12 --features lance
```


# References
candle tensor: /home/ling/ws/rustWS/candle/candle-core/src/tensor.rs
