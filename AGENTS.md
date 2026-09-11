## Build resource limits

To avoid saturating the development machine, do not run Rust builds with the default CPU-wide parallelism.

For all Cargo commands that may compile code, limit parallel jobs to 16:

```bash
cargo check -j 16
cargo build -j 16
cargo test -j 16