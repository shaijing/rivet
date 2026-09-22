//! Continuously run a CUDA tensor matmul so it can be inspected with
//! `nvidia-smi`.
//!
//! Run with:
//!
//! ```text
//! cargo run -j 12 -p rivet-core --features cuda --example cuda_tensor --release
//! ```
//!
//! The optional arguments are `CUDA_ORDINAL` and `MATRIX_SIZE`:
//!
//! ```text
//! cargo run -j 12 -p rivet-core --features cuda --example cuda_tensor --release -- 1 2048
//! ```

use std::env;
use std::time::{Duration, Instant};

use rivet_core::{DType, Device, Tensor};

const DEFAULT_MATRIX_SIZE: usize = 4096;

fn parse_arg<T>(args: &[String], index: usize, default: T, name: &str) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match args.get(index) {
        Some(value) => value.parse().unwrap_or_else(|error| {
            eprintln!("invalid {name} `{value}`: {error}");
            std::process::exit(2);
        }),
        None => default,
    }
}

fn main() -> rivet_core::Result<()> {
    let args: Vec<_> = env::args().collect();
    let ordinal = parse_arg(&args, 1, 0usize, "CUDA ordinal");
    let size = parse_arg(&args, 2, DEFAULT_MATRIX_SIZE, "matrix size");

    if size == 0 {
        eprintln!("matrix size must be greater than zero");
        std::process::exit(2);
    }

    let device = Device::cuda(ordinal)?;
    let lhs = Tensor::ones([size, size], DType::F32, &device)?;
    let rhs = Tensor::ones([size, size], DType::F32, &device)?;

    eprintln!(
        "running CUDA matmul on GPU {ordinal}: [{size}, {size}] x [{size}, {size}] (Ctrl-C to stop)"
    );

    // Keep the most recent output alive so nvidia-smi also shows its device
    // allocation. Synchronizing once per iteration keeps the loop bounded and
    // makes the reported iteration rate meaningful while cuBLAS does the work.
    let mut output = lhs.matmul(&rhs)?;
    device.synchronize()?;
    let mut iterations = 1u64;
    let mut report_at = Instant::now() + Duration::from_secs(1);
    let mut reported_iterations = iterations;

    loop {
        let next = lhs.matmul(&rhs)?;
        device.synchronize()?;
        // Replace the previous output only after the stream has completed, so
        // its allocation stays alive for the queued matmul.
        drop(std::mem::replace(&mut output, next));
        iterations += 1;

        let now = Instant::now();
        if now >= report_at {
            let elapsed = now.duration_since(report_at - Duration::from_secs(1));
            let window_iterations = iterations - reported_iterations;
            let rate = window_iterations as f64 / elapsed.as_secs_f64();
            eprintln!("{rate:.1} matmuls/s ({iterations} total)");
            reported_iterations = iterations;
            report_at = now + Duration::from_secs(1);
        }
    }
}
