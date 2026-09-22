use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda_utils.cuh");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");

    // Keep the generated PTX architecture configurable for CI and machines
    // without an accessible GPU. Candle uses the same cudaforge default when
    // the architecture is not supplied; sm_80 is a conservative fallback for
    // these small baseline kernels.
    let compute_cap = cudaforge::detect_compute_cap()
        .map(|arch| arch.base())
        .unwrap_or(80);

    let bindings = KernelBuilder::new()
        .compute_cap(compute_cap)
        .source_dir("src")
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;
    bindings.write(ptx_path)?;

    Ok(())
}
