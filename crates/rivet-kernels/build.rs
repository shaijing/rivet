use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::{Path, PathBuf};

fn kernel_builder() -> KernelBuilder {
    println!("cargo:rerun-if-env-changed=RIVET_CUDA_ROOT");

    if let Ok(cuda_root) = env::var("RIVET_CUDA_ROOT") {
        return KernelBuilder::new().cuda_root(cuda_root);
    }

    // rivet-core's cudarc feature is pinned to the CUDA 13.3 API profile.
    // Prefer the matching toolkit when it is installed even if a newer nvcc
    // happens to appear first in PATH: newer toolchains can emit PTX that an
    // older driver rejects before a kernel is launched.
    let compatible_root = Path::new("/usr/local/cuda-13.3");
    if compatible_root.join("bin/nvcc").is_file() {
        return KernelBuilder::new().cuda_root(compatible_root);
    }

    KernelBuilder::new()
}

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo:rerun-if-changed=src/compatibility.cuh");
    println!("cargo:rerun-if-changed=src/vision_normalize.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");

    // Keep the generated PTX architecture configurable for CI and machines
    // without an accessible GPU. Candle uses the same cudaforge default when
    // the architecture is not supplied; sm_80 is a conservative fallback for
    // these small baseline kernels.
    let compute_cap = cudaforge::detect_compute_cap()
        .map(|arch| arch.base())
        .unwrap_or(80);

    let bindings = kernel_builder()
        .compute_cap(compute_cap)
        .source_dir("src")
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-allow-unsupported-compiler")
        .arg("-O3")
        .build_ptx()?;
    bindings.write(ptx_path)?;

    Ok(())
}
