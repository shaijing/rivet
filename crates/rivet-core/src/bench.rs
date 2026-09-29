//! Internal measurement helpers for the Criterion targets.
//!
//! This module is feature-gated so the low-level paths being compared do not
//! become part of Rivet's supported public API.

use crate::cpu_backend::buffer::{AlignedBuffer, AlignedBufferBuilder};
use crate::cpu_backend::matmul;
use crate::cpu_backend::utils::ValidatedValues;
use crate::storage::validate_layout_for_storage;
use crate::{Error, Layout, Result};

/// Builds an aligned buffer by writing one byte at a time.
pub fn builder_sequential(values: &[u8]) -> Result<usize> {
    let mut builder = AlignedBufferBuilder::new(values.len())?;
    for &value in values {
        builder.write_next(value)?;
    }
    let output = builder.finish()?;
    // Keep the constructed bytes observable in optimized Criterion builds;
    // returning only len lets LLVM remove the allocation and every write.
    std::hint::black_box(output.as_slice());
    Ok(output.len())
}

/// Builds an aligned buffer with one bulk copy.
pub fn builder_extend_from_slice(values: &[u8]) -> Result<usize> {
    let mut builder = AlignedBufferBuilder::new(values.len())?;
    builder.extend_from_slice(values)?;
    let output = builder.finish()?;
    // Keep the constructed bytes observable in optimized Criterion builds;
    // returning only len lets LLVM remove the allocation and copy.
    std::hint::black_box(output.as_slice());
    Ok(output.len())
}

/// Sums a strided view using a checked slice lookup for every element.
pub fn sum_per_element_get(values: &[u8], layout: &Layout) -> Result<u64> {
    validate_layout_for_storage(layout, values.len())?;
    let mut sum = 0u64;
    for offset in layout.strided_index() {
        sum += u64::from(*values.get(offset).ok_or(Error::StorageOutOfBounds)?);
    }
    Ok(sum)
}

/// Sums a strided view after its layout has been validated once.
pub fn sum_validated_strided(values: &[u8], layout: &Layout) -> Result<u64> {
    let values = ValidatedValues::new(values, layout)?;
    let mut sum = 0u64;
    for offset in layout.strided_index() {
        sum += u64::from(values.read(offset));
    }
    Ok(sum)
}

/// Sums a contiguous slice without strided indexing.
pub fn sum_contiguous(values: &[u8]) -> u64 {
    values.iter().map(|&value| u64::from(value)).sum()
}

/// Identical aligned inputs for both CPU matmul implementations. Construction
/// is outside the timed loop; each multiply allocates its own aligned output.
pub struct MatmulCase {
    lhs: AlignedBuffer<f32>,
    rhs: AlignedBuffer<f32>,
    lhs_layout: Layout,
    rhs_layout: Layout,
}

impl MatmulCase {
    pub fn new(m: usize, k: usize, n: usize) -> Result<Self> {
        let lhs_len = m.checked_mul(k).ok_or(Error::StorageOutOfBounds)?;
        let rhs_len = k.checked_mul(n).ok_or(Error::StorageOutOfBounds)?;
        let mut lhs = AlignedBufferBuilder::new(lhs_len)?;
        let mut rhs = AlignedBufferBuilder::new(rhs_len)?;
        for index in 0..lhs_len {
            lhs.write_next((index % 17) as f32 * 0.125 - 1.0)?;
        }
        for index in 0..rhs_len {
            rhs.write_next((index % 13) as f32 * 0.25 - 1.5)?;
        }
        Ok(Self {
            lhs: lhs.finish()?,
            rhs: rhs.finish()?,
            lhs_layout: Layout::contiguous((m, k)),
            rhs_layout: Layout::contiguous((k, n)),
        })
    }

    /// Original Rivet GEMM path, including its Rayon parallelism policy.
    pub fn gemm(&self) -> Result<impl AsRef<[f32]>> {
        matmul::f32_gemm(&self.lhs, &self.lhs_layout, &self.rhs, &self.rhs_layout)
    }

    /// Production BLAS path, including GEMV for a single output column.
    #[cfg(feature = "blas")]
    pub fn blas(&self) -> Result<impl AsRef<[f32]>> {
        matmul::f32(&self.lhs, &self.lhs_layout, &self.rhs, &self.rhs_layout)
    }
}
