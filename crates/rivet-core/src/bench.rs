//! Internal measurement helpers for the Criterion targets.
//!
//! This module is feature-gated so the low-level paths being compared do not
//! become part of Rivet's supported public API.

use crate::cpu_backend::buffer::AlignedBufferBuilder;
use crate::cpu_backend::utils::ValidatedValues;
use crate::storage::validate_layout_for_storage;
use crate::{Error, Layout, Result};

/// Builds an aligned buffer by writing one byte at a time.
pub fn builder_sequential(values: &[u8]) -> Result<usize> {
    let mut builder = AlignedBufferBuilder::new(values.len())?;
    for &value in values {
        builder.write_next(value)?;
    }
    Ok(builder.finish()?.len())
}

/// Builds an aligned buffer with one bulk copy.
pub fn builder_extend_from_slice(values: &[u8]) -> Result<usize> {
    let mut builder = AlignedBufferBuilder::new(values.len())?;
    builder.extend_from_slice(values)?;
    Ok(builder.finish()?.len())
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
