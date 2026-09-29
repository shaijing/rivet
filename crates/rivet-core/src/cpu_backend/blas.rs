//! Native vector reductions. Unsupported layouts and ABI sizes fall back to
//! the existing kernels without materializing inputs.
use rivet_blas_sys::cblas::prelude::{CBlasInt, cblas_ddot, cblas_dnrm2, cblas_dsdot, cblas_snrm2};

use super::buffer::AlignedBuffer;
use super::storage::{CpuStorage, aligned};
use crate::{Error, Layout, Result};

struct Vector<'a, T> {
    values: &'a [T],
    len: CBlasInt,
    step: CBlasInt,
}

/// Only a contiguous flattening or a positive-stride rank-1 view can be
/// represented by one BLAS vector. Zero-stride broadcasts use the fallback.
fn vector<'a, T>(values: &'a [T], layout: &Layout) -> Result<Option<Vector<'a, T>>> {
    let count = layout.checked_elem_count()?;
    if count == 0 {
        // Empty views may have arbitrary offsets; do not form offset pointers.
        return Ok(Some(Vector {
            values: &values[..0],
            len: 0,
            step: 1,
        }));
    }
    let step = if count == 1 || layout.is_contiguous() {
        1
    } else if layout.dims().len() == 1 && layout.stride()[0] > 0 {
        layout.stride()[0]
    } else {
        return Ok(None);
    };
    let (Ok(len), Ok(step)) = (CBlasInt::try_from(count), CBlasInt::try_from(step)) else {
        return Ok(None);
    };
    let end = layout
        .max_storage_offset()
        .and_then(|offset| offset.checked_add(1))
        .ok_or(Error::StorageOutOfBounds)?;
    let values = values
        .get(layout.start_offset()..end)
        .ok_or(Error::StorageOutOfBounds)?;
    Ok(Some(Vector { values, len, step }))
}

macro_rules! reductions {
    ($dot:ident, $norm:ident, $ty:ty, $dot_ffi:ident, $norm_ffi:ident) => {
        fn $dot(
            lhs: &[$ty],
            lhs_layout: &Layout,
            rhs: &[$ty],
            rhs_layout: &Layout,
        ) -> Result<Option<AlignedBuffer<$ty>>> {
            let (Some(lhs), Some(rhs)) = (vector(lhs, lhs_layout)?, vector(rhs, rhs_layout)?)
            else {
                return Ok(None);
            };
            // The caller checked equal rank-1 shapes. Both complete storage
            // spans and ABI integers are validated before entering BLAS.
            debug_assert_eq!(lhs.len, rhs.len);
            let value = if lhs.len == 0 {
                0.0
            } else {
                // SAFETY: vector() checked both positive-stride spans.
                unsafe {
                    $dot_ffi(
                        lhs.len,
                        lhs.values.as_ptr(),
                        lhs.step,
                        rhs.values.as_ptr(),
                        rhs.step,
                    ) as $ty
                }
            };
            Ok(Some(AlignedBuffer::from_slice(&[value])?))
        }

        fn $norm(values: &[$ty], layout: &Layout) -> Result<Option<AlignedBuffer<$ty>>> {
            let Some(vector) = vector(values, layout)? else {
                return Ok(None);
            };
            let value = if vector.len == 0 {
                0.0
            } else {
                // SAFETY: vector() checked the entire positive-stride span.
                unsafe { $norm_ffi(vector.len, vector.values.as_ptr(), vector.step) }
            };
            Ok(Some(AlignedBuffer::from_slice(&[value])?))
        }
    };
}

// F32 dot retains double-precision products and accumulation via DSDOT.
reductions!(dot_f32, norm_f32, f32, cblas_dsdot, cblas_snrm2);
reductions!(dot_f64, norm_f64, f64, cblas_ddot, cblas_dnrm2);

impl CpuStorage {
    pub(super) fn blas_dot(
        &self,
        lhs_layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
    ) -> Result<Option<Self>> {
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => dot_f32(lhs, lhs_layout, rhs, rhs_layout)?
                .map(|output| Ok(Self::F32(aligned(output)?)))
                .transpose(),
            (Self::F64(lhs), Self::F64(rhs)) => dot_f64(lhs, lhs_layout, rhs, rhs_layout)?
                .map(|output| Ok(Self::F64(aligned(output)?)))
                .transpose(),
            _ => Ok(None),
        }
    }

    pub(super) fn blas_norm(&self, layout: &Layout) -> Result<Option<Self>> {
        match self {
            Self::F32(values) => norm_f32(values, layout)?
                .map(|output| Ok(Self::F32(aligned(output)?)))
                .transpose(),
            Self::F64(values) => norm_f64(values, layout)?
                .map(|output| Ok(Self::F64(aligned(output)?)))
                .transpose(),
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_checks_bounds_and_abi_without_forming_invalid_pointers() {
        assert!(matches!(
            vector(&[1.0f32], &Layout::contiguous_with_offset(2, 1)),
            Err(Error::StorageOutOfBounds)
        ));
        let empty = Layout::contiguous_with_offset(0, usize::MAX);
        assert_eq!(vector::<f32>(&[], &empty).unwrap().unwrap().len, 0);
        let broadcast = Layout::new(3.into(), vec![0], 0).unwrap();
        assert!(vector(&[1.0f32], &broadcast).unwrap().is_none());
        let oversized = Layout::contiguous(CBlasInt::MAX as usize + 1);
        assert!(vector::<f32>(&[], &oversized).unwrap().is_none());
        let overflow = Layout::contiguous_with_offset(1, usize::MAX);
        assert!(matches!(
            vector::<f32>(&[], &overflow),
            Err(Error::StorageOutOfBounds)
        ));
    }
}
