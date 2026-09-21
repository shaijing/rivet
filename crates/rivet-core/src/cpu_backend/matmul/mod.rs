use crate::cpu_backend::buffer::{AlignedBuffer, AlignedBufferBuilder};
use crate::cpu_backend::storage::{aligned, CpuStorage};
use crate::{Error, Layout, Result};

impl CpuStorage {
    pub(crate) fn matmul(
        &self,
        lhs_layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => {
                Ok(Self::F32(aligned(f32(lhs, lhs_layout, rhs, rhs_layout)?)?))
            }
            _ => Err(Error::UnsupportedMatmulDType {
                dtype: self.dtype(),
            }),
        }
    }
}

const GEMM_POINTER_ALIGNMENT: usize = 16;

/// Executes a rank-2 row-major F32 matrix multiplication without copying the
/// input allocations. The `gemm` crate accepts explicit strides, so the
/// contiguous layouts may still point into a non-zero storage offset.
pub(crate) fn f32(
    lhs: &[f32],
    lhs_layout: &Layout,
    rhs: &[f32],
    rhs_layout: &Layout,
) -> Result<AlignedBuffer<f32>> {
    if lhs_layout.dims().len() != 2 || rhs_layout.dims().len() != 2 {
        return Err(Error::MatmulShapeMismatch {
            lhs: lhs_layout.dims().to_vec(),
            rhs: rhs_layout.dims().to_vec(),
        });
    }

    let [m, k] = [lhs_layout.dims()[0], lhs_layout.dims()[1]];
    let [rhs_k, n] = [rhs_layout.dims()[0], rhs_layout.dims()[1]];
    if k != rhs_k {
        return Err(Error::MatmulShapeMismatch {
            lhs: lhs_layout.dims().to_vec(),
            rhs: rhs_layout.dims().to_vec(),
        });
    }
    if !lhs_layout.is_contiguous() || !rhs_layout.is_contiguous() {
        return Err(Error::UnsupportedMatmulLayout);
    }

    let output_len = m.checked_mul(n).ok_or(Error::StorageOutOfBounds)?;
    let mut builder = AlignedBufferBuilder::new(output_len)?;
    for _ in 0..output_len {
        builder.write_next(0.0)?;
    }
    let mut output = builder.finish()?;
    // No input or output element is addressed for these cases. Returning
    // early also avoids forming pointers from an empty allocation with an
    // arbitrary empty-view offset.
    if m == 0 || n == 0 || k == 0 {
        return Ok(output);
    }
    let lhs_start = lhs_layout.start_offset();
    let rhs_start = rhs_layout.start_offset();
    let lhs_end = lhs_start
        .checked_add(m.checked_mul(k).ok_or(Error::StorageOutOfBounds)?)
        .ok_or(Error::StorageOutOfBounds)?;
    let rhs_end = rhs_start
        .checked_add(k.checked_mul(n).ok_or(Error::StorageOutOfBounds)?)
        .ok_or(Error::StorageOutOfBounds)?;
    if lhs.get(lhs_start..lhs_end).is_none() || rhs.get(rhs_start..rhs_end).is_none() {
        return Err(Error::StorageOutOfBounds);
    }

    let lhs_cs =
        isize::try_from(lhs_layout.stride()[1]).map_err(|_| Error::UnsupportedMatmulLayout)?;
    let lhs_rs =
        isize::try_from(lhs_layout.stride()[0]).map_err(|_| Error::UnsupportedMatmulLayout)?;
    let rhs_cs =
        isize::try_from(rhs_layout.stride()[1]).map_err(|_| Error::UnsupportedMatmulLayout)?;
    let rhs_rs =
        isize::try_from(rhs_layout.stride()[0]).map_err(|_| Error::UnsupportedMatmulLayout)?;

    // `gemm` computes alpha * dst + beta * lhs * rhs. The output is fresh,
    // so alpha=0 and beta=1 produce exactly lhs @ rhs.
    let lhs_ptr = unsafe { lhs.as_ptr().add(lhs_start) };
    let rhs_ptr = unsafe { rhs.as_ptr().add(rhs_start) };
    let output_ptr = output.as_mut_slice().as_mut_ptr();
    let aligned = pointer_alignment(lhs_ptr) >= GEMM_POINTER_ALIGNMENT
        && pointer_alignment(rhs_ptr) >= GEMM_POINTER_ALIGNMENT
        && pointer_alignment(output_ptr) >= GEMM_POINTER_ALIGNMENT;

    if aligned {
        // The aligned planner branch is used only when every effective
        // pointer, not merely every backing allocation, meets the threshold.
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                output_ptr,
                1,
                isize::try_from(n).map_err(|_| Error::UnsupportedMatmulLayout)?,
                false,
                lhs_ptr,
                lhs_cs,
                lhs_rs,
                rhs_ptr,
                rhs_cs,
                rhs_rs,
                0.0,
                1.0,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }
    } else {
        // The generic scalar branch avoids making alignment assumptions for
        // offset views. Layout and storage bounds were checked above.
        for row in 0..m {
            for column in 0..n {
                let mut sum = 0.0;
                for inner in 0..k {
                    let lhs_offset = row as isize * lhs_rs + inner as isize * lhs_cs;
                    let rhs_offset = inner as isize * rhs_rs + column as isize * rhs_cs;
                    unsafe {
                        sum += *lhs_ptr.offset(lhs_offset) * *rhs_ptr.offset(rhs_offset);
                    }
                }
                output.as_mut_slice()[row * n + column] = sum;
            }
        }
    }
    Ok(output)
}

fn pointer_alignment<T>(ptr: *const T) -> usize {
    let address = ptr as usize;
    if address == 0 {
        return 1;
    }
    1usize << address.trailing_zeros().min(usize::BITS - 1)
}
