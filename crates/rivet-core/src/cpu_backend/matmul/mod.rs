use crate::{Error, Layout, Result};

/// Executes a rank-2 row-major F32 matrix multiplication without copying the
/// input allocations. The `gemm` crate accepts explicit strides, so the
/// contiguous layouts may still point into a non-zero storage offset.
pub(crate) fn f32(
    lhs: &[f32],
    lhs_layout: &Layout,
    rhs: &[f32],
    rhs_layout: &Layout,
) -> Result<Vec<f32>> {
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
    let mut output = vec![0.0f32; output_len];
    // No input or output element is addressed for these cases. Returning
    // early also avoids forming pointers from an empty allocation with an
    // arbitrary empty-view offset.
    if m == 0 || n == 0 || k == 0 {
        return Ok(output);
    }
    if lhs_layout.start_offset() > lhs.len() || rhs_layout.start_offset() > rhs.len() {
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
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            output.as_mut_ptr(),
            1,
            isize::try_from(n).map_err(|_| Error::UnsupportedMatmulLayout)?,
            false,
            lhs.as_ptr().add(lhs_layout.start_offset()),
            lhs_cs,
            lhs_rs,
            rhs.as_ptr().add(rhs_layout.start_offset()),
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
    Ok(output)
}
