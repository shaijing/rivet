use rivet_blas_sys::cblas::prelude::{
    CBlasInt, CBlasLayout, CBlasTranspose, cblas_sgemm, cblas_sgemv,
};

use crate::{Error, Result};

/// The caller has checked shapes, row-major contiguity, storage bounds, and
/// non-zero dimensions. The output is fresh and cannot alias either input.
pub(super) fn f32(
    lhs: &[f32],
    rhs: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    // Never truncate dimensions at the FFI boundary. CBlasInt follows the
    // selected binding's ABI (LP64 by default).
    let m = CBlasInt::try_from(m).map_err(|_| Error::UnsupportedMatmulLayout)?;
    let n = CBlasInt::try_from(n).map_err(|_| Error::UnsupportedMatmulLayout)?;
    let k = CBlasInt::try_from(k).map_err(|_| Error::UnsupportedMatmulLayout)?;

    if n == 1 {
        // SAFETY: the same bounds checks cover A[m,k], x[k], and y[m].
        // This also accelerates Tensor::mv, which delegates to matmul.
        unsafe {
            cblas_sgemv(
                CBlasLayout::CBlasRowMajor,
                CBlasTranspose::CBlasNoTrans,
                m,
                k,
                1.0,
                lhs.as_ptr(),
                k,
                rhs.as_ptr(),
                1,
                0.0,
                output.as_mut_ptr(),
                1,
            );
        }
        return Ok(());
    }

    // SAFETY: the caller validated the complete input and output ranges.
    // Row-major, non-transposed matrices use lda=k and ldb=ldc=n. Native
    // CBLAS accepts offset pointers without a SIMD alignment requirement.
    unsafe {
        cblas_sgemm(
            CBlasLayout::CBlasRowMajor,
            CBlasTranspose::CBlasNoTrans,
            CBlasTranspose::CBlasNoTrans,
            m,
            n,
            k,
            1.0,
            lhs.as_ptr(),
            k,
            rhs.as_ptr(),
            n,
            0.0,
            output.as_mut_ptr(),
            n,
        );
    }
    Ok(())
}
