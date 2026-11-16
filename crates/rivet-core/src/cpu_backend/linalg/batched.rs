//! Allocation-free traversal of regular batches. This uses the linked BLAS
//! provider's ordinary GEMM/GEMV entry points, so it also works through
//! FlexiBLAS and does not require vendor-specific batch symbols.
use super::*;

pub(crate) struct StridedBatch<'a, T> {
    pub(super) lhs: Input<'a, T>,
    pub(super) rhs: Input<'a, T>,
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) count: usize,
    lhs_step: usize,
    rhs_step: usize,
    output_step: usize,
}

/// Flatten only batches whose logical iteration has one constant increment.
/// Singleton dimensions do not constrain the stride; fully broadcast inputs
/// have increment zero. Partial broadcasts/interleaving retain the fallback.
fn batch_step(dims: &[usize], strides: &[usize]) -> Option<usize> {
    let mut step = None;
    let mut span = 0usize;
    for (&dim, &stride) in dims.iter().zip(strides).rev() {
        if dim <= 1 {
            continue;
        }
        if step.is_some() && stride != span {
            return None;
        }
        step.get_or_insert(stride);
        span = stride.checked_mul(dim)?;
    }
    Some(step.unwrap_or(0))
}

impl<'a, T: LinearFloat> StridedBatch<'a, T> {
    /// The caller validated matching batch/inner dimensions, full storage
    /// spans, non-empty matrices, and the exact output length.
    pub(super) fn new(
        lhs: Input<'a, T>,
        rhs: Input<'a, T>,
        count: usize,
        reduce: bool,
    ) -> Option<Self> {
        let rank = lhs.layout.dims().len();
        let batch_dims = &lhs.layout.dims()[..rank - 2];
        let (m, k, n) = (
            lhs.layout.dims()[rank - 2],
            lhs.layout.dims()[rank - 1],
            rhs.layout.dims()[rank - 1],
        );
        let mut batch = Self {
            lhs,
            rhs,
            m,
            n,
            k,
            count,
            lhs_step: batch_step(batch_dims, &lhs.layout.stride()[..rank - 2])?,
            rhs_step: batch_step(batch_dims, &rhs.layout.stride()[..rank - 2])?,
            output_step: if reduce { 0 } else { m.checked_mul(n)? },
        };
        // A shared RHS and consecutive LHS rows are one larger matrix
        // product: stack(A_i * B) == stack(A_i) * B. This eliminates all but
        // one GEMM/GEMV call without a vendor batch extension or packing.
        // A reduction cannot share this transformation: its outputs overlap.
        if !reduce
            && batch.rhs_step == 0
            && batch.lhs_step == m.checked_mul(lhs.layout.stride()[rank - 2])?
        {
            batch.m = m.checked_mul(count)?;
            batch.count = 1;
        }
        Some(batch)
    }

    pub(super) fn lhs_offset(&self, index: usize) -> usize {
        self.lhs.layout.start_offset() + index * self.lhs_step
    }

    pub(super) fn rhs_offset(&self, index: usize) -> usize {
        self.rhs.layout.start_offset() + index * self.rhs_step
    }

    pub(super) fn output_offset(&self, index: usize) -> usize {
        index * self.output_step
    }

    #[cfg(not(feature = "blas"))]
    pub(super) fn gemm(&self, output: &mut [T], alpha: T) -> bool {
        let rank = self.lhs.layout.dims().len();
        let a = self.lhs.values.as_ptr().wrapping_add(self.lhs_offset(0));
        let b = self.rhs.values.as_ptr().wrapping_add(self.rhs_offset(0));
        let c = output.as_mut_ptr();
        // Match gemm_product's alignment requirement for every batch, including
        // odd matrix sizes and reductions sharing one output matrix.
        let element_size = std::mem::size_of::<T>();
        let pointer_bits = (a as usize) | (b as usize) | (c as usize);
        let steps = self.lhs_step | self.rhs_step | self.output_step;
        if pointer_bits & 15 != 0 || (self.count > 1 && steps & (16 / element_size - 1) != 0) {
            return false;
        }
        let (Ok(a_rs), Ok(a_cs), Ok(b_rs), Ok(b_cs), Ok(c_rs)) = (
            isize::try_from(self.lhs.layout.stride()[rank - 2]),
            isize::try_from(self.lhs.layout.stride()[rank - 1]),
            isize::try_from(self.rhs.layout.stride()[rank - 2]),
            isize::try_from(self.rhs.layout.stride()[rank - 1]),
            isize::try_from(self.n),
        ) else {
            return false;
        };
        for index in 0..self.count {
            // SAFETY: complete input spans were validated before constructing
            // this descriptor. The checked constant strides reproduce logical
            // batch order. Output is exclusive and has the exact checked length.
            unsafe {
                gemm::gemm(
                    self.m,
                    self.n,
                    self.k,
                    c.add(self.output_offset(index)),
                    1,
                    c_rs,
                    true,
                    self.lhs.values.as_ptr().add(self.lhs_offset(index)),
                    a_cs,
                    a_rs,
                    self.rhs.values.as_ptr().add(self.rhs_offset(index)),
                    b_cs,
                    b_rs,
                    T::from_f64(1.0),
                    alpha,
                    false,
                    false,
                    false,
                    gemm::Parallelism::Rayon(0),
                );
            }
        }
        true
    }

    #[cfg(feature = "blas")]
    pub(super) fn native_parameters(&self) -> Option<NativeParameters> {
        let rank = self.lhs.layout.dims().len();
        let (ta, lda) = matrix_strides(
            self.m,
            self.k,
            self.lhs.layout.stride()[rank - 2],
            self.lhs.layout.stride()[rank - 1],
        )?;
        let (tb, ldb) = matrix_strides(
            self.k,
            self.n,
            self.rhs.layout.stride()[rank - 2],
            self.rhs.layout.stride()[rank - 1],
        )?;
        let (m, n, k) = (
            CBlasInt::try_from(self.m).ok()?,
            CBlasInt::try_from(self.n).ok()?,
            CBlasInt::try_from(self.k).ok()?,
        );
        let vector_step = if self.n == 1 && self.k > 1 {
            CBlasInt::try_from(self.rhs.layout.stride()[rank - 2]).ok()?
        } else {
            1
        };
        if vector_step == 0 {
            return None;
        }
        let (vector_rows, vector_cols) = if matches!(ta, CBlasTranspose::CBlasNoTrans) {
            (m, k)
        } else {
            (k, m)
        };
        Some(NativeParameters {
            ta,
            tb,
            lda,
            ldb,
            m,
            n,
            k,
            vector_rows,
            vector_cols,
            vector_step,
        })
    }
}

#[cfg(feature = "blas")]
pub(super) struct NativeParameters {
    pub(super) ta: CBlasTranspose,
    pub(super) tb: CBlasTranspose,
    pub(super) lda: CBlasInt,
    pub(super) ldb: CBlasInt,
    pub(super) m: CBlasInt,
    pub(super) n: CBlasInt,
    pub(super) k: CBlasInt,
    pub(super) vector_rows: CBlasInt,
    pub(super) vector_cols: CBlasInt,
    pub(super) vector_step: CBlasInt,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflowing_output_is_rejected_before_allocating_or_entering_blas() {
        let lhs_layout = Layout::new((1, usize::MAX, 1).into(), vec![0, 0, 0], 0).unwrap();
        let rhs_layout = Layout::new((1, 1, 2).into(), vec![0, 0, 0], 0).unwrap();
        let lhs = Input::new(&[0.0f32], &lhs_layout).unwrap();
        let rhs = Input::new(&[0.0f32], &rhs_layout).unwrap();
        assert!(matches!(
            batched_fused(lhs, rhs, None, 1.0, 0.0, false),
            Err(Error::StorageOutOfBounds)
        ));
    }

    #[test]
    fn constant_batch_stride_requires_the_same_increment_across_carries() {
        assert_eq!(batch_step(&[2, 3], &[24, 8]), Some(8));
        assert_eq!(batch_step(&[2, 1, 3], &[24, usize::MAX, 8]), Some(8));
        assert_eq!(batch_step(&[2, 3], &[0, 0]), Some(0));
        assert_eq!(batch_step(&[1, 1], &[usize::MAX, usize::MAX]), Some(0));
        assert_eq!(batch_step(&[2, 3], &[0, 8]), None);
        assert_eq!(batch_step(&[2, 3], &[24, 0]), None);
        assert_eq!(batch_step(&[3, 2], &[8, 24]), None);
        assert_eq!(batch_step(&[2], &[usize::MAX]), None);
    }

    #[cfg(feature = "blas")]
    #[test]
    fn native_dimensions_that_exceed_the_abi_keep_the_fallback() {
        // No allocation is necessary to check descriptor metadata. This test
        // never calls a kernel or forms an out-of-bounds input pointer.
        let huge = CBlasInt::MAX as usize + 1;
        let layout = Layout::contiguous((1, huge, 1));
        let rhs_layout = Layout::contiguous((1, 1, 1));
        let batch = StridedBatch::new(
            Input {
                values: &[0.0f32],
                layout: &layout,
            },
            Input {
                values: &[0.0f32],
                layout: &rhs_layout,
            },
            1,
            false,
        )
        .unwrap();
        assert!(batch.native_parameters().is_none());
    }
}
