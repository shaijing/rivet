//! Typed fused linear algebra. All writes target final aligned storage.
use super::buffer::{AlignedBuffer, AlignedBufferBuilder};
use crate::dtype::IntoCpuStorageBuffer;
use crate::storage::validate_layout_for_storage;
use crate::{Error, Layout, Result};
use std::ops::{Add, Div, Mul, Sub};

#[cfg(feature = "blas")]
use rivet_blas_sys::cblas::prelude::*;

pub(crate) trait LinearFloat:
    IntoCpuStorageBuffer
    + Copy
    + Send
    + Sync
    + 'static
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    #[cfg(feature = "blas")]
    fn native_asum(input: Input<'_, Self>) -> Option<Self>;
    #[cfg(feature = "blas")]
    fn native_gram(input: Input<'_, Self>, output: &mut [Self], transpose: bool) -> bool;
    #[cfg(feature = "blas")]
    fn native_symmetric(
        matrix: Input<'_, Self>,
        rhs: Input<'_, Self>,
        output: &mut [Self],
        upper: bool,
    ) -> bool;
    #[cfg(feature = "blas")]
    fn native_triangular(
        matrix: Input<'_, Self>,
        output: &mut [Self],
        cols: usize,
        upper: bool,
        unit: bool,
    ) -> bool;
    #[cfg(feature = "blas")]
    fn native_product(product: &Product<'_, Self>, output: &mut [Self], alpha: Self) -> bool;
    #[cfg(feature = "blas")]
    fn native_outer(
        x: Input<'_, Self>,
        y: Input<'_, Self>,
        output: &mut [Self],
        alpha: Self,
    ) -> bool;
    #[cfg(feature = "blas")]
    fn native_axpy(output: &mut [Self], x: &[Self], alpha: Self) -> bool;
    #[cfg(feature = "blas")]
    fn native_scale(output: &mut [Self], alpha: Self) -> bool;
}

#[derive(Clone, Copy)]
pub(crate) struct Input<'a, T> {
    values: &'a [T],
    layout: &'a Layout,
}

impl<'a, T: Copy> Input<'a, T> {
    pub(crate) fn new(values: &'a [T], layout: &'a Layout) -> Result<Self> {
        validate_layout_for_storage(layout, values.len())?;
        Ok(Self { values, layout })
    }

    fn read(&self, row: usize, col: usize) -> T {
        self.values[self.layout.start_offset()
            + row * self.layout.stride()[0]
            + col * self.layout.stride()[1]]
    }

    #[cfg(feature = "blas")]
    fn ptr(&self) -> *const T {
        // Only called after non-empty dimensions and complete spans are checked.
        unsafe { self.values.as_ptr().add(self.layout.start_offset()) }
    }
}

pub(crate) struct Product<'a, T> {
    lhs: Input<'a, T>,
    rhs: Input<'a, T>,
    m: usize,
    n: usize,
    k: usize,
}

fn seed<T: LinearFloat>(
    input: Option<Input<'_, T>>,
    len: usize,
    beta: f64,
) -> Result<AlignedBuffer<T>> {
    let mut builder = AlignedBufferBuilder::new(len)?;
    let beta = T::from_f64(beta);
    if let Some(input) = input.filter(|_| beta != T::from_f64(0.0)) {
        if let Some((start, end)) = input.layout.contiguous_offsets() {
            let values = &input.values[start..end];
            if beta == T::from_f64(1.0) {
                builder.extend_from_slice(values)?;
            } else {
                for &value in values {
                    builder.write_next(beta * value)?;
                }
            }
        } else {
            for offset in input.layout.strided_index() {
                builder.write_next(beta * input.values[offset])?;
            }
        }
    } else {
        for _ in 0..len {
            builder.write_next(T::from_f64(0.0))?;
        }
    }
    builder.finish()
}

pub(crate) fn addmm<T: LinearFloat>(
    lhs: Input<'_, T>,
    rhs: Input<'_, T>,
    add: Option<Input<'_, T>>,
    alpha: f64,
    beta: f64,
) -> Result<AlignedBuffer<T>> {
    let [m, k] = *lhs.layout.dims() else {
        return Err(Error::InvalidRank {
            expected: 2,
            actual: lhs.layout.dims().len(),
        });
    };
    let [rhs_k, n] = *rhs.layout.dims() else {
        return Err(Error::InvalidRank {
            expected: 2,
            actual: rhs.layout.dims().len(),
        });
    };
    if k != rhs_k || add.is_some_and(|add| add.layout.dims() != [m, n]) {
        return Err(Error::MatmulShapeMismatch {
            lhs: lhs.layout.dims().to_vec(),
            rhs: rhs.layout.dims().to_vec(),
        });
    }
    let mut output = seed(
        add,
        m.checked_mul(n).ok_or(Error::StorageOutOfBounds)?,
        beta,
    )?;
    if m == 0 || n == 0 || k == 0 || alpha == 0.0 {
        return Ok(output);
    }
    let product = Product { lhs, rhs, m, n, k };
    multiply(&product, output.as_mut_slice(), T::from_f64(alpha))?;
    Ok(output)
}

pub(crate) fn addmv<T: LinearFloat>(
    matrix: Input<'_, T>,
    vector: Input<'_, T>,
    add: Input<'_, T>,
    alpha: f64,
    beta: f64,
) -> Result<AlignedBuffer<T>> {
    let [k] = *vector.layout.dims() else {
        return Err(Error::InvalidRank {
            expected: 1,
            actual: vector.layout.dims().len(),
        });
    };
    let [m] = *add.layout.dims() else {
        return Err(Error::InvalidRank {
            expected: 1,
            actual: add.layout.dims().len(),
        });
    };
    let rhs_layout = Layout::new(
        (k, 1).into(),
        vec![vector.layout.stride()[0], 1],
        vector.layout.start_offset(),
    )?;
    let add_layout = Layout::new(
        (m, 1).into(),
        vec![add.layout.stride()[0], 1],
        add.layout.start_offset(),
    )?;
    addmm(
        matrix,
        Input::new(vector.values, &rhs_layout)?,
        Some(Input::new(add.values, &add_layout)?),
        alpha,
        beta,
    )
}

fn multiply<T: LinearFloat>(product: &Product<'_, T>, output: &mut [T], alpha: T) -> Result<()> {
    let Product { lhs, rhs, m, n, k } = *product;
    if m == 0 || n == 0 || k == 0 || alpha == T::from_f64(0.0) {
        return Ok(());
    }
    #[cfg(feature = "blas")]
    if T::native_product(product, output, alpha) {
        return Ok(());
    }
    #[cfg(not(feature = "blas"))]
    if gemm_product(product, output, alpha)? {
        return Ok(());
    }
    for row in 0..m {
        for col in 0..n {
            let mut sum = T::from_f64(0.0);
            for inner in 0..k {
                sum = sum + lhs.read(row, inner) * rhs.read(inner, col);
            }
            let offset = row * n + col;
            output[offset] = output[offset] + alpha * sum;
        }
    }
    Ok(())
}

#[cfg(not(feature = "blas"))]
fn gemm_product<T: LinearFloat>(
    product: &Product<'_, T>,
    output: &mut [T],
    alpha: T,
) -> Result<bool> {
    let Product { lhs, rhs, m, n, k } = *product;
    let lhs_ptr = unsafe { lhs.values.as_ptr().add(lhs.layout.start_offset()) };
    let rhs_ptr = unsafe { rhs.values.as_ptr().add(rhs.layout.start_offset()) };
    let pointer_bits = (lhs_ptr as usize) | (rhs_ptr as usize) | (output.as_ptr() as usize);
    if pointer_bits & 15 != 0 {
        return Ok(false);
    }
    let (Ok(lhs_cs), Ok(lhs_rs), Ok(rhs_cs), Ok(rhs_rs), Ok(output_rs)) = (
        isize::try_from(lhs.layout.stride()[1]),
        isize::try_from(lhs.layout.stride()[0]),
        isize::try_from(rhs.layout.stride()[1]),
        isize::try_from(rhs.layout.stride()[0]),
        isize::try_from(n),
    ) else {
        return Ok(false);
    };
    // SAFETY: Input::new validated every addressed element; the caller checked
    // shapes, non-empty dimensions, and exact output size. Pointers meet the
    // same alignment requirement as Rivet's original GEMM path.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            output.as_mut_ptr(),
            1,
            output_rs,
            true,
            lhs_ptr,
            lhs_cs,
            lhs_rs,
            rhs_ptr,
            rhs_cs,
            rhs_rs,
            T::from_f64(1.0),
            alpha,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(0),
        );
    }
    Ok(true)
}

pub(crate) fn outer<T: LinearFloat>(
    x: Input<'_, T>,
    y: Input<'_, T>,
    add: Option<Input<'_, T>>,
    alpha: f64,
    beta: f64,
) -> Result<AlignedBuffer<T>> {
    let [m] = *x.layout.dims() else {
        return Err(Error::InvalidRank {
            expected: 1,
            actual: x.layout.dims().len(),
        });
    };
    let [n] = *y.layout.dims() else {
        return Err(Error::InvalidRank {
            expected: 1,
            actual: y.layout.dims().len(),
        });
    };
    if let Some(add) = add
        && add.layout.dims() != [m, n]
    {
        return Err(Error::ShapeMismatchBinary {
            lhs: add.layout.dims().to_vec(),
            rhs: vec![m, n],
        });
    }
    let mut output = seed(
        add,
        m.checked_mul(n).ok_or(Error::StorageOutOfBounds)?,
        beta,
    )?;
    if m == 0 || n == 0 || alpha == 0.0 {
        return Ok(output);
    }
    let alpha = T::from_f64(alpha);
    if alpha == T::from_f64(0.0) {
        return Ok(output);
    }
    #[cfg(feature = "blas")]
    if T::native_outer(x, y, output.as_mut_slice(), alpha) {
        return Ok(output);
    }
    for row in 0..m {
        let value = alpha * x.values[x.layout.start_offset() + row * x.layout.stride()[0]];
        for col in 0..n {
            let index = row * n + col;
            output.as_mut_slice()[index] = output.as_slice()[index]
                + value * y.values[y.layout.start_offset() + col * y.layout.stride()[0]];
        }
    }
    Ok(output)
}

pub(crate) fn axpy<T: LinearFloat>(output: &mut [T], x: Input<'_, T>, alpha: f64) {
    if output.is_empty() || alpha == 0.0 {
        return;
    }
    let alpha = T::from_f64(alpha);
    if alpha == T::from_f64(0.0) {
        return;
    }
    #[cfg(feature = "blas")]
    if x.layout.is_contiguous() {
        let start = x.layout.start_offset();
        if T::native_axpy(output, &x.values[start..start + output.len()], alpha) {
            return;
        }
    }
    for (dst, offset) in output.iter_mut().zip(x.layout.strided_index()) {
        *dst = *dst + alpha * x.values[offset];
    }
}

pub(crate) fn scale<T: LinearFloat>(output: &mut [T], alpha: f64) {
    if output.is_empty() {
        return;
    }
    let alpha = T::from_f64(alpha);
    if alpha == T::from_f64(0.0) {
        output.fill(T::from_f64(0.0));
        return;
    }
    #[cfg(feature = "blas")]
    if T::native_scale(output, alpha) {
        return;
    }
    for value in output {
        *value = *value * alpha
    }
}

pub(crate) fn norm_l1<T: LinearFloat>(input: Input<'_, T>) -> Result<AlignedBuffer<T>> {
    #[cfg(feature = "blas")]
    if let Some(value) = T::native_asum(input) {
        return AlignedBuffer::from_slice(&[value]);
    }
    let mut sum = 0.0;
    for offset in input.layout.strided_index() {
        sum += input.values[offset].to_f64().abs();
    }
    AlignedBuffer::from_slice(&[T::from_f64(sum)])
}

pub(crate) fn gram<T: LinearFloat>(
    input: Input<'_, T>,
    transpose: bool,
) -> Result<AlignedBuffer<T>> {
    let [rows, cols] = *input.layout.dims() else {
        return Err(Error::InvalidRank {
            expected: 2,
            actual: input.layout.dims().len(),
        });
    };
    let n = if transpose { cols } else { rows };
    let k = if transpose { rows } else { cols };
    let mut output = seed(
        None,
        n.checked_mul(n).ok_or(Error::StorageOutOfBounds)?,
        0.0,
    )?;
    if n == 0 || k == 0 {
        return Ok(output);
    }
    #[cfg(feature = "blas")]
    let native = T::native_gram(input, output.as_mut_slice(), transpose);
    #[cfg(not(feature = "blas"))]
    let native = false;
    if !native {
        let transposed = input.layout.transpose(0, 1)?;
        let transposed = Input::new(input.values, &transposed)?;
        let (lhs, rhs) = if transpose {
            (transposed, input)
        } else {
            (input, transposed)
        };
        multiply(
            &Product {
                lhs,
                rhs,
                m: n,
                n,
                k,
            },
            output.as_mut_slice(),
            T::from_f64(1.0),
        )?;
    }
    // SYRK writes only the requested triangle. Publish a complete symmetric
    // tensor, also enforcing the same symmetry contract on fallback kernels.
    for row in 0..n {
        for col in 0..row {
            output.as_mut_slice()[row * n + col] = output.as_slice()[col * n + row];
        }
    }
    Ok(output)
}

fn square_rhs<T: LinearFloat>(matrix: Input<'_, T>, rhs: Input<'_, T>) -> Result<(usize, usize)> {
    let [rows, cols] = *matrix.layout.dims() else {
        return Err(Error::InvalidRank {
            expected: 2,
            actual: matrix.layout.dims().len(),
        });
    };
    let [rhs_rows, rhs_cols] = *rhs.layout.dims() else {
        return Err(Error::InvalidRank {
            expected: 2,
            actual: rhs.layout.dims().len(),
        });
    };
    if rows != cols || rows != rhs_rows {
        return Err(Error::MatmulShapeMismatch {
            lhs: matrix.layout.dims().to_vec(),
            rhs: rhs.layout.dims().to_vec(),
        });
    }
    Ok((rows, rhs_cols))
}

pub(crate) fn symmetric<T: LinearFloat>(
    matrix: Input<'_, T>,
    rhs: Input<'_, T>,
    upper: bool,
) -> Result<AlignedBuffer<T>> {
    let (n, cols) = square_rhs(matrix, rhs)?;
    let mut output = seed(
        None,
        n.checked_mul(cols).ok_or(Error::StorageOutOfBounds)?,
        0.0,
    )?;
    if n == 0 || cols == 0 {
        return Ok(output);
    }
    #[cfg(feature = "blas")]
    if T::native_symmetric(matrix, rhs, output.as_mut_slice(), upper) {
        return Ok(output);
    }
    for row in 0..n {
        for col in 0..cols {
            let mut sum = T::from_f64(0.0);
            for inner in 0..n {
                let value = if (upper && row <= inner) || (!upper && row >= inner) {
                    matrix.read(row, inner)
                } else {
                    matrix.read(inner, row)
                };
                sum = sum + value * rhs.read(inner, col);
            }
            output.as_mut_slice()[row * cols + col] = sum;
        }
    }
    Ok(output)
}

pub(crate) fn triangular<T: LinearFloat>(
    matrix: Input<'_, T>,
    rhs: Input<'_, T>,
    upper: bool,
    unit: bool,
) -> Result<AlignedBuffer<T>> {
    let (n, cols) = square_rhs(matrix, rhs)?;
    if !unit {
        for row in 0..n {
            if matrix.read(row, row) == T::from_f64(0.0) {
                return Err(Error::SingularMatrix { index: row });
            }
        }
    }
    let mut output = seed(
        Some(rhs),
        n.checked_mul(cols).ok_or(Error::StorageOutOfBounds)?,
        1.0,
    )?;
    if n == 0 || cols == 0 {
        return Ok(output);
    }
    #[cfg(feature = "blas")]
    if T::native_triangular(matrix, output.as_mut_slice(), cols, upper, unit) {
        return Ok(output);
    }
    for step in 0..n {
        let row = if upper { n - 1 - step } else { step };
        let range = if upper { row + 1..n } else { 0..row };
        for col in 0..cols {
            let mut value = output.as_slice()[row * cols + col];
            for inner in range.clone() {
                value = value - matrix.read(row, inner) * output.as_slice()[inner * cols + col];
            }
            if !unit {
                value = value / matrix.read(row, row)
            }
            output.as_mut_slice()[row * cols + col] = value;
        }
    }
    Ok(output)
}

pub(crate) fn batched<T: LinearFloat>(
    lhs: Input<'_, T>,
    rhs: Input<'_, T>,
) -> Result<AlignedBuffer<T>> {
    let rank = lhs.layout.dims().len();
    if rank < 2 || rhs.layout.dims().len() != rank {
        return Err(Error::MatmulShapeMismatch {
            lhs: lhs.layout.dims().to_vec(),
            rhs: rhs.layout.dims().to_vec(),
        });
    }
    let batch_dims = &lhs.layout.dims()[..rank - 2];
    let (m, k, n) = (
        lhs.layout.dims()[rank - 2],
        lhs.layout.dims()[rank - 1],
        rhs.layout.dims()[rank - 1],
    );
    if rhs.layout.dims()[..rank - 2] != *batch_dims || rhs.layout.dims()[rank - 2] != k {
        return Err(Error::MatmulShapeMismatch {
            lhs: lhs.layout.dims().to_vec(),
            rhs: rhs.layout.dims().to_vec(),
        });
    }
    let count = crate::Shape::from(batch_dims.to_vec()).checked_elem_count()?;
    let matrix_len = m.checked_mul(n).ok_or(Error::StorageOutOfBounds)?;
    let mut output = seed(
        None,
        count
            .checked_mul(matrix_len)
            .ok_or(Error::StorageOutOfBounds)?,
        0.0,
    )?;
    if count == 0 || m == 0 || n == 0 || k == 0 {
        return Ok(output);
    }
    let lhs_batch = Layout::new(
        batch_dims.to_vec().into(),
        lhs.layout.stride()[..rank - 2].to_vec(),
        lhs.layout.start_offset(),
    )?;
    let rhs_batch = Layout::new(
        batch_dims.to_vec().into(),
        rhs.layout.stride()[..rank - 2].to_vec(),
        rhs.layout.start_offset(),
    )?;
    for (batch, (lhs_offset, rhs_offset)) in lhs_batch
        .strided_index()
        .zip(rhs_batch.strided_index())
        .enumerate()
    {
        let lhs_layout = Layout::new(
            (m, k).into(),
            lhs.layout.stride()[rank - 2..].to_vec(),
            lhs_offset,
        )?;
        let rhs_layout = Layout::new(
            (k, n).into(),
            rhs.layout.stride()[rank - 2..].to_vec(),
            rhs_offset,
        )?;
        let lhs = Input::new(lhs.values, &lhs_layout)?;
        let rhs = Input::new(rhs.values, &rhs_layout)?;
        multiply(
            &Product { lhs, rhs, m, n, k },
            &mut output.as_mut_slice()[batch * matrix_len..(batch + 1) * matrix_len],
            T::from_f64(1.0),
        )?;
    }
    Ok(output)
}

#[cfg(feature = "blas")]
fn matrix(layout: &Layout) -> Option<(CBlasTranspose, CBlasInt)> {
    let [rows, cols] = *layout.dims() else {
        return None;
    };
    let [rs, cs] = *layout.stride() else {
        return None;
    };
    if (cols <= 1 || cs == 1) && (rows <= 1 || rs >= cols) {
        Some((
            CBlasTranspose::CBlasNoTrans,
            CBlasInt::try_from(if rows <= 1 { cols } else { rs }).ok()?,
        ))
    } else if (rows <= 1 || rs == 1) && (cols <= 1 || cs >= rows) {
        Some((
            CBlasTranspose::CBlasTrans,
            CBlasInt::try_from(if cols <= 1 { rows } else { cs }).ok()?,
        ))
    } else {
        None
    }
}

macro_rules! impl_float {
    ($ty:ty, $gemm:ident, $gemv:ident, $ger:ident, $axpy:ident, $scal:ident, $asum:ident, $syrk:ident, $symm:ident, $symv:ident, $trsm:ident) => {
        impl LinearFloat for $ty {
            fn from_f64(value: f64) -> Self {
                value as Self
            }
            fn to_f64(self) -> f64 {
                self as f64
            }

            #[cfg(feature = "blas")]
            fn native_asum(input: Input<'_, Self>) -> Option<Self> {
                let len = input.layout.checked_elem_count().ok()?;
                if len == 0 {
                    return Some(0.0);
                }
                let step = if len == 1 || input.layout.is_contiguous() {
                    1
                } else if input.layout.dims().len() == 1 && input.layout.stride()[0] > 0 {
                    input.layout.stride()[0]
                } else {
                    return None;
                };
                let (len, step) = (
                    CBlasInt::try_from(len).ok()?,
                    CBlasInt::try_from(step).ok()?,
                );
                // SAFETY: Input::new validated the whole span; increments
                // positive and both integers fit the native ABI.
                Some(unsafe { $asum(len, input.ptr(), step) })
            }

            #[cfg(feature = "blas")]
            fn native_gram(input: Input<'_, Self>, output: &mut [Self], transpose: bool) -> bool {
                let Some((stored_transpose, lda)) = matrix(input.layout) else {
                    return false;
                };
                let rows = input.layout.dims()[0];
                let cols = input.layout.dims()[1];
                let (n, k) = if transpose {
                    (cols, rows)
                } else {
                    (rows, cols)
                };
                let (Ok(n), Ok(k)) = (CBlasInt::try_from(n), CBlasInt::try_from(k)) else {
                    return false;
                };
                let trans = if (matches!(stored_transpose, CBlasTranspose::CBlasTrans)) ^ transpose
                {
                    CBlasTranspose::CBlasTrans
                } else {
                    CBlasTranspose::CBlasNoTrans
                };
                // SAFETY: matrix metadata, complete storage span, and output
                // size checked; the caller handles empty dimensions.
                unsafe {
                    $syrk(
                        CBlasLayout::CBlasRowMajor,
                        CBlasUplo::CblasUpper,
                        trans,
                        n,
                        k,
                        1.0,
                        input.ptr(),
                        lda,
                        0.0,
                        output.as_mut_ptr(),
                        n,
                    )
                }
                true
            }

            #[cfg(feature = "blas")]
            fn native_symmetric(
                a: Input<'_, Self>,
                b: Input<'_, Self>,
                output: &mut [Self],
                upper: bool,
            ) -> bool {
                let Some((ta, lda)) = matrix(a.layout) else {
                    return false;
                };
                let Some((tb, ldb)) = matrix(b.layout) else {
                    return false;
                };
                let n = a.layout.dims()[0];
                let cols = b.layout.dims()[1];
                let (Ok(n), Ok(cols)) = (CBlasInt::try_from(n), CBlasInt::try_from(cols)) else {
                    return false;
                };
                let uplo = if upper ^ (matches!(ta, CBlasTranspose::CBlasTrans)) {
                    CBlasUplo::CblasUpper
                } else {
                    CBlasUplo::CblasLower
                };
                if cols == 1 {
                    let step = if n == 1 { 1 } else { b.layout.stride()[0] };
                    let Ok(step) = CBlasInt::try_from(step) else {
                        return false;
                    };
                    if step == 0 {
                        return false;
                    }
                    // SAFETY: square symmetric matrix, selected triangle,
                    // positive-stride input vector, and fresh output checked.
                    unsafe {
                        $symv(
                            CBlasLayout::CBlasRowMajor,
                            uplo,
                            n,
                            1.0,
                            a.ptr(),
                            lda,
                            b.ptr(),
                            step,
                            0.0,
                            output.as_mut_ptr(),
                            1,
                        )
                    }
                    return true;
                }
                if !matches!(tb, CBlasTranspose::CBlasNoTrans) {
                    return false;
                }
                // SAFETY: validated square A, row-major B and output spans,
                // leading dimensions, and checked ABI sizes.
                unsafe {
                    $symm(
                        CBlasLayout::CBlasRowMajor,
                        CBlasSide::CblasLeft,
                        uplo,
                        n,
                        cols,
                        1.0,
                        a.ptr(),
                        lda,
                        b.ptr(),
                        ldb,
                        0.0,
                        output.as_mut_ptr(),
                        cols,
                    )
                }
                true
            }

            #[cfg(feature = "blas")]
            fn native_triangular(
                a: Input<'_, Self>,
                output: &mut [Self],
                cols: usize,
                upper: bool,
                unit: bool,
            ) -> bool {
                let Some((ta, lda)) = matrix(a.layout) else {
                    return false;
                };
                let (Ok(n), Ok(cols)) = (
                    CBlasInt::try_from(a.layout.dims()[0]),
                    CBlasInt::try_from(cols),
                ) else {
                    return false;
                };
                let uplo = if upper ^ (matches!(ta, CBlasTranspose::CBlasTrans)) {
                    CBlasUplo::CblasUpper
                } else {
                    CBlasUplo::CblasLower
                };
                let diagonal = if unit {
                    CBlasDiag::CblasUnit
                } else {
                    CBlasDiag::CblasNonUnit
                };
                // SAFETY: square triangular A, checked non-zero diagonal
                // unless unit, validated spans, and exclusive fresh output.
                unsafe {
                    $trsm(
                        CBlasLayout::CBlasRowMajor,
                        CBlasSide::CblasLeft,
                        uplo,
                        ta,
                        diagonal,
                        n,
                        cols,
                        1.0,
                        a.ptr(),
                        lda,
                        output.as_mut_ptr(),
                        cols,
                    )
                }
                true
            }

            #[cfg(feature = "blas")]
            fn native_product(
                product: &Product<'_, Self>,
                output: &mut [Self],
                alpha: Self,
            ) -> bool {
                let Product { lhs, rhs, m, n, k } = *product;
                let (Some((ta, lda)), Some((tb, ldb)), Ok(m), Ok(n), Ok(k)) = (
                    matrix(lhs.layout),
                    matrix(rhs.layout),
                    CBlasInt::try_from(m),
                    CBlasInt::try_from(n),
                    CBlasInt::try_from(k),
                ) else {
                    return false;
                };
                if n == 1 {
                    let step = if k == 1 { 1 } else { rhs.layout.stride()[0] };
                    let Ok(step) = CBlasInt::try_from(step) else {
                        return false;
                    };
                    if step == 0 {
                        return false;
                    }
                    let (rows, cols) = if matches!(ta, CBlasTranspose::CBlasNoTrans) {
                        (m, k)
                    } else {
                        (k, m)
                    };
                    // SAFETY: validated matrix span, positive vector stride,
                    // checked ABI sizes, and exactly m output elements.
                    unsafe {
                        $gemv(
                            CBlasLayout::CBlasRowMajor,
                            ta,
                            rows,
                            cols,
                            alpha,
                            lhs.ptr(),
                            lda,
                            rhs.ptr(),
                            step,
                            1.0,
                            output.as_mut_ptr(),
                            1,
                        )
                    }
                    return true;
                }
                // SAFETY: checked input spans and shapes, non-zero dimensions,
                // valid leading dimensions, fresh output, and checked ABI sizes.
                unsafe {
                    $gemm(
                        CBlasLayout::CBlasRowMajor,
                        ta,
                        tb,
                        m,
                        n,
                        k,
                        alpha,
                        lhs.ptr(),
                        lda,
                        rhs.ptr(),
                        ldb,
                        1.0,
                        output.as_mut_ptr(),
                        n,
                    );
                }
                true
            }

            #[cfg(feature = "blas")]
            fn native_outer(
                x: Input<'_, Self>,
                y: Input<'_, Self>,
                output: &mut [Self],
                alpha: Self,
            ) -> bool {
                let m = x.layout.dims()[0];
                let n = y.layout.dims()[0];
                let xs = if m <= 1 { 1 } else { x.layout.stride()[0] };
                let ys = if n <= 1 { 1 } else { y.layout.stride()[0] };
                if xs == 0 || ys == 0 {
                    return false;
                }
                let (Ok(m), Ok(n), Ok(xs), Ok(ys)) = (
                    CBlasInt::try_from(m),
                    CBlasInt::try_from(n),
                    CBlasInt::try_from(xs),
                    CBlasInt::try_from(ys),
                ) else {
                    return false;
                };
                // SAFETY: rank-1 input spans validated, dimensions positive,
                // increments positive, output fresh and exactly m*n elements.
                unsafe {
                    $ger(
                        CBlasLayout::CBlasRowMajor,
                        m,
                        n,
                        alpha,
                        x.ptr(),
                        xs,
                        y.ptr(),
                        ys,
                        output.as_mut_ptr(),
                        n,
                    )
                }
                true
            }

            #[cfg(feature = "blas")]
            fn native_axpy(output: &mut [Self], x: &[Self], alpha: Self) -> bool {
                let Ok(n) = CBlasInt::try_from(output.len()) else {
                    return false;
                };
                // SAFETY: the caller provides equal-length contiguous slices;
                // exclusive destination ownership rules out input aliasing.
                unsafe { $axpy(n, alpha, x.as_ptr(), 1, output.as_mut_ptr(), 1) }
                true
            }

            #[cfg(feature = "blas")]
            fn native_scale(output: &mut [Self], alpha: Self) -> bool {
                let Ok(n) = CBlasInt::try_from(output.len()) else {
                    return false;
                };
                // SAFETY: the entire mutable contiguous slice is owned.
                unsafe { $scal(n, alpha, output.as_mut_ptr(), 1) }
                true
            }
        }
    };
}

impl_float!(
    f32,
    cblas_sgemm,
    cblas_sgemv,
    cblas_sger,
    cblas_saxpy,
    cblas_sscal,
    cblas_sasum,
    cblas_ssyrk,
    cblas_ssymm,
    cblas_ssymv,
    cblas_strsm
);
impl_float!(
    f64,
    cblas_dgemm,
    cblas_dgemv,
    cblas_dger,
    cblas_daxpy,
    cblas_dscal,
    cblas_dasum,
    cblas_dsyrk,
    cblas_dsymm,
    cblas_dsymv,
    cblas_dtrsm
);
