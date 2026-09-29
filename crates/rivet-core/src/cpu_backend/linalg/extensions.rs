//! Additional typed kernels; inputs are borrowed and outputs stay aligned.
use super::*;
use crate::{MatrixSide, TriangularOptions};

#[derive(Clone, Copy)]
pub(crate) enum RankUpdate {
    One,
    Two,
    Gram,
    TwoK,
}

#[derive(Clone, Copy)]
pub(crate) struct RankOptions {
    pub kind: RankUpdate,
    pub transpose: bool,
    pub upper: bool,
    pub alpha: f64,
    pub beta: f64,
}

pub(crate) trait ExtendedFloat: LinearFloat {
    #[cfg(feature = "blas")]
    fn native_rank(
        output: &mut [Self],
        a: Input<'_, Self>,
        b: Option<Input<'_, Self>>,
        kind: RankUpdate,
        transpose: bool,
        alpha: Self,
    ) -> bool;
    #[cfg(feature = "blas")]
    fn native_structured(
        output: &mut [Self],
        a: Input<'_, Self>,
        b: Input<'_, Self>,
        options: TriangularOptions,
        solve: bool,
        symmetric: bool,
    ) -> bool;
    #[cfg(feature = "blas")]
    fn native_axpby(output: &mut [Self], x: Input<'_, Self>, alpha: Self, beta: Self) -> bool;
    #[cfg(feature = "blas")]
    fn native_copy(output: &mut [Self], x: Input<'_, Self>, transpose: bool, alpha: Self) -> bool;
    #[cfg(feature = "blas")]
    fn native_argmax(x: Input<'_, Self>) -> Option<usize>;
    #[cfg(feature = "blas")]
    fn native_rotate(x: &mut [Self], y: &mut [Self], c: Self, s: Self) -> bool;
    #[cfg(feature = "blas")]
    fn native_swap(x: &mut [Self], y: &mut [Self]) -> bool;
}

#[cfg(feature = "blas")]
pub(super) fn vector<T: Copy>(x: Input<'_, T>) -> Option<(CBlasInt, CBlasInt)> {
    let n = x.layout.checked_elem_count().ok()?;
    if n == 0 {
        return None;
    }
    let inc = if x.layout.is_contiguous() || n <= 1 {
        1
    } else if x.layout.dims().len() == 1 {
        x.layout.stride()[0]
    } else {
        return None;
    };
    if inc == 0 {
        return None;
    }
    Some((CBlasInt::try_from(n).ok()?, CBlasInt::try_from(inc).ok()?))
}

fn mismatch<T>(a: Input<'_, T>, dims: &[usize]) -> Error {
    Error::ShapeMismatchBinary {
        lhs: a.layout.dims().to_vec(),
        rhs: dims.to_vec(),
    }
}

fn mirror<T: Copy>(out: &mut [T], n: usize) {
    for row in 0..n {
        for col in 0..row {
            out[row * n + col] = out[col * n + row];
        }
    }
}

pub(crate) fn rank_update<T: ExtendedFloat>(
    c: Input<'_, T>,
    a: Input<'_, T>,
    b: Option<Input<'_, T>>,
    options: RankOptions,
) -> Result<AlignedBuffer<T>> {
    let RankOptions {
        kind,
        transpose,
        upper,
        alpha,
        beta,
    } = options;
    let (n, k) = match kind {
        RankUpdate::One | RankUpdate::Two => {
            let [n] = *a.layout.dims() else {
                return Err(Error::InvalidRank {
                    expected: 1,
                    actual: a.layout.dims().len(),
                });
            };
            if let Some(b) = b
                && b.layout.dims() != [n]
            {
                return Err(mismatch(b, &[n]));
            }
            (n, 1)
        }
        RankUpdate::Gram | RankUpdate::TwoK => {
            let [rows, cols] = *a.layout.dims() else {
                return Err(Error::InvalidRank {
                    expected: 2,
                    actual: a.layout.dims().len(),
                });
            };
            if let Some(b) = b
                && b.layout.dims() != [rows, cols]
            {
                return Err(mismatch(b, &[rows, cols]));
            }
            if transpose {
                (cols, rows)
            } else {
                (rows, cols)
            }
        }
    };
    if c.layout.dims() != [n, n] {
        return Err(mismatch(c, &[n, n]));
    }
    let beta = T::from_f64(beta);
    let alpha = T::from_f64(alpha);
    let len = n.checked_mul(n).ok_or(Error::StorageOutOfBounds)?;
    let mut out = seed(None, len, 0.0)?;
    // Read only the caller's chosen triangle, including when the other half
    // contains NaNs. beta=0 ignores all seed values.
    if beta != T::from_f64(0.0) {
        for row in 0..n {
            for col in row..n {
                out.as_mut_slice()[row * n + col] = beta
                    * if upper {
                        c.read(row, col)
                    } else {
                        c.read(col, row)
                    };
            }
        }
    }
    mirror(out.as_mut_slice(), n);
    if n == 0 || k == 0 || alpha == T::from_f64(0.0) {
        return Ok(out);
    }
    #[cfg(feature = "blas")]
    if T::native_rank(out.as_mut_slice(), a, b, kind, transpose, alpha) {
        mirror(out.as_mut_slice(), n);
        return Ok(out);
    }
    match kind {
        RankUpdate::One | RankUpdate::Two => {
            for row in 0..n {
                for col in row..n {
                    let ar = a.values[a.layout.start_offset() + row * a.layout.stride()[0]];
                    let ac = a.values[a.layout.start_offset() + col * a.layout.stride()[0]];
                    let product = if let Some(b) = b {
                        let br = b.values[b.layout.start_offset() + row * b.layout.stride()[0]];
                        let bc = b.values[b.layout.start_offset() + col * b.layout.stride()[0]];
                        ar * bc + br * ac
                    } else {
                        ar * ac
                    };
                    let index = row * n + col;
                    out.as_mut_slice()[index] = out.as_slice()[index] + alpha * product;
                }
            }
        }
        RankUpdate::Gram | RankUpdate::TwoK => {
            let other = b.unwrap_or(a);
            let at = a.layout.transpose(0, 1)?;
            let bt = other.layout.transpose(0, 1)?;
            let (lhs, rhs) = if transpose {
                (Input::new(a.values, &at)?, other)
            } else {
                (a, Input::new(other.values, &bt)?)
            };
            multiply(
                &Product {
                    lhs,
                    rhs,
                    m: n,
                    n,
                    k,
                },
                out.as_mut_slice(),
                alpha,
            )?;
            if matches!(kind, RankUpdate::TwoK) {
                let (lhs, rhs) = if transpose {
                    (Input::new(other.values, &bt)?, a)
                } else {
                    (other, Input::new(a.values, &at)?)
                };
                multiply(
                    &Product {
                        lhs,
                        rhs,
                        m: n,
                        n,
                        k,
                    },
                    out.as_mut_slice(),
                    alpha,
                )?;
            }
        }
    }
    mirror(out.as_mut_slice(), n);
    Ok(out)
}

fn structured_shape<T: Copy>(
    a: Input<'_, T>,
    b: Input<'_, T>,
    side: MatrixSide,
) -> Result<(usize, usize, usize)> {
    let [rows, cols] = *b.layout.dims() else {
        return Err(Error::InvalidRank {
            expected: 2,
            actual: b.layout.dims().len(),
        });
    };
    let n = if side == MatrixSide::Left { rows } else { cols };
    if a.layout.dims() != [n, n] {
        return Err(mismatch(a, &[n, n]));
    }
    Ok((rows, cols, n))
}

pub(crate) fn structured<T: ExtendedFloat>(
    a: Input<'_, T>,
    b: Input<'_, T>,
    options: TriangularOptions,
    solve: bool,
    symmetric: bool,
) -> Result<AlignedBuffer<T>> {
    let (rows, cols, n) = structured_shape(a, b, options.side)?;
    let count = rows.checked_mul(cols).ok_or(Error::StorageOutOfBounds)?;
    if solve && !options.unit_diagonal {
        for i in 0..n {
            if a.read(i, i) == T::from_f64(0.0) {
                return Err(Error::SingularMatrix { index: i });
            }
        }
    }
    if solve && options.side == MatrixSide::Left {
        let layout = if options.transpose {
            a.layout.transpose(0, 1)?
        } else {
            a.layout.clone()
        };
        return triangular(
            Input::new(a.values, &layout)?,
            b,
            options.upper ^ options.transpose,
            options.unit_diagonal,
        );
    }
    let mut out = seed(if symmetric { None } else { Some(b) }, count, 1.0)?;
    if count == 0 {
        return Ok(out);
    }
    #[cfg(feature = "blas")]
    if T::native_structured(out.as_mut_slice(), a, b, options, solve, symmetric) {
        return Ok(out);
    }
    let upper = options.upper ^ options.transpose;
    let read = |r, c| {
        if options.transpose {
            a.read(c, r)
        } else {
            a.read(r, c)
        }
    };
    if solve {
        // right-side solve X*op(A)=B
        for step in 0..n {
            let col = if upper { step } else { n - 1 - step };
            let range = if upper { 0..col } else { col + 1..n };
            for row in 0..rows {
                let mut value = out.as_slice()[row * cols + col];
                for i in range.clone() {
                    value = value - out.as_slice()[row * cols + i] * read(i, col);
                }
                if !options.unit_diagonal {
                    value = value / read(col, col);
                }
                out.as_mut_slice()[row * cols + col] = value;
            }
        }
        return Ok(out);
    }
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = T::from_f64(0.0);
            for i in 0..n {
                let (r, c) = if options.side == MatrixSide::Left {
                    (row, i)
                } else {
                    (i, col)
                };
                // Skip the unused triangle entirely: zero times NaN is not zero.
                if !symmetric && ((upper && r > c) || (!upper && r < c)) {
                    continue;
                }
                let value = if symmetric {
                    if (options.upper && r <= c) || (!options.upper && r >= c) {
                        a.read(r, c)
                    } else {
                        a.read(c, r)
                    }
                } else if options.unit_diagonal && r == c {
                    T::from_f64(1.0)
                } else {
                    read(r, c)
                };
                let rhs = if options.side == MatrixSide::Left {
                    b.read(i, col)
                } else {
                    b.read(row, i)
                };
                sum = sum + value * rhs;
            }
            out.as_mut_slice()[row * cols + col] = sum;
        }
    }
    Ok(out)
}

pub(crate) fn matmul_into<T: LinearFloat>(
    out: &mut [T],
    dims: &[usize],
    lhs: Input<'_, T>,
    rhs: Input<'_, T>,
    alpha: f64,
    beta: f64,
) -> Result<()> {
    let ([m, k], [rk, n]) = (lhs.layout.dims(), rhs.layout.dims()) else {
        return Err(Error::MatmulShapeMismatch {
            lhs: lhs.layout.dims().to_vec(),
            rhs: rhs.layout.dims().to_vec(),
        });
    };
    if k != rk || dims != [*m, *n] {
        return Err(Error::MatmulShapeMismatch {
            lhs: lhs.layout.dims().to_vec(),
            rhs: rhs.layout.dims().to_vec(),
        });
    }
    let expected = m.checked_mul(*n).ok_or(Error::StorageOutOfBounds)?;
    if out.len() != expected {
        return Err(Error::StorageOutOfBounds);
    }
    multiply_scaled(
        &Product {
            lhs,
            rhs,
            m: *m,
            n: *n,
            k: *k,
        },
        out,
        T::from_f64(alpha),
        T::from_f64(beta),
    )
}

pub(crate) fn outer_into<T: LinearFloat>(
    out: &mut [T],
    dims: &[usize],
    x: Input<'_, T>,
    y: Input<'_, T>,
    alpha: f64,
    beta: f64,
) -> Result<()> {
    let [m] = x.layout.dims() else {
        return Err(Error::InvalidRank {
            expected: 1,
            actual: x.layout.dims().len(),
        });
    };
    let [n] = y.layout.dims() else {
        return Err(Error::InvalidRank {
            expected: 1,
            actual: y.layout.dims().len(),
        });
    };
    if dims != [*m, *n] {
        return Err(mismatch(x, dims));
    }
    if out.len() != m.checked_mul(*n).ok_or(Error::StorageOutOfBounds)? {
        return Err(Error::StorageOutOfBounds);
    }
    scale(out, beta);
    let alpha = T::from_f64(alpha);
    if out.is_empty() || alpha == T::from_f64(0.0) {
        return Ok(());
    }
    #[cfg(feature = "blas")]
    if T::native_outer(x, y, out, alpha) {
        return Ok(());
    }
    for r in 0..*m {
        let v = alpha * x.values[x.layout.start_offset() + r * x.layout.stride()[0]];
        for c in 0..*n {
            let index = r * n + c;
            out[index] =
                out[index] + v * y.values[y.layout.start_offset() + c * y.layout.stride()[0]];
        }
    }
    Ok(())
}

pub(crate) fn axpby<T: ExtendedFloat>(out: &mut [T], x: Input<'_, T>, alpha: f64, beta: f64) {
    let (alpha, beta) = (T::from_f64(alpha), T::from_f64(beta));
    if out.is_empty() {
        return;
    }
    if alpha == T::from_f64(0.0) {
        scale(out, beta.to_f64());
        return;
    }
    // Standardize zero-beta behavior even for providers that read y anyway.
    if beta == T::from_f64(0.0) {
        out.fill(T::from_f64(0.0));
    }
    #[cfg(feature = "blas")]
    if T::native_axpby(out, x, alpha, beta) {
        return;
    }
    for (dst, i) in out.iter_mut().zip(x.layout.strided_index()) {
        *dst = if beta == T::from_f64(0.0) {
            alpha * x.values[i]
        } else {
            alpha * x.values[i] + beta * *dst
        };
    }
}

pub(crate) fn copy_into<T: ExtendedFloat>(
    out: &mut [T],
    x: Input<'_, T>,
    transpose: bool,
    alpha: f64,
) {
    let alpha = T::from_f64(alpha);
    if out.is_empty() {
        return;
    }
    if alpha == T::from_f64(0.0) {
        out.fill(T::from_f64(0.0));
        return;
    }
    #[cfg(feature = "blas")]
    if T::native_copy(out, x, transpose, alpha) {
        return;
    }
    if transpose {
        let [rows, cols] = *x.layout.dims() else {
            unreachable!("validated matrix");
        };
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = alpha * x.read(r, c);
            }
        }
    } else {
        for (dst, i) in out.iter_mut().zip(x.layout.strided_index()) {
            *dst = alpha * x.values[i];
        }
    }
}

pub(crate) fn matrix_copy<T: ExtendedFloat>(
    x: Input<'_, T>,
    transpose: bool,
    alpha: f64,
) -> Result<AlignedBuffer<T>> {
    if x.layout.dims().len() != 2 {
        return Err(Error::InvalidRank {
            expected: 2,
            actual: x.layout.dims().len(),
        });
    }
    let len = x.layout.checked_elem_count()?;
    let alpha = T::from_f64(alpha);
    if len == 0 || alpha == T::from_f64(0.0) {
        return seed(None, len, 0.0);
    }
    if !transpose
        && alpha == T::from_f64(1.0)
        && let Some((start, end)) = x.layout.contiguous_offsets()
    {
        return AlignedBuffer::from_slice(&x.values[start..end]);
    }
    #[cfg(feature = "blas")]
    if matrix(x.layout).is_some() {
        let mut out = seed(None, len, 0.0)?;
        copy_into(out.as_mut_slice(), x, transpose, alpha.to_f64());
        return Ok(out);
    }
    // Typed fallback writes directly into the final allocation once, without
    // initializing zeros before the copy or allocating an intermediate Vec.
    let mut builder = AlignedBufferBuilder::new(len)?;
    if transpose {
        let [rows, cols] = *x.layout.dims() else {
            unreachable!()
        };
        for col in 0..cols {
            for row in 0..rows {
                builder.write_next(alpha * x.read(row, col))?;
            }
        }
    } else {
        for offset in x.layout.strided_index() {
            builder.write_next(alpha * x.values[offset])?;
        }
    }
    builder.finish()
}

pub(crate) fn argmax_abs<T: ExtendedFloat>(x: Input<'_, T>) -> Result<usize> {
    let count = x.layout.checked_elem_count()?;
    if count == 0 {
        return Err(Error::EmptyReduction {
            op: "argmax_abs",
            dim: 0,
        });
    }
    if count == 1 || x.layout.stride().iter().all(|&s| s == 0) {
        return Ok(0);
    }
    #[cfg(feature = "blas")]
    if vector(x).is_some() {
        // Make NaN behavior independent of the BLAS provider.
        for (index, offset) in x.layout.strided_index().enumerate() {
            if x.values[offset].to_f64().is_nan() {
                return Ok(index);
            }
        }
        if let Some(index) = T::native_argmax(x) {
            return Ok(index);
        }
    }
    let mut best = 0;
    let mut magnitude = -1.0;
    for (index, offset) in x.layout.strided_index().enumerate() {
        let value = x.values[offset].to_f64().abs();
        if value.is_nan() {
            return Ok(index);
        }
        if value > magnitude {
            best = index;
            magnitude = value;
        }
    }
    Ok(best)
}

pub(crate) fn rotate<T: ExtendedFloat>(x: &mut [T], y: &mut [T], c: f64, s: f64) {
    if x.is_empty() {
        return;
    }
    let (c, s) = (T::from_f64(c), T::from_f64(s));
    if s == T::from_f64(0.0) {
        scale(x, c.to_f64());
        scale(y, c.to_f64());
        return;
    }
    if c == T::from_f64(0.0) {
        for (x, y) in x.iter_mut().zip(y) {
            let old = *x;
            *x = s * *y;
            *y = (T::from_f64(0.0) - s) * old;
        }
        return;
    }
    #[cfg(feature = "blas")]
    if T::native_rotate(x, y, c, s) {
        return;
    }
    for (x, y) in x.iter_mut().zip(y) {
        let old = *x;
        *x = c * old + s * *y;
        *y = c * *y - s * old;
    }
}

pub(crate) fn swap<T: ExtendedFloat>(x: &mut [T], y: &mut [T]) {
    if x.is_empty() {
        return;
    }
    #[cfg(feature = "blas")]
    if T::native_swap(x, y) {
        return;
    }
    x.swap_with_slice(y);
}

pub(crate) fn givens(a: f64, b: f64) -> (f64, f64, f64) {
    if a == 0.0 && b == 0.0 {
        return (1.0, 0.0, 0.0);
    }
    #[cfg(feature = "blas")]
    if a.is_finite() && b.is_finite() {
        let (mut r, mut z, mut c, mut s) = (a, b, 0.0, 0.0);
        // SAFETY: ROTG receives four distinct initialized scalar objects.
        unsafe { cblas_drotg(&mut r, &mut z, &mut c, &mut s) };
        if c.is_finite() && s.is_finite() && (c * c + s * s - 1.0).abs() < 1e-12 {
            return (c, s, r);
        }
    }
    let scale = a.abs().max(b.abs());
    let (u, v) = (a / scale, b / scale);
    let sign = if a.abs() > b.abs() {
        a.signum()
    } else {
        b.signum()
    };
    let norm = u.hypot(v) * sign;
    (u / norm, v / norm, scale * norm)
}

macro_rules! native {
    ($ty:ty,$syr:ident,$syr2:ident,$syrk:ident,$syr2k:ident,$trmm:ident,$trmv:ident,$trsm:ident,$symm:ident,$axpby:ident,$copy:ident,$omatcopy:ident,$iamax:ident,$rot:ident,$swap:ident) => {
        impl ExtendedFloat for $ty {
            #[cfg(feature = "blas")]
            fn native_rank(
                out: &mut [Self],
                a: Input<'_, Self>,
                b: Option<Input<'_, Self>>,
                kind: RankUpdate,
                transpose: bool,
                alpha: Self,
            ) -> bool {
                let order = CBlasLayout::CBlasRowMajor;
                let upper = CBlasUplo::CblasUpper;
                match kind {
                    RankUpdate::One | RankUpdate::Two => {
                        let Some((n, inc)) = vector(a) else {
                            return false;
                        };
                        let bv = if let Some(b) = b {
                            let Some((_, i)) = vector(b) else {
                                return false;
                            };
                            Some((b, i))
                        } else {
                            None
                        };
                        // SAFETY: checked rank-one spans and n*n owned output.
                        unsafe {
                            if let Some((b, i)) = bv {
                                $syr2(
                                    order,
                                    upper,
                                    n,
                                    alpha,
                                    a.ptr(),
                                    inc,
                                    b.ptr(),
                                    i,
                                    out.as_mut_ptr(),
                                    n,
                                )
                            } else {
                                $syr(order, upper, n, alpha, a.ptr(), inc, out.as_mut_ptr(), n)
                            }
                        }
                    }
                    RankUpdate::Gram | RankUpdate::TwoK => {
                        let Some((ta, lda)) = matrix(a.layout) else {
                            return false;
                        };
                        let transposed = matches!(ta, CBlasTranspose::CBlasTrans) ^ transpose;
                        let trans = if transposed {
                            CBlasTranspose::CBlasTrans
                        } else {
                            CBlasTranspose::CBlasNoTrans
                        };
                        let (n, k) = if transpose {
                            (a.layout.dims()[1], a.layout.dims()[0])
                        } else {
                            (a.layout.dims()[0], a.layout.dims()[1])
                        };
                        let (Ok(n), Ok(k)) = (CBlasInt::try_from(n), CBlasInt::try_from(k)) else {
                            return false;
                        };
                        let other = if let Some(b) = b {
                            let Some((tb, ldb)) = matrix(b.layout) else {
                                return false;
                            };
                            if matches!(ta, CBlasTranspose::CBlasTrans)
                                != matches!(tb, CBlasTranspose::CBlasTrans)
                            {
                                return false;
                            }
                            Some((b, ldb))
                        } else {
                            None
                        };
                        // SAFETY: dimensions, layouts and matching rank-k spans
                        // checked; update only the upper triangle of owned C.
                        unsafe {
                            if let Some((b, ldb)) = other {
                                $syr2k(
                                    order,
                                    upper,
                                    trans,
                                    n,
                                    k,
                                    alpha,
                                    a.ptr(),
                                    lda,
                                    b.ptr(),
                                    ldb,
                                    1.0,
                                    out.as_mut_ptr(),
                                    n,
                                )
                            } else {
                                $syrk(
                                    order,
                                    upper,
                                    trans,
                                    n,
                                    k,
                                    alpha,
                                    a.ptr(),
                                    lda,
                                    1.0,
                                    out.as_mut_ptr(),
                                    n,
                                )
                            }
                        }
                    }
                }
                true
            }
            #[cfg(feature = "blas")]
            fn native_structured(
                out: &mut [Self],
                a: Input<'_, Self>,
                b: Input<'_, Self>,
                o: TriangularOptions,
                solve: bool,
                symmetric: bool,
            ) -> bool {
                let Some((ta, lda)) = matrix(a.layout) else {
                    return false;
                };
                let (Ok(m), Ok(n)) = (
                    CBlasInt::try_from(b.layout.dims()[0]),
                    CBlasInt::try_from(b.layout.dims()[1]),
                ) else {
                    return false;
                };
                let physical_trans = matches!(ta, CBlasTranspose::CBlasTrans);
                let uplo = if o.upper ^ physical_trans {
                    CBlasUplo::CblasUpper
                } else {
                    CBlasUplo::CblasLower
                };
                let trans = if physical_trans ^ o.transpose {
                    CBlasTranspose::CBlasTrans
                } else {
                    CBlasTranspose::CBlasNoTrans
                };
                let diag = if o.unit_diagonal {
                    CBlasDiag::CblasUnit
                } else {
                    CBlasDiag::CblasNonUnit
                };
                let side = if o.side == MatrixSide::Left {
                    CBlasSide::CblasLeft
                } else {
                    CBlasSide::CblasRight
                };
                let order = CBlasLayout::CBlasRowMajor;
                if symmetric {
                    let Some((tb, ldb)) = matrix(b.layout) else {
                        return false;
                    };
                    if matches!(tb, CBlasTranspose::CBlasTrans) {
                        return false;
                    }
                    // SAFETY: square A, regular B, checked ABI sizes and spans;
                    // a fresh row-major output rules out source aliasing.
                    unsafe {
                        $symm(
                            order,
                            side,
                            uplo,
                            m,
                            n,
                            1.0,
                            a.ptr(),
                            lda,
                            b.ptr(),
                            ldb,
                            0.0,
                            out.as_mut_ptr(),
                            n,
                        )
                    }
                } else {
                    // SAFETY: B has already been copied into out; selected A
                    // triangle and its effective transpose are fully checked.
                    unsafe {
                        if solve {
                            $trsm(
                                order,
                                side,
                                uplo,
                                trans,
                                diag,
                                m,
                                n,
                                1.0,
                                a.ptr(),
                                lda,
                                out.as_mut_ptr(),
                                n,
                            )
                        } else if o.side == MatrixSide::Left && n == 1 {
                            $trmv(
                                order,
                                uplo,
                                trans,
                                diag,
                                m,
                                a.ptr(),
                                lda,
                                out.as_mut_ptr(),
                                1,
                            )
                        } else {
                            $trmm(
                                order,
                                side,
                                uplo,
                                trans,
                                diag,
                                m,
                                n,
                                1.0,
                                a.ptr(),
                                lda,
                                out.as_mut_ptr(),
                                n,
                            )
                        }
                    }
                }
                true
            }
            #[cfg(feature = "blas")]
            fn native_axpby(out: &mut [Self], x: Input<'_, Self>, alpha: Self, beta: Self) -> bool {
                let Some((n, inc)) = vector(x) else {
                    return false;
                };
                // SAFETY: exact peer shapes and exclusive contiguous output.
                unsafe { $axpby(n, alpha, x.ptr(), inc, beta, out.as_mut_ptr(), 1) };
                true
            }
            #[cfg(feature = "blas")]
            fn native_copy(
                out: &mut [Self],
                x: Input<'_, Self>,
                transpose: bool,
                alpha: Self,
            ) -> bool {
                if !transpose
                    && alpha == 1.0
                    && let Some((n, inc)) = vector(x)
                {
                    // SAFETY: logical input length matches exclusive output.
                    unsafe { $copy(n, x.ptr(), inc, out.as_mut_ptr(), 1) };
                    return true;
                }
                let Some((ta, lda)) = matrix(x.layout) else {
                    return false;
                };
                let physical_trans = matches!(ta, CBlasTranspose::CBlasTrans);
                let (rows, cols) = if physical_trans {
                    (x.layout.dims()[1], x.layout.dims()[0])
                } else {
                    (x.layout.dims()[0], x.layout.dims()[1])
                };
                let out_cols = if transpose {
                    x.layout.dims()[0]
                } else {
                    x.layout.dims()[1]
                };
                let (Ok(rows), Ok(cols), Ok(out_cols)) = (
                    CBlasInt::try_from(rows),
                    CBlasInt::try_from(cols),
                    CBlasInt::try_from(out_cols),
                ) else {
                    return false;
                };
                let trans = if physical_trans ^ transpose {
                    CBlasTranspose::CBlasTrans
                } else {
                    CBlasTranspose::CBlasNoTrans
                };
                // SAFETY: physical matrix dimensions and stride checked;
                // destination is separate, contiguous and correctly shaped.
                unsafe {
                    $omatcopy(
                        CBlasLayout::CBlasRowMajor,
                        trans,
                        rows,
                        cols,
                        alpha,
                        x.ptr(),
                        lda,
                        out.as_mut_ptr(),
                        out_cols,
                    )
                };
                true
            }
            #[cfg(feature = "blas")]
            fn native_argmax(x: Input<'_, Self>) -> Option<usize> {
                let (n, inc) = vector(x)?;
                // SAFETY: nonempty validated positive-stride vector.
                let index = unsafe { $iamax(n, x.ptr(), inc) };
                usize::try_from(index).ok().filter(|&i| i < n as usize)
            }
            #[cfg(feature = "blas")]
            fn native_rotate(x: &mut [Self], y: &mut [Self], c: Self, s: Self) -> bool {
                let Ok(n) = CBlasInt::try_from(x.len()) else {
                    return false;
                };
                // SAFETY: separate exclusive allocations of matching length.
                unsafe { $rot(n, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, c, s) };
                true
            }
            #[cfg(feature = "blas")]
            fn native_swap(x: &mut [Self], y: &mut [Self]) -> bool {
                let Ok(n) = CBlasInt::try_from(x.len()) else {
                    return false;
                };
                // SAFETY: separate exclusive allocations of matching length.
                unsafe { $swap(n, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1) };
                true
            }
        }
    };
}
native!(
    f32,
    cblas_ssyr,
    cblas_ssyr2,
    cblas_ssyrk,
    cblas_ssyr2k,
    cblas_strmm,
    cblas_strmv,
    cblas_strsm,
    cblas_ssymm,
    cblas_saxpby,
    cblas_scopy,
    cblas_somatcopy,
    cblas_isamax,
    cblas_srot,
    cblas_sswap
);
native!(
    f64,
    cblas_dsyr,
    cblas_dsyr2,
    cblas_dsyrk,
    cblas_dsyr2k,
    cblas_dtrmm,
    cblas_dtrmv,
    cblas_dtrsm,
    cblas_dsymm,
    cblas_daxpby,
    cblas_dcopy,
    cblas_domatcopy,
    cblas_idamax,
    cblas_drot,
    cblas_dswap
);
