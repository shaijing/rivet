//! Batched triangular solves with one final output, shared matrix metadata,
//! and a preflight diagonal scan before any exclusive destination is changed.
use super::extensions::ExtendedFloat;
use super::*;
use crate::{MatrixSide, StridedIndex, TriangularOptions};

struct SolveShape {
    count: usize,
    rows: usize,
    cols: usize,
    n: usize,
    len: usize,
    matrix_len: usize,
}

fn shape<T: ExtendedFloat>(
    a: Input<'_, T>,
    dims: &[usize],
    o: TriangularOptions,
) -> Result<SolveShape> {
    let rank = a.layout.dims().len();
    if rank < 2 || dims.len() != rank {
        return Err(Error::MatmulShapeMismatch {
            lhs: a.layout.dims().to_vec(),
            rhs: dims.to_vec(),
        });
    }
    let (rows, cols) = (dims[rank - 2], dims[rank - 1]);
    let n = if o.side == MatrixSide::Left {
        rows
    } else {
        cols
    };
    if a.layout.dims()[..rank - 2] != dims[..rank - 2] || a.layout.dims()[rank - 2..] != [n, n] {
        return Err(Error::MatmulShapeMismatch {
            lhs: a.layout.dims().to_vec(),
            rhs: dims.to_vec(),
        });
    }
    let count = dims[..rank - 2]
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or(Error::ShapeElementCountOverflow)?;
    let matrix_len = rows.checked_mul(cols).ok_or(Error::StorageOutOfBounds)?;
    let len = count
        .checked_mul(matrix_len)
        .ok_or(Error::StorageOutOfBounds)?;
    Ok(SolveShape {
        count,
        rows,
        cols,
        n,
        len,
        matrix_len,
    })
}

fn offsets<T>(a: Input<'_, T>) -> StridedIndex<'_> {
    let rank = a.layout.dims().len();
    StridedIndex::new(
        &a.layout.dims()[..rank - 2],
        &a.layout.stride()[..rank - 2],
        a.layout.start_offset(),
    )
}

fn shared_matrix<T>(a: Input<'_, T>) -> bool {
    let rank = a.layout.dims().len();
    a.layout.dims()[..rank - 2]
        .iter()
        .zip(&a.layout.stride()[..rank - 2])
        .all(|(&dim, &stride)| dim <= 1 || stride == 0)
}

fn validate_diagonals<T: ExtendedFloat>(
    a: Input<'_, T>,
    s: &SolveShape,
    o: TriangularOptions,
) -> Result<()> {
    if o.unit_diagonal || s.n == 0 || s.count == 0 {
        return Ok(());
    }
    let rank = a.layout.dims().len();
    let step = if s.n == 1 {
        0
    } else {
        a.layout.stride()[rank - 2]
            .checked_add(a.layout.stride()[rank - 1])
            .ok_or(Error::StorageOutOfBounds)?
    };
    let limit = if shared_matrix(a) { 1 } else { s.count };
    for (batch, offset) in offsets(a).take(limit).enumerate() {
        for index in 0..s.n {
            if a.values[offset + index * step] == T::from_f64(0.0) {
                return Err(Error::BatchedSingularMatrix { batch, index });
            }
        }
    }
    Ok(())
}

pub(crate) fn triangular_batch<T: ExtendedFloat>(
    a: Input<'_, T>,
    rhs: Input<'_, T>,
    o: TriangularOptions,
) -> Result<AlignedBuffer<T>> {
    let s = shape(a, rhs.layout.dims(), o)?;
    validate_diagonals(a, &s, o)?;
    let mut out = seed(Some(rhs), s.len, 1.0)?;
    solve(out.as_mut_slice(), a, &s, o)?;
    Ok(out)
}

pub(crate) fn triangular_batch_into<T: ExtendedFloat>(
    out: &mut [T],
    dims: &[usize],
    a: Input<'_, T>,
    o: TriangularOptions,
) -> Result<()> {
    let s = shape(a, dims, o)?;
    if out.len() != s.len {
        return Err(Error::StorageOutOfBounds);
    }
    validate_diagonals(a, &s, o)?;
    solve(out, a, &s, o)
}

fn solve<T: ExtendedFloat>(
    out: &mut [T],
    a: Input<'_, T>,
    s: &SolveShape,
    o: TriangularOptions,
) -> Result<()> {
    if s.len == 0 {
        return Ok(());
    }
    let rank = a.layout.dims().len();
    let layout = Layout::new((s.n, s.n).into(), a.layout.stride()[rank - 2..].to_vec(), 0)?;
    #[cfg(feature = "blas")]
    if shared_matrix(a) && (o.side == MatrixSide::Right || s.cols == 1) {
        // Shared right-side solves can stack RHS rows into one TRSM. Left
        // vectors are equivalent to right-side row solves with op(A)^T.
        // Both transformations preserve the existing contiguous output order.
        let left_vector = o.side == MatrixSide::Left;
        // n>0 and count*rows*cols already fits the validated output span.
        let rows = if left_vector {
            s.count
        } else {
            s.count * s.rows
        };
        let options = TriangularOptions {
            side: MatrixSide::Right,
            transpose: o.transpose ^ left_vector,
            ..o
        };
        let input = Input {
            values: &a.values[a.layout.start_offset()..],
            layout: &layout,
        };
        if T::native_solve_matrix(out, input, rows, s.n, options) {
            return Ok(());
        }
    }
    // Resolve transpose once in metadata, rather than branching on every
    // coefficient in the typed fallback. Keep its cache-friendly per-matrix
    // traversal when native grouping is unavailable or unsupported.
    let transposed = if o.transpose {
        Some(layout.transpose(0, 1)?)
    } else {
        None
    };
    let fallback_layout = transposed.as_ref().unwrap_or(&layout);
    let fallback_options = TriangularOptions {
        upper: o.upper ^ o.transpose,
        transpose: false,
        ..o
    };
    for (matrix, offset) in out.chunks_exact_mut(s.matrix_len).zip(offsets(a)) {
        // Input::new already validated the complete batched span. Slicing the
        // backing storage reuses one zero-offset matrix layout for every batch,
        // including partial broadcasts, padded matrices, and batch permutations.
        #[cfg(feature = "blas")]
        let input = Input {
            values: &a.values[offset..],
            layout: &layout,
        };
        #[cfg(feature = "blas")]
        if T::native_solve_matrix(matrix, input, s.rows, s.cols, o) {
            continue;
        }
        let input = Input {
            values: &a.values[offset..],
            layout: fallback_layout,
        };
        fallback(matrix, input, s, fallback_options);
    }
    Ok(())
}

fn fallback<T: ExtendedFloat>(
    out: &mut [T],
    a: Input<'_, T>,
    s: &SolveShape,
    o: TriangularOptions,
) {
    let upper = o.upper ^ o.transpose;
    if o.side == MatrixSide::Left {
        for step in 0..s.n {
            let row = if upper { s.n - 1 - step } else { step };
            let range = if upper { row + 1..s.n } else { 0..row };
            for col in 0..s.cols {
                let mut value = out[row * s.cols + col];
                for inner in range.clone() {
                    value = value - a.read(row, inner) * out[inner * s.cols + col];
                }
                if !o.unit_diagonal {
                    value = value / a.read(row, row);
                }
                out[row * s.cols + col] = value;
            }
        }
    } else {
        for step in 0..s.n {
            let col = if upper { step } else { s.n - 1 - step };
            let range = if upper { 0..col } else { col + 1..s.n };
            for row in 0..s.rows {
                let mut value = out[row * s.cols + col];
                for inner in range.clone() {
                    value = value - out[row * s.cols + inner] * a.read(inner, col);
                }
                if !o.unit_diagonal {
                    value = value / a.read(col, col);
                }
                out[row * s.cols + col] = value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_output_spans_are_rejected_before_mutation() {
        let layout = Layout::contiguous((1, 2, 2));
        let a = Input::new(&[2.0f64, 0.0, 0.5, 3.0], &layout).unwrap();
        let mut out = [7.0; 3];
        assert!(matches!(
            triangular_batch_into(&mut out, &[1, 2, 2], a, Default::default()),
            Err(Error::StorageOutOfBounds)
        ));
        assert!(triangular_batch_into(&mut out, &[1, 3, 1], a, Default::default()).is_err());
        assert_eq!(out, [7.0; 3]);
    }

    #[test]
    fn empty_batches_do_not_inspect_inactive_diagonal_strides() {
        let layout = Layout::new((0, 2, 2).into(), vec![0, usize::MAX, usize::MAX], 0).unwrap();
        let a = Input::new(&[] as &[f32], &layout).unwrap();
        triangular_batch_into(&mut [], &[0, 2, 1], a, Default::default()).unwrap();
        let layout = Layout::new((0, usize::MAX, usize::MAX).into(), vec![0, 0, 0], 0).unwrap();
        let a = Input::new(&[] as &[f32], &layout).unwrap();
        assert!(matches!(
            triangular_batch_into(&mut [], &[0, usize::MAX, 2], a, Default::default()),
            Err(Error::StorageOutOfBounds)
        ));
    }
}
