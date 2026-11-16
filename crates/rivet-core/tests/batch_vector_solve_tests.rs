use rivet_core::{
    CpuStorageRef, DType, Device, Error, MatrixSide, Shape, Tensor, TriangularOptions,
};

fn t(v: &[f64], shape: impl Into<Shape>, dtype: DType) -> Tensor {
    Tensor::from_vec(v.to_vec(), shape, &Device::Cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}
fn values(t: &Tensor) -> Vec<f64> {
    t.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap()
}
fn check(t: &Tensor, expected: &[f64]) {
    let actual = values(t);
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 1e-4 * (1.0 + b.abs()),
            "element {i}: {a} != {b}"
        );
    }
}
fn guarded(v: &[f64], shape: impl Into<Shape>, dtype: DType) -> (Tensor, *const u8) {
    let mut data = vec![99.0];
    data.extend_from_slice(v);
    data.push(99.0);
    let base = t(&data, data.len(), dtype);
    let view = base.narrow(0, 1, v.len()).unwrap().reshape(shape).unwrap();
    drop(base);
    let pointer = view.storage_base_ptr();
    (view, pointer)
}
fn check_guards(out: &Tensor, pointer: *const u8) {
    assert_eq!(out.storage_base_ptr(), pointer);
    assert_eq!(out.layout().start_offset(), 1);
    out.with_cpu_storage(|s, _| {
        match s {
            CpuStorageRef::F32(v) => {
                assert_eq!(v[0], 99.);
                assert_eq!(*v.last().unwrap(), 99.);
            }
            CpuStorageRef::F64(v) => {
                assert_eq!(v[0], 99.);
                assert_eq!(*v.last().unwrap(), 99.);
            }
            _ => unreachable!(),
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn batch_mv_matches_scalar_reference_with_broadcast_transpose_and_strides() {
    for dtype in [DType::F32, DType::F64] {
        let data = (0..24).map(|i| i as f64 * 0.125 - 1.0).collect::<Vec<_>>();
        let base = t(&data, (2, 1, 3, 4), dtype);
        let inputs = [
            base.clone(),
            base.transpose(2, 3)
                .unwrap()
                .contiguous()
                .unwrap()
                .transpose(2, 3)
                .unwrap(),
            t(
                &data.iter().flat_map(|v| [*v, 99.]).collect::<Vec<_>>(),
                (2, 1, 3, 4, 2),
                dtype,
            )
            .get_on_dim(4, 0)
            .unwrap(),
        ];
        for a in inputs {
            for transpose in [false, true] {
                let (m, k) = if transpose { (4, 3) } else { (3, 4) };
                let xdata = (0..5 * k)
                    .map(|i| (i % 7) as f64 * 0.25)
                    .collect::<Vec<_>>();
                let x = t(
                    &xdata.iter().flat_map(|v| [*v, 99.]).collect::<Vec<_>>(),
                    (1, 5, k, 2),
                    dtype,
                )
                .get_on_dim(3, 0)
                .unwrap();
                let av = values(&a);
                let mut expected = vec![0.; 2 * 5 * m];
                for batch in 0..2 {
                    for xb in 0..5 {
                        for row in 0..m {
                            for inner in 0..k {
                                let index = batch * 12
                                    + if transpose {
                                        inner * 4 + row
                                    } else {
                                        row * 4 + inner
                                    };
                                expected[(batch * 5 + xb) * m + row] +=
                                    av[index] * xdata[xb * k + inner];
                            }
                        }
                    }
                }
                let actual = a.batch_mv(&x, transpose).unwrap();
                assert_eq!(actual.dims(), &[2, 5, m]);
                check(&actual, &expected);
                let c = Tensor::ones((2, 5, m), dtype, &Device::Cpu).unwrap();
                let expected = expected.iter().map(|v| -0.5 * v + 2.).collect::<Vec<_>>();
                check(
                    &c.batch_addmv(&a, &x, -0.5, 2., transpose).unwrap(),
                    &expected,
                );
                let (dest, pointer) = guarded(&vec![1.; 2 * 5 * m], (2, 5, m), dtype);
                let mut dest = dest.try_into_exclusive().unwrap();
                dest.batch_addmv(&a, &x, -0.5, 2., transpose).unwrap();
                let out = dest.into_tensor();
                check(&out, &expected);
                check_guards(&out, pointer);
            }
        }
    }
}

fn residual(a: &Tensor, rhs: &Tensor, out: &Tensor, o: TriangularOptions, vector: bool) {
    let n = 3;
    let matrix_out = if vector {
        out.unsqueeze(out.rank() - usize::from(o.side == MatrixSide::Right))
            .unwrap()
    } else {
        out.clone()
    };
    let rows = matrix_out.dims()[matrix_out.rank() - 2];
    let cols = matrix_out.dims()[matrix_out.rank() - 1];
    let am = a.broadcast_as((2, 4, n, n)).unwrap();
    let rhs = if vector {
        rhs.unsqueeze(rhs.rank() - usize::from(o.side == MatrixSide::Right))
            .unwrap()
    } else {
        rhs.clone()
    };
    let rhs = rhs.broadcast_as((2, 4, rows, cols)).unwrap();
    let av = values(&am);
    let xv = values(&matrix_out);
    let bv = values(&rhs);
    assert!(xv.iter().all(|v| v.is_finite()));
    for batch in 0..8 {
        let read = |row: usize, col: usize| {
            let (r, c) = if o.transpose { (col, row) } else { (row, col) };
            if r == c && o.unit_diagonal {
                1.
            } else if (o.upper && r > c) || (!o.upper && r < c) {
                0.
            } else {
                av[batch * n * n + r * n + c]
            }
        };
        for row in 0..rows {
            for col in 0..cols {
                let mut sum = 0.;
                for inner in 0..n {
                    sum += if o.side == MatrixSide::Left {
                        read(row, inner) * xv[(batch * rows + inner) * cols + col]
                    } else {
                        xv[(batch * rows + row) * cols + inner] * read(inner, col)
                    };
                }
                let expected = bv[(batch * rows + row) * cols + col];
                assert!(
                    (sum - expected).abs() < 1e-4 * (1. + expected.abs()),
                    "residual batch={batch}, ({row},{col}): {sum} != {expected}, {o:?}"
                );
            }
        }
    }
}

#[test]
fn triangular_batches_cover_all_options_and_physical_layouts() {
    for dtype in [DType::F32, DType::F64] {
        for side in [MatrixSide::Left, MatrixSide::Right] {
            for upper in [false, true] {
                for transpose in [false, true] {
                    for unit_diagonal in [false, true] {
                        let o = TriangularOptions {
                            side,
                            upper,
                            transpose,
                            unit_diagonal,
                        };
                        let mut data = vec![f64::NAN; 18];
                        for b in 0..2 {
                            for r in 0..3 {
                                for c in 0..3 {
                                    if r == c {
                                        data[b * 9 + r * 3 + c] = if unit_diagonal {
                                            if r % 2 == 0 { 0. } else { f64::NAN }
                                        } else {
                                            2. + r as f64 + b as f64
                                        };
                                    } else if (upper && r < c) || (!upper && r > c) {
                                        data[b * 9 + r * 3 + c] = 0.05 * (r + c + 1) as f64;
                                    }
                                }
                            }
                        }
                        let base = t(&data, (2, 1, 3, 3), dtype);
                        let matrices = [
                            base.clone(),
                            base.get(0)
                                .unwrap()
                                .unsqueeze(0)
                                .unwrap()
                                .broadcast_as((2, 1, 3, 3))
                                .unwrap(),
                            base.get(0)
                                .unwrap()
                                .unsqueeze(0)
                                .unwrap()
                                .transpose(2, 3)
                                .unwrap()
                                .contiguous()
                                .unwrap()
                                .transpose(2, 3)
                                .unwrap()
                                .broadcast_as((2, 1, 3, 3))
                                .unwrap(),
                            base.transpose(2, 3)
                                .unwrap()
                                .contiguous()
                                .unwrap()
                                .transpose(2, 3)
                                .unwrap(),
                            t(
                                &data.iter().flat_map(|v| [*v, 99.]).collect::<Vec<_>>(),
                                (2, 1, 3, 3, 2),
                                dtype,
                            )
                            .get_on_dim(4, 0)
                            .unwrap(),
                        ];
                        for a in matrices {
                            for vector in [false, true] {
                                let (rows, cols) = if vector {
                                    if side == MatrixSide::Left {
                                        (3, 1)
                                    } else {
                                        (1, 3)
                                    }
                                } else if side == MatrixSide::Left {
                                    (3, 2)
                                } else {
                                    (2, 3)
                                };
                                let data = (0..4 * rows * cols)
                                    .map(|i| i as f64 * 0.125 - 1.)
                                    .collect::<Vec<_>>();
                                let rhs = if vector {
                                    t(
                                        &data.iter().flat_map(|v| [*v, 99.]).collect::<Vec<_>>(),
                                        (1, 4, 3, 2),
                                        dtype,
                                    )
                                    .get_on_dim(3, 0)
                                    .unwrap()
                                } else {
                                    t(&data, (1, 4, rows, cols), dtype)
                                        .transpose(2, 3)
                                        .unwrap()
                                        .contiguous()
                                        .unwrap()
                                        .transpose(2, 3)
                                        .unwrap()
                                };
                                let out = if vector {
                                    a.batch_triangular_solve_vector(&rhs, o)
                                } else {
                                    a.batch_triangular_solve(&rhs, o)
                                }
                                .unwrap();
                                residual(&a, &rhs, &out, o, vector);
                                let expanded = rhs.broadcast_as(out.shape().clone()).unwrap();
                                let (dest, pointer) =
                                    guarded(&values(&expanded), out.shape().clone(), dtype);
                                let mut dest = dest.try_into_exclusive().unwrap();
                                if vector {
                                    dest.batch_triangular_solve_vector(&a, o)
                                } else {
                                    dest.batch_triangular_solve(&a, o)
                                }
                                .unwrap();
                                let dest = dest.into_tensor();
                                residual(&a, &rhs, &dest, o, vector);
                                check(&dest, &values(&out));
                                check_guards(&dest, pointer);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn later_singular_batches_are_detected_before_any_inplace_write() {
    for dtype in [DType::F32, DType::F64] {
        let a = t(
            &[2., f64::NAN, 0.5, 3., 1., f64::NAN, 1., 0.],
            (2, 1, 2, 2),
            dtype,
        );
        let rhs = Tensor::ones((1, 3, 2), dtype, &Device::Cpu).unwrap();
        assert!(matches!(
            a.batch_triangular_solve_vector(&rhs, TriangularOptions::default()),
            Err(Error::BatchedSingularMatrix { batch: 3, index: 1 })
        ));
        let (dest, pointer) = guarded(&[7.; 12], (2, 3, 2), dtype);
        let mut dest = dest.try_into_exclusive().unwrap();
        assert!(matches!(
            dest.batch_triangular_solve_vector(&a, TriangularOptions::default()),
            Err(Error::BatchedSingularMatrix { batch: 3, index: 1 })
        ));
        let dest = dest.into_tensor();
        check(&dest, &[7.; 12]);
        check_guards(&dest, pointer);
        let unit = TriangularOptions {
            unit_diagonal: true,
            ..Default::default()
        };
        assert!(
            values(&a.batch_triangular_solve_vector(&rhs, unit).unwrap())
                .iter()
                .all(|v| v.is_finite())
        );
    }
}

#[test]
fn batch_vector_coefficients_and_empty_dimensions_follow_existing_contracts() {
    for dtype in [DType::F32, DType::F64] {
        let a = t(&[f64::NAN; 12], (2, 2, 3), dtype);
        let x = t(&[f64::NAN; 3], 3, dtype);
        let seed = Tensor::ones((2, 2), dtype, &Device::Cpu).unwrap();
        check(&seed.batch_addmv(&a, &x, 0., 2., false).unwrap(), &[2.; 4]);
        let a = Tensor::ones((2, 2, 3), dtype, &Device::Cpu).unwrap();
        let x = Tensor::ones(3, dtype, &Device::Cpu).unwrap();
        let seed = t(&[f64::NAN; 4], (2, 2), dtype);
        check(&seed.batch_addmv(&a, &x, 1., 0., false).unwrap(), &[3.; 4]);
        let a = Tensor::zeros((2, 3, 0), dtype, &Device::Cpu).unwrap();
        let x = Tensor::zeros(0, dtype, &Device::Cpu).unwrap();
        check(&a.batch_mv(&x, false).unwrap(), &[0.; 6]);
        let a = Tensor::zeros((0, 2, 2), dtype, &Device::Cpu).unwrap();
        let rhs = Tensor::zeros((0, 2), dtype, &Device::Cpu).unwrap();
        assert_eq!(
            a.batch_triangular_solve_vector(&rhs, TriangularOptions::default())
                .unwrap()
                .dims(),
            &[0, 2]
        );
        let a = Tensor::zeros((2, 0, 0), dtype, &Device::Cpu).unwrap();
        let rhs = Tensor::zeros((1, 0), dtype, &Device::Cpu).unwrap();
        assert_eq!(
            a.batch_triangular_solve_vector(&rhs, TriangularOptions::default())
                .unwrap()
                .dims(),
            &[2, 0]
        );
        let a = Tensor::zeros((2, 2, 2), dtype, &Device::Cpu).unwrap();
        let rhs = Tensor::zeros((2, 2, 0), dtype, &Device::Cpu).unwrap();
        assert!(matches!(
            a.batch_triangular_solve(&rhs, TriangularOptions::default()),
            Err(Error::BatchedSingularMatrix { batch: 0, index: 0 })
        ));
        let unit = TriangularOptions {
            unit_diagonal: true,
            ..Default::default()
        };
        assert_eq!(
            a.batch_triangular_solve(&rhs, unit).unwrap().dims(),
            &[2, 2, 0]
        );
    }
}

#[test]
fn vector_rhs_is_explicit_and_inplace_destinations_cannot_expand() {
    let a = Tensor::ones((3, 2, 2), DType::F64, &Device::Cpu).unwrap();
    let rhs = Tensor::ones((2, 2), DType::F64, &Device::Cpu).unwrap();
    let unit = TriangularOptions {
        unit_diagonal: true,
        ..Default::default()
    };
    assert_eq!(
        a.batch_triangular_solve(&rhs, unit).unwrap().dims(),
        &[3, 2, 2]
    );
    assert!(a.batch_triangular_solve_vector(&rhs, unit).is_err());
    let (dest, pointer) = guarded(&[1.; 4], (2, 2), DType::F64);
    let mut dest = dest.try_into_exclusive().unwrap();
    assert!(dest.batch_triangular_solve(&a, unit).is_err());
    let dest = dest.into_tensor();
    check(&dest, &[1.; 4]);
    check_guards(&dest, pointer);
}

#[test]
fn invalid_shapes_dtypes_and_noncontiguous_destinations_do_not_write() {
    for dtype in [DType::F32, DType::F64] {
        let a = Tensor::ones((2, 2, 2), dtype, &Device::Cpu).unwrap();
        let x = Tensor::ones((2, 2), dtype, &Device::Cpu).unwrap();
        let bad = Tensor::ones(3, dtype, &Device::Cpu).unwrap();
        assert!(a.batch_mv(&bad, false).is_err());
        assert!(
            a.batch_triangular_solve_vector(&bad, Default::default())
                .is_err()
        );
        let base = t(&[1., 2., 3., 4.], (2, 2), dtype);
        let view = base.transpose(0, 1).unwrap();
        drop(base);
        let mut dest = view.try_into_exclusive().unwrap();
        assert!(matches!(
            dest.batch_addmv(&a, &x, 1., 0., false),
            Err(Error::UnsupportedLayoutForOp { .. })
        ));
        assert!(
            dest.batch_triangular_solve_vector(&a, Default::default())
                .is_err()
        );
        check(&dest.into_tensor(), &[1., 3., 2., 4.]);
        let (dest, pointer) = guarded(&[1.; 4], (2, 2), dtype);
        let mut dest = dest.try_into_exclusive().unwrap();
        assert!(dest.batch_addmv(&a, &bad, 1., 0., false).is_err());
        let other = Tensor::ones((2, 2, 2), DType::I32, &Device::Cpu).unwrap();
        assert!(matches!(
            dest.batch_addmv(&other, &x, 1., 0., false),
            Err(Error::DTypeMismatch { .. })
        ));
        let dest = dest.into_tensor();
        check(&dest, &[1.; 4]);
        check_guards(&dest, pointer);
    }
    let a = Tensor::ones((2, 2, 2), DType::I32, &Device::Cpu).unwrap();
    let x = Tensor::ones((2, 2), DType::I32, &Device::Cpu).unwrap();
    assert!(matches!(
        a.batch_mv(&x, false),
        Err(Error::UnsupportedDTypeForOp { .. })
    ));
    assert!(matches!(
        a.batch_triangular_solve_vector(&x, Default::default()),
        Err(Error::UnsupportedDTypeForOp { .. })
    ));
}

#[test]
fn permuted_batch_axes_keep_logical_order_and_scalar_systems_work() {
    for dtype in [DType::F32, DType::F64] {
        let mut data = vec![f64::NAN; 8 * 9];
        for b in 0..8 {
            for r in 0..3 {
                for c in 0..=r {
                    data[b * 9 + r * 3 + c] = if r == c { 2. + b as f64 * 0.25 } else { 0.05 };
                }
            }
        }
        let a = t(&data, (4, 2, 3, 3), dtype).transpose(0, 1).unwrap();
        let rhs = Tensor::ones((2, 4, 3, 2), dtype, &Device::Cpu).unwrap();
        let o = TriangularOptions::default();
        let out = a.batch_triangular_solve(&rhs, o).unwrap();
        residual(&a, &rhs, &out, o, false);
        let a = t(&[2., 4.], (2, 1, 1), dtype);
        let rhs = t(&[6., 20.], (2, 1), dtype);
        for side in [MatrixSide::Left, MatrixSide::Right] {
            for transpose in [false, true] {
                let o = TriangularOptions {
                    side,
                    transpose,
                    ..Default::default()
                };
                check(
                    &a.batch_triangular_solve_vector(&rhs, o).unwrap(),
                    &[3., 5.],
                );
                let shared = t(&[2.], (), dtype).broadcast_as((2, 1, 1)).unwrap();
                check(
                    &shared.batch_triangular_solve_vector(&rhs, o).unwrap(),
                    &[3., 10.],
                );
            }
        }
        let a = t(&[2., 0., 1., 3.], (2, 2), dtype);
        let x = a.get_on_dim(1, 0).unwrap();
        check(&a.batch_mv(&x, false).unwrap(), &[4., 5.]);
        let out = a
            .batch_triangular_solve(&a.clone(), Default::default())
            .unwrap();
        assert!(!out.same_storage(&a));
        check(&out, &[1., 0., 0., 1.]);
        check(
            &a.batch_triangular_solve_vector(&x, Default::default())
                .unwrap(),
            &[1., 0.],
        );
        check(&a, &[2., 0., 1., 3.]);
    }
}
