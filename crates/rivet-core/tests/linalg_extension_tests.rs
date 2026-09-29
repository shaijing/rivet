use rivet_core::{
    DType, Device, Error, MatrixSide, Shape, Tensor, TriangularOptions, givens_rotation,
};
fn t(values: &[f64], shape: impl Into<Shape>, dtype: DType) -> Tensor {
    Tensor::from_vec(values.to_vec(), shape, &Device::Cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}
fn check(x: &Tensor, expected: &[f64]) {
    let actual = x.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() < 1e-5 * (1.0 + b.abs()),
            "index {i}: {a} != {b}"
        );
    }
}
fn same(x: &Tensor, y: &Tensor) {
    assert_eq!(x.dims(), y.dims());
    check(x, &y.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap());
}
#[test]
fn symmetric_rank_updates_read_selected_triangle_and_mirror() {
    for dtype in [DType::F32, DType::F64] {
        let c = t(&[1., 2., f64::NAN, 3.], (2, 2), dtype);
        let x = t(&[2., 3.], 2, dtype);
        let y = t(&[4., 5.], 2, dtype);
        check(
            &c.symmetric_rank1_update(&x, 2., 3., true).unwrap(),
            &[11., 18., 18., 27.],
        );
        check(
            &c.symmetric_rank2_update(&x, &y, 2., 3., true).unwrap(),
            &[35., 50., 50., 69.],
        );
        let lower = c.transpose(0, 1).unwrap();
        same(
            &lower.symmetric_rank1_update(&x, 2., 3., false).unwrap(),
            &c.symmetric_rank1_update(&x, 2., 3., true).unwrap(),
        );
        let zero = t(&[f64::NAN; 4], (2, 2), dtype);
        check(
            &zero.symmetric_rank1_update(&x, 1., 0., false).unwrap(),
            &[4., 6., 6., 9.],
        );
        let nan = t(&[f64::NAN; 2], 2, dtype);
        check(
            &c.symmetric_rank1_update(&nan, 0., 1., true).unwrap(),
            &[1., 2., 2., 3.],
        );
    }
}
#[test]
fn rank_k_updates_match_composed_products_for_views() {
    for dtype in [DType::F32, DType::F64] {
        for transpose in [false, true] {
            let a = t(&[1., 2., 3., 4., 5., 6.], (2, 3), dtype);
            let b = t(&[6., 5., 4., 3., 2., 1.], (2, 3), dtype);
            for (a, b) in [
                (a.clone(), b.clone()),
                (a.transpose(0, 1).unwrap(), b.transpose(0, 1).unwrap()),
                (
                    a.clone(),
                    b.transpose(0, 1)
                        .unwrap()
                        .contiguous()
                        .unwrap()
                        .transpose(0, 1)
                        .unwrap(),
                ),
            ] {
                let n = a.dims()[usize::from(transpose)];
                let c = Tensor::ones((n, n), dtype, &Device::Cpu).unwrap();
                same(
                    &c.gram_update(&a, transpose, 2., 3., true).unwrap(),
                    &a.gram(transpose).unwrap().affine(2., 3.).unwrap(),
                );
                let prod = if transpose {
                    a.transpose(0, 1).unwrap().matmul(&b).unwrap()
                } else {
                    a.matmul(&b.transpose(0, 1).unwrap()).unwrap()
                };
                let expected = prod
                    .add(&prod.transpose(0, 1).unwrap())
                    .unwrap()
                    .affine(2., 3.)
                    .unwrap();
                same(
                    &c.symmetric_rank2k_update(&a, &b, transpose, 2., 3., true)
                        .unwrap(),
                    &expected,
                );
            }
        }
    }
}
#[test]
fn fused_batches_broadcast_and_reduce_directly() {
    for dtype in [DType::F32, DType::F64] {
        let a = t(&[1., 2., 3., 4., 5., 6., 7., 8.], (2, 2, 2), dtype);
        let b = t(&[1., 0., 0., 2.], (1, 2, 2), dtype);
        let c = Tensor::ones((2, 2, 2), dtype, &Device::Cpu).unwrap();
        let products = a.broadcast_matmul(&b).unwrap();
        same(
            &c.baddbmm(&a, &b, 2., 3.).unwrap(),
            &products.affine(2., 3.).unwrap(),
        );
        let sum = Tensor::ones((2, 2), dtype, &Device::Cpu).unwrap();
        same(
            &sum.addbmm(&a, &b, 2., 3.).unwrap(),
            &products.sum(0).unwrap().affine(2., 3.).unwrap(),
        );
        let a = a.transpose(1, 2).unwrap();
        same(
            &c.baddbmm(&a, &b, 1., 1.).unwrap(),
            &a.broadcast_matmul(&b).unwrap().add(&c).unwrap(),
        );
        let a = t(&[], (0, 2, 2), dtype);
        let b = t(&[], (0, 2, 2), dtype);
        check(&sum.addbmm(&a, &b, 1., 2.).unwrap(), &[2.; 4]);
        let nan = Tensor::ones((2, 2, 2), dtype, &Device::Cpu)
            .unwrap()
            .affine(f64::NAN, 0.)
            .unwrap();
        check(&c.baddbmm(&nan, &nan, 0., 2.).unwrap(), &[2.; 8]);
    }
}
#[test]
fn fused_batches_handle_multiple_axes_and_singleton_outputs() {
    for dtype in [DType::F32, DType::F64] {
        let a = Tensor::ones((2, 1, 3, 2), dtype, &Device::Cpu).unwrap();
        let b = Tensor::ones((1, 4, 2, 1), dtype, &Device::Cpu).unwrap();
        let c = Tensor::ones((2, 4, 3, 1), dtype, &Device::Cpu).unwrap();
        check(&c.baddbmm(&a, &b, 2., 3.).unwrap(), &[7.; 24]);
        let c = Tensor::ones((3, 1), dtype, &Device::Cpu).unwrap();
        check(&c.addbmm(&a, &b, 2., 3.).unwrap(), &[35.; 3]);
    }
}
#[test]
fn structured_left_right_transposes_and_unit_diagonal() {
    for dtype in [DType::F32, DType::F64] {
        for upper in [false, true] {
            for transpose in [false, true] {
                for side in [MatrixSide::Left, MatrixSide::Right] {
                    for unit in [false, true] {
                        let data = if upper {
                            [2., 3., f64::NAN, 4.]
                        } else {
                            [2., f64::NAN, 3., 4.]
                        };
                        let a = t(&data, (2, 2), dtype);
                        let diag = if unit {
                            [1., 3., 0., 1.]
                        } else {
                            [2., 3., 0., 4.]
                        };
                        let full = t(&diag, (2, 2), dtype);
                        let full = if upper {
                            full
                        } else {
                            full.transpose(0, 1).unwrap()
                        };
                        let full = if transpose {
                            full.transpose(0, 1).unwrap()
                        } else {
                            full
                        };
                        let options = TriangularOptions {
                            upper,
                            unit_diagonal: unit,
                            transpose,
                            side,
                        };
                        let b = t(
                            &[1., 2., 3., 4., 5., 6.],
                            if side == MatrixSide::Left {
                                (2, 3)
                            } else {
                                (3, 2)
                            },
                            dtype,
                        );
                        let expected = if side == MatrixSide::Left {
                            full.matmul(&b).unwrap()
                        } else {
                            b.matmul(&full).unwrap()
                        };
                        for a in [
                            a.clone(),
                            a.transpose(0, 1)
                                .unwrap()
                                .contiguous()
                                .unwrap()
                                .transpose(0, 1)
                                .unwrap(),
                        ] {
                            let product = a.triangular_matmul_with_options(&b, options).unwrap();
                            same(&product, &expected);
                            same(
                                &a.triangular_solve_with_options(&product, options).unwrap(),
                                &b,
                            );
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn triangular_vectors_ignore_unused_values() {
    for dtype in [DType::F32, DType::F64] {
        let a = t(&[f64::NAN, 2., f64::NAN, f64::NAN], (2, 2), dtype);
        let x = t(&[3., 4.], 2, dtype);
        check(&a.triangular_mv(&x, true, true).unwrap(), &[11., 4.]);
        check(&a.triangular_solve(&x, true, true).unwrap(), &[-5., 4.]);
        let b = t(&[2., 3., f64::NAN, 4.], (2, 2), dtype);
        check(&b.triangular_mv(&x, true, false).unwrap(), &[18., 16.]);
        let options = TriangularOptions {
            upper: true,
            transpose: true,
            ..Default::default()
        };
        let out = b.triangular_solve_with_options(&x, options).unwrap();
        same(
            &b.matrix_copy(true, 1.)
                .unwrap()
                .triangular_mv(&out, false, false)
                .unwrap(),
            &x,
        );
    }
}
#[test]
fn symmetric_side_matches_dense_and_ignores_unused_triangle() {
    for dtype in [DType::F32, DType::F64] {
        for side in [MatrixSide::Left, MatrixSide::Right] {
            for upper in [false, true] {
                let a = t(
                    if upper {
                        &[1., 2., f64::NAN, 3.]
                    } else {
                        &[1., f64::NAN, 2., 3.]
                    },
                    (2, 2),
                    dtype,
                );
                let full = t(&[1., 2., 2., 3.], (2, 2), dtype);
                let b = t(
                    &[1., 2., 3., 4., 5., 6.],
                    if side == MatrixSide::Left {
                        (2, 3)
                    } else {
                        (3, 2)
                    },
                    dtype,
                );
                let expected = if side == MatrixSide::Left {
                    full.matmul(&b).unwrap()
                } else {
                    b.matmul(&full).unwrap()
                };
                same(
                    &a.symmetric_matmul_with_side(&b, upper, side).unwrap(),
                    &expected,
                );
                let b = b
                    .transpose(0, 1)
                    .unwrap()
                    .contiguous()
                    .unwrap()
                    .transpose(0, 1)
                    .unwrap();
                same(
                    &a.symmetric_matmul_with_side(&b, upper, side).unwrap(),
                    &expected,
                );
            }
        }
    }
}
#[test]
fn in_place_matrix_updates_keep_allocations_and_views() {
    for dtype in [DType::F32, DType::F64] {
        let a = t(&[1., 2., 3., 4.], (2, 2), dtype);
        let b = t(&[5., 6., 7., 8.], (2, 2), dtype);
        let base = t(&[9., 9., 1., 2., 3., 4., 9., 9.], (4, 2), dtype);
        let view = base.narrow(0, 1, 2).unwrap();
        drop(base);
        let ptr = view.storage_base_ptr();
        let expected = view.addmm(&a, &b, 2., 3.).unwrap();
        let mut owned = view.try_into_exclusive().unwrap();
        owned.addmm(&a, &b, 2., 3.).unwrap();
        let out = owned.into_tensor();
        assert_eq!(ptr, out.storage_base_ptr());
        same(&out, &expected);
        let x = t(&[2., 3.], 2, dtype);
        let seed = t(&[10., 20.], 2, dtype);
        let expected = seed.addmv(&a, &x, 2., 3.).unwrap();
        let mut owned = seed.try_into_exclusive().unwrap();
        owned.addmv(&a, &x, 2., 3.).unwrap();
        same(&owned.into_tensor(), &expected);
        let seed = t(&[1.; 4], (2, 2), dtype);
        let expected = seed.addr(&x, &x, 2., 3.).unwrap();
        let mut owned = seed.try_into_exclusive().unwrap();
        owned.addr(&x, &x, 2., 3.).unwrap();
        same(&owned.into_tensor(), &expected);
    }
}
#[test]
fn axpby_copy_and_positive_stride_axpy() {
    for dtype in [DType::F32, DType::F64] {
        let x = t(&[1., 9., 2., 9., 3., 9.], (3, 2), dtype)
            .get_on_dim(1, 0)
            .unwrap();
        let mut y = t(&[10., 20., 30.], 3, dtype).try_into_exclusive().unwrap();
        y.axpby(2., &x, 3.).unwrap();
        y.axpy(2., &x).unwrap();
        check(&y.into_tensor(), &[34., 68., 102.]);
        let mut y = t(&[f64::NAN; 3], 3, dtype).try_into_exclusive().unwrap();
        y.axpby(2., &x, 0.).unwrap();
        check(&y.into_tensor(), &[2., 4., 6.]);
        let nan = t(&[f64::NAN; 3], 3, dtype);
        let mut y = t(&[1., 2., 3.], 3, dtype).try_into_exclusive().unwrap();
        y.axpby(0., &nan, 2.).unwrap();
        y.copy_from(&x).unwrap();
        check(&y.into_tensor(), &[1., 2., 3.]);
        let x = t(&[2.], 1, dtype).broadcast_as(3).unwrap();
        let mut y = t(&[1.; 3], 3, dtype).try_into_exclusive().unwrap();
        y.axpby(2., &x, 3.).unwrap();
        check(&y.into_tensor(), &[7.; 3]);
    }
}
#[test]
fn matrix_copy_handles_transpose_padding_and_zero_scale() {
    for dtype in [DType::F32, DType::F64] {
        let x = t(&[9., 9., 9., 1., 2., 9., 3., 4., 9.], (3, 3), dtype)
            .narrow(0, 1, 2)
            .unwrap()
            .narrow(1, 0, 2)
            .unwrap();
        check(&x.matrix_copy(true, 2.).unwrap(), &[2., 6., 4., 8.]);
        check(
            &x.transpose(0, 1).unwrap().matrix_copy(true, 2.).unwrap(),
            &[2., 4., 6., 8.],
        );
        let x = t(&[f64::NAN; 6], (2, 3), dtype);
        check(&x.matrix_copy(true, 0.).unwrap(), &[0.; 6]);
        let x = t(&[2.], (1, 1), dtype).broadcast_as((2, 3)).unwrap();
        check(&x.matrix_copy(true, 3.).unwrap(), &[6.; 6]);
    }
}
#[test]
fn argmax_abs_has_stable_nan_tie_empty_and_stride_semantics() {
    for dtype in [DType::F32, DType::F64] {
        let x = t(&[1., -7., 7., 2.], 4, dtype);
        assert_eq!(x.argmax_abs().unwrap(), 1);
        assert_eq!(
            t(&[1., f64::NAN, f64::NAN], 3, dtype).argmax_abs().unwrap(),
            1
        );
        assert_eq!(
            t(&[9., 9., 1., -7., 2., 3.], (3, 2), dtype)
                .narrow(0, 1, 2)
                .unwrap()
                .get_on_dim(1, 1)
                .unwrap()
                .argmax_abs()
                .unwrap(),
            0
        );
        assert_eq!(
            t(&[2.], 1, dtype)
                .broadcast_as(3)
                .unwrap()
                .argmax_abs()
                .unwrap(),
            0
        );
        assert_eq!(
            t(&[1., 2., 3., 4.], (2, 2), dtype)
                .transpose(0, 1)
                .unwrap()
                .argmax_abs()
                .unwrap(),
            3
        );
        assert!(matches!(
            t(&[], 0, dtype).argmax_abs(),
            Err(Error::EmptyReduction { .. })
        ));
    }
}
#[test]
fn rotations_and_swaps_work_in_existing_allocations() {
    for dtype in [DType::F32, DType::F64] {
        let x = t(&[1., 2.], 2, dtype);
        let ptr = x.storage_base_ptr();
        let mut x = x.try_into_exclusive().unwrap();
        let mut y = t(&[3., 4.], 2, dtype).try_into_exclusive().unwrap();
        x.rotate(&mut y, 0.6, 0.8).unwrap();
        check(&x.into_tensor(), &[3., 4.4]);
        let y = y.into_tensor();
        assert_ne!(ptr, y.storage_base_ptr());
        check(&y, &[1., 0.8]);
        let mut x = t(&[1., 2.], 2, dtype).try_into_exclusive().unwrap();
        let mut y = t(&[3., 4.], 2, dtype).try_into_exclusive().unwrap();
        x.swap(&mut y).unwrap();
        check(&x.into_tensor(), &[3., 4.]);
        check(&y.into_tensor(), &[1., 2.]);
        let mut x = t(&[f64::NAN; 2], 2, dtype).try_into_exclusive().unwrap();
        let mut y = t(&[f64::NAN; 2], 2, dtype).try_into_exclusive().unwrap();
        x.rotate(&mut y, 0., 0.).unwrap();
        check(&x.into_tensor(), &[0.; 2]);
        check(&y.into_tensor(), &[0.; 2]);
    }
}
#[test]
fn givens_scaling_handles_huge_tiny_and_zero_inputs() {
    for (a, b) in [
        (3., 4.),
        (-3., 4.),
        (3., -4.),
        (0., 0.),
        (0., -2.),
        (1e300, 1e300),
        (1e-300, 1e-300),
        (f64::MAX, f64::MAX),
    ] {
        let (c, s, r) = givens_rotation(a, b).unwrap();
        assert!((c * c + s * s - 1.).abs() < 1e-12);
        if r.is_finite() && r != 0. {
            assert!(((c * a + s * b - r) / r).abs() < 1e-12);
        }
        assert!(
            (c * (b / a.abs().max(b.abs()).max(f64::MIN_POSITIVE))
                - s * (a / a.abs().max(b.abs()).max(f64::MIN_POSITIVE)))
            .abs()
                < 1e-12
        );
    }
    assert!(givens_rotation(f64::NAN, 1.).is_none());
    assert!(givens_rotation(1., f64::INFINITY).is_none());
}
#[test]
fn invalid_updates_do_not_modify_destination() {
    let mut c = t(&[1.; 4], (2, 2), DType::F32)
        .try_into_exclusive()
        .unwrap();
    let a = t(&[1.; 6], (2, 3), DType::F32);
    let b = t(&[1.; 4], (2, 2), DType::F32);
    assert!(c.addmm(&a, &b, 1., 0.).is_err());
    check(&c.into_tensor(), &[1.; 4]);
    let bad = t(&[1.; 4], (2, 2), DType::F32).transpose(0, 1).unwrap();
    let mut bad = bad.try_into_exclusive().unwrap();
    let x = t(&[1.; 4], (2, 2), DType::F32);
    assert!(bad.copy_from(&x).is_err());
    let a = t(&[0., 1., 0., 2.], (2, 2), DType::F32);
    assert!(
        a.triangular_solve_with_options(
            &x,
            TriangularOptions {
                upper: true,
                ..Default::default()
            }
        )
        .is_err()
    );
    let a = t(&[1.; 4], (2, 2), DType::F16);
    assert!(a.matrix_copy(false, 1.).is_err());
    assert!(a.argmax_abs().is_err());
}
#[test]
fn zero_shapes_and_offset_views_are_safe() {
    for dtype in [DType::F32, DType::F64] {
        let a = t(&[], (0, 3), dtype);
        let c = t(&[1.; 9], (3, 3), dtype);
        check(&c.gram_update(&a, true, 1., 2., true).unwrap(), &[2.; 9]);
        assert_eq!(a.matrix_copy(true, 1.).unwrap().dims(), &[3, 0]);
        let a = t(&[], (0, 0), dtype);
        let b = t(&[], (0, 2), dtype);
        assert_eq!(
            a.triangular_matmul(&b, true, false).unwrap().dims(),
            &[0, 2]
        );
        let x = t(&[], 0, dtype);
        let mut y = t(&[], 0, dtype).try_into_exclusive().unwrap();
        y.copy_from(&x).unwrap();
        y.axpby(2., &x, 3.).unwrap();
    }
}

#[test]
fn in_place_coefficients_ignore_nan_values_and_work_at_unaligned_offsets() {
    for dtype in [DType::F32, DType::F64] {
        let a = t(&[1., 2., 3., 4.], (2, 2), dtype);
        let b = t(&[5., 6., 7., 8.], (2, 2), dtype);
        let base = t(&[9., f64::NAN, f64::NAN, f64::NAN, f64::NAN, 9.], 6, dtype);
        let view = base.narrow(0, 1, 4).unwrap().reshape((2, 2)).unwrap();
        drop(base);
        let pointer = view.storage_base_ptr();
        let mut dst = view.try_into_exclusive().unwrap();
        dst.addmm(&a, &b, 2., 0.).unwrap();
        let out = dst.into_tensor();
        assert_eq!(pointer, out.storage_base_ptr());
        check(&out, &[38., 44., 86., 100.]);
        let nan = t(&[f64::NAN; 4], (2, 2), dtype);
        let mut dst = t(&[f64::NAN; 4], (2, 2), dtype)
            .try_into_exclusive()
            .unwrap();
        dst.addmm(&nan, &nan, 0., 0.).unwrap();
        check(&dst.into_tensor(), &[0.; 4]);
        let mut dst = t(&[10., 20.], 2, dtype).try_into_exclusive().unwrap();
        let x = t(&[2., 3.], 2, dtype);
        dst.addmv(&a.transpose(0, 1).unwrap(), &x, 2., 3.).unwrap();
        check(&dst.into_tensor(), &[52., 92.]);
    }
}
