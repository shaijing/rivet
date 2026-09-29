use rivet_core::{CpuStorageRef, DType, Device, Error, Shape, Tensor};

fn tensor(values: &[f64], shape: impl Into<Shape>, dtype: DType) -> Tensor {
    Tensor::from_vec(values.to_vec(), shape, &Device::Cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

fn check(tensor: &Tensor, expected: &[f64]) {
    let actual = tensor
        .to_dtype(DType::F64)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    assert_eq!(actual.len(), expected.len());
    for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5 * (1.0 + expected.abs()),
            "element {i}: {actual} != {expected}"
        );
    }
}

#[test]
fn fused_addmm_addmv_use_coefficients_and_allow_read_only_aliases() {
    for dtype in [DType::F32, DType::F64] {
        let a = tensor(&[1.0, 2.0, 3.0, 4.0], (2, 2), dtype);
        let b = tensor(&[5.0, 6.0, 7.0, 8.0], (2, 2), dtype);
        check(
            &a.addmm(&a, &b, 2.0, 3.0).unwrap(),
            &[41.0, 50.0, 95.0, 112.0],
        );
        check(
            &a.addmm(&a, &a, 1.0, 1.0).unwrap(),
            &[8.0, 12.0, 18.0, 26.0],
        );
        check(
            &a.transpose(0, 1).unwrap().addmm(&a, &b, 2.0, 3.0).unwrap(),
            &[41.0, 53.0, 92.0, 112.0],
        );
        let seed = tensor(&[10.0, 20.0], 2, dtype);
        let vector = tensor(&[2.0, 3.0], 2, dtype);
        check(&seed.addmv(&a, &vector, 2.0, 0.5).unwrap(), &[21.0, 46.0]);
        assert!(!a.addmm(&a, &b, 1.0, 1.0).unwrap().same_storage(&a));
    }
}

#[test]
fn outer_addr_support_strided_and_broadcast_vectors() {
    for dtype in [DType::F32, DType::F64] {
        let x = tensor(&[0.0, 1.0, 0.0, 2.0], (2, 2), dtype)
            .get_on_dim(1, 1)
            .unwrap();
        let y = tensor(&[3.0, 4.0, 5.0], 3, dtype);
        check(&x.outer(&y).unwrap(), &[3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
        let seed = Tensor::ones((2, 3), dtype, &Device::Cpu).unwrap();
        check(
            &seed.addr(&x, &y, 2.0, 3.0).unwrap(),
            &[9.0, 11.0, 13.0, 15.0, 19.0, 23.0],
        );
        let broadcast = tensor(&[2.0], 1, dtype).broadcast_as(2).unwrap();
        check(
            &broadcast.outer(&y).unwrap(),
            &[6.0, 8.0, 10.0, 6.0, 8.0, 10.0],
        );
        let empty = Tensor::zeros(0, dtype, &Device::Cpu).unwrap();
        assert_eq!(empty.outer(&y).unwrap().dims(), &[0, 3]);
    }
}

#[test]
fn f64_and_transposed_padded_matmul_and_mv_work() {
    for dtype in [DType::F32, DType::F64] {
        let base = tensor(&[1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0], (2, 4), dtype);
        let padded = base.narrow(1, 0, 3).unwrap();
        let identity = tensor(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            (3, 3),
            dtype,
        );
        check(
            &padded.matmul(&identity).unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let transposed = padded.transpose(0, 1).unwrap();
        let vector = tensor(&[2.0, 3.0], 2, dtype);
        check(&transposed.mv(&vector).unwrap(), &[14.0, 19.0, 24.0]);
        check(
            &transposed.matmul(&padded).unwrap(),
            &[17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0],
        );
        let broadcast = tensor(&[2.0], (1, 1), dtype).broadcast_as((2, 3)).unwrap();
        check(&broadcast.matmul(&identity).unwrap(), &[2.0; 6]);
    }
}

#[test]
fn gram_fills_both_triangles_for_transposed_views() {
    for dtype in [DType::F32, DType::F64] {
        let matrix = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), dtype);
        check(&matrix.gram(false).unwrap(), &[14.0, 32.0, 32.0, 77.0]);
        check(
            &matrix.gram(true).unwrap(),
            &[17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0],
        );
        check(
            &matrix.transpose(0, 1).unwrap().gram(true).unwrap(),
            &[14.0, 32.0, 32.0, 77.0],
        );
        check(
            &matrix.transpose(0, 1).unwrap().gram(false).unwrap(),
            &[17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0],
        );
        let empty = Tensor::zeros((0, 3), dtype, &Device::Cpu).unwrap();
        check(&empty.gram(true).unwrap(), &[0.0; 9]);
    }
}

#[test]
fn symmetric_operations_read_only_the_selected_triangle() {
    for dtype in [DType::F32, DType::F64] {
        let upper = tensor(&[2.0, 3.0, f64::NAN, 5.0], (2, 2), dtype);
        let x = tensor(&[1.0, 2.0], 2, dtype);
        check(&upper.symmetric_mv(&x, true).unwrap(), &[8.0, 13.0]);
        let lower = upper.transpose(0, 1).unwrap();
        check(&lower.symmetric_mv(&x, false).unwrap(), &[8.0, 13.0]);
        let b = tensor(&[1.0, 2.0, 3.0, 4.0], (2, 2), dtype);
        check(
            &upper.symmetric_matmul(&b, true).unwrap(),
            &[11.0, 16.0, 18.0, 26.0],
        );
        check(
            &lower.symmetric_matmul(&b, false).unwrap(),
            &[11.0, 16.0, 18.0, 26.0],
        );
        check(
            &upper
                .symmetric_matmul(&b.transpose(0, 1).unwrap(), true)
                .unwrap(),
            &[8.0, 18.0, 13.0, 29.0],
        );
    }
}

#[test]
fn triangular_solve_supports_upper_lower_unit_and_vector_rhs() {
    for dtype in [DType::F32, DType::F64] {
        let upper = tensor(&[2.0, 3.0, f64::NAN, 4.0], (2, 2), dtype);
        let b = tensor(&[8.0, 8.0], 2, dtype);
        check(
            &upper.triangular_solve(&b, true, false).unwrap(),
            &[1.0, 2.0],
        );
        let lower = upper.transpose(0, 1).unwrap();
        let b = tensor(&[2.0, 11.0], 2, dtype);
        check(
            &lower.triangular_solve(&b, false, false).unwrap(),
            &[1.0, 2.0],
        );
        let b = tensor(&[8.0, 4.0, 8.0, 4.0], (2, 2), dtype);
        check(
            &upper.triangular_solve(&b, true, false).unwrap(),
            &[1.0, 0.5, 2.0, 1.0],
        );
        let unit = tensor(&[f64::NAN, 3.0, f64::NAN, f64::NAN], (2, 2), dtype);
        let b = tensor(&[7.0, 2.0], 2, dtype);
        check(&unit.triangular_solve(&b, true, true).unwrap(), &[1.0, 2.0]);
        let singular = tensor(&[0.0, 1.0, 0.0, 1.0], (2, 2), dtype);
        assert!(matches!(
            singular.triangular_solve(&b, true, false),
            Err(Error::SingularMatrix { index: 0 })
        ));
    }
}

#[test]
fn l1_norm_handles_rank_strides_broadcasts_and_empty_values() {
    for dtype in [DType::F32, DType::F64] {
        let matrix = tensor(&[-1.0, 2.0, -3.0, 4.0], (2, 2), dtype);
        check(&matrix.norm_l1().unwrap(), &[10.0]);
        check(&matrix.transpose(0, 1).unwrap().norm_l1().unwrap(), &[10.0]);
        check(&matrix.get_on_dim(1, 1).unwrap().norm_l1().unwrap(), &[6.0]);
        check(&tensor(&[-2.0], (), dtype).norm_l1().unwrap(), &[2.0]);
        check(
            &tensor(&[-2.0], 1, dtype)
                .broadcast_as(3)
                .unwrap()
                .norm_l1()
                .unwrap(),
            &[6.0],
        );
        check(
            &Tensor::zeros(0, dtype, &Device::Cpu)
                .unwrap()
                .norm_l1()
                .unwrap(),
            &[0.0],
        );
    }
}

#[test]
fn exclusive_axpy_scale_preserve_allocation_offsets_and_ownership() {
    for dtype in [DType::F32, DType::F64] {
        let base = tensor(&[99.0, 1.0, 2.0, 99.0], 4, dtype);
        let view = base.narrow(0, 1, 2).unwrap();
        drop(base);
        let pointer = view.storage_base_ptr();
        let mut exclusive = view.try_into_exclusive().unwrap();
        let x = tensor(&[3.0, 4.0], 2, dtype);
        exclusive.axpy(2.0, &x).unwrap();
        exclusive.scale(0.5).unwrap();
        let output = exclusive.into_tensor();
        assert_eq!(output.storage_base_ptr(), pointer);
        assert_eq!(output.layout().start_offset(), 1);
        check(&output, &[3.5, 5.0]);
        output
            .with_cpu_storage(|storage, _| {
                match storage {
                    CpuStorageRef::F32(values) => assert_eq!(values, &[99.0, 3.5, 5.0, 99.0]),
                    CpuStorageRef::F64(values) => assert_eq!(values, &[99.0, 3.5, 5.0, 99.0]),
                    _ => unreachable!(),
                }
                Ok(())
            })
            .unwrap();
        let alias = output.clone();
        assert!(output.try_into_exclusive().is_err());
        drop(alias);
    }
}

#[test]
fn exclusive_axpy_accepts_strided_sources_and_rejects_strided_destinations() {
    let mut output = Tensor::ones(2, DType::F64, &Device::Cpu)
        .unwrap()
        .try_into_exclusive()
        .unwrap();
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64)
        .get_on_dim(1, 1)
        .unwrap();
    output.axpy(2.0, &x).unwrap();
    check(&output.into_tensor(), &[5.0, 9.0]);
    let base = tensor(&[1.0, 2.0, 3.0, 4.0], (2, 2), DType::F64);
    let transposed = base.transpose(0, 1).unwrap();
    drop(base);
    let mut exclusive = transposed.try_into_exclusive().unwrap();
    assert!(matches!(
        exclusive.scale(2.0),
        Err(Error::UnsupportedLayoutForOp { op: "scale" })
    ));
    check(&exclusive.into_tensor(), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn batched_matmul_handles_broadcast_transpose_and_zero_dimensions() {
    for dtype in [DType::F32, DType::F64] {
        let lhs = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (2, 2, 2), dtype);
        let identity = tensor(&[1.0, 0.0, 0.0, 1.0], (1, 2, 2), dtype);
        check(
            &lhs.broadcast_matmul(&identity).unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        check(
            &lhs.transpose(1, 2)
                .unwrap()
                .broadcast_matmul(&identity)
                .unwrap(),
            &[1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
        );
        let left = lhs.unsqueeze(1).unwrap();
        let right = identity.broadcast_as((1, 3, 2, 2)).unwrap();
        let output = left.broadcast_matmul(&right).unwrap();
        assert_eq!(output.dims(), &[2, 3, 2, 2]);
        check(
            &output,
            &[
                1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,
            ],
        );
        let zero = Tensor::zeros((2, 3, 0), dtype, &Device::Cpu).unwrap();
        let rhs = Tensor::zeros((1, 0, 4), dtype, &Device::Cpu).unwrap();
        check(&zero.broadcast_matmul(&rhs).unwrap(), &[0.0; 24]);
    }
}

#[test]
fn fused_operations_ignore_zero_coefficients_and_validate_shapes() {
    for dtype in [DType::F32, DType::F64] {
        let nan = tensor(&[f64::NAN; 4], (2, 2), dtype);
        let identity = tensor(&[1.0, 0.0, 0.0, 1.0], (2, 2), dtype);
        check(
            &nan.addmm(&identity, &identity, 1.0, 0.0).unwrap(),
            &[1.0, 0.0, 0.0, 1.0],
        );
        check(
            &identity.addmm(&nan, &nan, 0.0, 2.0).unwrap(),
            &[2.0, 0.0, 0.0, 2.0],
        );
        let zero_inner = Tensor::zeros((2, 0), dtype, &Device::Cpu).unwrap();
        let zero_rhs = Tensor::zeros((0, 2), dtype, &Device::Cpu).unwrap();
        check(
            &identity.addmm(&zero_inner, &zero_rhs, 1.0, 2.0).unwrap(),
            &[2.0, 0.0, 0.0, 2.0],
        );
        let bad = Tensor::ones((1, 2), dtype, &Device::Cpu).unwrap();
        assert!(identity.addmm(&bad, &identity, 1.0, 1.0).is_err());
        assert!(identity.gram(true).is_ok());
        assert!(identity.outer(&identity).is_err());
    }
    let integer = Tensor::ones((2, 2), DType::I32, &Device::Cpu).unwrap();
    assert!(matches!(
        integer.addmm(&integer, &integer, 1.0, 1.0),
        Err(Error::UnsupportedDTypeForOp { op: "addmm", .. })
    ));
    let mut integer = integer.try_into_exclusive().unwrap();
    assert!(integer.scale(2.0).is_err());
}

#[test]
fn fused_products_match_scalar_reference_for_padded_transposed_layouts() {
    for dtype in [DType::F32, DType::F64] {
        for (m, k, n) in [(1, 3, 4), (3, 1, 4), (3, 4, 1), (3, 4, 5)] {
            let av = (0..k * (m + 1))
                .map(|i| (i % 11) as f64 * 0.07 - 0.2)
                .collect::<Vec<_>>();
            let bv = (0..(n + 1) * (k + 1))
                .map(|i| (i % 13) as f64 * 0.05 - 0.3)
                .collect::<Vec<_>>();
            let a = tensor(&av, (k, m + 1), dtype)
                .narrow(1, 0, m)
                .unwrap()
                .transpose(0, 1)
                .unwrap();
            let b = tensor(&bv, (n + 1, k + 1), dtype)
                .narrow(0, 1, n)
                .unwrap()
                .narrow(1, 0, k)
                .unwrap()
                .transpose(0, 1)
                .unwrap();
            let c = Tensor::ones((m, n), dtype, &Device::Cpu).unwrap();
            let av = a.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
            let bv = b.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
            let mut expected = vec![0.0; m * n];
            for row in 0..m {
                for col in 0..n {
                    expected[row * n + col] = -0.5
                        + 1.25
                            * (0..k)
                                .map(|inner| av[row * k + inner] * bv[inner * n + col])
                                .sum::<f64>();
                }
            }
            check(&c.addmm(&a, &b, 1.25, -0.5).unwrap(), &expected);
        }
    }
}

#[test]
fn batch_output_handles_unaligned_small_matrix_offsets() {
    for dtype in [DType::F32, DType::F64] {
        let a = tensor(&[1.0, 2.0, 3.0, 4.0], (4, 1, 1), dtype);
        let b = tensor(&[2.0], (1, 1, 1), dtype);
        check(&a.broadcast_matmul(&b).unwrap(), &[2.0, 4.0, 6.0, 8.0]);
        let empty = Tensor::zeros((0, 2, 3), dtype, &Device::Cpu).unwrap();
        let b = Tensor::ones((1, 3, 4), dtype, &Device::Cpu).unwrap();
        assert_eq!(empty.broadcast_matmul(&b).unwrap().dims(), &[0, 2, 4]);
    }
}

#[test]
fn exclusive_zero_updates_and_invalid_inputs_are_consistent() {
    for dtype in [DType::F32, DType::F64] {
        let mut y = Tensor::ones(2, dtype, &Device::Cpu)
            .unwrap()
            .try_into_exclusive()
            .unwrap();
        let nan = tensor(&[f64::NAN, f64::NAN], 2, dtype);
        y.axpy(0.0, &nan).unwrap();
        check(&y.into_tensor(), &[1.0, 1.0]);
        let mut y = nan.try_into_exclusive().unwrap();
        y.scale(0.0).unwrap();
        check(&y.into_tensor(), &[0.0, 0.0]);
        let mut y = Tensor::ones(2, dtype, &Device::Cpu)
            .unwrap()
            .try_into_exclusive()
            .unwrap();
        let bad = Tensor::ones(3, dtype, &Device::Cpu).unwrap();
        assert!(matches!(
            y.axpy(1.0, &bad),
            Err(Error::ShapeMismatchBinary { .. })
        ));
        check(&y.into_tensor(), &[1.0, 1.0]);
        let mut empty = Tensor::zeros(0, dtype, &Device::Cpu)
            .unwrap()
            .try_into_exclusive()
            .unwrap();
        empty.scale(2.0).unwrap();
        empty
            .axpy(1.0, &Tensor::zeros(0, dtype, &Device::Cpu).unwrap())
            .unwrap();
        assert_eq!(empty.into_tensor().elem_count(), 0);
    }
}

#[test]
fn structured_operations_reject_invalid_shapes_and_handle_empty_matrices() {
    for dtype in [DType::F32, DType::F64] {
        let scalar = tensor(&[1.0], (), dtype);
        assert!(scalar.gram(true).is_err());
        let nonsquare = Tensor::ones((2, 3), dtype, &Device::Cpu).unwrap();
        let rhs = Tensor::ones((2, 2), dtype, &Device::Cpu).unwrap();
        assert!(nonsquare.triangular_solve(&rhs, true, false).is_err());
        assert!(nonsquare.symmetric_matmul(&rhs, true).is_err());
        let matrix = Tensor::zeros((0, 0), dtype, &Device::Cpu).unwrap();
        let rhs = Tensor::zeros((0, 2), dtype, &Device::Cpu).unwrap();
        assert_eq!(
            matrix.triangular_solve(&rhs, true, false).unwrap().dims(),
            &[0, 2]
        );
        assert_eq!(matrix.symmetric_matmul(&rhs, true).unwrap().dims(), &[0, 2]);
    }
}
