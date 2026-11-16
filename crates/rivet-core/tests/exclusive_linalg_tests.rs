use rivet_core::{CpuStorageRef, DType, Device, Error, ExclusiveTensor, Shape, Tensor};

fn t(values: &[f64], shape: impl Into<Shape>, dtype: DType) -> Tensor {
    Tensor::from_vec(values.to_vec(), shape, &Device::Cpu)
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
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 2e-5 * (1.0 + b.abs()),
            "element {index}: {a} != {b}"
        );
    }
}

fn destination(
    seed: &[f64],
    shape: impl Into<Shape>,
    dtype: DType,
    offset: usize,
) -> (ExclusiveTensor, *const u8) {
    let mut data = vec![99.0; offset];
    data.extend_from_slice(seed);
    data.extend_from_slice(&[99.0; 2]);
    let base = t(&data, data.len(), dtype);
    let view = base
        .narrow(0, offset, seed.len())
        .unwrap()
        .reshape(shape)
        .unwrap();
    drop(base);
    let pointer = view.storage_base_ptr();
    (view.try_into_exclusive().unwrap(), pointer)
}

fn finish(out: ExclusiveTensor, pointer: *const u8, offset: usize, expected: &[f64]) {
    let out = out.into_tensor();
    assert_eq!(
        out.storage_base_ptr(),
        pointer,
        "output allocation was replaced"
    );
    assert_eq!(out.layout().start_offset(), offset);
    check(&out, expected);
    out.with_cpu_storage(|storage, _| {
        match storage {
            CpuStorageRef::F32(v) => {
                assert!(v[..offset].iter().all(|v| *v == 99.0));
                assert_eq!(&v[offset + expected.len()..], &[99.0; 2]);
            }
            CpuStorageRef::F64(v) => {
                assert!(v[..offset].iter().all(|v| *v == 99.0));
                assert_eq!(&v[offset + expected.len()..], &[99.0; 2]);
            }
            _ => unreachable!(),
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn batches_reuse_offset_outputs_for_shared_and_partially_broadcast_inputs() {
    for dtype in [DType::F32, DType::F64] {
        let av = (0..24).map(|i| i as f64 * 0.125 - 1.0).collect::<Vec<_>>();
        let bv = (0..24).map(|i| i as f64 * 0.25 - 2.0).collect::<Vec<_>>();
        let pairs = [
            (
                t(&av, (2, 3, 4), dtype),
                t(&bv[..8], (1, 2, 4), dtype).transpose(1, 2).unwrap(),
            ),
            (t(&av, (2, 1, 3, 4), dtype), t(&bv, (1, 3, 4, 2), dtype)),
        ];
        for (a, b) in pairs {
            let av = values(&a);
            let bv = values(&b);
            let shape = a.broadcast_matmul(&b).unwrap().shape().clone();
            for offset in [1, 4] {
                let seed = (0..shape.elem_count())
                    .map(|i| i as f64 * 0.125)
                    .collect::<Vec<_>>();
                let mut expected = t(&seed, shape.clone(), dtype);
                let (mut out, pointer) = destination(&seed, shape.clone(), dtype, offset);
                for _ in 0..3 {
                    expected = expected.baddbmm(&a, &b, -0.5, 0.25).unwrap();
                    out.baddbmm(&a, &b, -0.5, 0.25).unwrap();
                }
                finish(out, pointer, offset, &values(&expected));
                let seed = vec![0.5; 6];
                let mut expected = t(&seed, (3, 2), dtype);
                let (mut out, pointer) = destination(&seed, (3, 2), dtype, offset);
                for _ in 0..3 {
                    expected = expected.addbmm(&a, &b, 0.5, -0.25).unwrap();
                    out.addbmm(&a, &b, 0.5, -0.25).unwrap();
                }
                finish(out, pointer, offset, &values(&expected));
            }
            check(&a, &av);
            check(&b, &bv);
        }
    }
}

#[test]
fn gram_overwrites_nan_outputs_for_transposed_and_inner_strided_sources() {
    for dtype in [DType::F32, DType::F64] {
        let data = (0..24).map(|i| i as f64 * 0.125 - 1.0).collect::<Vec<_>>();
        let inputs = [
            t(&data[..12], (4, 3), dtype),
            t(&data[..12], (3, 4), dtype).transpose(0, 1).unwrap(),
            t(&data, (4, 3, 2), dtype).get_on_dim(2, 0).unwrap(),
        ];
        for a in inputs {
            let av = values(&a);
            for transpose in [false, true] {
                let (rows, cols) = (a.dims()[0], a.dims()[1]);
                let (n, k) = if transpose {
                    (cols, rows)
                } else {
                    (rows, cols)
                };
                let mut expected = vec![0.0; n * n];
                for row in 0..n {
                    for col in 0..n {
                        for inner in 0..k {
                            let (x, y) = if transpose {
                                (av[inner * cols + row], av[inner * cols + col])
                            } else {
                                (av[row * cols + inner], av[col * cols + inner])
                            };
                            expected[row * n + col] += x * y;
                        }
                    }
                }
                for offset in [1, 4] {
                    let (mut out, pointer) =
                        destination(&vec![f64::NAN; n * n], (n, n), dtype, offset);
                    out.gram(&a, transpose).unwrap();
                    finish(out, pointer, offset, &expected);
                }
            }
        }
    }
}

#[test]
fn symmetric_updates_read_only_the_selected_seed_triangle() {
    for dtype in [DType::F32, DType::F64] {
        let x = t(&[2.0, 3.0], 2, dtype);
        let y = t(&[4.0, 5.0], 2, dtype);
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), dtype);
        for upper in [false, true] {
            let seed = if upper {
                [1.0, 2.0, f64::NAN, 3.0]
            } else {
                [1.0, f64::NAN, 2.0, 3.0]
            };
            for offset in [1, 4] {
                let (mut out, pointer) = destination(&seed, (2, 2), dtype, offset);
                out.symmetric_rank1_update(&x, 2.0, 3.0, upper).unwrap();
                finish(out, pointer, offset, &[11.0, 18.0, 18.0, 27.0]);
                let (mut out, pointer) = destination(&seed, (2, 2), dtype, offset);
                out.symmetric_rank2_update(&x, &y, 2.0, 3.0, upper).unwrap();
                finish(out, pointer, offset, &[35.0, 50.0, 50.0, 69.0]);
                let (mut out, pointer) = destination(&seed, (2, 2), dtype, offset);
                out.gram_update(&a, false, 2.0, 3.0, upper).unwrap();
                finish(out, pointer, offset, &[31.0, 70.0, 70.0, 163.0]);
                let (mut out, pointer) = destination(&seed, (2, 2), dtype, offset);
                // Read-only inputs may alias each other.
                out.symmetric_rank2k_update(&a, &a.clone(), false, 2.0, 3.0, upper)
                    .unwrap();
                finish(out, pointer, offset, &[59.0, 134.0, 134.0, 317.0]);
            }
        }
    }
}

#[test]
fn rank_k_updates_support_transpose_and_nonmatching_physical_layouts() {
    for dtype in [DType::F32, DType::F64] {
        let a = t(&[1., 2., 3., 4., 5., 6.], (2, 3), dtype);
        let b = t(&[6., 5., 4., 3., 2., 1.], (3, 2), dtype)
            .transpose(0, 1)
            .unwrap();
        for transpose in [false, true] {
            let n = a.dims()[usize::from(transpose)];
            for upper in [false, true] {
                let seed = (0..n * n).map(|i| i as f64).collect::<Vec<_>>();
                let c = t(&seed, (n, n), dtype);
                let expected = c
                    .symmetric_rank2k_update(&a, &b, transpose, -0.5, 2.0, upper)
                    .unwrap();
                let (mut out, pointer) = destination(&seed, (n, n), dtype, 1);
                out.symmetric_rank2k_update(&a, &b, transpose, -0.5, 2.0, upper)
                    .unwrap();
                finish(out, pointer, 1, &values(&expected));
                let expected = c.gram_update(&b, transpose, -0.5, 2.0, upper).unwrap();
                let (mut out, pointer) = destination(&seed, (n, n), dtype, 1);
                out.gram_update(&b, transpose, -0.5, 2.0, upper).unwrap();
                finish(out, pointer, 1, &values(&expected));
            }
        }
    }
}

#[test]
fn zero_coefficients_and_zero_inner_dimensions_preserve_seed_contracts() {
    for dtype in [DType::F32, DType::F64] {
        let a = Tensor::ones((3, 2, 2), dtype, &Device::Cpu).unwrap();
        let nan_batch = t(&[f64::NAN; 12], (3, 2, 2), dtype);
        let nan_matrix = t(&[f64::NAN; 4], (2, 2), dtype);
        let nan_vector = t(&[f64::NAN; 2], 2, dtype);
        let matrix = Tensor::ones((2, 2), dtype, &Device::Cpu).unwrap();
        let vector = Tensor::ones(2, dtype, &Device::Cpu).unwrap();
        for upper in [false, true] {
            let (mut out, pointer) = destination(&[f64::NAN; 4], (2, 2), dtype, 1);
            out.symmetric_rank1_update(&vector, 1.0, 0.0, upper)
                .unwrap();
            finish(out, pointer, 1, &[1.0; 4]);
            let (mut out, pointer) = destination(&[f64::NAN; 4], (2, 2), dtype, 1);
            out.symmetric_rank2_update(&vector, &vector, 1.0, 0.0, upper)
                .unwrap();
            finish(out, pointer, 1, &[2.0; 4]);
            let (mut out, pointer) = destination(&[f64::NAN; 4], (2, 2), dtype, 1);
            out.gram_update(&matrix, false, 1.0, 0.0, upper).unwrap();
            finish(out, pointer, 1, &[2.0; 4]);
            let (mut out, pointer) = destination(&[f64::NAN; 4], (2, 2), dtype, 1);
            out.symmetric_rank2k_update(&matrix, &matrix, false, 1.0, 0.0, upper)
                .unwrap();
            finish(out, pointer, 1, &[4.0; 4]);
        }
        let (mut out, pointer) = destination(&[f64::NAN; 12], (3, 2, 2), dtype, 1);
        out.baddbmm(&a, &a, 1.0, 0.0).unwrap();
        out.baddbmm(&nan_batch, &nan_batch, 0.0, 2.0).unwrap();
        finish(out, pointer, 1, &[4.0; 12]);
        let (mut out, pointer) = destination(&[f64::NAN; 4], (2, 2), dtype, 1);
        out.addbmm(&a, &a, 1.0, 0.0).unwrap();
        out.addbmm(&nan_batch, &nan_batch, 0.0, 2.0).unwrap();
        finish(out, pointer, 1, &[12.0; 4]);
        let (mut out, pointer) = destination(&[1.0; 4], (2, 2), dtype, 1);
        out.symmetric_rank1_update(&nan_vector, 0.0, 2.0, false)
            .unwrap();
        out.symmetric_rank2_update(&nan_vector, &nan_vector, 0.0, 2.0, true)
            .unwrap();
        out.gram_update(&nan_matrix, false, 0.0, 2.0, false)
            .unwrap();
        out.symmetric_rank2k_update(&nan_matrix, &nan_matrix, true, 0.0, 2.0, true)
            .unwrap();
        finish(out, pointer, 1, &[16.0; 4]);
        let empty = Tensor::zeros((2, 0), dtype, &Device::Cpu).unwrap();
        let (mut out, pointer) = destination(&[f64::NAN; 4], (2, 2), dtype, 1);
        out.gram(&empty, false).unwrap();
        finish(out, pointer, 1, &[0.0; 4]);
        let a = Tensor::zeros((3, 2, 0), dtype, &Device::Cpu).unwrap();
        let b = Tensor::zeros((3, 0, 2), dtype, &Device::Cpu).unwrap();
        let (mut out, pointer) = destination(&[1.0; 4], (2, 2), dtype, 1);
        out.addbmm(&a, &b, f64::NAN, 2.0).unwrap();
        finish(out, pointer, 1, &[2.0; 4]);
        let a = Tensor::zeros((0, 2, 2), dtype, &Device::Cpu).unwrap();
        let (mut out, pointer) = destination(&[1.0; 4], (2, 2), dtype, 1);
        out.addbmm(&a, &a, f64::NAN, 2.0).unwrap();
        finish(out, pointer, 1, &[2.0; 4]);
        let (mut out, pointer) = destination(&[], (0, 2, 2), dtype, 1);
        out.baddbmm(&a, &a, f64::NAN, 0.0).unwrap();
        finish(out, pointer, 1, &[]);
        let empty = Tensor::zeros(0, dtype, &Device::Cpu).unwrap();
        let (mut out, pointer) = destination(&[], (0, 0), dtype, 1);
        out.symmetric_rank1_update(&empty, 1.0, 0.0, true).unwrap();
        finish(out, pointer, 1, &[]);
    }
}

#[test]
fn invalid_sources_and_shapes_do_not_change_the_destination() {
    for dtype in [DType::F32, DType::F64] {
        let a = Tensor::ones((2, 3), dtype, &Device::Cpu).unwrap();
        let bad = Tensor::ones(3, dtype, &Device::Cpu).unwrap();
        let matrix = Tensor::ones((2, 2), dtype, &Device::Cpu).unwrap();
        let other_dtype = Tensor::ones((2, 2), DType::I32, &Device::Cpu).unwrap();
        let (mut out, pointer) = destination(&[1., 2., 3., 4.], (2, 2), dtype, 1);
        assert!(out.baddbmm(&a, &a, 1.0, 0.0).is_err());
        assert!(out.addbmm(&a, &a, 1.0, 0.0).is_err());
        assert!(out.gram(&bad, false).is_err());
        assert!(out.gram(&a, true).is_err());
        assert!(out.gram_update(&a, true, 1.0, 0.0, true).is_err());
        assert!(out.symmetric_rank1_update(&bad, 1.0, 0.0, true).is_err());
        assert!(
            out.symmetric_rank2_update(&bad, &bad, 1.0, 0.0, false)
                .is_err()
        );
        assert!(
            out.symmetric_rank2k_update(&a, &matrix, false, 1.0, 0.0, true)
                .is_err()
        );
        assert!(matches!(
            out.gram(&other_dtype, false),
            Err(Error::DTypeMismatch { .. })
        ));
        finish(out, pointer, 1, &[1., 2., 3., 4.]);
        // A valid product with a wrong batch output shape must not scale C.
        let batches = Tensor::ones((2, 2, 2), dtype, &Device::Cpu).unwrap();
        let (mut out, pointer) = destination(&[1.; 4], (2, 2), dtype, 1);
        assert!(out.baddbmm(&batches, &batches, 1.0, 0.0).is_err());
        finish(out, pointer, 1, &[1.; 4]);
    }
}

#[test]
fn noncontiguous_destinations_are_rejected_without_writes() {
    let base = t(&[1., 2., 3., 4.], (2, 2), DType::F64);
    let view = base.transpose(0, 1).unwrap();
    drop(base);
    let mut out = view.try_into_exclusive().unwrap();
    let a = Tensor::ones((2, 2), DType::F64, &Device::Cpu).unwrap();
    let x = Tensor::ones(2, DType::F64, &Device::Cpu).unwrap();
    assert!(matches!(
        out.gram(&a, false),
        Err(Error::UnsupportedLayoutForOp { .. })
    ));
    assert!(out.baddbmm(&a, &a, 1.0, 0.0).is_err());
    assert!(out.addbmm(&a, &a, 1.0, 0.0).is_err());
    assert!(out.gram_update(&a, false, 1.0, 0.0, true).is_err());
    assert!(out.symmetric_rank1_update(&x, 1.0, 0.0, true).is_err());
    assert!(out.symmetric_rank2_update(&x, &x, 1.0, 0.0, true).is_err());
    assert!(
        out.symmetric_rank2k_update(&a, &a, false, 1.0, 0.0, true)
            .is_err()
    );
    check(&out.into_tensor(), &[1., 3., 2., 4.]);
}

#[test]
fn ownership_transfer_prevents_destination_aliases_and_integer_updates_are_rejected() {
    let tensor = Tensor::ones((2, 2), DType::F64, &Device::Cpu).unwrap();
    let alias = tensor.clone();
    let tensor = tensor.try_into_exclusive().unwrap_err();
    drop(alias);
    let alias = tensor.narrow(0, 0, 1).unwrap();
    let tensor = tensor.try_into_exclusive().unwrap_err();
    drop(alias);
    assert!(tensor.try_into_exclusive().is_ok());
    let a = Tensor::ones((2, 2), DType::I32, &Device::Cpu).unwrap();
    let mut out = Tensor::ones((2, 2), DType::I32, &Device::Cpu)
        .unwrap()
        .try_into_exclusive()
        .unwrap();
    assert!(matches!(
        out.gram(&a, false),
        Err(Error::UnsupportedDTypeForOp { .. })
    ));
    assert!(out.baddbmm(&a, &a, 1.0, 0.0).is_err());
    assert_eq!(out.into_tensor().to_vec::<i32>().unwrap(), [1; 4]);
}
