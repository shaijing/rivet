use rivet_core::{DType, Device, Shape, Tensor};

fn values(shape: impl Into<Shape>, dtype: DType) -> Tensor {
    let shape = shape.into();
    Tensor::from_vec(
        (0..shape.elem_count())
            .map(|i| (i % 17) as f64 * 0.125 - 1.0)
            .collect::<Vec<_>>(),
        shape,
        &Device::Cpu,
    )
    .unwrap()
    .to_dtype(dtype)
    .unwrap()
}

fn check(actual: &Tensor, expected: &[f64]) {
    let actual = actual
        .to_dtype(DType::F64)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 2e-5 * (1.0 + e.abs()),
            "element {i}: {a} != {e}"
        );
    }
}

fn reference(a: &Tensor, b: &Tensor) -> Vec<f64> {
    let rank = a.rank();
    let (m, k, n) = (a.dims()[rank - 2], a.dims()[rank - 1], b.dims()[rank - 1]);
    let count: usize = a.dims()[..rank - 2].iter().product();
    let av = a.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
    let bv = b.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
    let mut out = vec![0.0; count * m * n];
    for batch in 0..count {
        for row in 0..m {
            for col in 0..n {
                for inner in 0..k {
                    out[(batch * m + row) * n + col] +=
                        av[(batch * m + row) * k + inner] * bv[(batch * k + inner) * n + col];
                }
            }
        }
    }
    out
}

#[test]
fn batched_products_match_scalar_reference_for_regular_and_fallback_views() {
    for dtype in [DType::F32, DType::F64] {
        for (m, k, n) in [(4, 4, 4), (3, 4, 5), (4, 4, 1), (4, 1, 4)] {
            let pairs = [
                (values((m, k), dtype), values((k, n), dtype)),
                (values((6, m, k), dtype), values((6, k, n), dtype)),
                (
                    values((1, m, k), dtype).broadcast_as((6, m, k)).unwrap(),
                    values((6, k, n), dtype),
                ),
                (
                    values((6, m, k), dtype),
                    values((1, n, k), dtype)
                        .transpose(1, 2)
                        .unwrap()
                        .broadcast_as((6, k, n))
                        .unwrap(),
                ),
                (
                    values((6, m, k, 2), dtype).get_on_dim(3, 0).unwrap(),
                    values((1, k, n), dtype).broadcast_as((6, k, n)).unwrap(),
                ),
                (
                    values((6, k, m), dtype).transpose(1, 2).unwrap(),
                    values((6, n, k), dtype).transpose(1, 2).unwrap(),
                ),
                (
                    values((7, m + 1, k + 1), dtype)
                        .narrow(0, 1, 6)
                        .unwrap()
                        .narrow(1, 1, m)
                        .unwrap()
                        .narrow(2, 1, k)
                        .unwrap(),
                    values((7, k + 1, n + 1), dtype)
                        .narrow(0, 1, 6)
                        .unwrap()
                        .narrow(1, 1, k)
                        .unwrap()
                        .narrow(2, 1, n)
                        .unwrap(),
                ),
                (
                    values((6, m, k, 2), dtype).get_on_dim(3, 0).unwrap(),
                    values((6, k, n, 2), dtype).get_on_dim(3, 0).unwrap(),
                ),
                (
                    values((2, 3, m, k), dtype),
                    values((1, 1, k, n), dtype)
                        .broadcast_as((2, 3, k, n))
                        .unwrap(),
                ),
                (
                    values((2, 1, m, k), dtype)
                        .broadcast_as((2, 3, m, k))
                        .unwrap(),
                    values((1, 3, k, n), dtype)
                        .broadcast_as((2, 3, k, n))
                        .unwrap(),
                ),
                (
                    values((2, 3, m, k), dtype),
                    values((3, 2, k, n), dtype).transpose(0, 1).unwrap(),
                ),
                (
                    values((2, 1, 3, m, k), dtype),
                    values((1, 1, 1, k, n), dtype)
                        .broadcast_as((2, 1, 3, k, n))
                        .unwrap(),
                ),
            ];
            for (a, b) in pairs {
                let expected = reference(&a, &b);
                let actual = a.broadcast_matmul(&b).unwrap();
                assert!(actual.is_contiguous());
                assert!(!actual.same_storage(&a));
                check(&actual, &expected);
                let c = values(actual.dims().to_vec(), dtype);
                let cv = c.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
                check(
                    &c.baddbmm(&a, &b, -0.75, 0.5).unwrap(),
                    &expected
                        .iter()
                        .zip(&cv)
                        .map(|(p, c)| -0.75 * p + 0.5 * c)
                        .collect::<Vec<_>>(),
                );
                let mut reduced = vec![0.0; m * n];
                for product in expected.chunks(m * n) {
                    for (out, p) in reduced.iter_mut().zip(product) {
                        *out += -0.75 * p;
                    }
                }
                let c = values((m, n), dtype);
                let cv = c.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
                for (out, c) in reduced.iter_mut().zip(cv) {
                    *out += 0.5 * c;
                }
                check(&c.addbmm(&a, &b, -0.75, 0.5).unwrap(), &reduced);
            }
        }
    }
}

#[test]
fn batched_zero_coefficients_ignore_nan_inputs_and_reduction_scales_seed_once() {
    for dtype in [DType::F32, DType::F64] {
        let a = Tensor::ones((3, 4, 4), dtype, &Device::Cpu).unwrap();
        let nan = Tensor::from_vec(vec![f64::NAN; 48], (3, 4, 4), &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let seed = Tensor::ones((4, 4), dtype, &Device::Cpu).unwrap();
        check(&seed.addbmm(&a, &a, 2.0, 3.0).unwrap(), &[27.0; 16]);
        check(&seed.addbmm(&nan, &nan, 0.0, 3.0).unwrap(), &[3.0; 16]);
        check(&nan.baddbmm(&a, &a, 1.0, 0.0).unwrap(), &[4.0; 48]);
        check(
            &nan.get(0).unwrap().addbmm(&a, &a, 1.0, 0.0).unwrap(),
            &[12.0; 16],
        );
        let propagated = a
            .broadcast_matmul(&nan)
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap();
        assert!(
            propagated
                .to_vec::<f64>()
                .unwrap()
                .iter()
                .all(|v| v.is_nan())
        );
    }
}

#[test]
fn empty_batches_and_zero_inner_dimensions_preserve_seed_semantics() {
    for dtype in [DType::F32, DType::F64] {
        for (count, m, k, n) in [(0, 4, 4, 4), (3, 0, 4, 4), (3, 4, 0, 4), (3, 4, 4, 0)] {
            let a = Tensor::zeros((count, m, k), dtype, &Device::Cpu).unwrap();
            let b = Tensor::zeros((count, k, n), dtype, &Device::Cpu).unwrap();
            let c = Tensor::ones((count, m, n), dtype, &Device::Cpu).unwrap();
            let sum = Tensor::ones((m, n), dtype, &Device::Cpu).unwrap();
            assert_eq!(a.broadcast_matmul(&b).unwrap().dims(), &[count, m, n]);
            check(
                &c.baddbmm(&a, &b, f64::NAN, 2.0).unwrap(),
                &vec![2.0; count * m * n],
            );
            check(
                &sum.addbmm(&a, &b, f64::NAN, 2.0).unwrap(),
                &vec![2.0; m * n],
            );
        }
    }
}

#[test]
fn batch_reduction_retains_logical_accumulation_order() {
    for (dtype, large) in [(DType::F32, 16777216.0), (DType::F64, 9007199254740992.0)] {
        let mut av = vec![0.0; 48];
        let mut identity = vec![0.0; 16];
        for i in 0..4 {
            identity[i * 4 + i] = 1.0;
            for (batch, value) in [large, 1.0, -large].into_iter().enumerate() {
                av[batch * 16 + i * 4 + i] = value;
            }
        }
        let a = Tensor::from_vec(av, (3, 4, 4), &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let b = Tensor::from_vec(identity, (1, 4, 4), &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let seed = Tensor::ones((4, 4), dtype, &Device::Cpu).unwrap();
        check(&seed.addbmm(&a, &b, 1.0, 0.0).unwrap(), &[0.0; 16]);
    }
}

#[test]
fn empty_batches_allow_huge_inactive_dimensions() {
    let a = Tensor::zeros((0, usize::MAX, 0), DType::F32, &Device::Cpu).unwrap();
    let b = Tensor::zeros((0, 0, 2), DType::F32, &Device::Cpu).unwrap();
    let output = a.broadcast_matmul(&b).unwrap();
    assert_eq!(output.dims(), &[0, usize::MAX, 2]);
    assert_eq!(output.elem_count(), 0);
}
