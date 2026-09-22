#![cfg(feature = "cuda")]

use std::sync::Arc;

use rivet_core::{CudaDevice, DType, Device, DeviceLocation, Error, Tensor};

fn cuda_runtime(device: &Device) -> Arc<CudaDevice> {
    match device {
        Device::Cuda(runtime) => Arc::clone(runtime),
        Device::Cpu => panic!("expected CUDA device"),
    }
}

#[test]
fn cuda_device_owns_a_default_stream_and_synchronizes() {
    let runtime = CudaDevice::new(0).expect("CUDA device 0 should initialize");
    assert_eq!(runtime.ordinal(), 0);
    runtime.synchronize().unwrap();

    let device = Device::cuda(0).unwrap();
    assert!(device.is_cuda());
    assert!(!device.is_cpu());
    assert_eq!(device.location(), DeviceLocation::Cuda { ordinal: 0 });
    assert!(device.same_device(&device.clone()));
}

#[test]
fn cuda_storage_allocates_typed_zeros_ones_and_host_copies() {
    let device = Device::cuda(0).unwrap();

    macro_rules! assert_zero_one {
        ($ty:ty, $dtype:expr, $zero:expr, $one:expr) => {{
            let zeros = Tensor::zeros([3], $dtype, &device).unwrap();
            let ones = Tensor::ones([3], $dtype, &device).unwrap();
            assert_eq!(zeros.dtype(), $dtype);
            assert_eq!(zeros.storage_len(), 3);
            assert_eq!(zeros.to_vec::<$ty>().unwrap(), vec![$zero; 3]);
            assert_eq!(ones.to_vec::<$ty>().unwrap(), vec![$one; 3]);
        }};
    }

    assert_zero_one!(u8, DType::U8, 0, 1);
    assert_zero_one!(u32, DType::U32, 0, 1);
    assert_zero_one!(i16, DType::I16, 0, 1);
    assert_zero_one!(i32, DType::I32, 0, 1);
    assert_zero_one!(i64, DType::I64, 0, 1);
    assert_zero_one!(
        half::bf16,
        DType::BF16,
        half::bf16::from_f32(0.0),
        half::bf16::from_f32(1.0)
    );
    assert_zero_one!(
        half::f16,
        DType::F16,
        half::f16::from_f32(0.0),
        half::f16::from_f32(1.0)
    );
    assert_zero_one!(f32, DType::F32, 0.0, 1.0);
    assert_zero_one!(f64, DType::F64, 0.0, 1.0);

    let values = vec![1.5f32, 2.5, 3.5];
    let from_vec = Tensor::from_vec(values.clone(), [3], &device).unwrap();
    let from_slice = Tensor::from_slice(&values, [3], &device).unwrap();
    assert_eq!(from_vec.to_vec::<f32>().unwrap(), values);
    assert_eq!(from_slice.to_vec::<f32>().unwrap(), values);
}

#[test]
fn unsupported_cuda_math_returns_an_explicit_error() {
    let device = Device::cuda(0).unwrap();
    let tensor = Tensor::ones([2], DType::F32, &device).unwrap();

    assert!(matches!(
        tensor.affine(2.0, 1.0),
        Err(Error::UnsupportedCudaOp { op: "affine" })
    ));
}

#[test]
fn cuda_f32_math_dispatches_static_kernels_without_intermediate_sync() {
    let device = Device::cuda(0).unwrap();
    let runtime = cuda_runtime(&device);
    runtime.reset_debug_stats();

    let lhs = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [4], &device).unwrap();
    let rhs = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], [4], &device).unwrap();
    let negated = {
        let sum = lhs.add(&rhs).unwrap();
        let product = sum.mul(&rhs).unwrap();
        product.neg().unwrap()
    };

    let queued = runtime.debug_stats();
    assert_eq!(queued.synchronize_count, 0);
    assert_eq!(queued.h2d_count, 2);
    assert_eq!(queued.d2h_count, 0);
    assert_eq!(queued.kernel_launch_count, 3);

    assert_eq!(
        negated.to_vec::<f32>().unwrap(),
        vec![-110.0, -440.0, -990.0, -1760.0]
    );
    let completed = runtime.debug_stats();
    assert_eq!(completed.synchronize_count, 1);
    assert_eq!(completed.d2h_count, 1);
}

#[test]
fn cuda_basic_ops_preserve_broadcast_and_view_layouts() {
    let device = Device::cuda(0).unwrap();

    let lhs = Tensor::from_vec(vec![1.0f32, 2.0], [2, 1], &device).unwrap();
    let rhs = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], [1, 3], &device).unwrap();
    let broadcast = lhs.broadcast_add(&rhs).unwrap();
    assert_eq!(
        broadcast.to_vec::<f32>().unwrap(),
        vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
    );

    let scalar = Tensor::from_vec(vec![2.0f32], (), &device).unwrap();
    assert_eq!(
        scalar
            .broadcast_add(&Tensor::from_vec(vec![1.0f32, 3.0], [2], &device).unwrap())
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        vec![3.0, 5.0]
    );

    let source =
        Tensor::from_vec((0..6).map(|value| value as f32).collect(), [2, 3], &device).unwrap();
    let view = source.transpose(0, 1).unwrap();
    assert_eq!(
        view.neg().unwrap().to_vec::<f32>().unwrap(),
        vec![0.0, -3.0, -1.0, -4.0, -2.0, -5.0]
    );
    assert_eq!(
        view.add(&view).unwrap().to_vec::<f32>().unwrap(),
        vec![0.0, 6.0, 2.0, 8.0, 4.0, 10.0]
    );
    assert_eq!(
        view.add_scalar(1.0f32).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    assert_eq!(
        view.gt_scalar(2.5f32).unwrap().to_vec::<u8>().unwrap(),
        vec![0, 1, 0, 1, 0, 1]
    );
}

#[test]
fn cuda_scalar_and_compare_ops_are_typed_and_async() {
    let device = Device::cuda(0).unwrap();
    let runtime = cuda_runtime(&device);
    runtime.reset_debug_stats();

    let values = Tensor::from_vec(vec![-2.0f32, 0.0, 3.0], [3], &device).unwrap();
    let absolute = values.abs().unwrap();
    let added = values.add_scalar(2.0f32).unwrap();
    let multiplied = values.mul_scalar(2.0f32).unwrap();
    let less_than = values.lt_scalar(1.0f32).unwrap();

    let queued = runtime.debug_stats();
    assert_eq!(queued.synchronize_count, 0);
    assert_eq!(queued.d2h_count, 0);
    assert_eq!(queued.kernel_launch_count, 4);

    assert_eq!(absolute.to_vec::<f32>().unwrap(), vec![2.0, 0.0, 3.0]);
    assert_eq!(added.to_vec::<f32>().unwrap(), vec![0.0, 2.0, 5.0]);
    assert_eq!(multiplied.to_vec::<f32>().unwrap(), vec![-4.0, 0.0, 6.0]);
    assert_eq!(less_than.to_vec::<u8>().unwrap(), vec![1, 1, 0]);
}

#[test]
fn cuda_typed_integer_and_half_ops_match_cpu_values() {
    let device = Device::cuda(0).unwrap();

    let ints = Tensor::from_vec(vec![1i32, 2, 3], [3], &device).unwrap();
    assert_eq!(
        ints.add(&ints).unwrap().to_vec::<i32>().unwrap(),
        vec![2, 4, 6]
    );
    assert_eq!(
        ints.sub_scalar(1i32).unwrap().to_vec::<i32>().unwrap(),
        vec![0, 1, 2]
    );
    assert_eq!(
        ints.ge_scalar(2i32).unwrap().to_vec::<u8>().unwrap(),
        vec![0, 1, 1]
    );

    let half = Tensor::from_vec(
        vec![half::f16::from_f32(1.5), half::f16::from_f32(-2.0)],
        [2],
        &device,
    )
    .unwrap();
    assert_eq!(
        half.abs().unwrap().to_vec::<half::f16>().unwrap(),
        vec![half::f16::from_f32(1.5), half::f16::from_f32(2.0)]
    );
}

#[test]
fn cuda_basic_ops_handle_empty_tensors_without_kernel_launches() {
    let device = Device::cuda(0).unwrap();
    let runtime = cuda_runtime(&device);
    let empty = Tensor::zeros([0, 3], DType::F32, &device).unwrap();
    let empty_ones = Tensor::ones([0, 3], DType::F32, &device).unwrap();
    runtime.reset_debug_stats();

    let negated = empty.neg().unwrap();
    let added = empty.add(&empty_ones).unwrap();
    let scalar_added = empty.add_scalar(1.0f32).unwrap();
    let compared = empty.eq_scalar(1.0f32).unwrap();
    let contiguous = empty.contiguous().unwrap();

    assert_eq!(runtime.debug_stats().kernel_launch_count, 0);
    assert!(negated.to_vec::<f32>().unwrap().is_empty());
    assert!(added.to_vec::<f32>().unwrap().is_empty());
    assert!(scalar_added.to_vec::<f32>().unwrap().is_empty());
    assert!(compared.to_vec::<u8>().unwrap().is_empty());
    assert!(contiguous.to_vec::<f32>().unwrap().is_empty());
}

#[test]
fn tensor_device_copy_preserves_contiguous_and_view_logical_order() {
    let cpu = Device::Cpu;
    let cuda = Device::cuda(0).unwrap();
    let source =
        Tensor::from_vec((0..6).map(|value| value as f32).collect(), [2, 3], &cpu).unwrap();

    let gpu = source.to_device(&cuda).unwrap();
    assert!(gpu.is_contiguous());
    assert_eq!(gpu.to_vec::<f32>().unwrap(), vec![0., 1., 2., 3., 4., 5.]);

    let roundtrip = gpu.to_device(&cpu).unwrap();
    assert_eq!(roundtrip.dims(), &[2, 3]);
    assert_eq!(
        roundtrip.to_vec::<f32>().unwrap(),
        vec![0., 1., 2., 3., 4., 5.]
    );

    let narrow = gpu.narrow(0, 1, 1).unwrap();
    let narrow_roundtrip = narrow.to_device(&cpu).unwrap();
    assert_eq!(narrow_roundtrip.dims(), &[1, 3]);
    assert_eq!(narrow_roundtrip.to_vec::<f32>().unwrap(), vec![3., 4., 5.]);

    let gpu_view_roundtrip = gpu.transpose(0, 1).unwrap().to_device(&cpu).unwrap();
    assert_eq!(gpu_view_roundtrip.dims(), &[3, 2]);
    assert_eq!(
        gpu_view_roundtrip.to_vec::<f32>().unwrap(),
        vec![0., 3., 1., 4., 2., 5.]
    );

    let transposed = source.transpose(0, 1).unwrap();
    assert!(!transposed.is_contiguous());
    let transposed_gpu = transposed.to_device(&cuda).unwrap();
    assert!(transposed_gpu.is_contiguous());
    assert_eq!(
        transposed_gpu.to_vec::<f32>().unwrap(),
        vec![0., 3., 1., 4., 2., 5.]
    );

    let transposed_roundtrip = transposed_gpu.to_device(&cpu).unwrap();
    assert_eq!(transposed_roundtrip.dims(), &[3, 2]);
    assert_eq!(
        transposed_roundtrip.to_vec::<f32>().unwrap(),
        vec![0., 3., 1., 4., 2., 5.]
    );

    let empty = Tensor::from_vec(Vec::<f32>::new(), [0, 3], &cpu).unwrap();
    let empty_gpu = empty.to_device(&cuda).unwrap();
    assert_eq!(empty_gpu.shape(), &rivet_core::Shape::from([0, 3]));
    assert!(empty_gpu.to_vec::<f32>().unwrap().is_empty());
}

#[test]
fn cuda_contiguous_materialization_and_same_device_copy_are_explicit() {
    let cuda = Device::cuda(0).unwrap();
    let same_cuda = Device::cuda(0).unwrap();
    let source = Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5], [2, 3], &cuda).unwrap();
    let view = source.transpose(0, 1).unwrap();

    let materialized = view.contiguous().unwrap();
    assert!(materialized.is_contiguous());
    assert!(!materialized.same_storage(&view));
    assert_eq!(
        materialized.to_vec::<u32>().unwrap(),
        vec![0, 3, 1, 4, 2, 5]
    );

    let shared = source.to_device(&same_cuda).unwrap();
    assert!(shared.same_storage(&source));
    assert_eq!(shared.to_vec::<u32>().unwrap(), vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn cuda_transfers_stay_async_until_readback_or_explicit_sync() {
    let device = Device::cuda(0).unwrap();
    let runtime = cuda_runtime(&device);
    runtime.reset_debug_stats();

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3], &device).unwrap();
    let output = input.force_contiguous().unwrap();
    let queued = runtime.debug_stats();

    assert_eq!(queued.synchronize_count, 0);
    assert_eq!(queued.h2d_count, 1);
    assert_eq!(queued.d2d_count, 1);
    assert_eq!(queued.d2h_count, 0);
    assert_eq!(queued.kernel_launch_count, 0);

    output.synchronize().unwrap();
    assert_eq!(runtime.debug_stats().synchronize_count, 1);

    assert_eq!(output.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    let readback = runtime.debug_stats();
    assert_eq!(readback.synchronize_count, 2);
    assert_eq!(readback.d2h_count, 1);
}

#[test]
fn cuda_allocation_lifetime_survives_input_drop_before_sync() {
    let device = Device::cuda(0).unwrap();
    let runtime = cuda_runtime(&device);
    let mut outputs = Vec::with_capacity(1000);

    for value in 0..1000 {
        let input = Tensor::from_vec(vec![value as f32; 64], [64], &device).unwrap();
        let output = input.force_contiguous().unwrap();
        drop(input);
        outputs.push(output);
    }

    runtime.synchronize().unwrap();

    for (value, output) in outputs.iter().enumerate() {
        assert_eq!(output.to_vec::<f32>().unwrap(), vec![value as f32; 64]);
    }
}
