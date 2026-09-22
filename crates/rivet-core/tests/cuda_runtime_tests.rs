#![cfg(feature = "cuda")]

use rivet_core::{CudaDevice, DType, Device, DeviceLocation, Error, Tensor};

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
        tensor.add(&tensor),
        Err(Error::UnsupportedCudaOp { op: "binary" })
    ));
}
