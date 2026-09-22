#![cfg(feature = "cuda")]

use rivet_core::{CudaDevice, Device, DeviceLocation};

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
