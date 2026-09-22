use cudarc::driver::{CudaSlice, CudaView, LaunchConfig, PushKernelArg};

use super::device::CudaDevice;
use crate::{Error, Result};

fn launch_config(numel: usize) -> Result<LaunchConfig> {
    let numel = u32::try_from(numel).map_err(|_| Error::ShapeElementCountOverflow)?;
    Ok(LaunchConfig::for_num_elems(numel))
}

fn launch_error(function: &'static str, error: impl std::fmt::Display) -> Error {
    Error::CudaOperationFailed {
        op: "kernel_launch",
        message: format!("{function}: {error}"),
    }
}

#[allow(dead_code)]
pub(crate) fn copy_f32(
    device: &CudaDevice,
    src: &CudaView<'_, f32>,
    dst: &mut CudaSlice<f32>,
    numel: usize,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    let function = device.get_or_load_func(rivet_kernels::COPY, "copy_f32")?;
    let stream = device.cuda_stream();
    let config = launch_config(numel)?;
    unsafe {
        stream
            .launch_builder(&function)
            .arg(src)
            .arg(dst)
            .arg(&numel)
            .launch(config)
    }
    .map_err(|error| launch_error("copy_f32", error))?;
    device.record_kernel_launch();
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn fill_f32(
    device: &CudaDevice,
    dst: &mut CudaSlice<f32>,
    value: f32,
    numel: usize,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    let function = device.get_or_load_func(rivet_kernels::FILL, "fill_f32")?;
    let stream = device.cuda_stream();
    let config = launch_config(numel)?;
    unsafe {
        stream
            .launch_builder(&function)
            .arg(dst)
            .arg(&value)
            .arg(&numel)
            .launch(config)
    }
    .map_err(|error| launch_error("fill_f32", error))?;
    device.record_kernel_launch();
    Ok(())
}

pub(crate) fn unary_f32(
    device: &CudaDevice,
    function_name: &'static str,
    src: &CudaView<'_, f32>,
    dst: &mut CudaSlice<f32>,
    numel: usize,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    let function = device.get_or_load_func(rivet_kernels::UNARY, function_name)?;
    let stream = device.cuda_stream();
    let config = launch_config(numel)?;
    unsafe {
        stream
            .launch_builder(&function)
            .arg(src)
            .arg(dst)
            .arg(&numel)
            .launch(config)
    }
    .map_err(|error| launch_error(function_name, error))?;
    device.record_kernel_launch();
    Ok(())
}

pub(crate) fn binary_f32(
    device: &CudaDevice,
    function_name: &'static str,
    lhs: &CudaView<'_, f32>,
    rhs: &CudaView<'_, f32>,
    dst: &mut CudaSlice<f32>,
    numel: usize,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    let function = device.get_or_load_func(rivet_kernels::BINARY, function_name)?;
    let stream = device.cuda_stream();
    let config = launch_config(numel)?;
    unsafe {
        stream
            .launch_builder(&function)
            .arg(lhs)
            .arg(rhs)
            .arg(dst)
            .arg(&numel)
            .launch(config)
    }
    .map_err(|error| launch_error(function_name, error))?;
    device.record_kernel_launch();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_cache_reuses_modules_and_functions() {
        let device = CudaDevice::new(0).unwrap();
        assert_eq!(device.debug_module_counts(), (0, 0));

        let _first = device
            .get_or_load_func(rivet_kernels::UNARY, "neg_f32")
            .unwrap();
        assert_eq!(device.debug_module_counts(), (1, 1));

        let _second = device
            .get_or_load_func(rivet_kernels::UNARY, "neg_f32")
            .unwrap();
        assert_eq!(device.debug_module_counts(), (1, 1));

        let _third = device
            .get_or_load_func(rivet_kernels::UNARY, "identity_f32")
            .unwrap();
        assert_eq!(device.debug_module_counts(), (1, 2));
    }

    #[test]
    fn static_kernels_preserve_single_stream_ordering() {
        let device = CudaDevice::new(0).unwrap();
        device.reset_debug_stats();
        let stream = device.cuda_stream();
        let input = stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let mut copied = stream.alloc_zeros::<f32>(4).unwrap();
        let mut filled = stream.alloc_zeros::<f32>(4).unwrap();
        let mut added = stream.alloc_zeros::<f32>(4).unwrap();
        let mut negated = stream.alloc_zeros::<f32>(4).unwrap();

        copy_f32(&device, &input.as_view(), &mut copied, 4).unwrap();
        fill_f32(&device, &mut filled, 2.0, 4).unwrap();
        binary_f32(
            &device,
            "add_f32",
            &copied.as_view(),
            &filled.as_view(),
            &mut added,
            4,
        )
        .unwrap();
        unary_f32(&device, "neg_f32", &added.as_view(), &mut negated, 4).unwrap();

        let queued = device.debug_stats();
        assert_eq!(queued.synchronize_count, 0);
        assert_eq!(queued.kernel_launch_count, 4);

        device.synchronize().unwrap();
        let values = stream.clone_dtoh(&negated).unwrap();
        device.synchronize().unwrap();
        assert_eq!(values, vec![-3.0, -4.0, -5.0, -6.0]);
    }
}
