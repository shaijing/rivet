use cudarc::driver::{CudaSlice, CudaView, DeviceRepr, LaunchConfig, PushKernelArg};

use super::device::CudaDevice;
use super::storage::{CudaStorageSlice, CudaStorageView};
use crate::cpu_backend::CpuStorageRef;
use crate::ops::{BinaryOp, CmpOp, UnaryOp};
use crate::{Error, Layout, Result, WithDType};

struct LayoutInfo {
    values: CudaSlice<usize>,
    rank: usize,
}

fn launch_config(numel: usize) -> Result<LaunchConfig> {
    let numel = u32::try_from(numel).map_err(|_| Error::ShapeElementCountOverflow)?;
    Ok(LaunchConfig::for_num_elems(numel))
}

fn cuda_error(op: &'static str, name: &'static str, error: impl std::fmt::Display) -> Error {
    Error::CudaOperationFailed {
        op,
        message: format!("{name}: {error}"),
    }
}

fn layout_info(device: &CudaDevice, layout: &Layout) -> Result<LayoutInfo> {
    let mut values = Vec::with_capacity(layout.dims().len() * 2);
    values.extend_from_slice(layout.dims());
    values.extend_from_slice(layout.stride());
    if !values.is_empty() {
        device.record_h2d();
    }
    let values = device
        .cuda_stream()
        .clone_htod(&values)
        .map_err(|error| cuda_error("kernel_layout_info", "layout", error))?;
    Ok(LayoutInfo {
        values,
        rank: layout.dims().len(),
    })
}

fn layout_view<'a>(data: &'a CudaStorageSlice, layout: &Layout) -> Result<CudaStorageView<'a>> {
    let start = layout.start_offset();
    let len = data
        .len()
        .checked_sub(start)
        .ok_or(Error::StorageOutOfBounds)?;
    data.view(start, len)
}

fn launch_fill<T: DeviceRepr>(
    device: &CudaDevice,
    function_name: &'static str,
    dst: &mut CudaSlice<T>,
    value: T,
    numel: usize,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    let function = device.get_or_load_func(rivet_kernels::FILL, function_name)?;
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
    .map_err(|error| cuda_error("kernel_launch", function_name, error))?;
    device.record_kernel_launch();
    Ok(())
}

fn launch_copy_layout<T: DeviceRepr>(
    device: &CudaDevice,
    function_name: &'static str,
    src: &CudaView<'_, T>,
    dst: &mut CudaSlice<T>,
    layout: &Layout,
    numel: usize,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    let info = layout_info(device, layout)?;
    let function = device.get_or_load_func(rivet_kernels::COPY, function_name)?;
    let stream = device.cuda_stream();
    let config = launch_config(numel)?;
    unsafe {
        stream
            .launch_builder(&function)
            .arg(&numel)
            .arg(&info.rank)
            .arg(&info.values)
            .arg(src)
            .arg(dst)
            .launch(config)
    }
    .map_err(|error| cuda_error("kernel_launch", function_name, error))?;
    device.record_kernel_launch();
    Ok(())
}

fn launch_unary<T: DeviceRepr>(
    device: &CudaDevice,
    function_name: &'static str,
    layout_function_name: &'static str,
    src: &CudaView<'_, T>,
    dst: &mut CudaSlice<T>,
    layout: &Layout,
    numel: usize,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    let stream = device.cuda_stream();
    let config = launch_config(numel)?;
    if layout.contiguous_offsets().is_some() {
        let function = device.get_or_load_func(rivet_kernels::UNARY, function_name)?;
        unsafe {
            stream
                .launch_builder(&function)
                .arg(src)
                .arg(dst)
                .arg(&numel)
                .launch(config)
        }
        .map_err(|error| cuda_error("kernel_launch", function_name, error))?;
    } else {
        let info = layout_info(device, layout)?;
        let function = device.get_or_load_func(rivet_kernels::UNARY, layout_function_name)?;
        unsafe {
            stream
                .launch_builder(&function)
                .arg(&numel)
                .arg(&info.rank)
                .arg(&info.values)
                .arg(src)
                .arg(dst)
                .launch(config)
        }
        .map_err(|error| cuda_error("kernel_launch", layout_function_name, error))?;
    }
    device.record_kernel_launch();
    Ok(())
}

fn launch_binary<T: DeviceRepr>(
    device: &CudaDevice,
    function_name: &'static str,
    layout_function_name: &'static str,
    lhs: &CudaView<'_, T>,
    lhs_layout: &Layout,
    rhs: &CudaView<'_, T>,
    rhs_layout: &Layout,
    dst: &mut CudaSlice<T>,
    numel: usize,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    let stream = device.cuda_stream();
    let config = launch_config(numel)?;
    if lhs_layout.contiguous_offsets().is_some() && rhs_layout.contiguous_offsets().is_some() {
        let function = device.get_or_load_func(rivet_kernels::BINARY, function_name)?;
        unsafe {
            stream
                .launch_builder(&function)
                .arg(lhs)
                .arg(rhs)
                .arg(dst)
                .arg(&numel)
                .launch(config)
        }
        .map_err(|error| cuda_error("kernel_launch", function_name, error))?;
    } else {
        let lhs_info = layout_info(device, lhs_layout)?;
        let rhs_info = layout_info(device, rhs_layout)?;
        let function = device.get_or_load_func(rivet_kernels::BINARY, layout_function_name)?;
        unsafe {
            stream
                .launch_builder(&function)
                .arg(&numel)
                .arg(&lhs_info.rank)
                .arg(&lhs_info.values)
                .arg(&rhs_info.values)
                .arg(lhs)
                .arg(rhs)
                .arg(dst)
                .launch(config)
        }
        .map_err(|error| cuda_error("kernel_launch", layout_function_name, error))?;
    }
    device.record_kernel_launch();
    Ok(())
}

fn launch_scalar_binary<T: DeviceRepr>(
    device: &CudaDevice,
    function_name: &'static str,
    layout_function_name: &'static str,
    src: &CudaView<'_, T>,
    dst: &mut CudaSlice<T>,
    scalar: &T,
    layout: &Layout,
    numel: usize,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    let stream = device.cuda_stream();
    let config = launch_config(numel)?;
    if layout.contiguous_offsets().is_some() {
        let function = device.get_or_load_func(rivet_kernels::BINARY, function_name)?;
        unsafe {
            stream
                .launch_builder(&function)
                .arg(src)
                .arg(dst)
                .arg(scalar)
                .arg(&numel)
                .launch(config)
        }
        .map_err(|error| cuda_error("kernel_launch", function_name, error))?;
    } else {
        let info = layout_info(device, layout)?;
        let function = device.get_or_load_func(rivet_kernels::BINARY, layout_function_name)?;
        unsafe {
            stream
                .launch_builder(&function)
                .arg(&numel)
                .arg(&info.rank)
                .arg(&info.values)
                .arg(scalar)
                .arg(src)
                .arg(dst)
                .launch(config)
        }
        .map_err(|error| cuda_error("kernel_launch", layout_function_name, error))?;
    }
    device.record_kernel_launch();
    Ok(())
}

fn launch_compare<T: DeviceRepr>(
    device: &CudaDevice,
    function_name: &'static str,
    layout_function_name: &'static str,
    lhs: &CudaView<'_, T>,
    lhs_layout: &Layout,
    rhs: &CudaView<'_, T>,
    rhs_layout: &Layout,
    dst: &mut CudaSlice<u8>,
    numel: usize,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    let stream = device.cuda_stream();
    let config = launch_config(numel)?;
    if lhs_layout.contiguous_offsets().is_some() && rhs_layout.contiguous_offsets().is_some() {
        let function = device.get_or_load_func(rivet_kernels::BINARY, function_name)?;
        unsafe {
            stream
                .launch_builder(&function)
                .arg(lhs)
                .arg(rhs)
                .arg(dst)
                .arg(&numel)
                .launch(config)
        }
        .map_err(|error| cuda_error("kernel_launch", function_name, error))?;
    } else {
        let lhs_info = layout_info(device, lhs_layout)?;
        let rhs_info = layout_info(device, rhs_layout)?;
        let function = device.get_or_load_func(rivet_kernels::BINARY, layout_function_name)?;
        unsafe {
            stream
                .launch_builder(&function)
                .arg(&numel)
                .arg(&lhs_info.rank)
                .arg(&lhs_info.values)
                .arg(&rhs_info.values)
                .arg(lhs)
                .arg(rhs)
                .arg(dst)
                .launch(config)
        }
        .map_err(|error| cuda_error("kernel_launch", layout_function_name, error))?;
    }
    device.record_kernel_launch();
    Ok(())
}

fn launch_scalar_compare<T: DeviceRepr>(
    device: &CudaDevice,
    function_name: &'static str,
    layout_function_name: &'static str,
    src: &CudaView<'_, T>,
    dst: &mut CudaSlice<u8>,
    scalar: &T,
    layout: &Layout,
    numel: usize,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    let stream = device.cuda_stream();
    let config = launch_config(numel)?;
    if layout.contiguous_offsets().is_some() {
        let function = device.get_or_load_func(rivet_kernels::BINARY, function_name)?;
        unsafe {
            stream
                .launch_builder(&function)
                .arg(src)
                .arg(dst)
                .arg(scalar)
                .arg(&numel)
                .launch(config)
        }
        .map_err(|error| cuda_error("kernel_launch", function_name, error))?;
    } else {
        let info = layout_info(device, layout)?;
        let function = device.get_or_load_func(rivet_kernels::BINARY, layout_function_name)?;
        unsafe {
            stream
                .launch_builder(&function)
                .arg(&numel)
                .arg(&info.rank)
                .arg(&info.values)
                .arg(scalar)
                .arg(src)
                .arg(dst)
                .launch(config)
        }
        .map_err(|error| cuda_error("kernel_launch", layout_function_name, error))?;
    }
    device.record_kernel_launch();
    Ok(())
}

macro_rules! unary_names {
    ($op:expr, $suffix:literal) => {
        match $op {
            UnaryOp::Neg => (concat!("neg_", $suffix), concat!("neg_layout_", $suffix)),
            UnaryOp::Abs => (concat!("abs_", $suffix), concat!("abs_layout_", $suffix)),
        }
    };
}

macro_rules! binary_names {
    ($op:expr, $suffix:literal) => {
        match $op {
            BinaryOp::Add => (concat!("add_", $suffix), concat!("add_layout_", $suffix)),
            BinaryOp::Sub => (concat!("sub_", $suffix), concat!("sub_layout_", $suffix)),
            BinaryOp::Mul => (concat!("mul_", $suffix), concat!("mul_layout_", $suffix)),
            BinaryOp::Div => (concat!("div_", $suffix), concat!("div_layout_", $suffix)),
            BinaryOp::Minimum => (
                concat!("minimum_", $suffix),
                concat!("minimum_layout_", $suffix),
            ),
            BinaryOp::Maximum => (
                concat!("maximum_", $suffix),
                concat!("maximum_layout_", $suffix),
            ),
        }
    };
}

macro_rules! scalar_binary_names {
    ($op:expr, $suffix:literal) => {
        match $op {
            BinaryOp::Add => (
                concat!("add_scalar_", $suffix),
                concat!("add_scalar_layout_", $suffix),
            ),
            BinaryOp::Sub => (
                concat!("sub_scalar_", $suffix),
                concat!("sub_scalar_layout_", $suffix),
            ),
            BinaryOp::Mul => (
                concat!("mul_scalar_", $suffix),
                concat!("mul_scalar_layout_", $suffix),
            ),
            BinaryOp::Div => (
                concat!("div_scalar_", $suffix),
                concat!("div_scalar_layout_", $suffix),
            ),
            BinaryOp::Minimum => (
                concat!("minimum_scalar_", $suffix),
                concat!("minimum_scalar_layout_", $suffix),
            ),
            BinaryOp::Maximum => (
                concat!("maximum_scalar_", $suffix),
                concat!("maximum_scalar_layout_", $suffix),
            ),
        }
    };
}

macro_rules! compare_names {
    ($op:expr, $suffix:literal) => {
        match $op {
            CmpOp::Eq => (concat!("eq_", $suffix), concat!("eq_layout_", $suffix)),
            CmpOp::Ne => (concat!("ne_", $suffix), concat!("ne_layout_", $suffix)),
            CmpOp::Lt => (concat!("lt_", $suffix), concat!("lt_layout_", $suffix)),
            CmpOp::Le => (concat!("le_", $suffix), concat!("le_layout_", $suffix)),
            CmpOp::Gt => (concat!("gt_", $suffix), concat!("gt_layout_", $suffix)),
            CmpOp::Ge => (concat!("ge_", $suffix), concat!("ge_layout_", $suffix)),
        }
    };
}

macro_rules! scalar_compare_names {
    ($op:expr, $suffix:literal) => {
        match $op {
            CmpOp::Eq => (
                concat!("eq_scalar_", $suffix),
                concat!("eq_scalar_layout_", $suffix),
            ),
            CmpOp::Ne => (
                concat!("ne_scalar_", $suffix),
                concat!("ne_scalar_layout_", $suffix),
            ),
            CmpOp::Lt => (
                concat!("lt_scalar_", $suffix),
                concat!("lt_scalar_layout_", $suffix),
            ),
            CmpOp::Le => (
                concat!("le_scalar_", $suffix),
                concat!("le_scalar_layout_", $suffix),
            ),
            CmpOp::Gt => (
                concat!("gt_scalar_", $suffix),
                concat!("gt_scalar_layout_", $suffix),
            ),
            CmpOp::Ge => (
                concat!("ge_scalar_", $suffix),
                concat!("ge_scalar_layout_", $suffix),
            ),
        }
    };
}

macro_rules! dispatch_fill {
    ($device:expr, $dst:expr, $numel:expr) => {
        match $dst {
            CudaStorageSlice::U8(dst) => launch_fill($device, "fill_u8", dst, 1u8, $numel),
            CudaStorageSlice::U32(dst) => launch_fill($device, "fill_u32", dst, 1u32, $numel),
            CudaStorageSlice::I16(dst) => launch_fill($device, "fill_i16", dst, 1i16, $numel),
            CudaStorageSlice::I32(dst) => launch_fill($device, "fill_i32", dst, 1i32, $numel),
            CudaStorageSlice::I64(dst) => launch_fill($device, "fill_i64", dst, 1i64, $numel),
            CudaStorageSlice::BF16(dst) => {
                launch_fill($device, "fill_bf16", dst, half::bf16::from_f32(1.0), $numel)
            }
            CudaStorageSlice::F16(dst) => {
                launch_fill($device, "fill_f16", dst, half::f16::from_f32(1.0), $numel)
            }
            CudaStorageSlice::F32(dst) => launch_fill($device, "fill_f32", dst, 1.0f32, $numel),
            CudaStorageSlice::F64(dst) => launch_fill($device, "fill_f64", dst, 1.0f64, $numel),
        }
    };
}

macro_rules! dispatch_copy {
    ($device:expr, $view:expr, $dst:expr, $layout:expr, $numel:expr, $src_dtype:expr, $dst_dtype:expr) => {
        match ($view, $dst) {
            (CudaStorageView::U8(src), CudaStorageSlice::U8(dst)) => {
                launch_copy_layout($device, "copy_layout_u8", &src, dst, $layout, $numel)
            }
            (CudaStorageView::U32(src), CudaStorageSlice::U32(dst)) => {
                launch_copy_layout($device, "copy_layout_u32", &src, dst, $layout, $numel)
            }
            (CudaStorageView::I16(src), CudaStorageSlice::I16(dst)) => {
                launch_copy_layout($device, "copy_layout_i16", &src, dst, $layout, $numel)
            }
            (CudaStorageView::I32(src), CudaStorageSlice::I32(dst)) => {
                launch_copy_layout($device, "copy_layout_i32", &src, dst, $layout, $numel)
            }
            (CudaStorageView::I64(src), CudaStorageSlice::I64(dst)) => {
                launch_copy_layout($device, "copy_layout_i64", &src, dst, $layout, $numel)
            }
            (CudaStorageView::BF16(src), CudaStorageSlice::BF16(dst)) => {
                launch_copy_layout($device, "copy_layout_bf16", &src, dst, $layout, $numel)
            }
            (CudaStorageView::F16(src), CudaStorageSlice::F16(dst)) => {
                launch_copy_layout($device, "copy_layout_f16", &src, dst, $layout, $numel)
            }
            (CudaStorageView::F32(src), CudaStorageSlice::F32(dst)) => {
                launch_copy_layout($device, "copy_layout_f32", &src, dst, $layout, $numel)
            }
            (CudaStorageView::F64(src), CudaStorageSlice::F64(dst)) => {
                launch_copy_layout($device, "copy_layout_f64", &src, dst, $layout, $numel)
            }
            _ => Err(Error::DTypeMismatch {
                lhs: $src_dtype,
                rhs: $dst_dtype,
            }),
        }
    };
}

macro_rules! dispatch_unary {
    ($device:expr, $view:expr, $dst:expr, $layout:expr, $numel:expr, $op:expr, $src_dtype:expr, $dst_dtype:expr) => {
        match ($view, $dst) {
            (CudaStorageView::U8(src), CudaStorageSlice::U8(dst)) => {
                let (function, layout_function) = unary_names!($op, "u8");
                launch_unary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::U32(src), CudaStorageSlice::U32(dst)) => {
                let (function, layout_function) = unary_names!($op, "u32");
                launch_unary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::I16(src), CudaStorageSlice::I16(dst)) => {
                let (function, layout_function) = unary_names!($op, "i16");
                launch_unary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::I32(src), CudaStorageSlice::I32(dst)) => {
                let (function, layout_function) = unary_names!($op, "i32");
                launch_unary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::I64(src), CudaStorageSlice::I64(dst)) => {
                let (function, layout_function) = unary_names!($op, "i64");
                launch_unary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::BF16(src), CudaStorageSlice::BF16(dst)) => {
                let (function, layout_function) = unary_names!($op, "bf16");
                launch_unary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::F16(src), CudaStorageSlice::F16(dst)) => {
                let (function, layout_function) = unary_names!($op, "f16");
                launch_unary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::F32(src), CudaStorageSlice::F32(dst)) => {
                let (function, layout_function) = unary_names!($op, "f32");
                launch_unary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::F64(src), CudaStorageSlice::F64(dst)) => {
                let (function, layout_function) = unary_names!($op, "f64");
                launch_unary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    $layout,
                    $numel,
                )
            }
            _ => Err(Error::DTypeMismatch {
                lhs: $src_dtype,
                rhs: $dst_dtype,
            }),
        }
    };
}

macro_rules! dispatch_binary {
    ($device:expr, $lhs_view:expr, $rhs_view:expr, $dst:expr, $lhs_layout:expr, $rhs_layout:expr, $numel:expr, $op:expr, $lhs_dtype:expr, $rhs_dtype:expr) => {
        match ($lhs_view, $rhs_view, $dst) {
            (CudaStorageView::U8(lhs), CudaStorageView::U8(rhs), CudaStorageSlice::U8(dst)) => {
                let (function, layout_function) = binary_names!($op, "u8");
                launch_binary(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    dst,
                    $numel,
                )
            }
            (CudaStorageView::U32(lhs), CudaStorageView::U32(rhs), CudaStorageSlice::U32(dst)) => {
                let (function, layout_function) = binary_names!($op, "u32");
                launch_binary(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    dst,
                    $numel,
                )
            }
            (CudaStorageView::I16(lhs), CudaStorageView::I16(rhs), CudaStorageSlice::I16(dst)) => {
                let (function, layout_function) = binary_names!($op, "i16");
                launch_binary(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    dst,
                    $numel,
                )
            }
            (CudaStorageView::I32(lhs), CudaStorageView::I32(rhs), CudaStorageSlice::I32(dst)) => {
                let (function, layout_function) = binary_names!($op, "i32");
                launch_binary(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    dst,
                    $numel,
                )
            }
            (CudaStorageView::I64(lhs), CudaStorageView::I64(rhs), CudaStorageSlice::I64(dst)) => {
                let (function, layout_function) = binary_names!($op, "i64");
                launch_binary(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    dst,
                    $numel,
                )
            }
            (
                CudaStorageView::BF16(lhs),
                CudaStorageView::BF16(rhs),
                CudaStorageSlice::BF16(dst),
            ) => {
                let (function, layout_function) = binary_names!($op, "bf16");
                launch_binary(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    dst,
                    $numel,
                )
            }
            (CudaStorageView::F16(lhs), CudaStorageView::F16(rhs), CudaStorageSlice::F16(dst)) => {
                let (function, layout_function) = binary_names!($op, "f16");
                launch_binary(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    dst,
                    $numel,
                )
            }
            (CudaStorageView::F32(lhs), CudaStorageView::F32(rhs), CudaStorageSlice::F32(dst)) => {
                let (function, layout_function) = binary_names!($op, "f32");
                launch_binary(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    dst,
                    $numel,
                )
            }
            (CudaStorageView::F64(lhs), CudaStorageView::F64(rhs), CudaStorageSlice::F64(dst)) => {
                let (function, layout_function) = binary_names!($op, "f64");
                launch_binary(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    dst,
                    $numel,
                )
            }
            _ => Err(Error::DTypeMismatch {
                lhs: $lhs_dtype,
                rhs: $rhs_dtype,
            }),
        }
    };
}

macro_rules! dispatch_compare {
    ($device:expr, $lhs_view:expr, $rhs_view:expr, $dst:expr, $lhs_layout:expr, $rhs_layout:expr, $numel:expr, $op:expr, $lhs_dtype:expr, $rhs_dtype:expr) => {
        match ($lhs_view, $rhs_view) {
            (CudaStorageView::U8(lhs), CudaStorageView::U8(rhs)) => {
                let (function, layout_function) = compare_names!($op, "u8");
                launch_compare(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    $dst,
                    $numel,
                )
            }
            (CudaStorageView::U32(lhs), CudaStorageView::U32(rhs)) => {
                let (function, layout_function) = compare_names!($op, "u32");
                launch_compare(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    $dst,
                    $numel,
                )
            }
            (CudaStorageView::I16(lhs), CudaStorageView::I16(rhs)) => {
                let (function, layout_function) = compare_names!($op, "i16");
                launch_compare(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    $dst,
                    $numel,
                )
            }
            (CudaStorageView::I32(lhs), CudaStorageView::I32(rhs)) => {
                let (function, layout_function) = compare_names!($op, "i32");
                launch_compare(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    $dst,
                    $numel,
                )
            }
            (CudaStorageView::I64(lhs), CudaStorageView::I64(rhs)) => {
                let (function, layout_function) = compare_names!($op, "i64");
                launch_compare(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    $dst,
                    $numel,
                )
            }
            (CudaStorageView::BF16(lhs), CudaStorageView::BF16(rhs)) => {
                let (function, layout_function) = compare_names!($op, "bf16");
                launch_compare(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    $dst,
                    $numel,
                )
            }
            (CudaStorageView::F16(lhs), CudaStorageView::F16(rhs)) => {
                let (function, layout_function) = compare_names!($op, "f16");
                launch_compare(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    $dst,
                    $numel,
                )
            }
            (CudaStorageView::F32(lhs), CudaStorageView::F32(rhs)) => {
                let (function, layout_function) = compare_names!($op, "f32");
                launch_compare(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    $dst,
                    $numel,
                )
            }
            (CudaStorageView::F64(lhs), CudaStorageView::F64(rhs)) => {
                let (function, layout_function) = compare_names!($op, "f64");
                launch_compare(
                    $device,
                    function,
                    layout_function,
                    &lhs,
                    $lhs_layout,
                    &rhs,
                    $rhs_layout,
                    $dst,
                    $numel,
                )
            }
            _ => Err(Error::DTypeMismatch {
                lhs: $lhs_dtype,
                rhs: $rhs_dtype,
            }),
        }
    };
}

macro_rules! dispatch_binary_scalar {
    ($device:expr, $view:expr, $dst:expr, $scalar:expr, $layout:expr, $numel:expr, $op:expr, $src_dtype:expr, $scalar_dtype:expr) => {
        match ($view, $dst, $scalar) {
            (CudaStorageView::U8(src), CudaStorageSlice::U8(dst), CpuStorageRef::U8(scalar)) => {
                let (function, layout_function) = scalar_binary_names!($op, "u8");
                launch_scalar_binary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::U32(src), CudaStorageSlice::U32(dst), CpuStorageRef::U32(scalar)) => {
                let (function, layout_function) = scalar_binary_names!($op, "u32");
                launch_scalar_binary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::I16(src), CudaStorageSlice::I16(dst), CpuStorageRef::I16(scalar)) => {
                let (function, layout_function) = scalar_binary_names!($op, "i16");
                launch_scalar_binary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::I32(src), CudaStorageSlice::I32(dst), CpuStorageRef::I32(scalar)) => {
                let (function, layout_function) = scalar_binary_names!($op, "i32");
                launch_scalar_binary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::I64(src), CudaStorageSlice::I64(dst), CpuStorageRef::I64(scalar)) => {
                let (function, layout_function) = scalar_binary_names!($op, "i64");
                launch_scalar_binary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (
                CudaStorageView::BF16(src),
                CudaStorageSlice::BF16(dst),
                CpuStorageRef::BF16(scalar),
            ) => {
                let (function, layout_function) = scalar_binary_names!($op, "bf16");
                launch_scalar_binary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::F16(src), CudaStorageSlice::F16(dst), CpuStorageRef::F16(scalar)) => {
                let (function, layout_function) = scalar_binary_names!($op, "f16");
                launch_scalar_binary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::F32(src), CudaStorageSlice::F32(dst), CpuStorageRef::F32(scalar)) => {
                let (function, layout_function) = scalar_binary_names!($op, "f32");
                launch_scalar_binary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::F64(src), CudaStorageSlice::F64(dst), CpuStorageRef::F64(scalar)) => {
                let (function, layout_function) = scalar_binary_names!($op, "f64");
                launch_scalar_binary(
                    $device,
                    function,
                    layout_function,
                    &src,
                    dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            _ => Err(Error::DTypeMismatch {
                lhs: $src_dtype,
                rhs: $scalar_dtype,
            }),
        }
    };
}

macro_rules! dispatch_scalar_compare {
    ($device:expr, $view:expr, $dst:expr, $scalar:expr, $layout:expr, $numel:expr, $op:expr, $src_dtype:expr, $scalar_dtype:expr) => {
        match ($view, $scalar) {
            (CudaStorageView::U8(src), CpuStorageRef::U8(scalar)) => {
                let (function, layout_function) = scalar_compare_names!($op, "u8");
                launch_scalar_compare(
                    $device,
                    function,
                    layout_function,
                    &src,
                    $dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::U32(src), CpuStorageRef::U32(scalar)) => {
                let (function, layout_function) = scalar_compare_names!($op, "u32");
                launch_scalar_compare(
                    $device,
                    function,
                    layout_function,
                    &src,
                    $dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::I16(src), CpuStorageRef::I16(scalar)) => {
                let (function, layout_function) = scalar_compare_names!($op, "i16");
                launch_scalar_compare(
                    $device,
                    function,
                    layout_function,
                    &src,
                    $dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::I32(src), CpuStorageRef::I32(scalar)) => {
                let (function, layout_function) = scalar_compare_names!($op, "i32");
                launch_scalar_compare(
                    $device,
                    function,
                    layout_function,
                    &src,
                    $dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::I64(src), CpuStorageRef::I64(scalar)) => {
                let (function, layout_function) = scalar_compare_names!($op, "i64");
                launch_scalar_compare(
                    $device,
                    function,
                    layout_function,
                    &src,
                    $dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::BF16(src), CpuStorageRef::BF16(scalar)) => {
                let (function, layout_function) = scalar_compare_names!($op, "bf16");
                launch_scalar_compare(
                    $device,
                    function,
                    layout_function,
                    &src,
                    $dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::F16(src), CpuStorageRef::F16(scalar)) => {
                let (function, layout_function) = scalar_compare_names!($op, "f16");
                launch_scalar_compare(
                    $device,
                    function,
                    layout_function,
                    &src,
                    $dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::F32(src), CpuStorageRef::F32(scalar)) => {
                let (function, layout_function) = scalar_compare_names!($op, "f32");
                launch_scalar_compare(
                    $device,
                    function,
                    layout_function,
                    &src,
                    $dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            (CudaStorageView::F64(src), CpuStorageRef::F64(scalar)) => {
                let (function, layout_function) = scalar_compare_names!($op, "f64");
                launch_scalar_compare(
                    $device,
                    function,
                    layout_function,
                    &src,
                    $dst,
                    &scalar[0],
                    $layout,
                    $numel,
                )
            }
            _ => Err(Error::DTypeMismatch {
                lhs: $src_dtype,
                rhs: $scalar_dtype,
            }),
        }
    };
}

pub(crate) fn fill_one(device: &CudaDevice, dst: &mut CudaStorageSlice) -> Result<()> {
    let numel = dst.len();
    dispatch_fill!(device, dst, numel)
}

pub(crate) fn copy_layout(
    device: &CudaDevice,
    src: &CudaStorageSlice,
    dst: &mut CudaStorageSlice,
    layout: &Layout,
) -> Result<()> {
    let numel = layout.checked_elem_count()?;
    let view = layout_view(src, layout)?;
    let src_dtype = src.dtype();
    let dst_dtype = dst.dtype();
    dispatch_copy!(device, view, dst, layout, numel, src_dtype, dst_dtype)
}

pub(crate) fn unary(
    device: &CudaDevice,
    src: &CudaStorageSlice,
    dst: &mut CudaStorageSlice,
    layout: &Layout,
    op: UnaryOp,
) -> Result<()> {
    let numel = layout.checked_elem_count()?;
    let view = layout_view(src, layout)?;
    let src_dtype = src.dtype();
    let dst_dtype = dst.dtype();
    dispatch_unary!(device, view, dst, layout, numel, op, src_dtype, dst_dtype)
}

pub(crate) fn binary(
    device: &CudaDevice,
    lhs: &CudaStorageSlice,
    lhs_layout: &Layout,
    rhs: &CudaStorageSlice,
    rhs_layout: &Layout,
    dst: &mut CudaStorageSlice,
    op: BinaryOp,
) -> Result<()> {
    let numel = lhs_layout.checked_elem_count()?;
    if numel != rhs_layout.checked_elem_count()? {
        return Err(Error::ShapeMismatchBinary {
            lhs: lhs_layout.dims().to_vec(),
            rhs: rhs_layout.dims().to_vec(),
        });
    }
    let lhs_view = layout_view(lhs, lhs_layout)?;
    let rhs_view = layout_view(rhs, rhs_layout)?;
    dispatch_binary!(
        device,
        lhs_view,
        rhs_view,
        dst,
        lhs_layout,
        rhs_layout,
        numel,
        op,
        lhs.dtype(),
        rhs.dtype()
    )
}

pub(crate) fn binary_scalar<T: WithDType>(
    device: &CudaDevice,
    src: &CudaStorageSlice,
    layout: &Layout,
    dst: &mut CudaStorageSlice,
    scalar: T,
    op: BinaryOp,
) -> Result<()> {
    let numel = layout.checked_elem_count()?;
    let scalar_storage = T::into_cpu_storage(vec![scalar])?;
    let scalar = scalar_storage.as_ref();
    let view = layout_view(src, layout)?;
    dispatch_binary_scalar!(
        device,
        view,
        dst,
        scalar,
        layout,
        numel,
        op,
        src.dtype(),
        T::DTYPE
    )
}

pub(crate) fn compare(
    device: &CudaDevice,
    lhs: &CudaStorageSlice,
    lhs_layout: &Layout,
    rhs: &CudaStorageSlice,
    rhs_layout: &Layout,
    dst: &mut CudaStorageSlice,
    op: CmpOp,
) -> Result<()> {
    let numel = lhs_layout.checked_elem_count()?;
    if numel != rhs_layout.checked_elem_count()? {
        return Err(Error::ShapeMismatchBinary {
            lhs: lhs_layout.dims().to_vec(),
            rhs: rhs_layout.dims().to_vec(),
        });
    }
    let lhs_view = layout_view(lhs, lhs_layout)?;
    let rhs_view = layout_view(rhs, rhs_layout)?;
    let CudaStorageSlice::U8(dst) = dst else {
        return Err(Error::UnexpectedDType {
            expected: crate::DType::U8,
            actual: dst.dtype(),
        });
    };
    dispatch_compare!(
        device,
        lhs_view,
        rhs_view,
        dst,
        lhs_layout,
        rhs_layout,
        numel,
        op,
        lhs.dtype(),
        rhs.dtype()
    )
}

pub(crate) fn compare_scalar<T: WithDType>(
    device: &CudaDevice,
    src: &CudaStorageSlice,
    layout: &Layout,
    dst: &mut CudaStorageSlice,
    scalar: T,
    op: CmpOp,
) -> Result<()> {
    let numel = layout.checked_elem_count()?;
    let scalar_storage = T::into_cpu_storage(vec![scalar])?;
    let scalar = scalar_storage.as_ref();
    let view = layout_view(src, layout)?;
    let CudaStorageSlice::U8(dst) = dst else {
        return Err(Error::UnexpectedDType {
            expected: crate::DType::U8,
            actual: dst.dtype(),
        });
    };
    dispatch_scalar_compare!(
        device,
        view,
        dst,
        scalar,
        layout,
        numel,
        op,
        src.dtype(),
        T::DTYPE
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn kernel_module_is_available() {
        assert!(rivet_kernels::BINARY.ptx().contains("add_f32"));
    }
}
