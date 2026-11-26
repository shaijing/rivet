//! Channel permutation and conversion to RGB.

use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::ImageAxisOrder;
use rivet_core::{CpuStorageRef, DType, Tensor};

pub(in crate::transforms) fn permute(
    input: &Tensor,
    dims: &[usize],
    axis: ImageAxisOrder,
    order: &[usize],
) -> RivetResult<Tensor> {
    let (h, w) = match axis {
        ImageAxisOrder::Hwc => (dims[0], dims[1]),
        ImageAxisOrder::Chw => (dims[1], dims[2]),
    };
    input
        .with_cpu_storage(|storage, layout| {
            macro_rules! run {
                ($values:expr,$t:ty) => {{
                    let src = $values;
                    let st = layout.stride();
                    let base = layout.start_offset();
                    Tensor::from_exact_writer::<$t, _, _>(dims, input.device(), |out| {
                        match axis {
                            ImageAxisOrder::Hwc => {
                                for y in 0..h {
                                    for x in 0..w {
                                        let offset = base + y * st[0] + x * st[1];
                                        if order.len() == 3 {
                                            out.write_next(src[offset + order[0] * st[2]])?;
                                            out.write_next(src[offset + order[1] * st[2]])?;
                                            out.write_next(src[offset + order[2] * st[2]])?;
                                        } else {
                                            for &c in order {
                                                out.write_next(src[offset + c * st[2]])?;
                                            }
                                        }
                                    }
                                }
                            }
                            ImageAxisOrder::Chw => {
                                for &c in order {
                                    for y in 0..h {
                                        for x in 0..w {
                                            out.write_next(
                                                src[base + c * st[0] + y * st[1] + x * st[2]],
                                            )?;
                                        }
                                    }
                                }
                            }
                        }
                        Ok(())
                    })
                }};
            }
            match storage {
                CpuStorageRef::U8(v) => run!(v, u8),
                CpuStorageRef::F32(v) => run!(v, f32),
                _ => Err(rivet_core::Error::UnexpectedDType {
                    expected: DType::U8,
                    actual: input.dtype(),
                }),
            }
        })
        .map_err(Into::into)
}

pub(in crate::transforms) fn convert_rgb(
    input: &Tensor,
    axis: ImageAxisOrder,
    height: usize,
    width: usize,
    channels: usize,
) -> RivetResult<Tensor> {
    if !(1..=4).contains(&channels) {
        return Err(invalid_argument("RGB requires 1, 2, 3, or 4 channels"));
    }
    if !matches!(input.dtype(), DType::U8 | DType::F32) {
        return Err(invalid_argument("RGB requires uint8 or float32 pixels"));
    }
    if channels == 3 {
        return Ok(input.clone());
    }
    let dims = match axis {
        ImageAxisOrder::Hwc => [height, width, 3],
        ImageAxisOrder::Chw => [3, height, width],
    };
    input
        .with_cpu_storage(|storage, layout| {
            macro_rules! convert {
                ($values:expr, $dtype:ty) => {{
                    let values = $values;
                    let stride = layout.stride();
                    let start = layout.start_offset();
                    Tensor::from_exact_writer::<$dtype, _, _>(dims, input.device(), |output| {
                        match axis {
                            ImageAxisOrder::Hwc => {
                                for y in 0..height {
                                    for x in 0..width {
                                        let offset = start + y * stride[0] + x * stride[1];
                                        if channels <= 2 {
                                            let value = *values
                                                .get(offset)
                                                .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                                            output.write_next(value)?;
                                            output.write_next(value)?;
                                            output.write_next(value)?;
                                        } else {
                                            for channel in 0..3 {
                                                output.write_next(
                                                    *values
                                                        .get(offset + channel * stride[2])
                                                        .ok_or(
                                                            rivet_core::Error::StorageOutOfBounds,
                                                        )?,
                                                )?;
                                            }
                                        }
                                    }
                                }
                            }
                            ImageAxisOrder::Chw => {
                                for channel in 0..3 {
                                    let source_channel = if channels <= 2 { 0 } else { channel };
                                    let channel_offset = start + source_channel * stride[0];
                                    for y in 0..height {
                                        let row_offset = channel_offset + y * stride[1];
                                        for x in 0..width {
                                            output.write_next(
                                                *values
                                                    .get(row_offset + x * stride[2])
                                                    .ok_or(rivet_core::Error::StorageOutOfBounds)?,
                                            )?;
                                        }
                                    }
                                }
                            }
                        }
                        Ok(())
                    })
                }};
            }
            match storage {
                CpuStorageRef::U8(values) => convert!(values, u8),
                CpuStorageRef::F32(values) => convert!(values, f32),
                _ => unreachable!("validated RGB dtype"),
            }
        })
        .map_err(Into::into)
}

#[cfg(test)]
mod rgb_tests {
    use super::*;
    use rivet_core::Device;

    #[test]
    fn rgb_modes_preserve_layout_and_strided_offsets() {
        for channels in 1..=4 {
            let data: Vec<u8> = (0..2 * 3 * channels).map(|value| value as u8).collect();
            let image = Tensor::from_vec(data.clone(), [2, 3, channels], &Device::Cpu).unwrap();
            let cropped = image.narrow(1, 1, 2).unwrap();
            for axis in [ImageAxisOrder::Hwc, ImageAxisOrder::Chw] {
                let input = match axis {
                    ImageAxisOrder::Hwc => cropped.clone(),
                    ImageAxisOrder::Chw => cropped.permute(&[2, 0, 1]).unwrap(),
                };
                let output = convert_rgb(&input, axis, 2, 2, channels).unwrap();
                let hwc = match axis {
                    ImageAxisOrder::Hwc => output,
                    ImageAxisOrder::Chw => output.permute(&[1, 2, 0]).unwrap(),
                };
                let expected: Vec<u8> = (0..2)
                    .flat_map(|y| {
                        let data = &data;
                        (1..3).flat_map(move |x| {
                            (0..3).map(move |channel| {
                                data[(y * 3 + x) * channels
                                    + if channels <= 2 { 0 } else { channel }]
                            })
                        })
                    })
                    .collect();
                assert_eq!(hwc.to_vec::<u8>().unwrap(), expected);
                let f32_base = image.to_dtype(DType::F32).unwrap().narrow(1, 1, 2).unwrap();
                let f32_input = match axis {
                    ImageAxisOrder::Hwc => f32_base,
                    ImageAxisOrder::Chw => f32_base.permute(&[2, 0, 1]).unwrap(),
                };
                let output = convert_rgb(&f32_input, axis, 2, 2, channels).unwrap();
                let hwc = match axis {
                    ImageAxisOrder::Hwc => output,
                    ImageAxisOrder::Chw => output.permute(&[1, 2, 0]).unwrap(),
                };
                assert_eq!(
                    hwc.to_vec::<f32>().unwrap(),
                    expected.into_iter().map(f32::from).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn rgb_identity_shares_storage_and_rejects_unknown_modes() {
        let image = Tensor::from_vec(vec![1u8, 2, 3], [1, 1, 3], &Device::Cpu).unwrap();
        let output = convert_rgb(&image, ImageAxisOrder::Hwc, 1, 1, 3).unwrap();
        assert!(output.same_storage(&image));
        let invalid = Tensor::from_vec(vec![0u8; 5], [1, 1, 5], &Device::Cpu).unwrap();
        assert!(convert_rgb(&invalid, ImageAxisOrder::Hwc, 1, 1, 5).is_err());
    }
}
