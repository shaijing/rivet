use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{ImageAxisOrder, ImageSample};
use crate::transforms::{
    PaddingMode, from_rgb_image, into_rgb_image, logical_offset, with_rgb_pixels,
};
use image::imageops::replace;
use image::{Rgb, RgbImage};
use rivet_core::{CpuStorageRef, DType, Tensor};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadConfig {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
    pub value: f32,
    pub fill_channels: Option<[f32; 3]>,
    pub mode: PaddingMode,
}

impl PadConfig {
    pub const fn new(padding: u32) -> Self {
        Self {
            left: padding,
            right: padding,
            top: padding,
            bottom: padding,
            value: 0.0,
            fill_channels: None,
            mode: PaddingMode::Constant,
        }
    }

    pub const fn with_sides(left: u32, top: u32, right: u32, bottom: u32) -> Self {
        Self {
            left,
            right,
            top,
            bottom,
            value: 0.0,
            fill_channels: None,
            mode: PaddingMode::Constant,
        }
    }

    pub const fn with_fill(mut self, value: f32) -> Self {
        self.value = value;
        self.fill_channels = None;
        self
    }

    pub const fn with_fill_channels(mut self, value: [f32; 3]) -> Self {
        self.fill_channels = Some(value);
        self
    }

    pub const fn with_mode(mut self, mode: PaddingMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        if !self.value.is_finite()
            || self
                .fill_channels
                .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(invalid_argument("padding fill value must be finite"));
        }
        Ok(())
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = sample.into_decoded()?;
        let input = &sample.image;
        if input.rank() != 3 {
            return Err(invalid_shape(format!(
                "pad requires a rank-3 image, got shape {:?}",
                input.dims()
            )));
        }
        let dims = input.dims().to_vec();
        let (height, width, channels) = match axis_order {
            ImageAxisOrder::Hwc => (dims[0], dims[1], dims[2]),
            ImageAxisOrder::Chw => (dims[1], dims[2], dims[0]),
        };
        if self.fill_channels.is_some() && channels != 3 {
            return Err(invalid_argument(format!(
                "per-channel padding fill requires a 3-channel image, got {channels} channels"
            )));
        }
        let output_height = height
            .checked_add(self.top as usize)
            .and_then(|value| value.checked_add(self.bottom as usize))
            .ok_or_else(|| invalid_shape("pad height overflows usize"))?;
        let output_width = width
            .checked_add(self.left as usize)
            .and_then(|value| value.checked_add(self.right as usize))
            .ok_or_else(|| invalid_shape("pad width overflows usize"))?;
        let output_dims = match axis_order {
            ImageAxisOrder::Hwc => vec![output_height, output_width, channels],
            ImageAxisOrder::Chw => vec![channels, output_height, output_width],
        };

        if self.mode != PaddingMode::Constant {
            if height == 0 || width == 0 {
                return Err(invalid_shape(
                    "nonconstant padding requires a nonempty image",
                ));
            }
            let indices = |len: usize, count: usize, before: u32| {
                (0..count)
                    .map(|i| {
                        let offset = i as i64 - i64::from(before);
                        match self.mode {
                            PaddingMode::Edge => offset.clamp(0, len as i64 - 1) as usize,
                            PaddingMode::Reflect if len == 1 => 0,
                            PaddingMode::Reflect => {
                                let period = 2 * (len as i64 - 1);
                                let i = offset.rem_euclid(period);
                                i.min(period - i) as usize
                            }
                            PaddingMode::Symmetric => {
                                let period = 2 * len as i64;
                                let i = offset.rem_euclid(period);
                                i.min(period - 1 - i) as usize
                            }
                            PaddingMode::Constant => unreachable!(),
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let xs = indices(width, output_width, self.left);
            let ys = indices(height, output_height, self.top);
            let image = input.with_cpu_storage(|storage, layout| match storage {
                CpuStorageRef::U8(values) => mapped_padding(
                    values,
                    layout,
                    input.device(),
                    axis_order,
                    &xs,
                    &ys,
                    channels,
                ),
                CpuStorageRef::F32(values) => mapped_padding(
                    values,
                    layout,
                    input.device(),
                    axis_order,
                    &xs,
                    &ys,
                    channels,
                ),
                _ => Err(rivet_core::Error::UnexpectedDType {
                    expected: DType::U8,
                    actual: input.dtype(),
                }),
            })?;
            return Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
                image,
                label: sample.label,
            }));
        }

        // RGB HWC writes rows directly; retain the image backend for CHW.
        // Float images and arbitrary channel counts use the generic kernel.
        if input.dtype() == DType::U8 && channels == 3 && self.fill_channels.is_none() {
            let output_width = u32::try_from(output_width)
                .map_err(|_| invalid_shape("pad output width is too large for image backend"))?;
            let output_height = u32::try_from(output_height)
                .map_err(|_| invalid_shape("pad output height is too large for image backend"))?;
            if axis_order == ImageAxisOrder::Hwc {
                let image = with_rgb_pixels(input, "pad", |pixels, width, height| {
                    let fill = self.value.clamp(0.0, 255.0) as u8;
                    let block = [fill; 256];
                    Ok(Tensor::from_exact_writer::<u8, _, _>(
                        output_dims.clone(),
                        input.device(),
                        |out| {
                            let fill = |out: &mut rivet_core::ExactOutput<u8>, mut count: usize| {
                                while count >= block.len() {
                                    out.extend_from_slice(&block)?;
                                    count -= block.len();
                                }
                                out.extend_from_slice(&block[..count])
                            };
                            fill(out, self.top as usize * output_width as usize * 3)?;
                            for y in 0..height {
                                fill(out, self.left as usize * 3)?;
                                out.extend_from_slice(&pixels[y * width * 3..(y + 1) * width * 3])?;
                                fill(out, self.right as usize * 3)?;
                            }
                            fill(out, self.bottom as usize * output_width as usize * 3)?;
                            Ok(())
                        },
                    )?)
                })?;
                return Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
                    image,
                    label: sample.label,
                }));
            }
            let image = match axis_order {
                ImageAxisOrder::Hwc => input.clone(),
                ImageAxisOrder::Chw => input.permute(&[1, 2, 0])?,
            };
            let (image, label) = into_rgb_image(
                crate::sample::image::DecodedSample {
                    image,
                    label: sample.label,
                },
                "pad",
            )?;
            let fill = self.value.clamp(0.0, 255.0) as u8;
            let mut padded = RgbImage::from_pixel(output_width, output_height, Rgb([fill; 3]));
            replace(
                &mut padded,
                &image,
                i64::from(self.left),
                i64::from(self.top),
            );
            let output = from_rgb_image(padded, label)?;
            let output = if axis_order == ImageAxisOrder::Hwc {
                output
            } else {
                crate::sample::image::DecodedSample {
                    image: output.image.permute(&[2, 0, 1])?.contiguous()?,
                    label: output.label,
                }
            };
            return Ok(ImageSample::Decoded(output));
        }

        let output = match input.dtype() {
            DType::U8 => {
                let fill = self
                    .fill_channels
                    .unwrap_or([self.value; 3])
                    .map(|value| value.clamp(0.0, 255.0) as u8);
                let values = input.with_cpu_storage(|storage, layout| {
                    let CpuStorageRef::U8(values) = storage else {
                        return Err(rivet_core::Error::UnexpectedDType {
                            expected: DType::U8,
                            actual: input.dtype(),
                        });
                    };
                    let read = |coords: [usize; 3]| {
                        values
                            .get(logical_offset(layout, &coords)?)
                            .copied()
                            .ok_or(rivet_core::Error::StorageOutOfBounds)
                    };
                    Tensor::from_exact_fn::<u8, _, _>(
                        output_dims.clone(),
                        input.device(),
                        |write| {
                            append_padded_u8(
                                write,
                                axis_order,
                                (height, width, channels),
                                (output_height, output_width),
                                (self.top as usize, self.left as usize),
                                fill,
                                read,
                            )
                        },
                    )
                })?;
                values
            }
            DType::F32 => {
                let fill = self.fill_channels.unwrap_or([self.value; 3]);
                let values = input.with_cpu_storage(|storage, layout| {
                    let CpuStorageRef::F32(values) = storage else {
                        return Err(rivet_core::Error::UnexpectedDType {
                            expected: DType::F32,
                            actual: input.dtype(),
                        });
                    };
                    let read = |coords: [usize; 3]| {
                        values
                            .get(logical_offset(layout, &coords)?)
                            .copied()
                            .ok_or(rivet_core::Error::StorageOutOfBounds)
                    };
                    Tensor::from_exact_fn::<f32, _, _>(
                        output_dims.clone(),
                        input.device(),
                        |write| {
                            append_padded_f32(
                                write,
                                axis_order,
                                (height, width, channels),
                                (output_height, output_width),
                                (self.top as usize, self.left as usize),
                                fill,
                                read,
                            )
                        },
                    )
                })?;
                values
            }
            dtype => {
                return Err(invalid_argument(format!(
                    "pad supports uint8 or float32 input, got {dtype:?}"
                )));
            }
        };

        Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
            image: output,
            label: sample.label,
        }))
    }
}

fn mapped_padding<T: rivet_core::WithDType + Copy>(
    values: &[T],
    layout: &rivet_core::Layout,
    device: &rivet_core::Device,
    order: ImageAxisOrder,
    xs: &[usize],
    ys: &[usize],
    channels: usize,
) -> rivet_core::Result<Tensor> {
    let shape = match order {
        ImageAxisOrder::Hwc => [ys.len(), xs.len(), channels],
        ImageAxisOrder::Chw => [channels, ys.len(), xs.len()],
    };
    let read = |coords: [usize; 3]| {
        values
            .get(logical_offset(layout, &coords)?)
            .copied()
            .ok_or(rivet_core::Error::StorageOutOfBounds)
    };
    Tensor::from_exact_writer::<T, _, _>(shape, device, |out| {
        match order {
            ImageAxisOrder::Hwc => {
                for &y in ys {
                    for &x in xs {
                        for c in 0..channels {
                            out.write_next(read([y, x, c])?)?;
                        }
                    }
                }
            }
            ImageAxisOrder::Chw => {
                for c in 0..channels {
                    for &y in ys {
                        for &x in xs {
                            out.write_next(read([c, y, x])?)?;
                        }
                    }
                }
            }
        }
        Ok(())
    })
}

fn append_padded_u8<F>(
    output: &mut dyn FnMut(u8) -> rivet_core::Result<()>,
    axis_order: ImageAxisOrder,
    (height, width, channels): (usize, usize, usize),
    (output_height, output_width): (usize, usize),
    (top, left): (usize, usize),
    fill: [u8; 3],
    mut read: F,
) -> rivet_core::Result<()>
where
    F: FnMut([usize; 3]) -> rivet_core::Result<u8>,
{
    match axis_order {
        ImageAxisOrder::Hwc => {
            for y in 0..output_height {
                for x in 0..output_width {
                    let inside_y = y >= top && y - top < height;
                    let inside_x = x >= left && x - left < width;
                    for c in 0..channels {
                        output(if inside_y && inside_x {
                            read([y - top, x - left, c])?
                        } else {
                            fill.get(c).copied().unwrap_or(fill[0])
                        })?;
                    }
                }
            }
        }
        ImageAxisOrder::Chw => {
            for c in 0..channels {
                for y in 0..output_height {
                    for x in 0..output_width {
                        let inside_y = y >= top && y - top < height;
                        let inside_x = x >= left && x - left < width;
                        output(if inside_y && inside_x {
                            read([c, y - top, x - left])?
                        } else {
                            fill.get(c).copied().unwrap_or(fill[0])
                        })?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn append_padded_f32<F>(
    output: &mut dyn FnMut(f32) -> rivet_core::Result<()>,
    axis_order: ImageAxisOrder,
    dims: (usize, usize, usize),
    output_dims: (usize, usize),
    offset: (usize, usize),
    fill: [f32; 3],
    read: F,
) -> rivet_core::Result<()>
where
    F: FnMut([usize; 3]) -> rivet_core::Result<f32>,
{
    let mut read = read;
    append_padded_u8_like(
        output,
        axis_order,
        dims,
        output_dims,
        offset,
        fill,
        &mut read,
    )
}

fn append_padded_u8_like<T, F>(
    output: &mut dyn FnMut(T) -> rivet_core::Result<()>,
    axis_order: ImageAxisOrder,
    (height, width, channels): (usize, usize, usize),
    (output_height, output_width): (usize, usize),
    (top, left): (usize, usize),
    fill: [T; 3],
    read: &mut F,
) -> rivet_core::Result<()>
where
    T: Copy,
    F: FnMut([usize; 3]) -> rivet_core::Result<T>,
{
    match axis_order {
        ImageAxisOrder::Hwc => {
            for y in 0..output_height {
                for x in 0..output_width {
                    let inside_y = y >= top && y - top < height;
                    let inside_x = x >= left && x - left < width;
                    for c in 0..channels {
                        output(if inside_y && inside_x {
                            read([y - top, x - left, c])?
                        } else {
                            fill.get(c).copied().unwrap_or(fill[0])
                        })?;
                    }
                }
            }
        }
        ImageAxisOrder::Chw => {
            for c in 0..channels {
                for y in 0..output_height {
                    for x in 0..output_width {
                        let inside_y = y >= top && y - top < height;
                        let inside_x = x >= left && x - left < width;
                        output(if inside_y && inside_x {
                            read([c, y - top, x - left])?
                        } else {
                            fill.get(c).copied().unwrap_or(fill[0])
                        })?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::PadConfig;
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{Device, Tensor};

    #[test]
    fn direct_rgb_padding_matches_backend_with_views_and_fractional_fill() {
        let base = Tensor::from_vec(
            (0..2 * 5 * 7 * 3).map(|v| v as u8).collect::<Vec<_>>(),
            [2, 5, 7, 3],
            &Device::Cpu,
        )
        .unwrap()
        .get(1)
        .unwrap();
        for image in [
            base.clone(),
            base.permute(&[1, 0, 2]).unwrap(),
            base.narrow(0, 1, 3).unwrap().narrow(1, 2, 4).unwrap(),
        ] {
            for fill in [-1.0, 37.75, 1000.0] {
                let config = PadConfig::with_sides(1, 2, 3, 4).with_fill(fill);
                let actual = config
                    .apply(
                        ImageSample::Decoded(DecodedSample {
                            image: image.clone(),
                            label: 7,
                        }),
                        ImageAxisOrder::Hwc,
                    )
                    .unwrap()
                    .into_decoded()
                    .unwrap();
                let chw = image.permute(&[2, 0, 1]).unwrap();
                let expected = config
                    .apply(
                        ImageSample::Decoded(DecodedSample {
                            image: chw,
                            label: 7,
                        }),
                        ImageAxisOrder::Chw,
                    )
                    .unwrap()
                    .into_decoded()
                    .unwrap()
                    .image
                    .permute(&[1, 2, 0])
                    .unwrap();
                assert_eq!(
                    actual.image.to_vec::<u8>().unwrap(),
                    expected.to_vec::<u8>().unwrap()
                );
                assert!(!actual.image.same_storage(&image));
                assert_eq!(actual.label, 7);
            }
        }
    }

    #[test]
    fn constant_pad_preserves_layout_semantics_and_copies() {
        let input = Tensor::from_vec(vec![1u8, 2, 3, 4], [1, 2, 2], &Device::Cpu).unwrap();
        let output = PadConfig::with_sides(1, 1, 0, 0)
            .with_fill(9.0)
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: input.clone(),
                    label: 5,
                }),
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dims(), [2, 3, 2]);
        assert_eq!(
            output.image.to_vec::<u8>().unwrap(),
            [9, 9, 9, 9, 9, 9, 9, 9, 1, 2, 3, 4]
        );
        assert!(!output.image.same_storage(&input));
        assert_eq!(output.label, 5);
    }

    #[test]
    fn constant_pad_supports_chw() {
        let input = Tensor::from_vec(vec![1u8, 2, 3, 4], [1, 2, 2], &Device::Cpu).unwrap();
        let output = PadConfig::new(1)
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: input,
                    label: 0,
                }),
                ImageAxisOrder::Chw,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dims(), [1, 4, 4]);
        assert!(
            PadConfig::new(1)
                .with_mode(crate::transforms::PaddingMode::Reflect)
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn constant_pad_uses_per_channel_fill_for_rgb() {
        let input = Tensor::from_vec(vec![1u8, 2, 3], [1, 1, 3], &Device::Cpu).unwrap();
        let output = PadConfig::new(1)
            .with_fill_channels([10.0, 20.0, 30.0])
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: input,
                    label: 0,
                }),
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        let values = output.image.to_vec::<u8>().unwrap();
        assert_eq!(&values[0..3], &[10, 20, 30]);
        assert_eq!(&values[3..6], &[10, 20, 30]);
        assert_eq!(&values[12..15], &[1, 2, 3]);
        assert_eq!(&values[18..21], &[10, 20, 30]);
    }
}
