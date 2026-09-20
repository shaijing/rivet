use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{ImageAxisOrder, ImageSample};
use crate::transforms::{PaddingMode, logical_offset};
use rivet_core::{CpuStorageRef, DType, Tensor};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadConfig {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
    pub value: f32,
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
            mode: PaddingMode::Constant,
        }
    }

    pub const fn with_fill(mut self, value: f32) -> Self {
        self.value = value;
        self
    }

    pub const fn with_mode(mut self, mode: PaddingMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self.mode != PaddingMode::Constant {
            return Err(invalid_argument(
                "only constant padding is implemented in Phase 2",
            ));
        }
        if !self.value.is_finite() {
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

        let output = match input.dtype() {
            DType::U8 => {
                let fill = self.value.clamp(0.0, 255.0).round() as u8;
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
                    let mut output = Vec::with_capacity(output_height * output_width * channels);
                    append_padded_u8(
                        &mut output,
                        axis_order,
                        (height, width, channels),
                        (output_height, output_width),
                        (self.top as usize, self.left as usize),
                        fill,
                        read,
                    )?;
                    Ok(output)
                })?;
                Tensor::from_vec(values, output_dims, input.device())?
            }
            DType::F32 => {
                let fill = self.value;
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
                    let mut output = Vec::with_capacity(output_height * output_width * channels);
                    append_padded_f32(
                        &mut output,
                        axis_order,
                        (height, width, channels),
                        (output_height, output_width),
                        (self.top as usize, self.left as usize),
                        fill,
                        read,
                    )?;
                    Ok(output)
                })?;
                Tensor::from_vec(values, output_dims, input.device())?
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

fn append_padded_u8<F>(
    output: &mut Vec<u8>,
    axis_order: ImageAxisOrder,
    (height, width, channels): (usize, usize, usize),
    (output_height, output_width): (usize, usize),
    (top, left): (usize, usize),
    fill: u8,
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
                        output.push(if inside_y && inside_x {
                            read([y - top, x - left, c])?
                        } else {
                            fill
                        });
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
                        output.push(if inside_y && inside_x {
                            read([c, y - top, x - left])?
                        } else {
                            fill
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

fn append_padded_f32<F>(
    output: &mut Vec<f32>,
    axis_order: ImageAxisOrder,
    dims: (usize, usize, usize),
    output_dims: (usize, usize),
    offset: (usize, usize),
    fill: f32,
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
    output: &mut Vec<T>,
    axis_order: ImageAxisOrder,
    (height, width, channels): (usize, usize, usize),
    (output_height, output_width): (usize, usize),
    (top, left): (usize, usize),
    fill: T,
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
                        output.push(if inside_y && inside_x {
                            read([y - top, x - left, c])?
                        } else {
                            fill
                        });
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
                        output.push(if inside_y && inside_x {
                            read([c, y - top, x - left])?
                        } else {
                            fill
                        });
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
    fn constant_pad_supports_chw_and_rejects_other_modes() {
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
                .is_err()
        );
    }
}
