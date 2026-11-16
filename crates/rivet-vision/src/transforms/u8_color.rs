use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use crate::transforms::{contiguous_values, logical_offset};
use rivet_core::{CpuStorageRef, DType, Tensor};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InvertConfig;

impl InvertConfig {
    pub const fn new() -> Self {
        Self
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        map_u8(sample, axis_order, "invert", |value| 255 - value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PosterizeConfig {
    pub bits: u8,
}

impl PosterizeConfig {
    pub const fn new(bits: u8) -> Self {
        Self { bits }
    }

    pub fn validate(&self) -> RivetResult<()> {
        if (0..=8).contains(&self.bits) {
            Ok(())
        } else {
            Err(invalid_argument("posterize bits must be in [0, 8]"))
        }
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let mask = if self.bits == 0 {
            0
        } else {
            u8::MAX << (8 - self.bits)
        };
        map_u8(sample, axis_order, "posterize", |value| value & mask)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolarizeConfig {
    pub threshold: u8,
}

impl SolarizeConfig {
    pub const fn new(threshold: u8) -> Self {
        Self { threshold }
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        map_u8(sample, axis_order, "solarize", |value| {
            if value >= self.threshold {
                255 - value
            } else {
                value
            }
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AutocontrastConfig;

impl AutocontrastConfig {
    pub const fn new() -> Self {
        Self
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        let sample = require_u8_image(sample, "autocontrast")?;
        let input = &sample.image;
        let dims = input.dims().to_vec();
        let (height, width, channels) = image_dims(&dims, axis_order)?;
        let values = input.with_cpu_storage(|storage, layout| {
            let CpuStorageRef::U8(values) = storage else {
                return Err(rivet_core::Error::UnexpectedDType {
                    expected: DType::U8,
                    actual: input.dtype(),
                });
            };
            if axis_order == ImageAxisOrder::Hwc && channels == 3 && layout.is_contiguous() {
                let pixels = contiguous_values(values, layout)?;
                let mut min = [255u8; 3];
                let mut max = [0u8; 3];
                for rgb in pixels.chunks_exact(3) {
                    for c in 0..3 {
                        min[c] = min[c].min(rgb[c]);
                        max[c] = max[c].max(rgb[c]);
                    }
                }
                let luts = std::array::from_fn(|c| {
                    std::array::from_fn(|v| {
                        if min[c] >= max[c] {
                            v as u8
                        } else {
                            ((v.saturating_sub(min[c] as usize) * 255)
                                / usize::from(max[c] - min[c]))
                            .min(255) as u8
                        }
                    })
                });
                return map_rgb_luts(pixels, &dims, input.device(), &luts);
            }
            let read = |coords: [usize; 3]| {
                values
                    .get(logical_offset(layout, &coords)?)
                    .copied()
                    .ok_or(rivet_core::Error::StorageOutOfBounds)
            };
            let mut min = vec![u8::MAX; channels];
            let mut max = vec![0u8; channels];
            for channel in 0..channels {
                for y in 0..height {
                    for x in 0..width {
                        let value = read(pixel_coords(axis_order, channel, y, x))?;
                        min[channel] = min[channel].min(value);
                        max[channel] = max[channel].max(value);
                    }
                }
            }
            Tensor::from_exact_fn::<u8, _, _>(dims.clone(), input.device(), |write| {
                append_mapped_pixels(
                    write,
                    axis_order,
                    (height, width, channels),
                    |channel, y, x| {
                        let value = read(pixel_coords(axis_order, channel, y, x))?;
                        if min[channel] == max[channel] {
                            Ok(value)
                        } else {
                            Ok(((u16::from(value.saturating_sub(min[channel])) * 255)
                                / u16::from(max[channel] - min[channel]))
                                as u8)
                        }
                    },
                )
            })
        })?;
        Ok(ImageSample::Decoded(DecodedSample {
            image: values,
            label: sample.label,
        }))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EqualizeConfig;

impl EqualizeConfig {
    pub const fn new() -> Self {
        Self
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        let sample = require_u8_image(sample, "equalize")?;
        let input = &sample.image;
        let dims = input.dims().to_vec();
        let (height, width, channels) = image_dims(&dims, axis_order)?;
        let values = input.with_cpu_storage(|storage, layout| {
            let CpuStorageRef::U8(values) = storage else {
                return Err(rivet_core::Error::UnexpectedDType {
                    expected: DType::U8,
                    actual: input.dtype(),
                });
            };
            if axis_order == ImageAxisOrder::Hwc && channels == 3 && layout.is_contiguous() {
                let pixels = contiguous_values(values, layout)?;
                let mut histograms = [[0u64; 256]; 3];
                for rgb in pixels.chunks_exact(3) {
                    histograms[0][rgb[0] as usize] += 1;
                    histograms[1][rgb[1] as usize] += 1;
                    histograms[2][rgb[2] as usize] += 1;
                }
                let luts = histograms.map(|histogram| equalize_lut(&histogram));
                return map_rgb_luts(pixels, &dims, input.device(), &luts);
            }
            let read = |coords: [usize; 3]| {
                values
                    .get(logical_offset(layout, &coords)?)
                    .copied()
                    .ok_or(rivet_core::Error::StorageOutOfBounds)
            };
            let mut histograms = vec![[0u64; 256]; channels];
            for channel in 0..channels {
                for y in 0..height {
                    for x in 0..width {
                        histograms[channel]
                            [usize::from(read(pixel_coords(axis_order, channel, y, x))?)] += 1;
                    }
                }
            }
            let mut luts = vec![[0u8; 256]; channels];
            for channel in 0..channels {
                luts[channel] = equalize_lut(&histograms[channel]);
            }
            Tensor::from_exact_fn::<u8, _, _>(dims.clone(), input.device(), |write| {
                append_mapped_pixels(
                    write,
                    axis_order,
                    (height, width, channels),
                    |channel, y, x| {
                        let value = read(pixel_coords(axis_order, channel, y, x))?;
                        Ok(luts[channel][usize::from(value)])
                    },
                )
            })
        })?;
        Ok(ImageSample::Decoded(DecodedSample {
            image: values,
            label: sample.label,
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharpnessConfig {
    /// torchvision sharpness factor: 0 is smooth, 1 is identity.
    pub amount: f32,
}

impl SharpnessConfig {
    pub const fn new(amount: f32) -> Self {
        Self { amount }
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self.amount.is_finite() && self.amount >= 0.0 {
            Ok(())
        } else {
            Err(invalid_argument(
                "sharpness amount must be finite and non-negative",
            ))
        }
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = require_u8_image(sample, "sharpness")?;
        let input = &sample.image;
        let dims = input.dims().to_vec();
        let (height, width, channels) = image_dims(&dims, axis_order)?;
        let values = input.with_cpu_storage(|storage, layout| {
            let CpuStorageRef::U8(values) = storage else {
                return Err(rivet_core::Error::UnexpectedDType {
                    expected: DType::U8,
                    actual: input.dtype(),
                });
            };
            if axis_order == ImageAxisOrder::Hwc && channels == 3 && layout.is_contiguous() {
                return sharpness_rgb(
                    contiguous_values(values, layout)?,
                    width,
                    height,
                    self.amount,
                    input.device(),
                );
            }
            let read = |channel: usize, y: usize, x: usize| {
                values
                    .get(logical_offset(
                        layout,
                        &pixel_coords(axis_order, channel, y, x),
                    )?)
                    .copied()
                    .ok_or(rivet_core::Error::StorageOutOfBounds)
            };
            let sharpen = |channel: usize, y: usize, x: usize| {
                let center = read(channel, y, x)?;
                if y == 0 || x == 0 || y + 1 == height || x + 1 == width {
                    return Ok(center); // Pillow SMOOTH leaves border pixels alone.
                }
                let weight = 1.0f32 / 13.0;
                let mut sum = 0.5f32;
                for dy in (y - 1..=y + 1).rev() {
                    let center_weight = if dy == y { 5.0f32 / 13.0 } else { weight };
                    sum += f32::from(read(channel, dy, x - 1)?) * weight
                        + f32::from(read(channel, dy, x)?) * center_weight
                        + f32::from(read(channel, dy, x + 1)?) * weight;
                }
                let blurred = sum.clamp(0.0, 255.0) as u8;
                Ok(super::color::blend(blurred, center, self.amount))
            };
            Tensor::from_exact_writer::<u8, _, _>(dims.clone(), input.device(), |write| {
                match axis_order {
                    ImageAxisOrder::Hwc => {
                        for y in 0..height {
                            for x in 0..width {
                                for c in 0..channels {
                                    write.write_next(sharpen(c, y, x)?)?;
                                }
                            }
                        }
                    }
                    ImageAxisOrder::Chw => {
                        for c in 0..channels {
                            for y in 0..height {
                                for x in 0..width {
                                    write.write_next(sharpen(c, y, x)?)?;
                                }
                            }
                        }
                    }
                }
                Ok(())
            })
        })?;
        Ok(ImageSample::Decoded(DecodedSample {
            image: values,
            label: sample.label,
        }))
    }
}

fn map_rgb_luts(
    pixels: &[u8],
    shape: &[usize],
    device: &rivet_core::Device,
    luts: &[[u8; 256]; 3],
) -> rivet_core::Result<Tensor> {
    Tensor::from_exact_writer::<u8, _, _>(shape, device, |out| {
        for rgb in pixels.chunks_exact(3) {
            out.write_next(luts[0][rgb[0] as usize])?;
            out.write_next(luts[1][rgb[1] as usize])?;
            out.write_next(luts[2][rgb[2] as usize])?;
        }
        Ok(())
    })
}

fn equalize_lut(histogram: &[u64; 256]) -> [u8; 256] {
    let total = histogram.iter().sum::<u64>();
    let last = histogram.iter().rposition(|&count| count != 0);
    let step = last.map_or(0, |last| (total - histogram[last]) / 255);
    if step == 0 {
        return std::array::from_fn(|v| v as u8);
    }
    let mut cumulative = step / 2;
    std::array::from_fn(|v| {
        let output = (cumulative / step).min(255) as u8;
        cumulative += histogram[v];
        output
    })
}

fn sharpness_rgb(
    pixels: &[u8],
    width: usize,
    height: usize,
    amount: f32,
    device: &rivet_core::Device,
) -> rivet_core::Result<Tensor> {
    let shape = [height, width, 3];
    if width < 3 || height < 3 || amount == 1.0 {
        return Tensor::from_exact_iter(pixels.iter().copied(), shape, device);
    }
    let stride = width * 3;
    Tensor::from_exact_writer::<u8, _, _>(shape, device, |out| {
        out.extend_from_slice(&pixels[..stride])?;
        for y in 1..height - 1 {
            let top = &pixels[(y - 1) * stride..y * stride];
            let center = &pixels[y * stride..(y + 1) * stride];
            let bottom = &pixels[(y + 1) * stride..(y + 2) * stride];
            out.extend_from_slice(&center[..3])?;
            let sharpen = |i: usize| {
                let weighted_row = |row: &[u8], middle: f32| {
                    f32::from(row[i - 3]) * (1.0f32 / 13.0)
                        + f32::from(row[i]) * middle
                        + f32::from(row[i + 3]) * (1.0f32 / 13.0)
                };
                // Keep Pillow's row grouping and bottom/center/top order.
                let mut sum = 0.5f32;
                sum += weighted_row(bottom, 1.0f32 / 13.0);
                sum += weighted_row(center, 5.0f32 / 13.0);
                sum += weighted_row(top, 1.0f32 / 13.0);
                super::color::blend(sum.clamp(0.0, 255.0) as u8, center[i], amount)
            };
            for x in 1..width - 1 {
                out.write_next(sharpen(x * 3))?;
                out.write_next(sharpen(x * 3 + 1))?;
                out.write_next(sharpen(x * 3 + 2))?;
            }
            out.extend_from_slice(&center[stride - 3..])?;
        }
        out.extend_from_slice(&pixels[(height - 1) * stride..])?;
        Ok(())
    })
}

fn require_u8_image(sample: ImageSample, op_name: &str) -> RivetResult<DecodedSample> {
    let sample = sample.into_decoded()?;
    if sample.image.dtype() != DType::U8 || sample.image.rank() != 3 {
        return Err(invalid_argument(format!(
            "{op_name} requires a rank-3 uint8 image, got dtype {:?} shape {:?}",
            sample.image.dtype(),
            sample.image.dims()
        )));
    }
    Ok(sample)
}

fn image_dims(dims: &[usize], axis_order: ImageAxisOrder) -> RivetResult<(usize, usize, usize)> {
    if dims.len() != 3 {
        return Err(invalid_shape(format!(
            "U8 color kernel requires a rank-3 image, got shape {dims:?}"
        )));
    }
    Ok(match axis_order {
        ImageAxisOrder::Hwc => (dims[0], dims[1], dims[2]),
        ImageAxisOrder::Chw => (dims[1], dims[2], dims[0]),
    })
}

fn pixel_coords(axis_order: ImageAxisOrder, channel: usize, y: usize, x: usize) -> [usize; 3] {
    match axis_order {
        ImageAxisOrder::Hwc => [y, x, channel],
        ImageAxisOrder::Chw => [channel, y, x],
    }
}

fn append_mapped_pixels<T, F>(
    output: &mut dyn FnMut(T) -> rivet_core::Result<()>,
    axis_order: ImageAxisOrder,
    (height, width, channels): (usize, usize, usize),
    mut read: F,
) -> rivet_core::Result<()>
where
    F: FnMut(usize, usize, usize) -> rivet_core::Result<T>,
{
    match axis_order {
        ImageAxisOrder::Hwc => {
            for y in 0..height {
                for x in 0..width {
                    for channel in 0..channels {
                        output(read(channel, y, x)?)?;
                    }
                }
            }
        }
        ImageAxisOrder::Chw => {
            for channel in 0..channels {
                for y in 0..height {
                    for x in 0..width {
                        output(read(channel, y, x)?)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn map_u8<F>(
    sample: ImageSample,
    _axis_order: ImageAxisOrder,
    op_name: &str,
    mut map: F,
) -> RivetResult<ImageSample>
where
    F: FnMut(u8) -> u8,
{
    let sample = require_u8_image(sample, op_name)?;
    let dims = sample.image.dims().to_vec();
    let image = sample.image.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::U8(values) = storage else {
            return Err(rivet_core::Error::UnexpectedDType {
                expected: DType::U8,
                actual: sample.image.dtype(),
            });
        };
        if layout.is_contiguous() {
            return Tensor::from_exact_iter(
                contiguous_values(values, layout)?.iter().copied().map(map),
                dims.clone(),
                sample.image.device(),
            );
        }
        let output = layout.strided_index().map(|index| {
            values
                .get(index)
                .copied()
                .map(&mut map)
                .ok_or(rivet_core::Error::StorageOutOfBounds)
        });
        Tensor::from_exact_try_iter(output, dims.clone(), sample.image.device())
    })?;
    Ok(ImageSample::Decoded(DecodedSample {
        image,
        label: sample.label,
    }))
}

#[cfg(test)]
mod tests {
    use super::{
        AutocontrastConfig, EqualizeConfig, InvertConfig, PosterizeConfig, SharpnessConfig,
        SolarizeConfig,
    };
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{DType, Device, Tensor};

    fn sample() -> ImageSample {
        ImageSample::Decoded(DecodedSample {
            image: Tensor::from_vec(
                vec![0u8, 1, 2, 100, 101, 102, 254, 255, 128],
                [1, 3, 3],
                &Device::Cpu,
            )
            .unwrap(),
            label: 4,
        })
    }

    #[test]
    fn elementwise_u8_kernels_preserve_shape_and_materialize() {
        let input = sample().into_decoded().unwrap().image;
        let inverted = InvertConfig::new()
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: input.clone(),
                    label: 0,
                }),
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(
            inverted.image.to_vec::<u8>().unwrap(),
            [255, 254, 253, 155, 154, 153, 1, 0, 127]
        );
        assert_eq!(inverted.image.dtype(), DType::U8);
        assert!(!inverted.image.same_storage(&input));

        let posterized = PosterizeConfig::new(4)
            .apply(sample(), ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(
            posterized.image.to_vec::<u8>().unwrap(),
            [0, 0, 0, 96, 96, 96, 240, 240, 128]
        );

        let solarized = SolarizeConfig::new(100)
            .apply(sample(), ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(
            solarized.image.to_vec::<u8>().unwrap(),
            [0, 1, 2, 155, 154, 153, 1, 0, 127]
        );
    }

    #[test]
    fn rgb_fast_paths_match_chw_fallback_with_offsets_and_small_images() {
        for (h, w) in [(1, 1), (2, 7), (5, 3), (11, 13), (32, 33)] {
            for pattern in 0..3 {
                let pixels: Vec<u8> = (0..h * w * 3)
                    .map(|i| match pattern {
                        0 => 37,
                        1 => [0, 127, 255][(i / 7) % 3],
                        _ => ((i * 17 + i / 31) % 256) as u8,
                    })
                    .collect();
                let mut storage = vec![19u8; pixels.len()];
                storage.extend_from_slice(&pixels);
                let image = Tensor::from_vec(storage, [2, h, w, 3], &Device::Cpu)
                    .unwrap()
                    .get(1)
                    .unwrap();
                let chw = image.permute(&[2, 0, 1]).unwrap();
                for op in 0..6 {
                    let apply = |tensor, axis| {
                        let sample = ImageSample::Decoded(DecodedSample {
                            image: tensor,
                            label: 4,
                        });
                        match op {
                            0 => AutocontrastConfig::new().apply(sample, axis),
                            1 => EqualizeConfig::new().apply(sample, axis),
                            2 => SharpnessConfig::new(0.0).apply(sample, axis),
                            3 => SharpnessConfig::new(0.3).apply(sample, axis),
                            4 => SharpnessConfig::new(1.0).apply(sample, axis),
                            _ => SharpnessConfig::new(2.0).apply(sample, axis),
                        }
                        .unwrap()
                        .into_decoded()
                        .unwrap()
                    };
                    let actual = apply(image.clone(), ImageAxisOrder::Hwc);
                    let expected = apply(chw.clone(), ImageAxisOrder::Chw)
                        .image
                        .permute(&[1, 2, 0])
                        .unwrap();
                    assert_eq!(
                        actual.image.to_vec::<u8>().unwrap(),
                        expected.to_vec::<u8>().unwrap(),
                        "shape={h}x{w} pattern={pattern} op={op}"
                    );
                    assert!(!actual.image.same_storage(&image));
                    assert_eq!(actual.label, 4);
                }
            }
        }
    }

    #[test]
    fn equalize_uses_integer_step_and_half_step_rounding() {
        let pixels: Vec<u8> = [10u8, 100, 200]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, 256))
            .collect();
        let output = EqualizeConfig::new()
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: Tensor::from_vec(pixels, [1, 768, 1], &Device::Cpu).unwrap(),
                    label: 4,
                }),
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        let expected: Vec<u8> = [0u8, 128, 255]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, 256))
            .collect();
        assert_eq!(output.image.to_vec::<u8>().unwrap(), expected);
        assert_eq!(output.label, 4);
    }

    #[test]
    fn autocontrast_equalize_and_sharpness_support_chw() {
        let input =
            Tensor::from_vec(vec![0u8, 10, 20, 30, 40, 50], [3, 1, 2], &Device::Cpu).unwrap();
        let auto = AutocontrastConfig::new()
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: input.clone(),
                    label: 0,
                }),
                ImageAxisOrder::Chw,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(auto.image.dims(), [3, 1, 2]);
        assert_eq!(auto.image.to_vec::<u8>().unwrap(), [0, 255, 0, 255, 0, 255]);

        let equalized = EqualizeConfig::new()
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: input.clone(),
                    label: 0,
                }),
                ImageAxisOrder::Chw,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(equalized.image.dims(), [3, 1, 2]);
        assert_eq!(equalized.image.dtype(), DType::U8);
        assert_eq!(
            equalized.image.to_vec::<u8>().unwrap(),
            [0, 10, 20, 30, 40, 50]
        );

        let sharp = SharpnessConfig::new(0.0)
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
        assert_eq!(sharp.image.to_vec::<u8>().unwrap(), [0, 10, 20, 30, 40, 50]);
    }

    #[test]
    fn kernels_read_non_contiguous_views_and_reject_non_u8() {
        let base = Tensor::from_vec((0..18).collect::<Vec<u8>>(), [2, 3, 3], &Device::Cpu).unwrap();
        let view = base.permute(&[1, 0, 2]).unwrap();
        assert!(!view.is_contiguous());

        let output = InvertConfig::new()
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: view.clone(),
                    label: 0,
                }),
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dims(), [3, 2, 3]);
        assert_eq!(
            output.image.to_vec::<u8>().unwrap(),
            [
                255, 254, 253, 246, 245, 244, 252, 251, 250, 243, 242, 241, 249, 248, 247, 240,
                239, 238
            ]
        );
        assert!(!output.image.same_storage(&view));

        let f32_image = Tensor::from_vec(vec![0.0f32; 3], [1, 1, 3], &Device::Cpu).unwrap();
        let err = InvertConfig::new()
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: f32_image,
                    label: 0,
                }),
                ImageAxisOrder::Hwc,
            )
            .unwrap_err();
        assert!(err.to_string().contains("rank-3 uint8 image"));
    }

    #[test]
    fn posterize_and_sharpness_validate_arguments() {
        assert!(PosterizeConfig::new(0).validate().is_ok());
        assert!(PosterizeConfig::new(9).validate().is_err());
        assert!(SharpnessConfig::new(-1.0).validate().is_err());
        assert!(SharpnessConfig::new(f32::NAN).validate().is_err());
    }
}
