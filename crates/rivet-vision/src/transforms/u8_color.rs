use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use crate::transforms::{from_rgb_image, into_rgb_image, logical_offset};
use image::imageops::invert;
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
        let sample = require_u8_image(sample, "invert")?;
        if image_dims(sample.image.dims(), axis_order)?.2 == 3 {
            return Ok(ImageSample::Decoded(invert_rgb(sample, axis_order)?));
        }
        map_u8(
            ImageSample::Decoded(sample),
            axis_order,
            "invert",
            |value| 255 - value,
        )
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
        if (1..=8).contains(&self.bits) {
            Ok(())
        } else {
            Err(invalid_argument("posterize bits must be in [1, 8]"))
        }
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let mask = u8::MAX << (8 - self.bits);
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
            if value > self.threshold {
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
            let mut output = Vec::with_capacity(input.elem_count());
            append_mapped_pixels(
                &mut output,
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
            )?;
            Ok(output)
        })?;
        Ok(ImageSample::Decoded(new_sample(sample, values, dims)?))
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
                let histogram = &histograms[channel];
                let total = histogram.iter().sum::<u64>();
                let cdf_min = histogram
                    .iter()
                    .copied()
                    .find(|&count| count > 0)
                    .unwrap_or(0);
                if total == 0 || total == cdf_min {
                    for value in 0..256 {
                        luts[channel][value] = value as u8;
                    }
                    continue;
                }
                let mut cumulative = 0u64;
                for value in 0..256 {
                    cumulative += histogram[value];
                    luts[channel][value] = (cumulative.saturating_sub(cdf_min).saturating_mul(255)
                        / (total - cdf_min)) as u8;
                }
            }
            let mut output = Vec::with_capacity(input.elem_count());
            append_mapped_pixels(
                &mut output,
                axis_order,
                (height, width, channels),
                |channel, y, x| {
                    let value = read(pixel_coords(axis_order, channel, y, x))?;
                    Ok(luts[channel][usize::from(value)])
                },
            )?;
            Ok(output)
        })?;
        Ok(ImageSample::Decoded(new_sample(sample, values, dims)?))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharpnessConfig {
    /// Unsharp amount. `0` is identity; positive values increase local edge
    /// contrast using a 3x3 box blur reference. This is a Rivet contract and
    /// does not claim torchvision/image backend parity.
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
            let read = |channel: usize, y: usize, x: usize| {
                values
                    .get(logical_offset(
                        layout,
                        &pixel_coords(axis_order, channel, y, x),
                    )?)
                    .copied()
                    .ok_or(rivet_core::Error::StorageOutOfBounds)
            };
            let mut output = Vec::with_capacity(input.elem_count());
            append_mapped_pixels(
                &mut output,
                axis_order,
                (height, width, channels),
                |channel, y, x| {
                    let center = f32::from(read(channel, y, x)?);
                    let mut sum = 0.0;
                    let mut count = 0.0;
                    for dy in
                        y.saturating_sub(1)..=y.saturating_add(1).min(height.saturating_sub(1))
                    {
                        for dx in
                            x.saturating_sub(1)..=x.saturating_add(1).min(width.saturating_sub(1))
                        {
                            sum += f32::from(read(channel, dy, dx)?);
                            count += 1.0;
                        }
                    }
                    let blurred = sum / count;
                    Ok((center + self.amount * (center - blurred))
                        .clamp(0.0, 255.0)
                        .round() as u8)
                },
            )?;
            Ok(output)
        })?;
        Ok(ImageSample::Decoded(new_sample(sample, values, dims)?))
    }
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

fn invert_rgb(sample: DecodedSample, axis_order: ImageAxisOrder) -> RivetResult<DecodedSample> {
    let image = match axis_order {
        ImageAxisOrder::Hwc => sample.image,
        ImageAxisOrder::Chw => sample.image.permute(&[1, 2, 0])?,
    };
    let (mut image, label) = into_rgb_image(
        DecodedSample {
            image,
            label: sample.label,
        },
        "invert",
    )?;
    invert(&mut image);
    let output = from_rgb_image(image, label)?;
    if axis_order == ImageAxisOrder::Hwc {
        Ok(output)
    } else {
        Ok(DecodedSample {
            image: output.image.permute(&[2, 0, 1])?.contiguous()?,
            label: output.label,
        })
    }
}

fn append_mapped_pixels<T, F>(
    output: &mut Vec<T>,
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
                        output.push(read(channel, y, x)?);
                    }
                }
            }
        }
        ImageAxisOrder::Chw => {
            for channel in 0..channels {
                for y in 0..height {
                    for x in 0..width {
                        output.push(read(channel, y, x)?);
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
    let values = sample.image.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::U8(values) = storage else {
            return Err(rivet_core::Error::UnexpectedDType {
                expected: DType::U8,
                actual: sample.image.dtype(),
            });
        };
        layout
            .strided_index()
            .map(|index| {
                values
                    .get(index)
                    .copied()
                    .map(&mut map)
                    .ok_or(rivet_core::Error::StorageOutOfBounds)
            })
            .collect::<rivet_core::Result<Vec<_>>>()
    })?;
    Ok(ImageSample::Decoded(new_sample(sample, values, dims)?))
}

fn new_sample(
    sample: DecodedSample,
    values: Vec<u8>,
    dims: Vec<usize>,
) -> RivetResult<DecodedSample> {
    Ok(DecodedSample {
        image: Tensor::from_vec(values, dims, sample.image.device())?,
        label: sample.label,
    })
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
            [0, 1, 2, 100, 154, 153, 1, 0, 127]
        );
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
            [0, 255, 0, 255, 0, 255]
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
        assert!(PosterizeConfig::new(0).validate().is_err());
        assert!(PosterizeConfig::new(9).validate().is_err());
        assert!(SharpnessConfig::new(-1.0).validate().is_err());
        assert!(SharpnessConfig::new(f32::NAN).validate().is_err());
    }
}
