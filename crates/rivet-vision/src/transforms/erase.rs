use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{ImageAxisOrder, ImageSample};
use crate::transforms::logical_offset;
use rivet_core::{CpuStorageRef, DType, Tensor};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EraseRegion {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EraseConfig {
    pub region: EraseRegion,
    pub value: Vec<f32>,
}
impl EraseConfig {
    pub fn new(x: usize, y: usize, width: usize, height: usize, value: Vec<f32>) -> Self {
        Self {
            region: EraseRegion {
                x,
                y,
                width,
                height,
            },
            value,
        }
    }
    pub fn validate(&self) -> RivetResult<()> {
        if self.value.is_empty() || self.value.iter().any(|v| !v.is_finite()) {
            return Err(invalid_argument(
                "erase fill must be a nonempty finite scalar or channel list",
            ));
        }
        if self.region.x.checked_add(self.region.width).is_none()
            || self.region.y.checked_add(self.region.height).is_none()
        {
            return Err(invalid_argument("erase rectangle overflows usize"));
        }
        Ok(())
    }
    pub fn apply(&self, sample: ImageSample, axis: ImageAxisOrder) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = sample.into_decoded()?;
        let d = sample.image.dims();
        if d.len() != 3 {
            return Err(invalid_shape("erase requires a rank-3 image"));
        }
        let (h, w, c) = match axis {
            ImageAxisOrder::Hwc => (d[0], d[1], d[2]),
            ImageAxisOrder::Chw => (d[1], d[2], d[0]),
        };
        if self.value.len() != 1 && self.value.len() != c {
            return Err(invalid_argument(
                "erase fill count must be one or match the channel count",
            ));
        }
        let image = sample.image.with_cpu_storage(|storage, layout| {
            let run_u8 = |values: &[u8]| {
                erase_values(
                    values,
                    layout,
                    sample.image.device(),
                    axis,
                    (h, w, c),
                    self.region,
                    &self
                        .value
                        .iter()
                        .map(|v| v.trunc().rem_euclid(256.0) as u8)
                        .collect::<Vec<_>>(),
                )
            };
            match storage {
                CpuStorageRef::U8(v) => run_u8(v),
                CpuStorageRef::F32(v) => erase_values(
                    v,
                    layout,
                    sample.image.device(),
                    axis,
                    (h, w, c),
                    self.region,
                    &self.value,
                ),
                _ => Err(rivet_core::Error::UnexpectedDType {
                    expected: DType::U8,
                    actual: sample.image.dtype(),
                }),
            }
        })?;
        Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
            image,
            label: sample.label,
        }))
    }
}
fn erase_values<T: rivet_core::WithDType>(
    values: &[T],
    layout: &rivet_core::Layout,
    device: &rivet_core::Device,
    axis: ImageAxisOrder,
    (h, w, c): (usize, usize, usize),
    region: EraseRegion,
    fill: &[T],
) -> rivet_core::Result<Tensor> {
    let read = |channel, y, x| {
        if in_region(x, y, region) {
            return Ok(fill[if fill.len() == 1 { 0 } else { channel }]);
        }
        let coords = match axis {
            ImageAxisOrder::Hwc => [y, x, channel],
            ImageAxisOrder::Chw => [channel, y, x],
        };
        values
            .get(logical_offset(layout, &coords)?)
            .copied()
            .ok_or(rivet_core::Error::StorageOutOfBounds)
    };
    Tensor::from_exact_writer::<T, _, _>(layout.dims(), device, |out| {
        match axis {
            ImageAxisOrder::Hwc => {
                for y in 0..h {
                    for x in 0..w {
                        if c == 3 {
                            out.write_next(read(0, y, x)?)?;
                            out.write_next(read(1, y, x)?)?;
                            out.write_next(read(2, y, x)?)?;
                        } else {
                            for channel in 0..c {
                                out.write_next(read(channel, y, x)?)?;
                            }
                        }
                    }
                }
            }
            ImageAxisOrder::Chw => {
                for channel in 0..c {
                    for y in 0..h {
                        for x in 0..w {
                            out.write_next(read(channel, y, x)?)?;
                        }
                    }
                }
            }
        }
        Ok(())
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct RandomErasingConfig {
    pub probability: f64,
    pub scale_min: f32,
    pub scale_max: f32,
    pub ratio_min: f32,
    pub ratio_max: f32,
    pub value: ErasingValue,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ErasingValue {
    Constant(f32),
    Channels(Vec<f32>),
    Random,
}

impl RandomErasingConfig {
    pub const fn new(probability: f64) -> Self {
        Self {
            probability,
            scale_min: 0.02,
            scale_max: 0.33,
            ratio_min: 0.3,
            ratio_max: 3.3333333,
            value: ErasingValue::Constant(0.0),
        }
    }

    pub const fn with_params(
        mut self,
        scale_min: f32,
        scale_max: f32,
        ratio_min: f32,
        ratio_max: f32,
    ) -> Self {
        self.scale_min = scale_min;
        self.scale_max = scale_max;
        self.ratio_min = ratio_min;
        self.ratio_max = ratio_max;
        self
    }

    pub fn with_value(mut self, value: f32) -> Self {
        self.value = ErasingValue::Constant(value);
        self
    }

    pub fn with_values(mut self, values: Vec<f32>) -> Self {
        self.value = ErasingValue::Channels(values);
        self
    }

    pub fn with_random_value(mut self) -> Self {
        self.value = ErasingValue::Random;
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        if !(0.0..=1.0).contains(&self.probability) {
            return Err(invalid_argument(
                "random_erasing probability must be in [0.0, 1.0]",
            ));
        }
        if !self.scale_min.is_finite()
            || !self.scale_max.is_finite()
            || self.scale_min < 0.0
            || self.scale_min > self.scale_max
            || self.scale_max > 1.0
            || !self.ratio_min.is_finite()
            || !self.ratio_max.is_finite()
            || self.ratio_min <= 0.0
            || self.ratio_min > self.ratio_max
            || matches!(&self.value, ErasingValue::Constant(value) if !value.is_finite())
            || matches!(&self.value, ErasingValue::Channels(values) if values.is_empty() || values.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid_argument(
                "random_erasing scale/ratio/value parameters are invalid",
            ));
        }
        Ok(())
    }

    pub fn resolve(
        &self,
        width: usize,
        height: usize,
        rng: &mut RandomStream,
    ) -> Option<EraseRegion> {
        if uniform(rng) >= self.probability {
            return None;
        }
        self.sample_region(width, height, rng)
    }

    fn sample_region(
        &self,
        width: usize,
        height: usize,
        rng: &mut RandomStream,
    ) -> Option<EraseRegion> {
        if width == 0 || height == 0 {
            return None;
        }
        let area = (width * height) as f64;
        let log_ratio_min = f64::from(self.ratio_min).ln();
        let log_ratio_max = f64::from(self.ratio_max).ln();
        for _ in 0..10 {
            let scale = interpolate(
                uniform(rng),
                f64::from(self.scale_min),
                f64::from(self.scale_max),
            );
            let ratio = interpolate(uniform(rng), log_ratio_min, log_ratio_max).exp();
            let target = area * scale;
            let erase_width = (target / ratio).sqrt().round_ties_even() as usize;
            let erase_height = (target * ratio).sqrt().round_ties_even() as usize;
            if erase_width > 0 && erase_height > 0 && erase_width < width && erase_height < height {
                let x = if width == erase_width {
                    0
                } else {
                    rng.gen_range_usize(0..width - erase_width + 1).ok()?
                };
                let y = if height == erase_height {
                    0
                } else {
                    rng.gen_range_usize(0..height - erase_height + 1).ok()?
                };
                return Some(EraseRegion {
                    x,
                    y,
                    width: erase_width,
                    height: erase_height,
                });
            }
        }
        None // torchvision returns the original image after ten failed attempts.
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        rng: &mut RandomStream,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = sample.into_decoded()?;
        if sample.image.rank() != 3 {
            return Err(invalid_shape(format!(
                "random_erasing requires a rank-3 image, got shape {:?}",
                sample.image.dims()
            )));
        }
        let dims = sample.image.dims().to_vec();
        let (height, width, channels) = match axis_order {
            ImageAxisOrder::Hwc => (dims[0], dims[1], dims[2]),
            ImageAxisOrder::Chw => (dims[1], dims[2], dims[0]),
        };
        if uniform(rng) >= self.probability {
            return Ok(ImageSample::Decoded(sample));
        }
        if let ErasingValue::Channels(values) = &self.value {
            if values.len() != 1 && values.len() != channels {
                return Err(invalid_argument(format!(
                    "random_erasing value must be scalar or contain {channels} channel values"
                )));
            }
        }
        let Some(region) = self.sample_region(width, height, rng) else {
            return Ok(ImageSample::Decoded(sample));
        };

        let output = match sample.image.dtype() {
            DType::U8 => {
                let fills = match &self.value {
                    ErasingValue::Constant(value) => vec![value.trunc().rem_euclid(256.0) as u8],
                    ErasingValue::Channels(values) => values
                        .iter()
                        .map(|v| v.trunc().rem_euclid(256.0) as u8)
                        .collect::<Vec<_>>(),
                    ErasingValue::Random => Vec::new(),
                };
                let random = matches!(&self.value, ErasingValue::Random);
                sample.image.with_cpu_storage(|storage, layout| {
                    let CpuStorageRef::U8(pixels) = storage else {
                        return Err(rivet_core::Error::UnexpectedDType {
                            expected: DType::U8,
                            actual: sample.image.dtype(),
                        });
                    };
                    erase_random_values(
                        pixels,
                        layout,
                        sample.image.device(),
                        &dims,
                        axis_order,
                        (height, width, channels),
                        region,
                        |c, _, _| {
                            if random {
                                normal(rng).trunc().rem_euclid(256.0) as u8
                            } else {
                                fills[if fills.len() == 1 { 0 } else { c }]
                            }
                        },
                    )
                })?
            }
            DType::F32 => {
                let fills = match &self.value {
                    ErasingValue::Constant(value) => vec![*value],
                    ErasingValue::Channels(values) => values.clone(),
                    ErasingValue::Random => Vec::new(),
                };
                let random = matches!(&self.value, ErasingValue::Random);
                sample.image.with_cpu_storage(|storage, layout| {
                    let CpuStorageRef::F32(pixels) = storage else {
                        return Err(rivet_core::Error::UnexpectedDType {
                            expected: DType::F32,
                            actual: sample.image.dtype(),
                        });
                    };
                    erase_random_values(
                        pixels,
                        layout,
                        sample.image.device(),
                        &dims,
                        axis_order,
                        (height, width, channels),
                        region,
                        |c, _, _| {
                            if random {
                                normal(rng)
                            } else {
                                fills[if fills.len() == 1 { 0 } else { c }]
                            }
                        },
                    )
                })?
            }
            dtype => {
                return Err(invalid_argument(format!(
                    "random_erasing supports uint8 or float32 input, got {dtype:?}"
                )));
            }
        };

        Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
            image: output,
            label: sample.label,
        }))
    }
}

fn erase_random_values<T: rivet_core::WithDType, F: FnMut(usize, usize, usize) -> T>(
    values: &[T],
    layout: &rivet_core::Layout,
    device: &rivet_core::Device,
    dims: &[usize],
    axis_order: ImageAxisOrder,
    (height, width, channels): (usize, usize, usize),
    region: EraseRegion,
    mut fill: F,
) -> rivet_core::Result<Tensor> {
    let read = |coords: [usize; 3]| {
        values
            .get(logical_offset(layout, &coords)?)
            .copied()
            .ok_or(rivet_core::Error::StorageOutOfBounds)
    };
    Tensor::from_exact_writer::<T, _, _>(dims, device, |out| {
        match axis_order {
            ImageAxisOrder::Hwc => {
                for y in 0..height {
                    for x in 0..width {
                        if channels == 3 {
                            if in_region(x, y, region) {
                                out.write_next(fill(0, y, x))?;
                                out.write_next(fill(1, y, x))?;
                                out.write_next(fill(2, y, x))?;
                            } else {
                                out.write_next(read([y, x, 0])?)?;
                                out.write_next(read([y, x, 1])?)?;
                                out.write_next(read([y, x, 2])?)?;
                            }
                        } else if in_region(x, y, region) {
                            for c in 0..channels {
                                out.write_next(fill(c, y, x))?;
                            }
                        } else {
                            for c in 0..channels {
                                out.write_next(read([y, x, c])?)?;
                            }
                        }
                    }
                }
            }
            ImageAxisOrder::Chw => {
                for c in 0..channels {
                    for y in 0..height {
                        for x in 0..width {
                            out.write_next(if in_region(x, y, region) {
                                fill(c, y, x)
                            } else {
                                read([c, y, x])?
                            })?;
                        }
                    }
                }
            }
        }
        Ok(())
    })
}

fn normal(rng: &mut RandomStream) -> f32 {
    let u1 = rng.next_f32().max(f32::MIN_POSITIVE);
    let u2 = rng.next_f32();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn in_region(x: usize, y: usize, region: EraseRegion) -> bool {
    x >= region.x && x < region.x + region.width && y >= region.y && y < region.y + region.height
}

fn uniform(rng: &mut RandomStream) -> f64 {
    rng.next_f64()
}

fn interpolate(unit: f64, min: f64, max: f64) -> f64 {
    min + (max - min) * unit
}

#[cfg(test)]
mod tests {
    use super::RandomErasingConfig;
    use crate::pipeline::op::SampleContext;
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{DType, Device, Tensor};
    use rivet_data::random::{OpKey, RandomContext};

    fn sample() -> ImageSample {
        ImageSample::Decoded(DecodedSample {
            image: Tensor::from_vec(vec![1u8, 2, 3, 4, 5, 6], [2, 1, 3], &Device::Cpu).unwrap(),
            label: 7,
        })
    }

    #[test]
    fn random_erasing_probability_zero_preserves_storage() {
        let input = sample().into_decoded().unwrap().image;
        let ctx = SampleContext::with_random(0, RandomContext::new(1));
        let mut rng = ctx.stream(OpKey::from_parts("RandomErasing", 0));
        let output = RandomErasingConfig::new(0.0)
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: input.clone(),
                    label: 0,
                }),
                &mut rng,
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert!(output.image.same_storage(&input));
    }

    #[test]
    fn random_erasing_failed_sampling_preserves_original_storage() {
        let input = sample().into_decoded().unwrap().image;
        let ctx = SampleContext::new(0);
        let mut rng = ctx.stream(OpKey::from_parts("RandomErasing", 0));
        let output = RandomErasingConfig::new(1.0)
            .with_params(1.0, 1.0, 0.5, 0.5)
            .with_value(9.0)
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: input.clone(),
                    label: 0,
                }),
                &mut rng,
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dtype(), DType::U8);
        assert_eq!(output.image.to_vec::<u8>().unwrap(), [1, 2, 3, 4, 5, 6]);
        assert!(output.image.same_storage(&input));
    }
}
