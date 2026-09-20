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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomErasingConfig {
    pub probability: f64,
    pub scale_min: f32,
    pub scale_max: f32,
    pub ratio_min: f32,
    pub ratio_max: f32,
    pub value: f32,
}

impl RandomErasingConfig {
    pub const fn new(probability: f64) -> Self {
        Self {
            probability,
            scale_min: 0.02,
            scale_max: 0.33,
            ratio_min: 0.3,
            ratio_max: 3.3333333,
            value: 0.0,
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

    pub const fn with_value(mut self, value: f32) -> Self {
        self.value = value;
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
            || !self.value.is_finite()
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
        let draw = uniform(rng);
        if draw >= self.probability || width == 0 || height == 0 {
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
            let erase_width = (target * ratio).sqrt().round() as usize;
            let erase_height = (target / ratio).sqrt().round() as usize;
            if erase_width > 0 && erase_height > 0 && erase_width <= width && erase_height <= height
            {
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
        Some(EraseRegion {
            x: 0,
            y: 0,
            width,
            height,
        })
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
        let Some(region) = self.resolve(width, height, rng) else {
            return Ok(ImageSample::Decoded(sample));
        };

        let output = match sample.image.dtype() {
            DType::U8 => {
                let fill = self.value.clamp(0.0, 255.0).round() as u8;
                let values = sample.image.with_cpu_storage(|storage, layout| {
                    let CpuStorageRef::U8(values) = storage else {
                        return Err(rivet_core::Error::UnexpectedDType {
                            expected: DType::U8,
                            actual: sample.image.dtype(),
                        });
                    };
                    map_erased(
                        values,
                        layout,
                        axis_order,
                        (height, width, channels),
                        region,
                        fill,
                    )
                })?;
                Tensor::from_vec(values, dims, sample.image.device())?
            }
            DType::F32 => {
                let fill = self.value;
                let values = sample.image.with_cpu_storage(|storage, layout| {
                    let CpuStorageRef::F32(values) = storage else {
                        return Err(rivet_core::Error::UnexpectedDType {
                            expected: DType::F32,
                            actual: sample.image.dtype(),
                        });
                    };
                    map_erased(
                        values,
                        layout,
                        axis_order,
                        (height, width, channels),
                        region,
                        fill,
                    )
                })?;
                Tensor::from_vec(values, dims, sample.image.device())?
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

fn map_erased<T: Copy>(
    values: &[T],
    layout: &rivet_core::Layout,
    axis_order: ImageAxisOrder,
    (height, width, channels): (usize, usize, usize),
    region: EraseRegion,
    fill: T,
) -> rivet_core::Result<Vec<T>> {
    let read = |coords: [usize; 3]| {
        values
            .get(logical_offset(layout, &coords)?)
            .copied()
            .ok_or(rivet_core::Error::StorageOutOfBounds)
    };
    let mut output = Vec::with_capacity(height * width * channels);
    match axis_order {
        ImageAxisOrder::Hwc => {
            for y in 0..height {
                for x in 0..width {
                    for c in 0..channels {
                        output.push(if in_region(x, y, region) {
                            fill
                        } else {
                            read([y, x, c])?
                        });
                    }
                }
            }
        }
        ImageAxisOrder::Chw => {
            for c in 0..channels {
                for y in 0..height {
                    for x in 0..width {
                        output.push(if in_region(x, y, region) {
                            fill
                        } else {
                            read([c, y, x])?
                        });
                    }
                }
            }
        }
    }
    Ok(output)
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
    fn random_erasing_can_erase_full_image_without_in_place_mutation() {
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
        assert_eq!(output.image.to_vec::<u8>().unwrap(), [9, 9, 9, 9, 9, 9]);
        assert!(!output.image.same_storage(&input));
    }
}
