use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use rivet_core::{CpuStorageRef, DType, Tensor};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlipDirection {
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlipConfig {
    pub direction: FlipDirection,
}

impl FlipConfig {
    pub const fn new(direction: FlipDirection) -> Self {
        Self { direction }
    }

    pub const fn horizontal() -> Self {
        Self::new(FlipDirection::Horizontal)
    }

    pub const fn vertical() -> Self {
        Self::new(FlipDirection::Vertical)
    }

    /// Apply to the historical HWC representation.
    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        self.apply_with_axis_order(sample, ImageAxisOrder::Hwc)
    }

    pub fn apply_with_axis_order(
        &self,
        sample: ImageSample,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        Ok(ImageSample::Decoded(flip_decoded(
            sample,
            axis_order,
            self.direction,
            "flip",
        )?))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomHorizontalFlipConfig {
    /// Probability of flipping a sample; one draw from the pipeline RNG
    /// stream is compared against it per image.
    pub probability: f64,
}

impl RandomHorizontalFlipConfig {
    pub const fn new(probability: f64) -> Self {
        Self { probability }
    }

    /// Apply to the historical HWC representation.
    pub fn apply(&self, sample: ImageSample, rng: &mut RandomStream) -> RivetResult<ImageSample> {
        self.apply_with_axis_order(sample, rng, ImageAxisOrder::Hwc)
    }

    pub fn apply_with_axis_order(
        &self,
        sample: ImageSample,
        rng: &mut RandomStream,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        if !rng.gen_bool(self.probability)? {
            return Ok(ImageSample::Decoded(sample));
        }

        Ok(ImageSample::Decoded(flip_decoded(
            sample,
            axis_order,
            FlipDirection::Horizontal,
            "random_horizontal_flip",
        )?))
    }
}

fn flip_decoded(
    sample: DecodedSample,
    axis_order: ImageAxisOrder,
    direction: FlipDirection,
    op_name: &str,
) -> RivetResult<DecodedSample> {
    if sample.image.dtype() != DType::U8 || sample.image.rank() != 3 {
        return Err(invalid_argument(format!(
            "{op_name} requires a rank-3 uint8 image, got dtype {:?} shape {:?}",
            sample.image.dtype(),
            sample.image.dims()
        )));
    }
    let dims = sample.image.dims().to_vec();
    let (height, width, channels) = match axis_order {
        ImageAxisOrder::Hwc => (dims[0], dims[1], dims[2]),
        ImageAxisOrder::Chw => (dims[1], dims[2], dims[0]),
    };
    if height == 0 || width == 0 || channels == 0 {
        return Err(invalid_shape(format!(
            "{op_name} does not support zero-sized image dimensions"
        )));
    }

    let output = sample.image.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::U8(values) = storage else {
            return Err(rivet_core::Error::UnexpectedDType {
                expected: DType::U8,
                actual: sample.image.dtype(),
            });
        };
        let stride = layout.stride();
        let offset = |coords: &[usize]| {
            coords.iter().zip(stride).try_fold(
                layout.start_offset(),
                |offset, (&coord, &stride)| {
                    let delta = coord
                        .checked_mul(stride)
                        .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                    offset
                        .checked_add(delta)
                        .ok_or(rivet_core::Error::StorageOutOfBounds)
                },
            )
        };
        let mut output = Vec::with_capacity(sample.image.elem_count());
        match axis_order {
            ImageAxisOrder::Hwc => {
                for y in 0..height {
                    for x in 0..width {
                        for channel in 0..channels {
                            let (source_y, source_x) = match direction {
                                FlipDirection::Horizontal => (y, width - 1 - x),
                                FlipDirection::Vertical => (height - 1 - y, x),
                            };
                            let coords = [source_y, source_x, channel];
                            output.push(
                                *values
                                    .get(offset(&coords)?)
                                    .ok_or(rivet_core::Error::StorageOutOfBounds)?,
                            );
                        }
                    }
                }
            }
            ImageAxisOrder::Chw => {
                for channel in 0..channels {
                    for y in 0..height {
                        for x in 0..width {
                            let (source_y, source_x) = match direction {
                                FlipDirection::Horizontal => (y, width - 1 - x),
                                FlipDirection::Vertical => (height - 1 - y, x),
                            };
                            let coords = [channel, source_y, source_x];
                            output.push(
                                *values
                                    .get(offset(&coords)?)
                                    .ok_or(rivet_core::Error::StorageOutOfBounds)?,
                            );
                        }
                    }
                }
            }
        }
        Ok(output)
    })?;

    Ok(DecodedSample {
        image: Tensor::from_vec(output, dims, sample.image.device())?,
        label: sample.label,
    })
}

#[cfg(test)]
mod tests {
    use super::FlipConfig;
    use super::RandomHorizontalFlipConfig;
    use crate::pipeline::op::SampleContext;
    use crate::sample::image::ImageSample::Decoded;
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{Device, Tensor};
    use rivet_data::random::{OpKey, RandomContext};

    /// 2x2 RGB image with asymmetric content so a flip is observable.
    fn asymmetric_image() -> DecodedSample {
        DecodedSample {
            image: Tensor::from_vec(
                vec![
                    1u8, 2, 3, 4, 5, 6, //
                    7, 8, 9, 10, 11, 12,
                ],
                [2, 2, 3],
                &Device::Cpu,
            )
            .unwrap(),
            label: 3,
        }
    }

    fn raw_pixels(sample: ImageSample) -> Vec<u8> {
        let sample = sample.into_decoded().unwrap();
        sample.image.to_vec::<u8>().unwrap()
    }

    fn apply_with(probability: f64, seed: u64, index: usize) -> Vec<u8> {
        let ctx = SampleContext::with_random(index, RandomContext::new(seed));
        let mut rng = ctx.stream(OpKey::from_parts("RandomHorizontalFlip", 0));
        let out = RandomHorizontalFlipConfig::new(probability)
            .apply(Decoded(asymmetric_image()), &mut rng)
            .unwrap();
        raw_pixels(out)
    }

    #[test]
    fn flip_probability_zero_never_flips() {
        assert_eq!(
            apply_with(0.0, 0, 0),
            raw_pixels(Decoded(asymmetric_image()))
        );
    }

    #[test]
    fn flip_probability_one_always_flips() {
        let expected = raw_pixels(
            FlipConfig::horizontal()
                .apply(Decoded(asymmetric_image()))
                .unwrap(),
        );

        for seed in 0..32u64 {
            assert_eq!(apply_with(1.0, seed, seed as usize), expected);
        }
    }

    #[test]
    fn flip_is_deterministic_per_seed_and_sample() {
        assert_eq!(apply_with(0.5, 42, 0), apply_with(0.5, 42, 0));
        assert_eq!(apply_with(0.5, 7, 9), apply_with(0.5, 7, 9));
    }

    #[test]
    fn flip_mixes_flipped_and_original_samples() {
        // Across a fixed spread of seeds both outcomes must occur,
        // otherwise the "random" flip is degenerate.
        let mut distinct: Vec<Vec<u8>> = (0..64).map(|seed| apply_with(0.5, seed, 0)).collect();
        distinct.sort();
        distinct.dedup();
        assert!(distinct.len() > 1, "both flip outcomes must occur");
    }

    #[test]
    fn vertical_flip_supports_chw_and_materializes() {
        let image = Tensor::from_vec(
            vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [3, 2, 2],
            &Device::Cpu,
        )
        .unwrap();
        let owner = image.clone();
        let out = FlipConfig::vertical()
            .apply_with_axis_order(
                Decoded(DecodedSample { image, label: 8 }),
                ImageAxisOrder::Chw,
            )
            .unwrap()
            .into_decoded()
            .unwrap();

        assert_eq!(out.image.dims(), [3, 2, 2]);
        assert_eq!(
            out.image.to_vec::<u8>().unwrap(),
            [3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10]
        );
        assert!(!out.image.same_storage(&owner));
        assert_eq!(out.label, 8);
    }

    #[test]
    fn horizontal_flip_reads_non_contiguous_hwc_logically() {
        let base = Tensor::from_vec((0..18).collect::<Vec<u8>>(), [2, 3, 3], &Device::Cpu).unwrap();
        let image = base.permute(&[1, 0, 2]).unwrap();
        assert!(!image.is_contiguous());
        let expected = Tensor::from_vec(
            vec![
                9u8, 10, 11, 0, 1, 2, 12, 13, 14, 3, 4, 5, 15, 16, 17, 6, 7, 8,
            ],
            [3, 2, 3],
            &Device::Cpu,
        )
        .unwrap();
        let out = FlipConfig::horizontal()
            .apply_with_axis_order(
                Decoded(DecodedSample {
                    image: image.clone(),
                    label: 0,
                }),
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap();

        assert_eq!(
            out.image.to_vec::<u8>().unwrap(),
            expected.to_vec::<u8>().unwrap()
        );
        assert!(!out.image.same_storage(&image));
    }
}
