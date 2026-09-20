use crate::errors::RivetResult;
use crate::pipeline::op::SampleContext;
use crate::sample::image::ImageSample;
use crate::transforms::{from_rgb_image, into_rgb_image};
use image::imageops::{flip_horizontal, flip_vertical};

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

    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let (image, label) = into_rgb_image(sample, "flip")?;
        let flipped = match self.direction {
            FlipDirection::Horizontal => flip_horizontal(&image),
            FlipDirection::Vertical => flip_vertical(&image),
        };
        Ok(ImageSample::Decoded(from_rgb_image(flipped, label)?))
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

    pub fn apply(&self, sample: ImageSample, ctx: &mut SampleContext) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let draw = ctx.next_rng_u64();
        // Uniform [0, 1) over the full 53-bit mantissa.
        let value = (draw >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        if value >= self.probability {
            return Ok(ImageSample::Decoded(sample));
        }

        let (image, label) = into_rgb_image(sample, "random_horizontal_flip")?;
        let flipped = flip_horizontal(&image);
        Ok(ImageSample::Decoded(from_rgb_image(flipped, label)?))
    }
}

#[cfg(test)]
mod tests {
    use super::FlipConfig;
    use super::{RandomHorizontalFlipConfig, SampleContext};
    use crate::sample::image::ImageSample::Decoded;
    use crate::sample::image::{DecodedSample, ImageSample};
    use rivet_core::{Device, Tensor};

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
        let mut ctx = SampleContext::new(index);
        ctx.global_seed = seed;
        let out = RandomHorizontalFlipConfig::new(probability)
            .apply(Decoded(asymmetric_image()), &mut ctx)
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
}
