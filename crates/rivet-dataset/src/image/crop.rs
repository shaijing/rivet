use crate::errors::{RivetResult, invalid_shape};
use crate::image::{from_rgb_image, into_rgb_image};
use crate::pipeline::op::SampleContext;
use crate::sample::image::ImageSample;
use image::imageops::crop_imm;
use image::{Rgb, RgbImage};

#[derive(Clone, Copy)]
pub struct CropConfig {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl CropConfig {
    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;

        if self.x + self.width > sample.width || self.y + self.height > sample.height {
            return Err(invalid_shape(format!(
                "crop rectangle ({}, {}, {}, {}) exceeds image shape {}x{}",
                self.x, self.y, self.width, self.height, sample.width, sample.height
            )));
        }

        let (image, label) = into_rgb_image(sample, "crop")?;
        let cropped = crop_imm(&image, self.x, self.y, self.width, self.height).to_image();
        Ok(ImageSample::Decoded(from_rgb_image(cropped, label)))
    }
}

#[derive(Clone, Copy)]
pub struct CenterCropConfig {
    pub width: u32,
    pub height: u32,
}

impl CenterCropConfig {
    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;

        if self.width > sample.width || self.height > sample.height {
            return Err(invalid_shape(format!(
                "center_crop size {}x{} exceeds image shape {}x{}",
                self.width, self.height, sample.width, sample.height
            )));
        }

        let x = (sample.width - self.width) / 2;
        let y = (sample.height - self.height) / 2;
        CropConfig {
            x,
            y,
            width: self.width,
            height: self.height,
        }
        .apply(ImageSample::Decoded(sample))
    }
}

#[derive(Clone, Copy)]
pub struct RandomCropConfig {
    pub width: u32,
    pub height: u32,
    /// Zero-padding applied on every side before the crop window is drawn.
    pub padding: u32,
}

impl RandomCropConfig {
    /// Zero-pad the image by `padding` on each side, then crop a
    /// `width`x`height` window whose offset is drawn uniformly per sample
    /// from the pipeline RNG stream (two draws: y then x).
    pub fn apply(&self, sample: ImageSample, ctx: &mut SampleContext) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        let (image, label) = into_rgb_image(sample, "random_crop")?;
        let (width, height) = image.dimensions();

        let pad = u64::from(self.padding);
        let padded_width = u64::from(width) + 2 * pad;
        let padded_height = u64::from(height) + 2 * pad;
        let (padded_width, padded_height) = match (
            u32::try_from(padded_width),
            u32::try_from(padded_height),
        ) {
            (Ok(width), Ok(height)) => (width, height),
            _ => {
                return Err(invalid_shape(format!(
                    "random_crop padding {} makes image {}x{} too large",
                    self.padding, width, height
                )));
            }
        };
        let (crop_width, crop_height) = (u64::from(self.width), u64::from(self.height));
        if crop_width > u64::from(padded_width) || crop_height > u64::from(padded_height) {
            return Err(invalid_shape(format!(
                "random_crop size {}x{} exceeds padded image {}x{}",
                self.width, self.height, padded_width, padded_height
            )));
        }

        let x_count = u64::from(padded_width) - crop_width + 1;
        let y_count = u64::from(padded_height) - crop_height + 1;
        let y = (ctx.next_rng_u64() % y_count) as u32;
        let x = (ctx.next_rng_u64() % x_count) as u32;

        let mut padded = RgbImage::from_pixel(padded_width, padded_height, Rgb([0, 0, 0]));
        {
            let source = image.as_raw();
            let dest = padded.as_mut();
            let source_row = width as usize * 3;
            let dest_row = padded_width as usize * 3;
            let offset = self.padding as usize;
            for row in 0..height as usize {
                let start = (offset + row) * dest_row + offset * 3;
                dest[start..start + source_row]
                    .copy_from_slice(&source[row * source_row..(row + 1) * source_row]);
            }
        }

        let cropped = crop_imm(&padded, x, y, self.width, self.height).to_image();
        Ok(ImageSample::Decoded(from_rgb_image(cropped, label)))
    }
}

#[cfg(test)]
mod tests {
    use super::{RandomCropConfig, SampleContext};
    use crate::sample::image::{DecodedSample, ImageBuffer, ImageLayout};
    use crate::sample::image::ImageSample::Decoded;

    /// 4x4 RGB image where every pixel is unique, so any crop offset
    /// produces a distinguishable result.
    fn unique_image() -> DecodedSample {
        let values: Vec<u8> = (0..4 * 4 * 3).map(|value| value as u8).collect();
        DecodedSample {
            image: ImageBuffer::U8(values),
            width: 4,
            height: 4,
            channels: 3,
            label: 7,
            layout: ImageLayout::Hwc,
        }
    }

    fn crop_with(seed: u64, index: usize) -> Vec<u8> {
        let mut ctx = SampleContext::new(index);
        ctx.global_seed = seed;
        let out = RandomCropConfig {
            width: 4,
            height: 4,
            padding: 2,
        }
        .apply(Decoded(unique_image()), &mut ctx)
        .unwrap()
        .into_decoded()
        .unwrap();
        match out.image {
            ImageBuffer::U8(values) => values,
            ImageBuffer::F32(_) => panic!("expected u8 output"),
        }
    }

    #[test]
    fn random_crop_reproduces_same_seed_and_sample() {
        assert_eq!(crop_with(42, 0), crop_with(42, 0));
        assert_eq!(crop_with(42, 5), crop_with(42, 5));
    }

    #[test]
    fn random_crop_varies_with_seed_and_sample() {
        let outputs: Vec<Vec<u8>> = (0..20).map(|seed| crop_with(seed, 0)).collect();
        let mut distinct = outputs.clone();
        distinct.sort();
        distinct.dedup();
        assert!(distinct.len() > 1, "augmentation must vary across seeds");

        let outputs: Vec<Vec<u8>> = (0..20).map(|index| crop_with(1, index)).collect();
        let mut distinct = outputs.clone();
        distinct.sort();
        distinct.dedup();
        assert!(distinct.len() > 1, "augmentation must vary across samples");
    }

    #[test]
    fn random_crop_outputs_requested_size_and_preserves_labels() {
        let mut ctx = SampleContext::new(0);
        ctx.global_seed = 99;
        let sample = RandomCropConfig {
            width: 4,
            height: 4,
            padding: 2,
        }
        .apply(Decoded(unique_image()), &mut ctx)
        .unwrap();
        let out = sample.into_decoded().unwrap();
        assert_eq!(out.width, 4);
        assert_eq!(out.height, 4);
        assert_eq!(out.label, 7);
    }

    #[test]
    fn random_crop_size_beyond_padded_image_rejected() {
        let mut ctx = SampleContext::new(0);
        ctx.global_seed = 1;
        let result = RandomCropConfig {
            width: 9,
            height: 4,
            padding: 2,
        }
        .apply(Decoded(unique_image()), &mut ctx);
        assert!(result.is_err(), "oversized random crop must fail");
    }
}
