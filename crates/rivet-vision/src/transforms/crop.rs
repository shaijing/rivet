use crate::errors::{RivetResult, invalid_shape};
use crate::pipeline::op::SampleContext;
use crate::sample::image::{ImageAxisOrder, ImageSample};
use crate::transforms::{from_rgb_image, into_rgb_image};
use image::imageops::crop_imm;
use image::{Rgb, RgbImage};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropConfig {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl CropConfig {
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn apply(&self, sample: ImageSample, layout: ImageAxisOrder) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        if sample.image.rank() != 3 {
            return Err(invalid_shape(format!(
                "crop requires a rank-3 image, got shape {:?}",
                sample.image.dims()
            )));
        }
        let (height, width) = match layout {
            ImageAxisOrder::Hwc => (sample.image.dims()[0], sample.image.dims()[1]),
            ImageAxisOrder::Chw => (sample.image.dims()[1], sample.image.dims()[2]),
        };
        let x = self.x as usize;
        let y = self.y as usize;
        let crop_width = self.width as usize;
        let crop_height = self.height as usize;
        if x.checked_add(crop_width).is_none_or(|end| end > width)
            || y.checked_add(crop_height).is_none_or(|end| end > height)
        {
            return Err(invalid_shape(format!(
                "crop rectangle ({}, {}, {}, {}) exceeds image shape {}x{}",
                self.x, self.y, self.width, self.height, width, height
            )));
        }
        let image = match layout {
            ImageAxisOrder::Hwc => sample
                .image
                .narrow(0, y, crop_height)?
                .narrow(1, x, crop_width)?,
            ImageAxisOrder::Chw => sample
                .image
                .narrow(1, y, crop_height)?
                .narrow(2, x, crop_width)?,
        };
        Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
            image,
            label: sample.label,
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CenterCropConfig {
    pub width: u32,
    pub height: u32,
}

impl CenterCropConfig {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn apply(&self, sample: ImageSample, layout: ImageAxisOrder) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        if sample.image.rank() != 3 {
            return Err(invalid_shape(format!(
                "center_crop requires a rank-3 image, got shape {:?}",
                sample.image.dims()
            )));
        }
        let (image_height, image_width) = match layout {
            ImageAxisOrder::Hwc => (sample.image.dims()[0], sample.image.dims()[1]),
            ImageAxisOrder::Chw => (sample.image.dims()[1], sample.image.dims()[2]),
        };
        if self.width as usize > image_width || self.height as usize > image_height {
            return Err(invalid_shape(format!(
                "center_crop size {}x{} exceeds image shape {}x{}",
                self.width, self.height, image_width, image_height
            )));
        }

        let x = (image_width as u32 - self.width) / 2;
        let y = (image_height as u32 - self.height) / 2;
        CropConfig::new(x, y, self.width, self.height).apply(ImageSample::Decoded(sample), layout)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RandomCropConfig {
    pub width: u32,
    pub height: u32,
    /// Zero-padding applied on every side before the crop window is drawn.
    pub padding: u32,
}

impl RandomCropConfig {
    pub const fn new(width: u32, height: u32, padding: u32) -> Self {
        Self {
            width,
            height,
            padding,
        }
    }

    /// Zero-pad the image by `padding` on each side, then crop a
    /// `width`x`height` window whose offset is drawn uniformly per sample
    /// from the pipeline RNG stream (two draws: y then x).
    pub fn apply(
        &self,
        sample: ImageSample,
        ctx: &mut SampleContext,
        layout: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        if layout != ImageAxisOrder::Hwc {
            return Err(crate::errors::invalid_argument(
                "random_crop currently requires uint8 HWC input",
            ));
        }
        let (image, label) = into_rgb_image(sample, "random_crop")?;
        let (width, height) = image.dimensions();

        let pad = u64::from(self.padding);
        let padded_width = u64::from(width) + 2 * pad;
        let padded_height = u64::from(height) + 2 * pad;
        let (padded_width, padded_height) =
            match (u32::try_from(padded_width), u32::try_from(padded_height)) {
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
        Ok(ImageSample::Decoded(from_rgb_image(cropped, label)?))
    }
}

#[cfg(test)]
mod tests {
    use super::{CenterCropConfig, CropConfig, RandomCropConfig, SampleContext};
    use crate::sample::image::ImageSample::Decoded;
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{Device, Tensor};

    /// 4x4 RGB image where every pixel is unique, so any crop offset
    /// produces a distinguishable result.
    fn unique_image() -> DecodedSample {
        let values: Vec<u8> = (0..4 * 4 * 3).map(|value| value as u8).collect();
        DecodedSample {
            image: Tensor::from_vec(values, [4, 4, 3], &Device::Cpu).unwrap(),
            label: 7,
        }
    }

    fn crop_with(seed: u64, index: usize) -> Vec<u8> {
        let mut ctx = SampleContext::new(index);
        ctx.global_seed = seed;
        let out = RandomCropConfig::new(4, 4, 2)
            .apply(Decoded(unique_image()), &mut ctx, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        out.image.to_vec::<u8>().unwrap()
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
        let sample = RandomCropConfig::new(4, 4, 2)
            .apply(Decoded(unique_image()), &mut ctx, ImageAxisOrder::Hwc)
            .unwrap();
        let out = sample.into_decoded().unwrap();
        assert_eq!(out.image.dims(), [4, 4, 3]);
        assert_eq!(out.label, 7);
    }

    #[test]
    fn random_crop_size_beyond_padded_image_rejected() {
        let mut ctx = SampleContext::new(0);
        ctx.global_seed = 1;
        let result = RandomCropConfig::new(9, 4, 2).apply(
            Decoded(unique_image()),
            &mut ctx,
            ImageAxisOrder::Hwc,
        );
        assert!(result.is_err(), "oversized random crop must fail");
    }

    #[test]
    fn crop_and_center_crop_are_shared_views_for_hwc_and_chw() {
        let image =
            Tensor::from_vec((0..24).collect::<Vec<u8>>(), [4, 2, 3], &Device::Cpu).unwrap();
        let owner = image.clone();
        let out = CropConfig::new(0, 1, 2, 2)
            .apply(
                ImageSample::Decoded(DecodedSample { image, label: 0 }),
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(out.image.dims(), [2, 2, 3]);
        assert_eq!(
            out.image.to_vec::<u8>().unwrap(),
            (6..18).collect::<Vec<_>>()
        );
        assert!(out.image.same_storage(&owner));

        let chw = Tensor::from_vec((0..24).collect::<Vec<u8>>(), [3, 4, 2], &Device::Cpu).unwrap();
        let owner = chw.clone();
        let out = CenterCropConfig::new(1, 2)
            .apply(
                Decoded(DecodedSample {
                    image: chw,
                    label: 0,
                }),
                ImageAxisOrder::Chw,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert!(out.image.same_storage(&owner));
        assert_eq!(out.image.dims(), [3, 2, 1]);
    }
}
