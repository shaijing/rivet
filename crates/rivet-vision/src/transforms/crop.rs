use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{ImageAxisOrder, ImageSample};
use crate::transforms::with_rgb_pixels;
use rivet_core::{DType, ExactOutput, Tensor};
use rivet_data::random::RandomStream;

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
        let right = x
            .checked_add(crop_width)
            .ok_or_else(|| invalid_shape("crop width overflows usize"))?
            .saturating_sub(width);
        let bottom = y
            .checked_add(crop_height)
            .ok_or_else(|| invalid_shape("crop height overflows usize"))?
            .saturating_sub(height);
        let sample = if right != 0 || bottom != 0 {
            super::pad::PadConfig::with_sides(
                0,
                0,
                u32::try_from(right).map_err(|_| invalid_shape("crop padding too large"))?,
                u32::try_from(bottom).map_err(|_| invalid_shape("crop padding too large"))?,
            )
            .apply(ImageSample::Decoded(sample), layout)?
            .into_decoded()?
        } else {
            sample
        };
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
        let pad_width = (self.width as usize).saturating_sub(image_width) as u32;
        let pad_height = (self.height as usize).saturating_sub(image_height) as u32;
        let sample = if pad_width != 0 || pad_height != 0 {
            super::pad::PadConfig::with_sides(
                pad_width / 2,
                pad_height / 2,
                pad_width - pad_width / 2,
                pad_height - pad_height / 2,
            )
            .apply(ImageSample::Decoded(sample), layout)?
            .into_decoded()?
        } else {
            sample
        };
        let image_width = image_width.max(self.width as usize);
        let image_height = image_height.max(self.height as usize);

        // Python round uses ties-to-even for half-pixel center offsets.
        let center_offset = |difference: u32| difference / 2 + (difference % 4 == 3) as u32;
        let x = center_offset(image_width as u32 - self.width);
        let y = center_offset(image_height as u32 - self.height);
        CropConfig::new(x, y, self.width, self.height).apply(ImageSample::Decoded(sample), layout)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RandomCropConfig {
    pub width: u32,
    pub height: u32,
    /// Zero-padding applied on every side before the crop window is drawn.
    pub padding: u32,
    pub padding_sides: Option<[u32; 4]>,
    pub pad_if_needed: bool,
    pub fill: Vec<f32>,
    pub padding_mode: super::PaddingMode,
}

impl RandomCropConfig {
    pub fn new(width: u32, height: u32, padding: u32) -> Self {
        Self {
            width,
            height,
            padding,
            padding_sides: None,
            pad_if_needed: false,
            fill: vec![0.0],
            padding_mode: super::PaddingMode::Constant,
        }
    }

    pub fn with_options(
        mut self,
        padding_sides: Option<[u32; 4]>,
        pad_if_needed: bool,
        fill: Vec<f32>,
        padding_mode: super::PaddingMode,
    ) -> Self {
        self.padding_sides = padding_sides;
        self.pad_if_needed = pad_if_needed;
        self.fill = fill;
        self.padding_mode = padding_mode;
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(invalid_argument(
                "random_crop width and height must be greater than 0",
            ));
        }
        if self.fill.is_empty()
            || (self.fill.len() != 1 && self.fill.len() != 3)
            || self.fill.iter().any(|v| !v.is_finite())
        {
            return Err(invalid_argument(
                "random_crop fill must be a finite number or a 3-value RGB sequence",
            ));
        }
        Ok(())
    }

    /// Apply explicit padding and optional minimum-size padding, then crop a
    /// `width`x`height` window with a per-sample random offset (y then x).
    /// Default options retain the direct zero-padding and crop writer.
    pub fn apply(
        &self,
        sample: ImageSample,
        rng: &mut RandomStream,
        layout: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = sample.into_decoded()?;
        if layout != ImageAxisOrder::Hwc {
            return Err(crate::errors::invalid_argument(
                "random_crop currently requires uint8 HWC input",
            ));
        }
        if sample.image.dtype() != DType::U8
            || sample.image.rank() != 3
            || sample.image.dims()[2] != 3
        {
            return Err(crate::errors::invalid_argument(format!(
                "random_crop requires uint8 HWC RGB input, got dtype {:?} shape {:?}",
                sample.image.dtype(),
                sample.image.dims()
            )));
        }
        if self.padding_sides.is_some()
            || self.pad_if_needed
            || self.fill != [0.0]
            || self.padding_mode != super::PaddingMode::Constant
        {
            return self.apply_with_padding_options(sample, rng);
        }
        let image = with_rgb_pixels(&sample.image, "random_crop", |source, width, height| {
            let pad = u64::from(self.padding);
            let padded_width = width as u64 + 2 * pad;
            let padded_height = height as u64 + 2 * pad;
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
            if self.width > padded_width || self.height > padded_height {
                return Err(invalid_shape(format!(
                    "random_crop size {}x{} exceeds padded image {}x{}",
                    self.width, self.height, padded_width, padded_height
                )));
            }
            // Keep both draws, in y/x order, even for a single possible window.
            let y = rng.gen_range_usize(
                0..(u64::from(padded_height) - u64::from(self.height) + 1) as usize,
            )?;
            let x = rng.gen_range_usize(
                0..(u64::from(padded_width) - u64::from(self.width) + 1) as usize,
            )?;
            let (out_width, out_height, pad) = (
                self.width as usize,
                self.height as usize,
                self.padding as usize,
            );
            let left = pad.saturating_sub(x).min(out_width);
            let source_x = x.saturating_sub(pad);
            let copy_width = width.saturating_sub(source_x).min(out_width - left);
            Ok(Tensor::from_exact_writer::<u8, _, _>(
                [out_height, out_width, 3],
                sample.image.device(),
                |out| {
                    for row in 0..out_height {
                        let source_y = y + row;
                        if source_y < pad || source_y - pad >= height || copy_width == 0 {
                            write_zeros(out, out_width * 3)?;
                        } else {
                            write_zeros(out, left * 3)?;
                            let start = ((source_y - pad) * width + source_x) * 3;
                            out.extend_from_slice(&source[start..start + copy_width * 3])?;
                            write_zeros(out, (out_width - left - copy_width) * 3)?;
                        }
                    }
                    Ok(())
                },
            )?)
        })?;
        Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
            image,
            label: sample.label,
        }))
    }

    fn apply_with_padding_options(
        &self,
        mut sample: crate::sample::image::DecodedSample,
        rng: &mut RandomStream,
    ) -> RivetResult<ImageSample> {
        let label = sample.label.clone();
        let [left, top, right, bottom] = self.padding_sides.unwrap_or([self.padding; 4]);
        // torchvision applies the requested padding first. It then pads a
        // second time on the axis that is still smaller than the crop.
        sample = apply_crop_padding_stage(
            sample,
            [left, top, right, bottom],
            &self.fill,
            self.padding_mode,
        )?;
        if self.pad_if_needed {
            let width = sample.image.dims()[1];
            if width < self.width as usize {
                let deficit = self.width - width as u32;
                sample = apply_crop_padding_stage(
                    sample,
                    [deficit, 0, deficit, 0],
                    &self.fill,
                    self.padding_mode,
                )?;
            }
            let height = sample.image.dims()[0];
            if height < self.height as usize {
                let deficit = self.height - height as u32;
                sample = apply_crop_padding_stage(
                    sample,
                    [0, deficit, 0, deficit],
                    &self.fill,
                    self.padding_mode,
                )?;
            }
        }
        let height = sample.image.dims()[0] as usize;
        let width = sample.image.dims()[1] as usize;
        if self.width as usize > width || self.height as usize > height {
            return Err(invalid_shape(format!(
                "random_crop size {}x{} exceeds padded image {}x{}",
                self.width, self.height, width, height
            )));
        }
        let y = rng.gen_range_usize(0..height - self.height as usize + 1)?;
        let x = rng.gen_range_usize(0..width - self.width as usize + 1)?;
        Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
            image: super::CropConfig::new(x as u32, y as u32, self.width, self.height)
                .apply(ImageSample::Decoded(sample), ImageAxisOrder::Hwc)?
                .into_decoded()?
                .image,
            label,
        }))
    }
}

fn apply_crop_padding_stage(
    sample: crate::sample::image::DecodedSample,
    sides: [u32; 4],
    fill: &[f32],
    mode: super::PaddingMode,
) -> RivetResult<crate::sample::image::DecodedSample> {
    if sides == [0; 4] {
        return Ok(sample);
    }
    let mut pad =
        super::PadConfig::with_sides(sides[0], sides[1], sides[2], sides[3]).with_mode(mode);
    if fill.len() == 3 {
        pad = pad.with_fill_channels([fill[0], fill[1], fill[2]]);
    } else {
        pad = pad.with_fill(fill.first().copied().unwrap_or(0.0));
    }
    pad.apply(ImageSample::Decoded(sample), ImageAxisOrder::Hwc)?
        .into_decoded()
}

#[inline]
fn write_zeros(out: &mut ExactOutput<u8>, mut count: usize) -> rivet_core::Result<()> {
    const ZEROS: [u8; 256] = [0; 256];
    while count >= ZEROS.len() {
        out.extend_from_slice(&ZEROS)?;
        count -= ZEROS.len();
    }
    out.extend_from_slice(&ZEROS[..count])
}

#[cfg(test)]
mod tests {
    use super::{CenterCropConfig, CropConfig, RandomCropConfig};
    use crate::pipeline::op::SampleContext;
    use crate::sample::image::ImageSample::Decoded;
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{Device, Tensor};
    use rivet_data::random::{OpKey, RandomContext};

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
        let ctx = SampleContext::with_random(index, RandomContext::new(seed));
        let mut rng = ctx.stream(OpKey::from_parts("RandomCrop", 0));
        let out = RandomCropConfig::new(4, 4, 2)
            .apply(Decoded(unique_image()), &mut rng, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        out.image.to_vec::<u8>().unwrap()
    }

    #[test]
    fn direct_random_crop_matches_padded_backend_for_views_and_rng() {
        use crate::transforms::into_rgb_image;
        use image::{Rgb, RgbImage};
        let base = Tensor::from_vec(
            (0..2 * 5 * 7 * 3).map(|v| v as u8).collect::<Vec<_>>(),
            [2, 5, 7, 3],
            &Device::Cpu,
        )
        .unwrap();
        let image = base.get(1).unwrap();
        let views = [
            image.clone(),
            image.permute(&[1, 0, 2]).unwrap(),
            image.narrow(0, 1, 3).unwrap().narrow(1, 2, 4).unwrap(),
        ];
        for view in views {
            for pad in [0, 1, 4, 9] {
                let width = view.dims()[1] as u32;
                let height = view.dims()[0] as u32;
                for (cw, ch) in [(1, 1), (width, height), (width + 2 * pad, height + 2 * pad)] {
                    for seed in 0..20 {
                        let sample = DecodedSample {
                            image: view.clone(),
                            label: 7,
                        };
                        let ctx = RandomContext::new(seed);
                        let mut actual_rng = ctx.stream(
                            rivet_data::random::SampleKey::from_index(3),
                            OpKey::from_parts("RandomCrop", 0),
                        );
                        let mut expected_rng = actual_rng.clone();
                        let actual = RandomCropConfig::new(cw, ch, pad)
                            .apply(
                                Decoded(sample.clone()),
                                &mut actual_rng,
                                ImageAxisOrder::Hwc,
                            )
                            .unwrap()
                            .into_decoded()
                            .unwrap();
                        let (rgb, _) = into_rgb_image(sample, "reference").unwrap();
                        let mut padded =
                            RgbImage::from_pixel(width + 2 * pad, height + 2 * pad, Rgb([0; 3]));
                        image::imageops::replace(&mut padded, &rgb, i64::from(pad), i64::from(pad));
                        let y = expected_rng
                            .gen_range_usize(0..(height + 2 * pad - ch + 1) as usize)
                            .unwrap() as u32;
                        let x = expected_rng
                            .gen_range_usize(0..(width + 2 * pad - cw + 1) as usize)
                            .unwrap() as u32;
                        let expected = image::imageops::crop_imm(&padded, x, y, cw, ch).to_image();
                        assert_eq!(actual.image.to_vec::<u8>().unwrap(), expected.into_raw());
                        assert_eq!(actual_rng.next_u64(), expected_rng.next_u64());
                        assert_eq!(actual.label, 7);
                        assert!(actual.image.is_contiguous());
                        assert!(!actual.image.same_storage(&view));
                    }
                }
            }
        }
    }

    #[test]
    fn random_crop_reproduces_same_seed_and_sample() {
        assert_eq!(crop_with(42, 0), crop_with(42, 0));
        assert_eq!(crop_with(42, 5), crop_with(42, 5));
    }

    #[test]
    fn random_crop_pad_if_needed_supports_side_fill() {
        let config = RandomCropConfig::new(7, 6, 0).with_options(
            Some([1, 2, 3, 4]),
            true,
            vec![11.0, 22.0, 33.0],
            crate::transforms::PaddingMode::Constant,
        );
        let mut rng = RandomContext::new(17).stream(
            rivet_data::random::SampleKey::from_index(0),
            OpKey::from_parts("RandomCrop", 0),
        );
        let out = config
            .apply(Decoded(unique_image()), &mut rng, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(out.image.dims(), &[6, 7, 3]);
        assert_eq!(out.label, 7);
        assert!(
            out.image
                .to_vec::<u8>()
                .unwrap()
                .chunks_exact(3)
                .any(|pixel| pixel == [11, 22, 33])
        );
    }

    #[test]
    fn random_crop_exposes_nonconstant_padding_modes() {
        for mode in [
            crate::transforms::PaddingMode::Edge,
            crate::transforms::PaddingMode::Reflect,
            crate::transforms::PaddingMode::Symmetric,
        ] {
            let sample = unique_image();
            let expected = crate::transforms::PadConfig::with_sides(1, 1, 1, 1)
                .with_mode(mode)
                .apply(Decoded(sample.clone()), ImageAxisOrder::Hwc)
                .unwrap()
                .into_decoded()
                .unwrap();
            let config = RandomCropConfig::new(6, 6, 0).with_options(
                Some([1, 1, 1, 1]),
                false,
                vec![0.0],
                mode,
            );
            let mut rng = RandomContext::new(8).stream(
                rivet_data::random::SampleKey::from_index(0),
                OpKey::from_parts("RandomCrop", 0),
            );
            let actual = config
                .apply(Decoded(sample), &mut rng, ImageAxisOrder::Hwc)
                .unwrap()
                .into_decoded()
                .unwrap();
            assert_eq!(
                actual.image.to_vec::<u8>().unwrap(),
                expected.image.to_vec::<u8>().unwrap()
            );
        }
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
        let ctx = SampleContext::with_random(0, RandomContext::new(99));
        let mut rng = ctx.stream(OpKey::from_parts("RandomCrop", 0));
        let sample = RandomCropConfig::new(4, 4, 2)
            .apply(Decoded(unique_image()), &mut rng, ImageAxisOrder::Hwc)
            .unwrap();
        let out = sample.into_decoded().unwrap();
        assert_eq!(out.image.dims(), [4, 4, 3]);
        assert_eq!(out.label, 7);
    }

    #[test]
    fn random_crop_size_beyond_padded_image_rejected() {
        let ctx = SampleContext::with_random(0, RandomContext::new(1));
        let mut rng = ctx.stream(OpKey::from_parts("RandomCrop", 0));
        let result = RandomCropConfig::new(9, 4, 2).apply(
            Decoded(unique_image()),
            &mut rng,
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
