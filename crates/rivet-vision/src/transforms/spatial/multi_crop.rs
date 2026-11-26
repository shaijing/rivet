use crate::errors::RivetResult;
use crate::sample::image::{ImageAxisOrder, ImageSample};
use crate::transforms::spatial::crop::CropConfig;
use crate::transforms::spatial::flip::FlipConfig;
use rivet_core::DType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FiveCropConfig {
    pub width: u32,
    pub height: u32,
}

impl FiveCropConfig {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(crate::errors::invalid_argument(
                "five_crop width and height must be greater than 0",
            ));
        }
        Ok(())
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<Vec<ImageSample>> {
        self.validate()?;
        let decoded = sample.into_decoded()?;
        if decoded.image.rank() != 3 {
            return Err(crate::errors::invalid_shape(format!(
                "five_crop requires a rank-3 image, got shape {:?}",
                decoded.image.dims()
            )));
        }
        let (height, width) = match axis_order {
            ImageAxisOrder::Hwc => (decoded.image.dims()[0], decoded.image.dims()[1]),
            ImageAxisOrder::Chw => (decoded.image.dims()[1], decoded.image.dims()[2]),
        };
        let width = u32::try_from(width)
            .map_err(|_| crate::errors::invalid_shape("five_crop image width is too large"))?;
        let height = u32::try_from(height)
            .map_err(|_| crate::errors::invalid_shape("five_crop image height is too large"))?;
        if self.width > width || self.height > height {
            return Err(crate::errors::invalid_shape(format!(
                "five_crop size {}x{} exceeds image shape {}x{}",
                self.width, self.height, width, height
            )));
        }
        let x_right = width - self.width;
        let y_bottom = height - self.height;
        let x_center = x_right / 2 + (x_right % 4 == 3) as u32;
        let y_center = y_bottom / 2 + (y_bottom % 4 == 3) as u32;
        let regions = [
            (0, 0),
            (x_right, 0),
            (0, y_bottom),
            (x_right, y_bottom),
            (x_center, y_center),
        ];
        regions
            .into_iter()
            .map(|(x, y)| {
                CropConfig::new(x, y, self.width, self.height)
                    .apply(ImageSample::Decoded(decoded.clone()), axis_order)
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TenCropConfig {
    pub width: u32,
    pub height: u32,
    pub vertical_flip: bool,
}

impl TenCropConfig {
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            vertical_flip: false,
        }
    }

    pub const fn with_vertical_flip(mut self, vertical_flip: bool) -> Self {
        self.vertical_flip = vertical_flip;
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        FiveCropConfig::new(self.width, self.height).validate()
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<Vec<ImageSample>> {
        self.validate()?;
        let decoded = sample.into_decoded()?;
        let five = FiveCropConfig::new(self.width, self.height);
        let mut output = five.apply(ImageSample::Decoded(decoded.clone()), axis_order)?;
        let flipped = if decoded.image.dtype() == DType::U8 {
            let flip = if self.vertical_flip {
                FlipConfig::vertical()
            } else {
                FlipConfig::horizontal()
            };
            flip.apply_with_axis_order(ImageSample::Decoded(decoded), axis_order)?
        } else {
            let axis = match (axis_order, self.vertical_flip) {
                (ImageAxisOrder::Hwc, false) | (ImageAxisOrder::Chw, true) => 1,
                (ImageAxisOrder::Hwc, true) => 0,
                (ImageAxisOrder::Chw, false) => 2,
            };
            ImageSample::Decoded(crate::sample::image::DecodedSample {
                image: decoded.image.flip(&[axis])?,
                label: decoded.label,
            })
        };
        output.extend(five.apply(flipped, axis_order)?);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::{FiveCropConfig, TenCropConfig};
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{Device, Tensor};

    fn sample() -> ImageSample {
        ImageSample::Decoded(DecodedSample {
            image: Tensor::from_vec((0..48).collect::<Vec<u8>>(), [4, 4, 3], &Device::Cpu).unwrap(),
            label: 3,
        })
    }

    #[test]
    fn five_crop_returns_ordered_shared_views() {
        let outputs = FiveCropConfig::new(2, 2)
            .apply(sample(), ImageAxisOrder::Hwc)
            .unwrap();
        assert_eq!(outputs.len(), 5);
        let first = outputs[0].clone().into_decoded().unwrap();
        let second = outputs[1].clone().into_decoded().unwrap();
        assert_eq!(first.image.dims(), [2, 2, 3]);
        assert_eq!(
            first.image.to_vec::<u8>().unwrap(),
            (0..6).chain(12..18).collect::<Vec<_>>()
        );
        assert_eq!(
            second.image.to_vec::<u8>().unwrap(),
            (6..12).chain(18..24).collect::<Vec<_>>()
        );
        assert!(first.image.same_storage(&second.image));
    }

    #[test]
    fn five_crop_center_rounds_half_pixel_to_even() {
        let outputs = FiveCropConfig::new(1, 1)
            .apply(sample(), ImageAxisOrder::Hwc)
            .unwrap();
        let center = outputs[4].clone().into_decoded().unwrap();
        assert_eq!(center.image.to_vec::<u8>().unwrap(), [30, 31, 32]);
    }

    #[test]
    fn ten_crop_has_one_materialized_flip_branch() {
        let outputs = TenCropConfig::new(2, 2)
            .apply(sample(), ImageAxisOrder::Hwc)
            .unwrap();
        assert_eq!(outputs.len(), 10);
        let original = outputs[0].clone().into_decoded().unwrap();
        let flipped = outputs[5].clone().into_decoded().unwrap();
        assert!(
            original
                .image
                .same_storage(&outputs[1].clone().into_decoded().unwrap().image)
        );
        assert!(!original.image.same_storage(&flipped.image));
    }
}
