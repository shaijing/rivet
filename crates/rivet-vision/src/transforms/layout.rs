use crate::errors::{RivetResult, invalid_shape};
use crate::sample::image::{DecodedSample, ImageLayout, ImageSample};
use rivet_core::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutConfig {
    pub layout: ImageLayout,
}

impl LayoutConfig {
    pub const fn new(layout: ImageLayout) -> Self {
        Self { layout }
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        input_layout: ImageLayout,
    ) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        if input_layout == self.layout {
            return Ok(ImageSample::Decoded(sample));
        }
        if sample.image.rank() != 3 {
            return Err(invalid_shape(format!(
                "image layout conversion requires rank 3, got shape {:?}",
                sample.image.dims()
            )));
        }

        let image = match (input_layout, self.layout) {
            (ImageLayout::Hwc, ImageLayout::Chw) => sample.image.permute(&[2, 0, 1])?,
            (ImageLayout::Chw, ImageLayout::Hwc) => sample.image.permute(&[1, 2, 0])?,
            _ => sample.image,
        };

        Ok(ImageSample::Decoded(DecodedSample {
            image,
            label: sample.label,
        }))
    }

    pub fn apply_batch(&self, input: Tensor, input_layout: ImageLayout) -> RivetResult<Tensor> {
        if input_layout == self.layout {
            return Ok(input);
        }
        if input.rank() != 4 {
            return Err(invalid_shape(format!(
                "image batch layout conversion requires rank 4, got shape {:?}",
                input.dims()
            )));
        }

        match (input_layout, self.layout) {
            (ImageLayout::Hwc, ImageLayout::Chw) => Ok(input.permute(&[0, 3, 1, 2])?),
            (ImageLayout::Chw, ImageLayout::Hwc) => Ok(input.permute(&[0, 2, 3, 1])?),
            _ => Ok(input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LayoutConfig;
    use crate::sample::image::{DecodedSample, ImageLayout, ImageSample};
    use rivet_core::{Device, Tensor};

    #[test]
    fn converts_hwc_to_chw_as_a_shared_view() {
        let image = Tensor::from_vec(vec![1u8, 2, 3, 4, 5, 6], [1, 2, 3], &Device::Cpu).unwrap();
        let storage_owner = image.clone();
        let sample = ImageSample::Decoded(DecodedSample { image, label: 0 });
        let out = LayoutConfig::new(ImageLayout::Chw)
            .apply(sample, ImageLayout::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();

        assert_eq!(out.image.dims(), [3, 1, 2]);
        assert_eq!(out.image.to_vec::<u8>().unwrap(), [1, 4, 2, 5, 3, 6]);
        assert!(!out.image.is_contiguous());
        assert!(storage_owner.same_storage(&out.image));
    }

    #[test]
    fn converts_nhwc_to_nchw_as_a_shared_view() {
        let input = Tensor::from_vec(vec![1u8, 2, 3, 4, 5, 6], [1, 1, 2, 3], &Device::Cpu).unwrap();
        let storage_owner = input.clone();
        let output = LayoutConfig::new(ImageLayout::Chw)
            .apply_batch(input, ImageLayout::Hwc)
            .unwrap();

        assert_eq!(output.dims(), [1, 3, 1, 2]);
        assert_eq!(output.to_vec::<u8>().unwrap(), [1, 4, 2, 5, 3, 6]);
        assert!(!output.is_contiguous());
        assert!(storage_owner.same_storage(&output));
    }
}
