use crate::errors::{RivetResult, invalid_shape};
use crate::sample::image::{DecodedSample, ImageLayout, ImageSample};

#[derive(Clone, Copy)]
pub struct LayoutConfig {
    pub layout: ImageLayout,
}

impl LayoutConfig {
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
        let out = LayoutConfig {
            layout: ImageLayout::Chw,
        }
        .apply(sample, ImageLayout::Hwc)
        .unwrap()
        .into_decoded()
        .unwrap();

        assert_eq!(out.image.dims(), [3, 1, 2]);
        assert_eq!(out.image.to_vec::<u8>().unwrap(), [1, 4, 2, 5, 3, 6]);
        assert!(!out.image.is_contiguous());
        assert!(storage_owner.same_storage(&out.image));
    }
}
