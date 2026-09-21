use crate::errors::{invalid_shape, RivetResult};
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use rivet_core::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutConfig {
    pub axis_order: ImageAxisOrder,
}

impl LayoutConfig {
    pub const fn new(axis_order: ImageAxisOrder) -> Self {
        Self { axis_order }
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        input_layout: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        let sample = sample.into_decoded()?;
        if input_layout == self.axis_order {
            return Ok(ImageSample::Decoded(sample));
        }
        if sample.image.rank() != 3 {
            return Err(invalid_shape(format!(
                "image axis-order conversion requires rank 3, got shape {:?}",
                sample.image.dims()
            )));
        }

        let image = match (input_layout, self.axis_order) {
            (ImageAxisOrder::Hwc, ImageAxisOrder::Chw) => sample.image.permute(&[2, 0, 1])?,
            (ImageAxisOrder::Chw, ImageAxisOrder::Hwc) => sample.image.permute(&[1, 2, 0])?,
            _ => sample.image,
        };

        Ok(ImageSample::Decoded(DecodedSample {
            image,
            label: sample.label,
        }))
    }

    pub fn apply_batch(&self, input: Tensor, input_layout: ImageAxisOrder) -> RivetResult<Tensor> {
        if input_layout == self.axis_order {
            return Ok(input);
        }
        if input.rank() != 4 {
            return Err(invalid_shape(format!(
                "image batch axis-order conversion requires rank 4, got shape {:?}",
                input.dims()
            )));
        }

        match (input_layout, self.axis_order) {
            (ImageAxisOrder::Hwc, ImageAxisOrder::Chw) => Ok(input.permute(&[0, 3, 1, 2])?),
            (ImageAxisOrder::Chw, ImageAxisOrder::Hwc) => Ok(input.permute(&[0, 2, 3, 1])?),
            _ => Ok(input),
        }
    }

    /// Apply a layout change whose input rank and axis order were checked by
    /// pipeline compilation.
    pub(crate) fn apply_batch_trusted(
        &self,
        input: Tensor,
        input_layout: ImageAxisOrder,
    ) -> RivetResult<Tensor> {
        if input_layout == self.axis_order {
            return Ok(input);
        }
        debug_assert_eq!(input.rank(), 4);
        match (input_layout, self.axis_order) {
            (ImageAxisOrder::Hwc, ImageAxisOrder::Chw) => Ok(input.permute(&[0, 3, 1, 2])?),
            (ImageAxisOrder::Chw, ImageAxisOrder::Hwc) => Ok(input.permute(&[0, 2, 3, 1])?),
            _ => Ok(input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LayoutConfig;
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{Device, Tensor};

    #[test]
    fn converts_hwc_to_chw_as_a_shared_view() {
        let image = Tensor::from_vec(vec![1u8, 2, 3, 4, 5, 6], [1, 2, 3], &Device::Cpu).unwrap();
        let storage_owner = image.clone();
        let sample = ImageSample::Decoded(DecodedSample { image, label: 0 });
        let out = LayoutConfig::new(ImageAxisOrder::Chw)
            .apply(sample, ImageAxisOrder::Hwc)
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
        let output = LayoutConfig::new(ImageAxisOrder::Chw)
            .apply_batch(input, ImageAxisOrder::Hwc)
            .unwrap();

        assert_eq!(output.dims(), [1, 3, 1, 2]);
        assert_eq!(output.to_vec::<u8>().unwrap(), [1, 4, 2, 5, 3, 6]);
        assert!(!output.is_contiguous());
        assert!(storage_owner.same_storage(&output));
    }
}
