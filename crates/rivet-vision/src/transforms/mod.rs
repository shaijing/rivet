pub mod color;
pub mod geometry;
pub mod representation;

mod crop;
mod decode;
mod dtype;
mod flip;
mod layout;
mod normalize;
mod resize;
mod rotate;

/// Padding semantics shared by geometry transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaddingMode {
    Constant,
    Edge,
    Reflect,
    Symmetric,
}

pub use color::{BrightnessConfig, ContrastConfig, GrayscaleConfig, HueConfig};
pub use geometry::{
    CenterCropConfig, CropConfig, FlipConfig, FlipDirection, InterpolationMode, LayoutConfig,
    RandomCropConfig, RandomHorizontalFlipConfig, ResizeConfig, RotateConfig, RotationAngle,
};
pub use representation::{ConvertImageDtypeConfig, DecodeImageConfig, NormalizeConfig};

use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::DecodedSample;
use image::RgbImage;
use rivet_core::{CpuStorageRef, DType, Device, Tensor};

pub(crate) fn require_u8_hwc(sample: DecodedSample, op_name: &str) -> RivetResult<DecodedSample> {
    let dims = sample.image.dims();
    if sample.image.dtype() != DType::U8 || dims.len() != 3 || dims[2] != 3 {
        return Err(invalid_argument(format!(
            "{op_name} requires uint8 HWC RGB input, got dtype {:?} shape {:?}",
            sample.image.dtype(),
            dims
        )));
    }

    Ok(sample)
}

pub(crate) fn into_rgb_image(sample: DecodedSample, op_name: &str) -> RivetResult<(RgbImage, i64)> {
    let sample = require_u8_hwc(sample, op_name)?;
    let label = sample.label;
    let [height, width, channels] = sample.image.dims() else {
        return Err(invalid_shape(format!(
            "{op_name} requires an HWC tensor with rank 3"
        )));
    };
    let width = u32::try_from(*width)
        .map_err(|_| invalid_shape(format!("{op_name} image width is too large: {width}")))?;
    let height = u32::try_from(*height)
        .map_err(|_| invalid_shape(format!("{op_name} image height is too large: {height}")))?;
    if *channels != 3 {
        return Err(invalid_shape(format!(
            "{op_name} requires 3 channels, got {channels}"
        )));
    }
    // Backend transforms require an owned image buffer, so this is the
    // deliberate materialization boundary. Read through the borrowed CPU
    // storage and logical layout rather than flattening the input tensor.
    let values = sample.image.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::U8(values) = storage else {
            return Err(rivet_core::Error::UnexpectedDType {
                expected: DType::U8,
                actual: sample.image.dtype(),
            });
        };
        layout
            .strided_index()
            .map(|index| {
                values
                    .get(index)
                    .copied()
                    .ok_or(rivet_core::Error::StorageOutOfBounds)
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    let image = RgbImage::from_raw(width, height, values).ok_or_else(|| {
        invalid_shape(format!(
            "{op_name} received invalid image buffer for shape {}x{}x{}",
            width, height, channels
        ))
    })?;

    Ok((image, label))
}

pub(crate) fn from_rgb_image(image: RgbImage, label: i64) -> RivetResult<DecodedSample> {
    let (width, height) = image.dimensions();
    let tensor = Tensor::from_vec(
        image.into_raw(),
        [height as usize, width as usize, 3],
        &Device::Cpu,
    )?;

    Ok(DecodedSample {
        image: tensor,
        label,
    })
}

#[cfg(test)]
mod tests {
    use super::into_rgb_image;
    use crate::sample::image::DecodedSample;
    use rivet_core::{Device, Tensor};

    #[test]
    fn backend_bridge_reads_non_contiguous_input_in_logical_order() {
        let base = Tensor::from_vec((0..12).collect::<Vec<u8>>(), [2, 3, 2], &Device::Cpu).unwrap();
        let image = base.permute(&[0, 2, 1]).unwrap();
        assert!(!image.is_contiguous());

        let (rgb, label) = into_rgb_image(DecodedSample { image, label: 9 }, "test").unwrap();

        assert_eq!(label, 9);
        assert_eq!(rgb.dimensions(), (2, 2));
        assert_eq!(rgb.into_raw(), [0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11]);
    }
}
