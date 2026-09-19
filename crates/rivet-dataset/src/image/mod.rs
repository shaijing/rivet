pub mod color;
pub mod crop;
pub mod decode;
pub mod flip;
pub mod layout;
pub mod normalize;
pub mod resize;

use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::DecodedSample;
use image::RgbImage;
use rivet_core::{DType, Device, Tensor};

pub fn require_u8_hwc(sample: DecodedSample, op_name: &str) -> RivetResult<DecodedSample> {
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

pub fn into_rgb_image(sample: DecodedSample, op_name: &str) -> RivetResult<(RgbImage, i64)> {
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
    let values = sample.image.to_vec::<u8>()?;
    let image = RgbImage::from_raw(width, height, values).ok_or_else(|| {
        invalid_shape(format!(
            "{op_name} received invalid image buffer for shape {}x{}x{}",
            width, height, channels
        ))
    })?;

    Ok((image, label))
}

pub fn from_rgb_image(image: RgbImage, label: i64) -> RivetResult<DecodedSample> {
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
