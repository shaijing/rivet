use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::ImageAxisOrder;
use crate::transforms::contiguous_values;
use rivet_core::{CpuStorageRef, DType, Tensor};

pub(crate) fn webp(input: &Tensor, axis: ImageAxisOrder, quality: u8) -> RivetResult<Tensor> {
    if input.dtype() != DType::U8 || input.rank() != 3 || !(1..=100).contains(&quality) {
        return Err(invalid_argument(
            "WebP requires rank-3 uint8 pixels and quality 1..100",
        ));
    }
    let hwc = match axis {
        ImageAxisOrder::Hwc => input.clone(),
        ImageAxisOrder::Chw => input.permute(&[1, 2, 0])?,
    };
    let [height, width, channels] = *hwc.dims() else {
        unreachable!()
    };
    if !matches!(channels, 1 | 3) || !(1..=16383).contains(&height) || !(1..=16383).contains(&width)
    {
        return Err(invalid_argument(
            "WebP requires grayscale or RGB images with axes in 1..16383",
        ));
    }
    let hwc = hwc.contiguous()?;
    let encoded = hwc
        .with_cpu_storage(|storage, layout| {
            let CpuStorageRef::U8(values) = storage else {
                unreachable!()
            };
            let pixels = contiguous_values(values, layout)?;
            let rgb;
            let pixels = if channels == 1 {
                rgb = pixels.iter().flat_map(|&v| [v; 3]).collect::<Vec<_>>();
                rgb.as_slice()
            } else {
                pixels
            };
            Ok(webp::Encoder::from_rgb(pixels, width as u32, height as u32)
                .encode_simple(false, f32::from(quality)))
        })?
        .map_err(|error| invalid_argument(format!("WebP encoder failed: {error:?}")))?;
    let decoded = image::load_from_memory_with_format(&encoded, image::ImageFormat::WebP)
        .map_err(|error| invalid_argument(format!("WebP decoder failed: {error}")))?;
    if decoded.width() as usize != width || decoded.height() as usize != height {
        return Err(invalid_argument("WebP round-trip changed the dimensions"));
    }
    let shape = [height, width, channels];
    let result = if channels == 1 {
        Tensor::from_exact_iter(
            decoded.to_luma8().into_raw().into_iter(),
            shape,
            input.device(),
        )?
    } else {
        Tensor::from_exact_iter(
            decoded.into_rgb8().into_raw().into_iter(),
            shape,
            input.device(),
        )?
    };
    Ok(match axis {
        ImageAxisOrder::Hwc => result,
        ImageAxisOrder::Chw => result.permute(&[2, 0, 1])?,
    })
}
