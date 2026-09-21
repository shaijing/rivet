pub mod color;
pub mod filter;
pub mod geometry;
pub mod representation;

mod advanced_geometry;
mod blur;
mod crop;
mod decode;
mod dtype;
mod erase;
mod flip;
mod layout;
mod multi_crop;
mod normalize;
mod pad;
mod resize;
mod resized_crop;
mod rotate;
mod u8_color;

/// Padding semantics shared by geometry transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaddingMode {
    Constant,
    Edge,
    Reflect,
    Symmetric,
}

pub use advanced_geometry::{
    ArbitraryRotateConfig, ElasticTransformConfig, PerspectiveConfig, Point2, RandomAffineConfig,
    RandomPerspectiveConfig,
};
pub use blur::GaussianBlurConfig;
pub use color::{
    AutocontrastConfig, BrightnessConfig, ColorJitterConfig, ColorJitterParams, ContrastConfig,
    EqualizeConfig, GrayscaleConfig, HueConfig, InvertConfig, PosterizeConfig,
    RandomGrayscaleConfig, SharpnessConfig, SolarizeConfig,
};
pub use erase::{EraseRegion, RandomErasingConfig};
pub use geometry::{
    CenterCropConfig, CropConfig, CropRegion, FiveCropConfig, FlipConfig, FlipDirection,
    InterpolationMode, LayoutConfig, PadConfig, RandomCropConfig, RandomHorizontalFlipConfig,
    RandomResizedCropConfig, ResizeConfig, RotateConfig, RotationAngle, TenCropConfig,
};
pub use representation::{ConvertImageDtypeConfig, DecodeImageConfig, NormalizeConfig};

use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::DecodedSample;
use image::RgbImage;
use rivet_core::{CpuStorageRef, DType, Device, Layout, Tensor};

pub(crate) fn logical_offset(layout: &Layout, coords: &[usize]) -> rivet_core::Result<usize> {
    if coords.len() != layout.dims().len() {
        return Err(rivet_core::Error::InvalidRank {
            expected: layout.dims().len(),
            actual: coords.len(),
        });
    }
    coords.iter().zip(layout.stride()).try_fold(
        layout.start_offset(),
        |offset, (&coord, &stride)| {
            let delta = coord
                .checked_mul(stride)
                .ok_or(rivet_core::Error::StorageOutOfBounds)?;
            offset
                .checked_add(delta)
                .ok_or(rivet_core::Error::StorageOutOfBounds)
        },
    )
}

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
    // deliberate materialization boundary. Decoded caches expose each HWC
    // sample as a contiguous view with a storage offset; copying its range
    // directly keeps that hot path a bulk memcpy. Non-contiguous views still
    // need logical traversal to preserve their element order.
    let values = sample.image.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::U8(values) = storage else {
            return Err(rivet_core::Error::UnexpectedDType {
                expected: DType::U8,
                actual: sample.image.dtype(),
            });
        };
        let element_count = layout.checked_elem_count()?;
        if element_count != 0 {
            let Some((_, max_offset)) = layout.storage_bounds() else {
                return Err(rivet_core::Error::StorageOutOfBounds);
            };
            if max_offset >= values.len() {
                return Err(rivet_core::Error::StorageOutOfBounds);
            }
        }
        if sample.image.is_contiguous() {
            let end = layout
                .start_offset()
                .checked_add(element_count)
                .ok_or(rivet_core::Error::StorageOutOfBounds)?;
            return values
                .get(layout.start_offset()..end)
                .map(ToOwned::to_owned)
                .ok_or(rivet_core::Error::StorageOutOfBounds);
        }
        layout
            .strided_index()
            .map(|index| {
                // SAFETY: the non-empty layout's maximum offset was checked
                // against `values.len()` above; strided_index only yields
                // offsets from that same layout.
                Ok(unsafe { *values.get_unchecked(index) })
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

    #[test]
    fn backend_bridge_reads_contiguous_view_at_storage_offset() {
        let base =
            Tensor::from_vec((0..24).collect::<Vec<u8>>(), [2, 2, 2, 3], &Device::Cpu).unwrap();
        let image = base.get(1).unwrap();
        assert!(image.is_contiguous());

        let (rgb, label) = into_rgb_image(DecodedSample { image, label: 4 }, "test").unwrap();

        assert_eq!(label, 4);
        assert_eq!(rgb.dimensions(), (2, 2));
        assert_eq!(rgb.into_raw(), (12..24).collect::<Vec<u8>>());
    }
}
