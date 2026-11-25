pub mod color;
pub mod filter;
pub mod geometry;
pub mod representation;

mod linear;
pub use linear::LinearTransformationConfig;
pub mod autoaugment;
pub use autoaugment::{
    AugMixConfig, AugMixPlan, AugmentOp, AugmentStep, AutoAugmentConfig, AutoAugmentPolicy,
    RandAugmentConfig, TrivialAugmentWideConfig,
};
mod random_extra;
pub use random_extra::{RandomGaussianBlurConfig, RandomRotationConfig};
mod advanced_geometry;
mod blur;
mod border_crop;
mod crop;
mod crop_pad;
pub(crate) use border_crop::sample_border_crop;
pub use crop_pad::{CropAndPadConfig, CropPadAmounts};
mod decode;
mod distribution;
mod dither;
mod dtype;
pub use distribution::{DistributionMode, PixelDistributionAdaptationConfig};
pub use dither::{DiffusionAlgorithm, DitherMethod, DitheringConfig};
mod elastic_map;
mod erase;
mod fda;
pub use fda::FdaConfig;
mod flare;
mod flip;
mod fog;
mod frequency;
mod grid_mask;
mod grid_shuffle;
mod histogram_match;
pub use grid_mask::GridMaskConfig;
pub use grid_shuffle::RandomGridShuffleConfig;
pub use histogram_match::{HistogramMatchingConfig, ReferenceImages};
mod layout;
mod letterbox;
mod local_color;
mod longest_size;
pub(crate) use longest_size::sample_scale;
pub use longest_size::{LongestMaxSizeConfig, LongestSizeLimit};
mod morphology;
mod superpixels;
mod thin_plate;
pub use superpixels::SuperpixelsConfig;
pub use thin_plate::ThinPlateSplineConfig;
mod multi_crop;
pub use morphology::{MorphologicalConfig, MorphologicalOperation};
mod noise;
mod normalize;
mod pad;
mod pad_needed;
pub use pad_needed::{PadIfNeededConfig, PadPosition};
mod pca;
mod piecewise;
mod pixel_mask;
mod pixel_spread;
pub use piecewise::{GridElasticDeformConfig, PiecewiseAffineConfig};
pub use pixel_mask::{
    ChannelDropoutConfig, ChannelDropoutFill, PixelDropoutConfig, StripLength, XyMaskingConfig,
};
pub(crate) use pixel_mask::{
    apply as replace_spatial_pixels, sample_coarse_dropout, sample_grid_dropout,
};
pub use pixel_spread::PixelSpreadConfig;
pub(crate) mod remap;
mod resize;
mod resized_crop;
mod rotate;
pub use remap::{GridDistortionConfig, OpticalDistortionConfig, OpticalDistortionMode};
pub use rotate::SquareSymmetry;
pub(crate) use rotate::quarter_turn;
mod sized_crop;
pub use sized_crop::RandomSizedCropConfig;
mod u8_color;
mod water;
pub use water::WaterRefractionConfig;
mod weather;
pub use fog::{AtmosphericFogConfig, FogDepthMode, RandomFogConfig};
pub use weather::{
    RainType, RandomGravelConfig, RandomRainConfig, RandomSnowConfig, SnowMethod, SpatterConfig,
    SpatterMode,
};

/// Padding semantics shared by geometry transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaddingMode {
    Constant,
    Edge,
    Reflect,
    Symmetric,
}

pub use advanced_geometry::{
    AffineConfig, ArbitraryRotateConfig, ElasticTransformConfig, PerspectiveConfig, Point2,
    RandomAffineConfig, RandomPerspectiveConfig, SafeRotateConfig,
};
pub use blur::{EnhanceMode, GaussianBlurConfig};
pub use color::{
    AutocontrastConfig, BrightnessConfig, ColorJitterConfig, ColorJitterParams, ColorizeConfig,
    ContrastConfig, EqualizeConfig, GammaConfig, GrayscaleConfig, HueConfig, InvertConfig,
    PosterizeConfig, RandomGrayscaleConfig, SaturationConfig, SharpnessConfig, SolarizeConfig,
};
pub use erase::{EraseConfig, EraseRegion, ErasingValue, RandomErasingConfig};
pub use geometry::{
    CenterCropConfig, CropConfig, CropRegion, FiveCropConfig, FlipConfig, FlipDirection,
    InterpolationMode, LayoutConfig, PadConfig, RandomCropConfig, RandomHorizontalFlipConfig,
    RandomResizedCropConfig, ResizeConfig, RotateConfig, RotationAngle, TenCropConfig,
};
pub use noise::{
    AdditiveDistribution, AdditiveNoiseConfig, NoiseSharing, gaussian_noise, shot_noise,
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

/// Borrow the active contiguous range, including a sample's storage offset.
/// Callers retain their strided fallback rather than treating a view as dense.
pub(crate) fn contiguous_values<'a, T>(
    values: &'a [T],
    layout: &Layout,
) -> rivet_core::Result<&'a [T]> {
    let end = layout
        .start_offset()
        .checked_add(layout.checked_elem_count()?)
        .ok_or(rivet_core::Error::StorageOutOfBounds)?;
    values
        .get(layout.start_offset()..end)
        .ok_or(rivet_core::Error::StorageOutOfBounds)
}

/// Native RGB kernels can borrow the input; only strided inputs need packing.
/// The callback is per image and statically dispatched, never per pixel.
pub(crate) fn with_rgb_pixels<T>(
    input: &Tensor,
    op_name: &str,
    f: impl FnOnce(&[u8], usize, usize) -> RivetResult<T>,
) -> RivetResult<T> {
    let dims = input.dims();
    if input.dtype() != DType::U8 || dims.len() != 3 || dims[2] != 3 {
        return Err(invalid_argument(format!(
            "{op_name} requires uint8 HWC RGB input, got dtype {:?} shape {:?}",
            input.dtype(),
            dims
        )));
    }
    let (height, width) = (dims[0], dims[1]);
    u32::try_from(width)
        .map_err(|_| invalid_shape(format!("{op_name} image width is too large: {width}")))?;
    u32::try_from(height)
        .map_err(|_| invalid_shape(format!("{op_name} image height is too large: {height}")))?;
    let packed;
    let input = if input.is_contiguous() {
        input
    } else {
        packed = input.contiguous()?;
        &packed
    };
    input.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::U8(values) = storage else {
            return Err(rivet_core::Error::UnexpectedDType {
                expected: DType::U8,
                actual: input.dtype(),
            });
        };
        Ok(f(contiguous_values(values, layout)?, width, height))
    })?
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

mod v2;
pub use v2::{
    AdvancedBlurConfig, IlluminationEffect, IlluminationMode, LensFlareConfig, RandomShadowConfig,
    RandomSunFlareConfig, V2Config,
};

mod v2_jpeg;
mod v2_numeric;

pub(crate) use advanced_geometry::{homography, pillow_affine_map, rotation_inverse};

pub use letterbox::LetterBoxConfig;
