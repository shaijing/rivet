//! Region erasure and pixel masking.

pub(in crate::transforms) mod erase;
pub(in crate::transforms) mod grid_mask;
pub(in crate::transforms) mod pixel_mask;

pub use erase::{EraseConfig, EraseRegion, ErasingValue, RandomErasingConfig};
pub use grid_mask::GridMaskConfig;
pub use pixel_mask::{
    ChannelDropoutConfig, ChannelDropoutFill, PixelDropoutConfig, StripLength, XyMaskingConfig,
};
pub(crate) use pixel_mask::{
    apply as replace_spatial_pixels, sample_coarse_dropout, sample_grid_dropout,
};
