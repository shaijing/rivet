//! Color, intensity, and color distribution transforms.

pub mod color;
pub use local_color::{IlluminationEffect, IlluminationMode, RandomShadowConfig};
pub(in crate::transforms) mod distribution;
pub(in crate::transforms) mod fda;
pub(in crate::transforms) mod histogram_match;
pub(in crate::transforms) mod local_color;
pub(in crate::transforms) mod pca;
pub(in crate::transforms) mod stain;
pub(in crate::transforms) mod u8_color;

pub use color::{
    AutocontrastConfig, BrightnessConfig, ColorJitterConfig, ColorJitterParams, ColorizeConfig,
    ContrastConfig, EqualizeConfig, GammaConfig, GrayscaleConfig, HueConfig, InvertConfig,
    PosterizeConfig, RandomGrayscaleConfig, SaturationConfig, SharpnessConfig, SolarizeConfig,
};
pub use distribution::{DistributionMode, PixelDistributionAdaptationConfig};
pub use fda::FdaConfig;
pub use histogram_match::{HistogramMatchingConfig, ReferenceImages};
pub use stain::{HEStainConfig, StainBasis, StainResidual};
