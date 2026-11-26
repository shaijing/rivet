//! Noise, weather, and visual artifacts.

pub(in crate::transforms) mod artifacts;
pub(in crate::transforms) mod flare;
pub(in crate::transforms) mod fog;
pub(crate) mod frequency;
pub use flare::{LensFlareConfig, RandomSunFlareConfig};
pub(crate) mod noise;
pub(in crate::transforms) mod weather;

pub use artifacts::{AnnotationArtifactsConfig, ArtifactElement};
pub use fog::{AtmosphericFogConfig, FogDepthMode, RandomFogConfig};
pub use noise::{
    AdditiveDistribution, AdditiveNoiseConfig, NoiseSharing, gaussian_noise, shot_noise,
};
pub use weather::{
    RainType, RandomGravelConfig, RandomRainConfig, RandomSnowConfig, SnowMethod, SpatterConfig,
    SpatterMode,
};
