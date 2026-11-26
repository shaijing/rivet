//! Decoded volumes use DHW or DHWC. Masks are DHW class/instance IDs,
//! and keypoints are stored as [N, 3] in X, Y, Z pixel-center coordinates.

mod bias;
mod callback;
mod crop_pad;
mod dropout;
mod elastic;
mod intensity;
mod motion;
mod reorient;
mod resample;
mod sample;
mod spectral;
mod warp;

pub use bias::BiasFieldConfig;
pub use crop_pad::{Crop3DConfig, Pad3DConfig, PadIfNeeded3DConfig, RandomResizedCrop3DConfig};
pub use dropout::CoarseDropout3DConfig;
pub use elastic::ElasticTransform3DConfig;
pub use motion::MotionArtifactConfig;
pub use reorient::volume_axis_mask;
pub use resample::{Anisotropy3DConfig, Resize3DConfig, VolumeInterpolation};
pub use sample::VolumeSample;
pub use spectral::{GhostingArtifactConfig, GibbsRingingConfig, KSpaceSpikeNoiseConfig};
pub use warp::{Affine3DConfig, VolumeBorder};
