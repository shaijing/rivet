//! Image geometry, coordinate remapping, interpolation, and padding.

pub(in crate::transforms) mod advanced_geometry;
pub(in crate::transforms) mod border_crop;
pub(in crate::transforms) mod crop;
pub(in crate::transforms) mod crop_pad;
pub(in crate::transforms) mod elastic_map;
pub(in crate::transforms) mod flip;
pub(in crate::transforms) mod grid_shuffle;
pub(in crate::transforms) mod letterbox;
pub(in crate::transforms) mod longest_size;
pub(in crate::transforms) mod multi_crop;
pub(in crate::transforms) mod pad;
pub(in crate::transforms) mod pad_needed;
pub(in crate::transforms) mod piecewise;
pub(in crate::transforms) mod random_rotation;
pub(crate) mod remap;
pub(in crate::transforms) mod resize;
pub(in crate::transforms) mod resized_crop;
pub(in crate::transforms) mod rotate;
pub(in crate::transforms) mod sized_crop;
pub(in crate::transforms) mod thin_plate;
pub(in crate::transforms) mod water;

pub use advanced_geometry::{
    AffineConfig, ArbitraryRotateConfig, ElasticTransformConfig, PerspectiveConfig, Point2,
    RandomAffineConfig, RandomPerspectiveConfig, SafeRotateConfig,
};
pub(crate) use advanced_geometry::{homography, pillow_affine_map, rotation_inverse};
pub(crate) use border_crop::sample_border_crop;
pub use crop::{CenterCropConfig, CropConfig, RandomCropConfig};
pub use crop_pad::{CropAndPadConfig, CropPadAmounts};
pub use flip::{FlipConfig, FlipDirection, RandomHorizontalFlipConfig};
pub use grid_shuffle::RandomGridShuffleConfig;
pub use letterbox::LetterBoxConfig;
pub(crate) use longest_size::sample_scale;
pub use longest_size::{LongestMaxSizeConfig, LongestSizeLimit};
pub use multi_crop::{FiveCropConfig, TenCropConfig};
pub use pad::PadConfig;
pub use pad_needed::{PadIfNeededConfig, PadPosition};
pub use piecewise::{GridElasticDeformConfig, PiecewiseAffineConfig};
pub use random_rotation::RandomRotationConfig;
pub use remap::{GridDistortionConfig, OpticalDistortionConfig, OpticalDistortionMode};
pub use resize::{InterpolationMode, ResizeConfig};
pub use resized_crop::{CropRegion, RandomResizedCropConfig};
pub use rotate::SquareSymmetry;
pub(crate) use rotate::quarter_turn;
pub use rotate::{RotateConfig, RotationAngle};
pub use sized_crop::RandomSizedCropConfig;
pub use thin_plate::ThinPlateSplineConfig;
pub use water::WaterRefractionConfig;

/// Padding semantics shared by geometry transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaddingMode {
    Constant,
    Edge,
    Reflect,
    Symmetric,
}
