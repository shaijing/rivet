//! Augmentation policies that compose and sample transforms.

pub mod autoaugment;

pub use autoaugment::{
    AugMixConfig, AugMixPlan, AugmentOp, AugmentStep, AutoAugmentConfig, AutoAugmentPolicy,
    RandAugmentConfig, TrivialAugmentWideConfig,
};
