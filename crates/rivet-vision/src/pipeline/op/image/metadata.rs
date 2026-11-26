use super::{ExecutionKind, ImageOp};

impl ImageOp {
    /// Stable semantic name used to assign per-operator random namespaces.
    pub fn random_key_kind(&self) -> Option<&'static str> {
        match self {
            Self::Callback(_) => Some("Lambda"),
            Self::V2(op) => op.is_random().then(|| op.name()),
            Self::RandomRotation(_) => Some("RandomRotation"),
            Self::RandomGaussianBlur(_) => Some("RandomGaussianBlur"),
            Self::AutoAugment(_) => Some("AutoAugment"),
            Self::RandAugment(_) => Some("RandAugment"),
            Self::TrivialAugmentWide(_) => Some("TrivialAugmentWide"),
            Self::AugMix(_) => Some("AugMix"),
            Self::RandomCrop(_) => Some("RandomCrop"),
            Self::RandomResizedCrop(_) => Some("RandomResizedCrop"),
            Self::RandomHorizontalFlip(_) => Some("RandomHorizontalFlip"),
            Self::ColorJitter(_) => Some("ColorJitter"),
            Self::RandomAffine(_) => Some("RandomAffine"),
            Self::RandomPerspective(_) => Some("RandomPerspective"),
            Self::ElasticTransform(_) => Some("ElasticTransform"),
            Self::RandomApply { .. } => Some("RandomApply"),
            Self::RandomChoice { .. } => Some("RandomChoice"),
            Self::RandomOrder { .. } => Some("RandomOrder"),
            Self::RandomGrayscale(_) => Some("RandomGrayscale"),
            Self::RandomErasing(_) => Some("RandomErasing"),
            _ => None,
        }
    }

    pub fn execution_kind(&self) -> ExecutionKind {
        match self {
            Self::V2(crate::transforms::V2Config::ToDtype { .. }) => ExecutionKind::Batch,
            Self::V2(_) | Self::Callback(_) => ExecutionKind::Sample,
            Self::Normalize(_) | Self::ConvertImageDtype(_) | Self::Layout(_) => {
                ExecutionKind::Batch
            }
            Self::Decode(_)
            | Self::LinearTransformation(_)
            | Self::AutoAugment(_)
            | Self::RandAugment(_)
            | Self::TrivialAugmentWide(_)
            | Self::AugMix(_)
            | Self::Saturation(_)
            | Self::Gamma(_)
            | Self::Affine(_)
            | Self::Erase(_)
            | Self::RandomRotation(_)
            | Self::RandomGaussianBlur(_)
            | Self::Resize(_)
            | Self::Crop(_)
            | Self::CenterCrop(_)
            | Self::FiveCrop(_)
            | Self::TenCrop(_)
            | Self::Pad(_)
            | Self::Flip(_)
            | Self::RandomCrop(_)
            | Self::RandomResizedCrop(_)
            | Self::RandomHorizontalFlip(_)
            | Self::Brightness(_)
            | Self::Contrast(_)
            | Self::Hue(_)
            | Self::ColorJitter(_)
            | Self::Invert(_)
            | Self::Posterize(_)
            | Self::Solarize(_)
            | Self::Autocontrast(_)
            | Self::Equalize(_)
            | Self::Sharpness(_)
            | Self::ArbitraryRotate(_)
            | Self::RandomAffine(_)
            | Self::Perspective(_)
            | Self::RandomPerspective(_)
            | Self::ElasticTransform(_)
            | Self::RandomApply { .. }
            | Self::RandomChoice { .. }
            | Self::RandomOrder { .. }
            | Self::GaussianBlur(_)
            | Self::Grayscale(_)
            | Self::RandomGrayscale(_)
            | Self::RandomErasing(_)
            | Self::Rotate(_) => ExecutionKind::Sample,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Callback(_) => "Lambda",
            Self::V2(op) => op.name(),
            Self::Saturation(_) => "Saturation",
            Self::Gamma(_) => "Gamma",
            Self::Affine(_) => "Affine",
            Self::Erase(_) => "Erase",
            Self::RandomRotation(_) => "RandomRotation",
            Self::RandomGaussianBlur(_) => "RandomGaussianBlur",
            Self::AutoAugment(_) => "AutoAugment",
            Self::RandAugment(_) => "RandAugment",
            Self::TrivialAugmentWide(_) => "TrivialAugmentWide",
            Self::AugMix(_) => "AugMix",
            Self::LinearTransformation(_) => "LinearTransformation",
            Self::Decode(_) => "Decode",
            Self::Resize(_) => "Resize",
            Self::Crop(_) => "Crop",
            Self::CenterCrop(_) => "CenterCrop",
            Self::FiveCrop(_) => "FiveCrop",
            Self::TenCrop(_) => "TenCrop",
            Self::Pad(_) => "Pad",
            Self::Flip(_) => "Flip",
            Self::RandomCrop(_) => "RandomCrop",
            Self::RandomResizedCrop(_) => "RandomResizedCrop",
            Self::RandomHorizontalFlip(_) => "RandomHorizontalFlip",
            Self::Brightness(_) => "Brightness",
            Self::Contrast(_) => "Contrast",
            Self::Hue(_) => "Hue",
            Self::ColorJitter(_) => "ColorJitter",
            Self::Invert(_) => "Invert",
            Self::Posterize(_) => "Posterize",
            Self::Solarize(_) => "Solarize",
            Self::Autocontrast(_) => "Autocontrast",
            Self::Equalize(_) => "Equalize",
            Self::Sharpness(_) => "Sharpness",
            Self::ArbitraryRotate(_) => "ArbitraryRotate",
            Self::RandomAffine(_) => "RandomAffine",
            Self::Perspective(_) => "Perspective",
            Self::RandomPerspective(_) => "RandomPerspective",
            Self::ElasticTransform(_) => "ElasticTransform",
            Self::RandomApply { .. } => "RandomApply",
            Self::RandomChoice { .. } => "RandomChoice",
            Self::RandomOrder { .. } => "RandomOrder",
            Self::GaussianBlur(_) => "GaussianBlur",
            Self::Grayscale(_) => "Grayscale",
            Self::RandomGrayscale(_) => "RandomGrayscale",
            Self::RandomErasing(_) => "RandomErasing",
            Self::ConvertImageDtype(_) => "ConvertImageDtype",
            Self::Rotate(_) => "Rotate",
            Self::Normalize(_) => "Normalize",
            Self::Layout(_) => "Layout",
        }
    }

    /// Stable semantic operation identity used by the domain planning
    /// protocol. Keep these values stable when variants are reordered.
    pub const fn planning_op_id(&self) -> u32 {
        match self {
            Self::Callback(_) => 143,
            Self::V2(op) => op.planning_op_id(),
            Self::Saturation(_) => 36,
            Self::Gamma(_) => 37,
            Self::Affine(_) => 38,
            Self::Erase(_) => 39,
            Self::RandomRotation(_) => 40,
            Self::RandomGaussianBlur(_) => 41,
            Self::AutoAugment(_) => 42,
            Self::RandAugment(_) => 43,
            Self::TrivialAugmentWide(_) => 44,
            Self::AugMix(_) => 45,
            Self::LinearTransformation(_) => 46,
            Self::FiveCrop(_) => 47,
            Self::TenCrop(_) => 48,
            Self::Decode(_) => 1,
            Self::Resize(_) => 2,
            Self::Crop(_) => 3,
            Self::CenterCrop(_) => 4,
            Self::Pad(_) => 5,
            Self::Flip(_) => 6,
            Self::RandomCrop(_) => 7,
            Self::RandomResizedCrop(_) => 8,
            Self::RandomHorizontalFlip(_) => 9,
            Self::Brightness(_) => 10,
            Self::Contrast(_) => 11,
            Self::Hue(_) => 12,
            Self::ColorJitter(_) => 13,
            Self::Invert(_) => 14,
            Self::Posterize(_) => 15,
            Self::Solarize(_) => 16,
            Self::Autocontrast(_) => 17,
            Self::Equalize(_) => 18,
            Self::Sharpness(_) => 19,
            Self::ArbitraryRotate(_) => 20,
            Self::RandomAffine(_) => 21,
            Self::Perspective(_) => 22,
            Self::RandomPerspective(_) => 23,
            Self::ElasticTransform(_) => 24,
            Self::RandomApply { .. } => 25,
            Self::RandomChoice { .. } => 26,
            Self::RandomOrder { .. } => 27,
            Self::GaussianBlur(_) => 28,
            Self::Grayscale(_) => 29,
            Self::RandomGrayscale(_) => 30,
            Self::RandomErasing(_) => 31,
            Self::ConvertImageDtype(_) => 32,
            Self::Rotate(_) => 33,
            Self::Normalize(_) => 34,
            Self::Layout(_) => 35,
        }
    }

    /// Number of output views per input image for this operator.
    pub fn output_multiplier(&self) -> usize {
        match self {
            Self::FiveCrop(_) => 5,
            Self::TenCrop(_) => 10,
            _ => 1,
        }
    }
}
