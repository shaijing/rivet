use super::CallbackConfig;
use crate::sample::image::ImageAxisOrder;
use crate::transforms::{
    AffineConfig, ArbitraryRotateConfig, AugMixConfig, AutoAugmentConfig, AutocontrastConfig,
    BrightnessConfig, CenterCropConfig, ColorJitterConfig, ContrastConfig, ConvertImageDtypeConfig,
    CropConfig, DecodeImageConfig, ElasticTransformConfig, EqualizeConfig, EraseConfig,
    FiveCropConfig, FlipConfig, GammaConfig, GaussianBlurConfig, GrayscaleConfig, HueConfig,
    InvertConfig, LayoutConfig, LinearTransformationConfig, NormalizeConfig, PadConfig,
    PerspectiveConfig, PosterizeConfig, RandAugmentConfig, RandomAffineConfig, RandomCropConfig,
    RandomErasingConfig, RandomGaussianBlurConfig, RandomGrayscaleConfig,
    RandomHorizontalFlipConfig, RandomPerspectiveConfig, RandomResizedCropConfig,
    RandomRotationConfig, ResizeConfig, RotateConfig, SaturationConfig, SharpnessConfig,
    SolarizeConfig, TenCropConfig, TrivialAugmentWideConfig,
};
use rivet_core::DType;

mod constructors;
mod execute;
mod metadata;
mod validate;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionKind {
    Sample,
    Batch,
}

#[derive(Clone)]
pub enum ImageOp {
    Callback(CallbackConfig),
    Decode(DecodeImageConfig),
    V2(crate::transforms::V2Config),
    LinearTransformation(LinearTransformationConfig),
    AutoAugment(AutoAugmentConfig),
    RandAugment(RandAugmentConfig),
    TrivialAugmentWide(TrivialAugmentWideConfig),
    AugMix(AugMixConfig),

    Saturation(SaturationConfig),
    Gamma(GammaConfig),
    Affine(AffineConfig),
    Erase(EraseConfig),
    RandomRotation(RandomRotationConfig),
    RandomGaussianBlur(RandomGaussianBlurConfig),

    Resize(ResizeConfig),
    Crop(CropConfig),
    CenterCrop(CenterCropConfig),
    Pad(PadConfig),
    Flip(FlipConfig),
    RandomCrop(RandomCropConfig),
    RandomResizedCrop(RandomResizedCropConfig),
    RandomHorizontalFlip(RandomHorizontalFlipConfig),
    Brightness(BrightnessConfig),
    Contrast(ContrastConfig),
    Hue(HueConfig),
    ColorJitter(ColorJitterConfig),
    Invert(InvertConfig),
    Posterize(PosterizeConfig),
    Solarize(SolarizeConfig),
    Autocontrast(AutocontrastConfig),
    Equalize(EqualizeConfig),
    Sharpness(SharpnessConfig),
    ArbitraryRotate(ArbitraryRotateConfig),
    RandomAffine(RandomAffineConfig),
    Perspective(PerspectiveConfig),
    RandomPerspective(RandomPerspectiveConfig),
    ElasticTransform(ElasticTransformConfig),
    RandomApply {
        probability: f64,
        ops: Vec<ImageOp>,
    },
    RandomChoice {
        choices: Vec<Vec<ImageOp>>,
        weights: Option<Vec<f64>>,
    },
    RandomOrder {
        ops: Vec<ImageOp>,
    },
    GaussianBlur(GaussianBlurConfig),
    Grayscale(GrayscaleConfig),
    RandomGrayscale(RandomGrayscaleConfig),
    RandomErasing(RandomErasingConfig),
    ConvertImageDtype(ConvertImageDtypeConfig),
    Rotate(RotateConfig),
    Normalize(NormalizeConfig),
    Layout(LayoutConfig),
    FiveCrop(FiveCropConfig),
    TenCrop(TenCropConfig),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineImageState {
    Encoded,
    Decoded {
        dtype: DType,
        axis_order: ImageAxisOrder,
    },
}
