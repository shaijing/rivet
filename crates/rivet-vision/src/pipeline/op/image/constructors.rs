use super::ImageOp;
use crate::sample::image::ImageAxisOrder;
use crate::transforms::{
    ArbitraryRotateConfig, AutocontrastConfig, BrightnessConfig, CenterCropConfig,
    ColorJitterConfig, ContrastConfig, ConvertImageDtypeConfig, CropConfig, DecodeImageConfig,
    ElasticTransformConfig, EqualizeConfig, FiveCropConfig, FlipConfig, GaussianBlurConfig,
    GrayscaleConfig, HueConfig, InterpolationMode, InvertConfig, LayoutConfig, NormalizeConfig,
    PadConfig, PerspectiveConfig, PosterizeConfig, RandomAffineConfig, RandomCropConfig,
    RandomErasingConfig, RandomGrayscaleConfig, RandomHorizontalFlipConfig,
    RandomPerspectiveConfig, RandomResizedCropConfig, ResizeConfig, RotateConfig, RotationAngle,
    SharpnessConfig, SolarizeConfig, TenCropConfig,
};
use rivet_core::DType;

impl ImageOp {
    pub fn decode() -> Self {
        Self::Decode(DecodeImageConfig::new())
    }

    pub fn resize(width: u32, height: u32) -> Self {
        Self::Resize(ResizeConfig::new(width, height))
    }

    pub fn resize_short_edge(
        size: u32,
        max_size: Option<u32>,
        interpolation: InterpolationMode,
    ) -> Self {
        Self::Resize(ResizeConfig::short_edge(size, max_size, interpolation))
    }

    pub fn resize_with_interpolation(
        width: u32,
        height: u32,
        interpolation: InterpolationMode,
    ) -> Self {
        Self::Resize(ResizeConfig::with_interpolation(
            width,
            height,
            interpolation,
        ))
    }

    pub fn crop(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self::Crop(CropConfig::new(x, y, width, height))
    }

    pub fn five_crop(width: u32, height: u32) -> Self {
        Self::FiveCrop(FiveCropConfig::new(width, height))
    }

    pub fn ten_crop(width: u32, height: u32, vertical_flip: bool) -> Self {
        Self::TenCrop(TenCropConfig::new(width, height).with_vertical_flip(vertical_flip))
    }

    pub fn center_crop(width: u32, height: u32) -> Self {
        Self::CenterCrop(CenterCropConfig::new(width, height))
    }

    pub fn pad(padding: u32) -> Self {
        Self::Pad(PadConfig::new(padding))
    }

    pub fn pad_with_fill(padding: u32, value: f32) -> Self {
        Self::Pad(PadConfig::new(padding).with_fill(value))
    }

    pub fn pad_with_sides(left: u32, top: u32, right: u32, bottom: u32, value: f32) -> Self {
        Self::Pad(PadConfig::with_sides(left, top, right, bottom).with_fill(value))
    }

    pub fn random_crop(width: u32, height: u32, padding: u32) -> Self {
        Self::RandomCrop(RandomCropConfig::new(width, height, padding))
    }

    pub fn random_crop_with_config(config: RandomCropConfig) -> Self {
        Self::RandomCrop(config)
    }

    pub fn random_resized_crop(width: u32, height: u32) -> Self {
        Self::RandomResizedCrop(RandomResizedCropConfig::new(width, height))
    }

    pub fn random_resized_crop_with_config(config: RandomResizedCropConfig) -> Self {
        Self::RandomResizedCrop(config)
    }

    pub fn horizontal_flip() -> Self {
        Self::Flip(FlipConfig::horizontal())
    }

    pub fn vertical_flip() -> Self {
        Self::Flip(FlipConfig::vertical())
    }

    pub fn random_horizontal_flip(probability: f64) -> Self {
        Self::RandomHorizontalFlip(RandomHorizontalFlipConfig::new(probability))
    }

    pub fn brightness(value: f32) -> Self {
        Self::Brightness(BrightnessConfig::new(value))
    }

    pub fn contrast(value: f32) -> Self {
        Self::Contrast(ContrastConfig::new(value))
    }

    pub fn hue(factor: f32) -> Self {
        Self::Hue(HueConfig::new(factor))
    }

    pub fn color_jitter(brightness: f32, contrast: f32, hue: f32) -> Self {
        Self::ColorJitter(ColorJitterConfig::new(brightness, contrast, hue))
    }

    pub fn invert() -> Self {
        Self::Invert(InvertConfig::new())
    }

    pub fn posterize(bits: u8) -> Self {
        Self::Posterize(PosterizeConfig::new(bits))
    }

    pub fn solarize(threshold: u8) -> Self {
        Self::Solarize(SolarizeConfig::new(threshold))
    }

    pub fn autocontrast() -> Self {
        Self::Autocontrast(AutocontrastConfig::new())
    }

    pub fn equalize() -> Self {
        Self::Equalize(EqualizeConfig::new())
    }

    pub fn sharpness(amount: f32) -> Self {
        Self::Sharpness(SharpnessConfig::new(amount))
    }

    pub fn arbitrary_rotate(angle: f32) -> Self {
        Self::ArbitraryRotate(ArbitraryRotateConfig::new(angle))
    }

    pub fn arbitrary_rotate_with_options(
        angle: f32,
        expand: bool,
        interpolation: InterpolationMode,
        fill: u8,
    ) -> Self {
        Self::ArbitraryRotate(ArbitraryRotateConfig::new(angle).with_options(
            expand,
            interpolation,
            fill,
        ))
    }

    pub fn arbitrary_rotate_with_config(config: ArbitraryRotateConfig) -> Self {
        Self::ArbitraryRotate(config)
    }

    pub fn random_affine(degrees: f32) -> Self {
        Self::RandomAffine(RandomAffineConfig::new(degrees))
    }

    pub fn random_affine_with_config(config: RandomAffineConfig) -> Self {
        Self::RandomAffine(config)
    }

    pub fn perspective(
        start_points: [crate::transforms::Point2; 4],
        end_points: [crate::transforms::Point2; 4],
    ) -> Self {
        Self::Perspective(PerspectiveConfig::new(start_points, end_points))
    }

    pub fn perspective_with_config(config: PerspectiveConfig) -> Self {
        Self::Perspective(config)
    }

    pub fn random_perspective(distortion_scale: f32, probability: f64) -> Self {
        Self::RandomPerspective(RandomPerspectiveConfig::new(distortion_scale, probability))
    }

    pub fn random_perspective_with_config(config: RandomPerspectiveConfig) -> Self {
        Self::RandomPerspective(config)
    }

    pub fn elastic_transform(alpha: f32, sigma: f32) -> Self {
        Self::ElasticTransform(ElasticTransformConfig::new(alpha, sigma))
    }

    pub fn elastic_transform_with_config(config: ElasticTransformConfig) -> Self {
        Self::ElasticTransform(config)
    }

    pub fn random_apply(probability: f64, ops: Vec<Self>) -> Self {
        Self::RandomApply { probability, ops }
    }

    pub fn random_choice(choices: Vec<Vec<Self>>) -> Self {
        Self::RandomChoice {
            choices,
            weights: None,
        }
    }

    pub fn random_choice_with_weights(choices: Vec<Vec<Self>>, weights: Vec<f64>) -> Self {
        Self::RandomChoice {
            choices,
            weights: Some(weights),
        }
    }

    pub fn random_order(ops: Vec<Self>) -> Self {
        Self::RandomOrder { ops }
    }

    pub fn gaussian_blur(sigma: f32) -> Self {
        Self::GaussianBlur(GaussianBlurConfig::new(sigma))
    }

    pub fn grayscale(num_output_channels: u8) -> Self {
        Self::Grayscale(GrayscaleConfig::new(num_output_channels))
    }

    pub fn random_grayscale(probability: f64, num_output_channels: u8) -> Self {
        Self::RandomGrayscale(
            RandomGrayscaleConfig::new(probability).with_output_channels(num_output_channels),
        )
    }

    pub fn random_erasing(probability: f64) -> Self {
        Self::RandomErasing(RandomErasingConfig::new(probability))
    }

    pub fn random_erasing_with_config(config: RandomErasingConfig) -> Self {
        Self::RandomErasing(config)
    }

    pub fn convert_image_dtype(dtype: DType) -> Self {
        Self::ConvertImageDtype(ConvertImageDtypeConfig::new(dtype))
    }

    pub fn rotate(angle: RotationAngle) -> Self {
        Self::Rotate(RotateConfig::new(angle))
    }

    pub fn normalize(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self::Normalize(NormalizeConfig::new(mean, std))
    }

    pub fn hwc_to_chw() -> Self {
        Self::Layout(LayoutConfig::new(ImageAxisOrder::Chw))
    }

    pub fn chw_to_hwc() -> Self {
        Self::Layout(LayoutConfig::new(ImageAxisOrder::Hwc))
    }
}
