use super::op::ImageOp;
use crate::sample::image::ImageAxisOrder;
use crate::transforms::{
    ElasticTransformConfig, InterpolationMode, PerspectiveConfig, Point2, RandomAffineConfig,
    RandomErasingConfig, RandomPerspectiveConfig, RandomResizedCropConfig, RotationAngle,
};
use rivet_core::DType;

/// A reusable, ordered sequence of image operations.
///
/// `ImagePipeline` remains the compiled execution owner. This type only
/// records operations so a sequence can be inserted into a pipeline or used
/// as a child of a random control operation; it does not create another
/// runtime or loader.
#[derive(Clone, Default)]
pub struct TransformSequence {
    pub(crate) ops: Vec<ImageOp>,
}

/// Compatibility name for callers that prefer the torchvision terminology.
pub type Compose = TransformSequence;

/// Descriptive alias for the public image transform sequence.
pub type ImageTransform = TransformSequence;

impl TransformSequence {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_ops(ops: Vec<ImageOp>) -> Self {
        Self { ops }
    }

    pub fn compose(mut self, other: Self) -> Self {
        self.ops.extend(other.ops);
        self
    }

    pub(crate) fn into_ops(self) -> Vec<ImageOp> {
        self.ops
    }

    pub fn decode_image(mut self) -> Self {
        self.ops.push(ImageOp::decode());
        self
    }

    pub fn resize(mut self, width: u32, height: u32) -> Self {
        self.ops.push(ImageOp::resize(width, height));
        self
    }

    pub fn resize_with_interpolation(
        mut self,
        width: u32,
        height: u32,
        interpolation: InterpolationMode,
    ) -> Self {
        self.ops.push(ImageOp::resize_with_interpolation(
            width,
            height,
            interpolation,
        ));
        self
    }

    pub fn crop(mut self, x: u32, y: u32, width: u32, height: u32) -> Self {
        self.ops.push(ImageOp::crop(x, y, width, height));
        self
    }

    pub fn center_crop(mut self, width: u32, height: u32) -> Self {
        self.ops.push(ImageOp::center_crop(width, height));
        self
    }

    pub fn pad(mut self, padding: u32) -> Self {
        self.ops.push(ImageOp::pad(padding));
        self
    }

    pub fn pad_with_fill(mut self, padding: u32, value: f32) -> Self {
        self.ops.push(ImageOp::pad_with_fill(padding, value));
        self
    }

    pub fn pad_with_sides(
        mut self,
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
        value: f32,
    ) -> Self {
        self.ops
            .push(ImageOp::pad_with_sides(left, top, right, bottom, value));
        self
    }

    pub fn horizontal_flip(mut self) -> Self {
        self.ops.push(ImageOp::horizontal_flip());
        self
    }

    pub fn vertical_flip(mut self) -> Self {
        self.ops.push(ImageOp::vertical_flip());
        self
    }

    pub fn random_crop(mut self, width: u32, height: u32, padding: u32) -> Self {
        self.ops.push(ImageOp::random_crop(width, height, padding));
        self
    }

    pub fn random_resized_crop(mut self, width: u32, height: u32) -> Self {
        self.ops.push(ImageOp::random_resized_crop(width, height));
        self
    }

    pub fn random_resized_crop_with_config(mut self, config: RandomResizedCropConfig) -> Self {
        self.ops
            .push(ImageOp::random_resized_crop_with_config(config));
        self
    }

    pub fn random_horizontal_flip(mut self, probability: f64) -> Self {
        self.ops.push(ImageOp::random_horizontal_flip(probability));
        self
    }

    pub fn brightness(mut self, value: i32) -> Self {
        self.ops.push(ImageOp::brightness(value));
        self
    }

    pub fn contrast(mut self, value: f32) -> Self {
        self.ops.push(ImageOp::contrast(value));
        self
    }

    pub fn hue(mut self, degrees: i32) -> Self {
        self.ops.push(ImageOp::hue(degrees));
        self
    }

    pub fn color_jitter(mut self, brightness: i32, contrast: f32, hue: i32) -> Self {
        self.ops
            .push(ImageOp::color_jitter(brightness, contrast, hue));
        self
    }

    pub fn invert(mut self) -> Self {
        self.ops.push(ImageOp::invert());
        self
    }

    pub fn posterize(mut self, bits: u8) -> Self {
        self.ops.push(ImageOp::posterize(bits));
        self
    }

    pub fn solarize(mut self, threshold: u8) -> Self {
        self.ops.push(ImageOp::solarize(threshold));
        self
    }

    pub fn autocontrast(mut self) -> Self {
        self.ops.push(ImageOp::autocontrast());
        self
    }

    pub fn equalize(mut self) -> Self {
        self.ops.push(ImageOp::equalize());
        self
    }

    pub fn sharpness(mut self, amount: f32) -> Self {
        self.ops.push(ImageOp::sharpness(amount));
        self
    }

    pub fn arbitrary_rotate(mut self, angle: f32) -> Self {
        self.ops.push(ImageOp::arbitrary_rotate(angle));
        self
    }

    pub fn arbitrary_rotate_with_options(
        mut self,
        angle: f32,
        expand: bool,
        interpolation: InterpolationMode,
        fill: u8,
    ) -> Self {
        self.ops.push(ImageOp::arbitrary_rotate_with_options(
            angle,
            expand,
            interpolation,
            fill,
        ));
        self
    }

    pub fn random_affine(mut self, degrees: f32) -> Self {
        self.ops.push(ImageOp::random_affine(degrees));
        self
    }

    pub fn random_affine_with_config(mut self, config: RandomAffineConfig) -> Self {
        self.ops.push(ImageOp::random_affine_with_config(config));
        self
    }

    pub fn perspective(mut self, start_points: [Point2; 4], end_points: [Point2; 4]) -> Self {
        self.ops
            .push(ImageOp::perspective(start_points, end_points));
        self
    }

    pub fn perspective_with_config(mut self, config: PerspectiveConfig) -> Self {
        self.ops.push(ImageOp::perspective_with_config(config));
        self
    }

    pub fn random_perspective(mut self, distortion_scale: f32, probability: f64) -> Self {
        self.ops
            .push(ImageOp::random_perspective(distortion_scale, probability));
        self
    }

    pub fn random_perspective_with_config(mut self, config: RandomPerspectiveConfig) -> Self {
        self.ops
            .push(ImageOp::random_perspective_with_config(config));
        self
    }

    pub fn elastic_transform(mut self, alpha: f32, sigma: f32) -> Self {
        self.ops.push(ImageOp::elastic_transform(alpha, sigma));
        self
    }

    pub fn elastic_transform_with_config(mut self, config: ElasticTransformConfig) -> Self {
        self.ops
            .push(ImageOp::elastic_transform_with_config(config));
        self
    }

    pub fn gaussian_blur(mut self, sigma: f32) -> Self {
        self.ops.push(ImageOp::gaussian_blur(sigma));
        self
    }

    pub fn grayscale(mut self, num_output_channels: u8) -> Self {
        self.ops.push(ImageOp::grayscale(num_output_channels));
        self
    }

    pub fn random_grayscale(mut self, probability: f64, num_output_channels: u8) -> Self {
        self.ops
            .push(ImageOp::random_grayscale(probability, num_output_channels));
        self
    }

    pub fn random_erasing(mut self, probability: f64) -> Self {
        self.ops.push(ImageOp::random_erasing(probability));
        self
    }

    pub fn random_erasing_with_config(mut self, config: RandomErasingConfig) -> Self {
        self.ops.push(ImageOp::random_erasing_with_config(config));
        self
    }

    pub fn convert_image_dtype(mut self, dtype: DType) -> Self {
        self.ops.push(ImageOp::convert_image_dtype(dtype));
        self
    }

    pub fn rotate(mut self, angle: RotationAngle) -> Self {
        self.ops.push(ImageOp::rotate(angle));
        self
    }

    pub fn normalize(mut self, mean: Vec<f32>, std: Vec<f32>) -> Self {
        self.ops.push(ImageOp::normalize(mean, std));
        self
    }

    pub fn hwc_to_chw(mut self) -> Self {
        self.ops
            .push(ImageOp::Layout(crate::transforms::LayoutConfig::new(
                ImageAxisOrder::Chw,
            )));
        self
    }

    pub fn chw_to_hwc(mut self) -> Self {
        self.ops
            .push(ImageOp::Layout(crate::transforms::LayoutConfig::new(
                ImageAxisOrder::Hwc,
            )));
        self
    }

    pub fn random_apply(mut self, probability: f64, nested: Self) -> Self {
        self.ops
            .push(ImageOp::random_apply(probability, nested.into_ops()));
        self
    }

    pub fn random_choice(mut self, choices: Vec<Self>) -> Self {
        self.ops.push(ImageOp::random_choice(
            choices.into_iter().map(Self::into_ops).collect(),
        ));
        self
    }

    pub fn random_order(mut self, nested: Self) -> Self {
        self.ops.push(ImageOp::random_order(nested.into_ops()));
        self
    }
}

impl From<Vec<ImageOp>> for TransformSequence {
    fn from(ops: Vec<ImageOp>) -> Self {
        Self::from_ops(ops)
    }
}

impl From<TransformSequence> for Vec<ImageOp> {
    fn from(sequence: TransformSequence) -> Self {
        sequence.into_ops()
    }
}
