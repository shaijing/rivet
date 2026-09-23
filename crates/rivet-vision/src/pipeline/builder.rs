use super::op::{BatchConfig, ImageOp, IndexOp, SourceOp};
use super::transform::TransformSequence;
use crate::runtime::RuntimeConfig;
use crate::sample::image::EncodedImageSample;
use crate::source::ImageSource;
use crate::transforms::{
    ElasticTransformConfig, InterpolationMode, PerspectiveConfig, Point2, RandomAffineConfig,
    RandomErasingConfig, RandomPerspectiveConfig, RandomResizedCropConfig,
};
use rivet_core::DType;
use rivet_data::dataset::Dataset;
use std::sync::Arc;

#[derive(Clone)]
pub struct ImagePipeline {
    pub source: SourceOp,
    pub index_ops: Vec<IndexOp>,
    pub ops: Vec<ImageOp>,
    pub batch: Option<BatchConfig>,
    pub runtime: RuntimeConfig,
    pub epoch: u64,
    pub global_seed: Option<u64>,
}

impl ImagePipeline {
    pub fn new<T>(dataset: Arc<T>) -> Self
    where
        T: Dataset<Item = EncodedImageSample> + 'static,
    {
        Self::from_source(ImageSource::from_encoded(dataset))
    }

    pub fn from_source(source: ImageSource) -> Self {
        Self {
            source: SourceOp::new(source),
            index_ops: Vec::new(),
            ops: Vec::new(),
            batch: None,
            runtime: RuntimeConfig::default(),
            epoch: 0,
            global_seed: None,
        }
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

    /// Zero-pad every decoded image by `padding` pixels on each side, then
    /// crop a `width`x`height` window at a per-sample random offset.
    ///
    /// Draws come from the pipeline's stochastic seed (the shuffle seed
    /// when the pipeline shuffles), so the same `(seed, epoch)` reproduces
    /// the same augmentations at any worker count.
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

    /// Randomly flip each decoded image horizontally with `probability`,
    /// per sample, from the pipeline's stochastic seed.
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

    pub fn gaussian_blur(mut self, sigma: f32) -> Self {
        self.ops.push(ImageOp::gaussian_blur(sigma));
        self
    }

    pub fn hue(mut self, degrees: i32) -> Self {
        self.ops.push(ImageOp::hue(degrees));
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

    pub fn rotate(mut self, angle: crate::transforms::RotationAngle) -> Self {
        self.ops.push(ImageOp::rotate(angle));
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

    pub fn normalize(mut self, mean: Vec<f32>, std: Vec<f32>) -> Self {
        self.ops.push(ImageOp::normalize(mean, std));
        self
    }

    pub fn hwc_to_chw(mut self) -> Self {
        self.ops.push(ImageOp::hwc_to_chw());
        self
    }

    pub fn chw_to_hwc(mut self) -> Self {
        self.ops.push(ImageOp::chw_to_hwc());
        self
    }

    /// Append a reusable transform sequence to this pipeline.
    pub fn compose(mut self, sequence: TransformSequence) -> Self {
        self.ops.extend(sequence.into_ops());
        self
    }

    pub fn random_apply(mut self, probability: f64, sequence: TransformSequence) -> Self {
        self.ops
            .push(ImageOp::random_apply(probability, sequence.into_ops()));
        self
    }

    pub fn random_choice(mut self, choices: Vec<TransformSequence>) -> Self {
        self.ops.push(ImageOp::random_choice(
            choices
                .into_iter()
                .map(TransformSequence::into_ops)
                .collect(),
        ));
        self
    }

    pub fn random_order(mut self, sequence: TransformSequence) -> Self {
        self.ops.push(ImageOp::random_order(sequence.into_ops()));
        self
    }

    pub fn skip(mut self, count: usize) -> Self {
        self.index_ops.push(IndexOp::Skip { count });
        self
    }

    pub fn take(mut self, count: usize) -> Self {
        self.index_ops.push(IndexOp::Take { count });
        self
    }

    /// Deterministically shuffle the sampled window with `seed`; the same
    /// semantic seed and epoch reproduce the same order at any worker count.
    /// Prefer `.seed(seed)` when sampler and transform randomness should share
    /// an explicit pipeline-owned namespace.
    pub fn shuffle(mut self, seed: u64) -> Self {
        self.index_ops.push(IndexOp::Shuffle { seed });
        self
    }

    /// Set the pipeline-owned semantic seed used by sampling and transforms.
    /// A legacy `.shuffle(seed)` remains a fallback when this is omitted.
    pub fn seed(mut self, seed: u64) -> Self {
        self.global_seed = Some(seed);
        self
    }

    /// Set the explicit epoch used by both sampler and transform namespaces.
    /// Use the same epoch with different worker counts to reproduce exactly;
    /// changing it produces a fresh permutation and per-sample RNG stream.
    pub fn epoch(mut self, epoch: u64) -> Self {
        self.epoch = epoch;
        self
    }

    pub fn batch(mut self, size: usize, drop_last: bool) -> Self {
        self.batch = Some(BatchConfig::new(size, drop_last));
        self
    }

    /// Request that batches be returned on the CUDA device at `ordinal`.
    /// Physical lowering inserts one explicit H2D transfer after all current
    /// CPU operations. This path uses one device and its default stream.
    #[cfg(feature = "cuda")]
    pub fn cuda_sink(mut self, ordinal: usize) -> Self {
        self.runtime.sink_device_ordinal = Some(ordinal);
        self
    }

    /// Execute sample loading on a persistent pool of `num_workers` threads
    /// (`0` keeps the synchronous inline path). Ordering, batching and
    /// sampling semantics are unaffected by the worker count.
    pub fn workers(mut self, num_workers: usize) -> Self {
        self.runtime.num_workers = num_workers;
        self
    }

    /// Prepare up to `prefetch_batches` future batches while the caller
    /// consumes the current one (worker pools only; the current batch is
    /// always in flight, so total in-flight = `prefetch_batches + 1`).
    /// Delivery stays in sampler order.
    pub fn prefetch_batches(mut self, prefetch_batches: usize) -> Self {
        self.runtime.prefetch_batches = prefetch_batches;
        self
    }
}
