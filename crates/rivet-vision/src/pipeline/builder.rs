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
use rivet_plan::DeviceTarget;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeviceCutPlacement {
    pub(crate) after_ops: usize,
    pub(crate) after_batch: bool,
    pub(crate) target: DeviceTarget,
}

/// Declared operation order. Public operation vectors remain available for
/// inspection; slots retain their order when their payload is edited.
#[derive(Clone)]
pub(crate) enum PipelineStep {
    Image { index: usize, identity: Option<u64> },
    Index(usize),
    Batch,
    DeviceCut(usize),
}

#[derive(Clone)]
pub struct ImagePipeline {
    pub(crate) steps: Vec<PipelineStep>,
    pub source: SourceOp,
    pub index_ops: Vec<IndexOp>,
    pub ops: Vec<ImageOp>,
    pub batch: Option<BatchConfig>,
    pub(crate) device_cuts: Vec<DeviceCutPlacement>,
    pub runtime: RuntimeConfig,
    pub epoch: u64,
    pub global_seed: Option<u64>,
}

impl ImagePipeline {
    fn push_image(&mut self, op: ImageOp) {
        self.steps.push(PipelineStep::Image {
            index: self.ops.len(),
            identity: None,
        });
        self.ops.push(op);
    }

    fn push_index(&mut self, op: IndexOp) {
        self.steps.push(PipelineStep::Index(self.index_ops.len()));
        self.index_ops.push(op);
    }

    pub fn new<T>(dataset: Arc<T>) -> Self
    where
        T: Dataset<Item = EncodedImageSample> + 'static,
    {
        Self::from_source(ImageSource::from_encoded(dataset))
    }

    pub fn from_source(source: ImageSource) -> Self {
        Self {
            steps: Vec::new(),
            source: SourceOp::new(source),
            index_ops: Vec::new(),
            ops: Vec::new(),
            batch: None,
            device_cuts: Vec::new(),
            runtime: RuntimeConfig::default(),
            epoch: 0,
            global_seed: None,
        }
    }

    pub fn decode_image(mut self) -> Self {
        self.push_image(ImageOp::decode());
        self
    }

    pub fn resize(mut self, width: u32, height: u32) -> Self {
        self.push_image(ImageOp::resize(width, height));
        self
    }

    pub fn resize_with_interpolation(
        mut self,
        width: u32,
        height: u32,
        interpolation: InterpolationMode,
    ) -> Self {
        self.push_image(ImageOp::resize_with_interpolation(
            width,
            height,
            interpolation,
        ));
        self
    }

    pub fn crop(mut self, x: u32, y: u32, width: u32, height: u32) -> Self {
        self.push_image(ImageOp::crop(x, y, width, height));
        self
    }

    pub fn center_crop(mut self, width: u32, height: u32) -> Self {
        self.push_image(ImageOp::center_crop(width, height));
        self
    }

    pub fn pad(mut self, padding: u32) -> Self {
        self.push_image(ImageOp::pad(padding));
        self
    }

    pub fn pad_with_fill(mut self, padding: u32, value: f32) -> Self {
        self.push_image(ImageOp::pad_with_fill(padding, value));
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
        self.push_image(ImageOp::pad_with_sides(left, top, right, bottom, value));
        self
    }

    pub fn horizontal_flip(mut self) -> Self {
        self.push_image(ImageOp::horizontal_flip());
        self
    }

    pub fn vertical_flip(mut self) -> Self {
        self.push_image(ImageOp::vertical_flip());
        self
    }

    /// Zero-pad every decoded image by `padding` pixels on each side, then
    /// crop a `width`x`height` window at a per-sample random offset.
    ///
    /// Draws come from the pipeline's stochastic seed (the shuffle seed
    /// when the pipeline shuffles), so the same `(seed, epoch)` reproduces
    /// the same augmentations at any worker count.
    pub fn random_crop(mut self, width: u32, height: u32, padding: u32) -> Self {
        self.push_image(ImageOp::random_crop(width, height, padding));
        self
    }

    pub fn random_resized_crop(mut self, width: u32, height: u32) -> Self {
        self.push_image(ImageOp::random_resized_crop(width, height));
        self
    }

    pub fn random_resized_crop_with_config(mut self, config: RandomResizedCropConfig) -> Self {
        self.push_image(ImageOp::random_resized_crop_with_config(config));
        self
    }

    /// Randomly flip each decoded image horizontally with `probability`,
    /// per sample, from the pipeline's stochastic seed.
    pub fn random_horizontal_flip(mut self, probability: f64) -> Self {
        self.push_image(ImageOp::random_horizontal_flip(probability));
        self
    }

    pub fn brightness(mut self, value: i32) -> Self {
        self.push_image(ImageOp::brightness(value));
        self
    }

    pub fn contrast(mut self, value: f32) -> Self {
        self.push_image(ImageOp::contrast(value));
        self
    }

    pub fn color_jitter(mut self, brightness: i32, contrast: f32, hue: i32) -> Self {
        self.push_image(ImageOp::color_jitter(brightness, contrast, hue));
        self
    }

    pub fn invert(mut self) -> Self {
        self.push_image(ImageOp::invert());
        self
    }

    pub fn posterize(mut self, bits: u8) -> Self {
        self.push_image(ImageOp::posterize(bits));
        self
    }

    pub fn solarize(mut self, threshold: u8) -> Self {
        self.push_image(ImageOp::solarize(threshold));
        self
    }

    pub fn autocontrast(mut self) -> Self {
        self.push_image(ImageOp::autocontrast());
        self
    }

    pub fn equalize(mut self) -> Self {
        self.push_image(ImageOp::equalize());
        self
    }

    pub fn sharpness(mut self, amount: f32) -> Self {
        self.push_image(ImageOp::sharpness(amount));
        self
    }

    pub fn gaussian_blur(mut self, sigma: f32) -> Self {
        self.push_image(ImageOp::gaussian_blur(sigma));
        self
    }

    pub fn hue(mut self, degrees: i32) -> Self {
        self.push_image(ImageOp::hue(degrees));
        self
    }

    pub fn grayscale(mut self, num_output_channels: u8) -> Self {
        self.push_image(ImageOp::grayscale(num_output_channels));
        self
    }

    pub fn random_grayscale(mut self, probability: f64, num_output_channels: u8) -> Self {
        self.push_image(ImageOp::random_grayscale(probability, num_output_channels));
        self
    }

    pub fn random_erasing(mut self, probability: f64) -> Self {
        self.push_image(ImageOp::random_erasing(probability));
        self
    }

    pub fn random_erasing_with_config(mut self, config: RandomErasingConfig) -> Self {
        self.push_image(ImageOp::random_erasing_with_config(config));
        self
    }

    pub fn convert_image_dtype(mut self, dtype: DType) -> Self {
        self.push_image(ImageOp::convert_image_dtype(dtype));
        self
    }

    pub fn rotate(mut self, angle: crate::transforms::RotationAngle) -> Self {
        self.push_image(ImageOp::rotate(angle));
        self
    }

    pub fn arbitrary_rotate(mut self, angle: f32) -> Self {
        self.push_image(ImageOp::arbitrary_rotate(angle));
        self
    }

    pub fn arbitrary_rotate_with_options(
        mut self,
        angle: f32,
        expand: bool,
        interpolation: InterpolationMode,
        fill: u8,
    ) -> Self {
        self.push_image(ImageOp::arbitrary_rotate_with_options(
            angle,
            expand,
            interpolation,
            fill,
        ));
        self
    }

    pub fn random_affine(mut self, degrees: f32) -> Self {
        self.push_image(ImageOp::random_affine(degrees));
        self
    }

    pub fn random_affine_with_config(mut self, config: RandomAffineConfig) -> Self {
        self.push_image(ImageOp::random_affine_with_config(config));
        self
    }

    pub fn perspective(mut self, start_points: [Point2; 4], end_points: [Point2; 4]) -> Self {
        self.push_image(ImageOp::perspective(start_points, end_points));
        self
    }

    pub fn perspective_with_config(mut self, config: PerspectiveConfig) -> Self {
        self.push_image(ImageOp::perspective_with_config(config));
        self
    }

    pub fn random_perspective(mut self, distortion_scale: f32, probability: f64) -> Self {
        self.push_image(ImageOp::random_perspective(distortion_scale, probability));
        self
    }

    pub fn random_perspective_with_config(mut self, config: RandomPerspectiveConfig) -> Self {
        self.push_image(ImageOp::random_perspective_with_config(config));
        self
    }

    pub fn elastic_transform(mut self, alpha: f32, sigma: f32) -> Self {
        self.push_image(ImageOp::elastic_transform(alpha, sigma));
        self
    }

    pub fn elastic_transform_with_config(mut self, config: ElasticTransformConfig) -> Self {
        self.push_image(ImageOp::elastic_transform_with_config(config));
        self
    }

    pub fn normalize(mut self, mean: Vec<f32>, std: Vec<f32>) -> Self {
        self.push_image(ImageOp::normalize(mean, std));
        self
    }

    pub fn hwc_to_chw(mut self) -> Self {
        self.push_image(ImageOp::hwc_to_chw());
        self
    }

    pub fn chw_to_hwc(mut self) -> Self {
        self.push_image(ImageOp::chw_to_hwc());
        self
    }

    /// Append a reusable transform sequence to this pipeline.
    pub fn compose(mut self, sequence: TransformSequence) -> Self {
        for op in sequence.into_ops() {
            self.push_image(op);
        }
        self
    }

    pub fn random_apply(mut self, probability: f64, sequence: TransformSequence) -> Self {
        self.push_image(ImageOp::random_apply(probability, sequence.into_ops()));
        self
    }

    pub fn random_choice(mut self, choices: Vec<TransformSequence>) -> Self {
        self.push_image(ImageOp::random_choice(
            choices
                .into_iter()
                .map(TransformSequence::into_ops)
                .collect(),
        ));
        self
    }

    pub fn random_order(mut self, sequence: TransformSequence) -> Self {
        self.push_image(ImageOp::random_order(sequence.into_ops()));
        self
    }

    pub fn skip(mut self, count: usize) -> Self {
        self.push_index(IndexOp::Skip { count });
        self
    }

    pub fn take(mut self, count: usize) -> Self {
        self.push_index(IndexOp::Take { count });
        self
    }

    /// Deterministically shuffle the sampled window with `seed`; the same
    /// semantic seed and epoch reproduce the same order at any worker count.
    /// Prefer `.seed(seed)` when sampler and transform randomness should share
    /// an explicit pipeline-owned namespace.
    pub fn shuffle(mut self, seed: u64) -> Self {
        self.push_index(IndexOp::Shuffle { seed });
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
        let config = BatchConfig::new(size, drop_last);
        self.steps.push(PipelineStep::Batch);
        self.batch = Some(config);
        self
    }

    /// Insert a fixed host-to-CUDA semantic boundary after the image
    /// transformations appended so far. If `.batch(...)` has already been
    /// configured, the cut follows the batch node; otherwise the batch node
    /// follows the image transformation chain.
    pub fn device_cut_to_cuda(self, ordinal: usize) -> Self {
        self.device_cut(DeviceTarget::cuda(ordinal))
    }

    /// Insert a fixed host-to-Metal semantic boundary after the image
    /// transformations appended so far, with the same batch ordering as
    /// [`Self::device_cut_to_cuda`]. Runtime execution requires a Metal
    /// backend implementation.
    pub fn device_cut_to_metal(self, ordinal: usize) -> Self {
        self.device_cut(DeviceTarget::metal(ordinal))
    }

    /// Insert a fixed host-to-device semantic boundary after the image
    /// transformations appended so far, with the same batch ordering as
    /// [`Self::device_cut_to_cuda`].
    pub fn device_cut(mut self, target: DeviceTarget) -> Self {
        self.steps
            .push(PipelineStep::DeviceCut(self.device_cuts.len()));
        self.device_cuts.push(DeviceCutPlacement {
            after_ops: self.ops.len(),
            after_batch: self.batch.is_some(),
            target,
        });
        self
    }

    /// Execute per-sample semantic transforms on a persistent pool of
    /// `num_workers` inter-sample workers (`0` executes them on the CPU
    /// transform stage thread). Decode runs on its own persistent stage;
    /// backend-internal parallelism remains controlled by each kernel/backend.
    /// Ordering, batching and sampling semantics do not depend on worker count.
    pub fn workers(mut self, num_workers: usize) -> Self {
        self.runtime.num_workers = num_workers;
        self
    }

    /// Allow up to `prefetch_batches` future batches behind the current
    /// delivery (`prefetch_batches + 1` outstanding sequences total). Stage
    /// queues apply the configured item and byte bounds; delivery stays in
    /// sampler order.
    pub fn prefetch_batches(mut self, prefetch_batches: usize) -> Self {
        self.runtime.prefetch_batches = prefetch_batches;
        self
    }

    /// Bound the retained payload size of each persistent stage queue.
    /// Defaults to 512 MiB. An individual encoded image or batch larger than
    /// this limit returns a runtime error instead of exceeding the budget.
    pub fn stage_queue_max_bytes(mut self, max_bytes: usize) -> Self {
        self.runtime.stage_queue_max_bytes = max_bytes;
        self
    }

    /// Enable detailed per-node profiling for diagnostics.
    /// Profiling is disabled by default to avoid steady-state timing and
    /// counter updates on the runtime hot path.
    pub fn profiling(mut self, enabled: bool) -> Self {
        self.runtime.profiling_enabled = enabled;
        self
    }
}
