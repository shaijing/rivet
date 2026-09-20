use super::op::{BatchConfig, ImageOp, IndexOp, SourceOp};
use crate::runtime::RuntimeConfig;
use crate::sample::image::EncodedImageSample;
use crate::source::ImageSource;
use crate::transforms::InterpolationMode;
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

    pub fn convert_image_dtype(mut self, dtype: DType) -> Self {
        self.ops.push(ImageOp::convert_image_dtype(dtype));
        self
    }

    pub fn rotate(mut self, angle: crate::transforms::RotationAngle) -> Self {
        self.ops.push(ImageOp::rotate(angle));
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

    pub fn skip(mut self, count: usize) -> Self {
        self.index_ops.push(IndexOp::Skip { count });
        self
    }

    pub fn take(mut self, count: usize) -> Self {
        self.index_ops.push(IndexOp::Take { count });
        self
    }

    /// Deterministically shuffle the sampled window with `seed`; the same
    /// seed reproduces the same order at any worker count. Use per-epoch
    /// seeds (e.g. `base_seed + epoch`) for reproducible shuffling across
    /// epochs.
    pub fn shuffle(mut self, seed: u64) -> Self {
        self.index_ops.push(IndexOp::Shuffle { seed });
        self
    }

    pub fn batch(mut self, size: usize, drop_last: bool) -> Self {
        self.batch = Some(BatchConfig::new(size, drop_last));
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
