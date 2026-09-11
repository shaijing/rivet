use crate::dataset::Source;
use crate::errors::{RivetResult, invalid_argument, invalid_pipeline};
use crate::image::color::{BrightnessConfig, ContrastConfig};
use crate::image::crop::{CenterCropConfig, CropConfig, RandomCropConfig};
use crate::image::decode::DecodeImageConfig;
use crate::image::flip::{FlipConfig, FlipDirection, RandomHorizontalFlipConfig};
use crate::image::layout::LayoutConfig;
use crate::image::normalize::NormalizeConfig;
use crate::image::resize::ResizeConfig;
use crate::sample::image::{DecodedSample, EncodedImageSample, ImageSample};
use crate::sample::image::{ImageDType, ImageLayout};
use crate::sampler::{SamplerPlan, permute};

#[derive(Clone)]
pub struct SourceOp {
    source: Source<EncodedImageSample>,
}

impl SourceOp {
    pub fn new(source: Source<EncodedImageSample>) -> Self {
        Self { source }
    }

    pub fn len(&self) -> usize {
        self.source.len()
    }

    pub fn get(&self, index: usize) -> RivetResult<EncodedImageSample> {
        self.source.get(index)
    }
}

#[derive(Clone)]
pub enum IndexOp {
    Skip { count: usize },
    Take { count: usize },
    /// Deterministically shuffle the selected index window with a seed.
    /// Skip/Take apply first, then the window is permuted, so a fixed
    /// `(seed, epoch)` always reproduces the same order regardless of
    /// worker count.
    Shuffle { seed: u64 },
}

impl IndexOp {
    pub fn apply_range(&self, start: &mut usize, end: &mut usize) {
        match self {
            Self::Skip { count } => {
                *start = (*start + *count).min(*end);
            }
            Self::Take { count } => {
                *end = (*start + *count).min(*end);
            }
            Self::Shuffle { .. } => {}
        }
    }
}

#[derive(Clone)]
pub enum ImageOp {
    Decode(DecodeImageConfig),
    Resize(ResizeConfig),
    Crop(CropConfig),
    CenterCrop(CenterCropConfig),
    Flip(FlipConfig),
    RandomCrop(RandomCropConfig),
    RandomHorizontalFlip(RandomHorizontalFlipConfig),
    Brightness(BrightnessConfig),
    Contrast(ContrastConfig),
    Normalize(NormalizeConfig),
    Layout(LayoutConfig),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineImageState {
    Encoded,
    Decoded {
        dtype: ImageDType,
        layout: ImageLayout,
    },
}

impl ImageOp {
    pub fn transition(&self, input: PipelineImageState) -> RivetResult<PipelineImageState> {
        use PipelineImageState::{Decoded, Encoded};

        match self {
            Self::Decode(_) => match input {
                Encoded => Ok(Decoded {
                    dtype: ImageDType::U8,
                    layout: ImageLayout::Hwc,
                }),
                Decoded { .. } => Err(invalid_pipeline(
                    "Decode requires an encoded image, current state is decoded",
                )),
            },
            Self::Resize(_) => require_u8_hwc(input, "Resize"),
            Self::Crop(_) => require_u8_hwc(input, "Crop"),
            Self::CenterCrop(_) => require_u8_hwc(input, "CenterCrop"),
            Self::Flip(_) => require_u8_hwc(input, "Flip"),
            Self::RandomCrop(_) => require_u8_hwc(input, "RandomCrop"),
            Self::RandomHorizontalFlip(_) => {
                require_u8_hwc(input, "RandomHorizontalFlip")
            }
            Self::Brightness(_) => require_u8_hwc(input, "Brightness"),
            Self::Contrast(_) => require_u8_hwc(input, "Contrast"),
            Self::Normalize(_) => match input {
                Encoded => Err(invalid_pipeline(
                    "Normalize requires a decoded image, current state is encoded",
                )),
                Decoded { layout, .. } => Ok(Decoded {
                    dtype: ImageDType::F32,
                    layout,
                }),
            },
            Self::Layout(op) => match input {
                Encoded => Err(invalid_pipeline(
                    "Layout requires a decoded image, current state is encoded",
                )),
                Decoded { dtype, .. } => Ok(Decoded {
                    dtype,
                    layout: op.layout,
                }),
            },
        }
    }

    pub fn apply(&self, sample: ImageSample, ctx: &mut SampleContext) -> RivetResult<ImageSample> {
        match self {
            Self::Decode(op) => op.apply(sample),
            Self::Resize(op) => op.apply(sample),
            Self::Crop(op) => op.apply(sample),
            Self::CenterCrop(op) => op.apply(sample),
            Self::Flip(op) => op.apply(sample),
            Self::RandomCrop(op) => op.apply(sample, ctx),
            Self::RandomHorizontalFlip(op) => op.apply(sample, ctx),
            Self::Brightness(op) => op.apply(sample),
            Self::Contrast(op) => op.apply(sample),
            Self::Normalize(op) => op.apply(sample),
            Self::Layout(op) => op.apply(sample),
        }
    }
}

fn require_u8_hwc(input: PipelineImageState, op_name: &str) -> RivetResult<PipelineImageState> {
    match input {
        PipelineImageState::Encoded => Err(invalid_pipeline(format!(
            "{op_name} requires a decoded image, current state is encoded"
        ))),
        PipelineImageState::Decoded {
            dtype: ImageDType::U8,
            layout: ImageLayout::Hwc,
        } => Ok(input),
        PipelineImageState::Decoded { dtype, layout } => Err(invalid_pipeline(format!(
            "{op_name} requires uint8 HWC input, current state is {} {}",
            dtype.as_str(),
            layout.as_str()
        ))),
    }
}

#[derive(Clone, Copy)]
pub struct BatchConfig {
    pub size: usize,
    pub drop_last: bool,
}

impl BatchConfig {
    /// Raw configuration; pipeline compilation validates it.
    pub fn new(size: usize, drop_last: bool) -> Self {
        Self { size, drop_last }
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self.size == 0 {
            return Err(invalid_argument("batch size must be greater than 0"));
        }

        Ok(())
    }
}

pub struct SampleContext {
    pub sample_index: usize,
    pub epoch: u64,
    pub global_seed: u64,
    rng_state: Option<u64>,
}

impl SampleContext {
    pub fn new(sample_index: usize) -> Self {
        Self {
            sample_index,
            epoch: 0,
            global_seed: 0,
            rng_state: None,
        }
    }

    pub fn sample_seed(&self) -> u64 {
        let mut seed = self.global_seed ^ 0x9E37_79B9_7F4A_7C15;
        seed = mix_seed(seed ^ self.epoch);
        mix_seed(seed ^ self.sample_index as u64)
    }

    /// Draw the next deterministic u64 for a stochastic op on this sample.
    ///
    /// The stream is seeded from `(global_seed, epoch, sample_index)`, so
    /// every sample gets a reproducible, distinct draw sequence regardless
    /// of worker count or scheduling; each stochastic op consumes one or
    /// more draws in pipeline order.
    pub fn next_rng_u64(&mut self) -> u64 {
        if self.rng_state.is_none() {
            self.rng_state = Some(self.sample_seed());
        }
        splitmix64(self.rng_state.as_mut().unwrap())
    }
}

/// SplitMix64 stream step: advance the state and return a mixed output.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[derive(Clone)]
pub struct ExecutionPlan {
    pub source: SourceOp,
    pub sampler: SamplerPlan,
    pub ops: Vec<ImageOp>,
    pub batch: BatchConfig,
    /// Seed for stochastic image ops. When the pipeline shuffles, this is
    /// the shuffle seed, so one `(seed, epoch)` reproduces both the sample
    /// order and every random augmentation.
    pub random_seed: u64,
    /// Compile-time image state after `ops`, so batch builders and bindings
    /// know the output dtype and layout before any sample is processed.
    pub output_state: PipelineImageState,
}

impl ExecutionPlan {
    pub fn apply_ops(
        &self,
        sample: EncodedImageSample,
        sample_index: usize,
    ) -> RivetResult<DecodedSample> {
        let mut ctx = SampleContext::new(sample_index);
        ctx.global_seed = self.random_seed;
        let mut sample = ImageSample::Encoded(sample);

        for op in &self.ops {
            sample = op.apply(sample, &mut ctx)?;
        }

        sample.into_decoded()
    }
}

pub fn compile_sampler(len: usize, index_ops: &[IndexOp]) -> RivetResult<SamplerPlan> {
    let mut start = 0usize;
    let mut end = len;
    let mut shuffle_seed: Option<u64> = None;

    for op in index_ops {
        match op {
            IndexOp::Shuffle { seed } => {
                if shuffle_seed.is_some() {
                    return Err(invalid_pipeline(
                        "multiple shuffle ops are not supported; combine seeds outside the pipeline",
                    ));
                }
                shuffle_seed = Some(*seed);
            }
            _ => op.apply_range(&mut start, &mut end),
        }
    }

    match shuffle_seed {
        None => Ok(SamplerPlan::Sequential { start, end }),
        Some(seed) => {
            let window_len = end - start;
            let mut indices = permute(window_len, seed);
            for index in &mut indices {
                *index += start;
            }
            Ok(SamplerPlan::Permutation { indices })
        }
    }
}

fn mix_seed(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

impl ImageOp {
    pub fn decode() -> Self {
        Self::Decode(DecodeImageConfig)
    }

    pub fn resize(width: u32, height: u32) -> Self {
        Self::Resize(ResizeConfig { width, height })
    }

    pub fn crop(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self::Crop(CropConfig {
            x,
            y,
            width,
            height,
        })
    }

    pub fn center_crop(width: u32, height: u32) -> Self {
        Self::CenterCrop(CenterCropConfig { width, height })
    }

    pub fn random_crop(width: u32, height: u32, padding: u32) -> Self {
        Self::RandomCrop(RandomCropConfig {
            width,
            height,
            padding,
        })
    }

    pub fn horizontal_flip() -> Self {
        Self::Flip(FlipConfig {
            direction: FlipDirection::Horizontal,
        })
    }

    pub fn vertical_flip() -> Self {
        Self::Flip(FlipConfig {
            direction: FlipDirection::Vertical,
        })
    }

    pub fn random_horizontal_flip(probability: f64) -> Self {
        Self::RandomHorizontalFlip(RandomHorizontalFlipConfig { probability })
    }

    pub fn brightness(value: i32) -> Self {
        Self::Brightness(BrightnessConfig { value })
    }

    pub fn contrast(value: f32) -> Self {
        Self::Contrast(ContrastConfig { value })
    }

    pub fn normalize(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self::Normalize(NormalizeConfig::new(mean, std))
    }

    pub fn hwc_to_chw() -> Self {
        Self::Layout(LayoutConfig {
            layout: ImageLayout::Chw,
        })
    }

    pub fn chw_to_hwc() -> Self {
        Self::Layout(LayoutConfig {
            layout: ImageLayout::Hwc,
        })
    }

    /// Validate the op's own configuration (not its position in the
    /// pipeline; that is `transition`'s job).
    pub fn validate(&self) -> RivetResult<()> {
        match self {
            Self::Decode(_)
            | Self::Flip(_)
            | Self::Brightness(_)
            | Self::Contrast(_)
            | Self::Layout(_) => Ok(()),
            Self::Resize(op) => {
                if op.width == 0 || op.height == 0 {
                    return Err(invalid_argument(
                        "resize width and height must be greater than 0",
                    ));
                }
                Ok(())
            }
            Self::Crop(op) => {
                if op.width == 0 || op.height == 0 {
                    return Err(invalid_argument(
                        "crop width and height must be greater than 0",
                    ));
                }
                Ok(())
            }
            Self::CenterCrop(op) => {
                if op.width == 0 || op.height == 0 {
                    return Err(invalid_argument(
                        "center_crop width and height must be greater than 0",
                    ));
                }
                Ok(())
            }
            Self::RandomCrop(op) => {
                if op.width == 0 || op.height == 0 {
                    return Err(invalid_argument(
                        "random_crop width and height must be greater than 0",
                    ));
                }
                Ok(())
            }
            Self::RandomHorizontalFlip(op) => {
                if !(0.0..=1.0).contains(&op.probability) {
                    return Err(invalid_argument(
                        "random_horizontal_flip probability must be in [0.0, 1.0]",
                    ));
                }
                Ok(())
            }
            Self::Normalize(op) => op.validate(),
        }
    }
}
