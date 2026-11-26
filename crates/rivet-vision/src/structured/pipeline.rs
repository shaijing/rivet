use super::CopyAndPasteConfig;
use super::GuidedCoarseDropoutConfig;
use super::RandomCropNearBBoxConfig;
use super::StructuredDropoutConfig;
use super::{
    AtLeastOneBBoxRandomCropConfig, BBoxSafeRandomCropConfig, BBoxSubsetSafeRandomCropConfig,
    BoxFormat, RandomIoUCropConfig, StructuredSample,
};
use super::{ConstrainedCoarseDropoutConfig, MaskDropoutConfig};
use super::{CropNonEmptyMaskConfig, MosaicConfig, OverlayElementsConfig};
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::pipeline::{
    TransformSequence,
    op::{ImageOp, SampleContext},
};
use rivet_core::Tensor;
use rivet_data::random::{OpKey, RandomContext, SampleKey};
use std::sync::Arc;

#[derive(Clone)]
pub enum StructuredOp {
    Image(ImageOp),
    ConvertBoxes(BoxFormat),
    ClampBoxes,
    ClampKeypoints,
    SanitizeKeypoints,
    Sanitize { min_size: f32, min_area: f32 },
    RandomIoUCrop(RandomIoUCropConfig),
    RandomCropNearBBox(RandomCropNearBBoxConfig),
    BBoxSafeRandomCrop(BBoxSafeRandomCropConfig),
    BBoxSubsetSafeRandomCrop(BBoxSubsetSafeRandomCropConfig),
    AtLeastOneBBoxRandomCrop(AtLeastOneBBoxRandomCropConfig),
    GuidedCoarseDropout(Arc<GuidedCoarseDropoutConfig>),
    MaskDropout(MaskDropoutConfig),
    OverlayElements(Arc<OverlayElementsConfig>),
    CopyAndPaste(Arc<CopyAndPasteConfig>),
    Mosaic(Arc<MosaicConfig>),
    SpatialDropout(StructuredDropoutConfig),
    ConstrainedCoarseDropout(std::sync::Arc<ConstrainedCoarseDropoutConfig>),
    CropNonEmptyMask(CropNonEmptyMaskConfig),
}
impl StructuredOp {
    fn name(&self) -> &'static str {
        match self {
            Self::Image(op) => op.name(),
            Self::ConvertBoxes(_) => "ConvertBoundingBoxFormat",
            Self::ClampBoxes => "ClampBoundingBoxes",
            Self::ClampKeypoints => "ClampKeyPoints",
            Self::SanitizeKeypoints => "SanitizeKeyPoints",
            Self::Sanitize { .. } => "SanitizeBoundingBoxes",
            Self::RandomIoUCrop(_) => "RandomIoUCrop",
            Self::ConstrainedCoarseDropout(_) => "ConstrainedCoarseDropout",
            Self::SpatialDropout(c) => c.name(),
            Self::CopyAndPaste(_) => "CopyAndPaste",
            Self::Mosaic(_) => "Mosaic",
            Self::OverlayElements(_) => "OverlayElements",

            Self::GuidedCoarseDropout(_) => "GuidedCoarseDropout",
            Self::MaskDropout(_) => "MaskDropout",
            Self::CropNonEmptyMask(_) => "CropNonEmptyMaskIfExists",
            Self::RandomCropNearBBox(_) => "RandomCropNearBBox",
            Self::BBoxSafeRandomCrop(c) if c.output.is_some() => "RandomSizedBBoxSafeCrop",
            Self::BBoxSafeRandomCrop(_) => "BBoxSafeRandomCrop",
            Self::BBoxSubsetSafeRandomCrop(_) => "BBoxSubsetSafeRandomCrop",
            Self::AtLeastOneBBoxRandomCrop(_) => "AtLeastOneBBoxRandomCrop",
        }
    }
    pub fn validate(&self) -> RivetResult<()> {
        match self {
            Self::Image(op) => validate_image(op),
            Self::RandomIoUCrop(c) => c.validate(),
            Self::ConstrainedCoarseDropout(c) => c.validate(),
            Self::SpatialDropout(c) => c.validate(),
            Self::CopyAndPaste(c) => c.validate(),
            Self::Mosaic(c) => c.validate(),
            Self::OverlayElements(c) => c.validate(),

            Self::GuidedCoarseDropout(c) => c.validate(),
            Self::MaskDropout(c) => c.validate(),
            Self::CropNonEmptyMask(c) => c.validate(),
            Self::RandomCropNearBBox(c) => c.validate(),
            Self::BBoxSafeRandomCrop(c) => c.validate(),
            Self::BBoxSubsetSafeRandomCrop(c) => c.validate(),
            Self::AtLeastOneBBoxRandomCrop(c) => c.validate(),
            Self::Sanitize { min_size, min_area }
                if !min_size.is_finite()
                    || *min_size < 1.0
                    || !min_area.is_finite()
                    || *min_area < 1.0 =>
            {
                Err(invalid_argument(
                    "sanitize thresholds must be finite and >=1",
                ))
            }
            _ => Ok(()),
        }
    }
}
fn validate_image(op: &ImageOp) -> RivetResult<()> {
    op.validate()?;
    match op {
        ImageOp::Decode(_)
        | ImageOp::FiveCrop(_)
        | ImageOp::TenCrop(_)
        | ImageOp::AutoAugment(_)
        | ImageOp::RandAugment(_)
        | ImageOp::TrivialAugmentWide(_)
        | ImageOp::AugMix(_) => Err(invalid_argument(format!(
            "{} is unsupported for structured samples",
            op.name()
        ))),
        ImageOp::RandomApply { ops, .. } | ImageOp::RandomOrder { ops } => {
            for op in ops {
                validate_image(op)?;
            }
            Ok(())
        }
        ImageOp::RandomChoice { choices, .. } => {
            for ops in choices {
                for op in ops {
                    validate_image(op)?;
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
#[derive(Clone)]
pub struct StructuredPipeline {
    pub samples: Arc<Vec<StructuredSample>>,
    pub ops: Vec<StructuredOp>,
    pub seed: u64,
    pub epoch: u64,
    pub workers: usize,
    pub batch_size: usize,
    pub drop_last: bool,
    pub shuffle: bool,
}
impl StructuredPipeline {
    pub fn new(samples: Vec<StructuredSample>) -> RivetResult<Self> {
        for sample in &samples {
            sample.validate()?;
        }
        Ok(Self {
            samples: Arc::new(samples),
            ops: Vec::new(),
            seed: 0,
            epoch: 0,
            workers: 0,
            batch_size: 1,
            drop_last: false,
            shuffle: false,
        })
    }
    pub fn compose(mut self, sequence: TransformSequence) -> Self {
        self.ops
            .extend(sequence.ops.into_iter().map(StructuredOp::Image));
        self
    }
    pub fn push(mut self, op: StructuredOp) -> Self {
        self.ops.push(op);
        self
    }
    pub fn compile(self) -> RivetResult<StructuredLoader> {
        if self.batch_size == 0 || self.workers > 256 {
            return Err(invalid_argument(
                "batch size must be positive and workers <=256",
            ));
        }
        for op in &self.ops {
            op.validate()?;
            if let StructuredOp::GuidedCoarseDropout(config) = op
                && config.region.rank() == 3
                && config.region.dims()[0] != self.samples.len()
            {
                return Err(invalid_shape(
                    "GuidedCoarseDropout requires one region per source sample",
                ));
            }
        }
        let mut indices: Vec<_> = (0..self.samples.len()).collect();
        if self.shuffle {
            let mut rng = RandomContext::new(self.seed).with_epoch(self.epoch).stream(
                SampleKey::from_index(0),
                OpKey::from_parts("StructuredShuffle", 0),
            );
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range_usize(0..i + 1)?;
                indices.swap(i, j);
            }
        }
        Ok(StructuredLoader {
            pipeline: self,
            indices,
            position: 0,
            failed: false,
        })
    }
    pub fn apply(
        &self,
        mut sample: StructuredSample,
        index: usize,
    ) -> RivetResult<StructuredSample> {
        sample.validate()?;
        let ctx =
            SampleContext::with_random(index, RandomContext::new(self.seed).with_epoch(self.epoch));
        let mut counts = std::collections::HashMap::<&str, u32>::new();
        for op in &self.ops {
            let count = counts.entry(op.name()).or_default();
            let key = OpKey::from_parts(op.name(), *count);
            *count += 1;
            let mut rng = ctx.stream(key);
            sample = match op {
                StructuredOp::Image(op) => super::dispatch::apply(sample, op, &mut rng, &ctx, key)?,
                StructuredOp::ConvertBoxes(format) => sample.convert_boxes(*format)?,
                StructuredOp::ClampBoxes => sample.clamp_boxes()?,
                StructuredOp::ClampKeypoints => sample.clamp_keypoints()?,
                StructuredOp::SanitizeKeypoints => sample.sanitize_keypoints()?,
                StructuredOp::Sanitize { min_size, min_area } => {
                    sample.sanitize(*min_size, *min_area)?
                }
                StructuredOp::RandomIoUCrop(c) => sample.random_iou_crop(c, &mut rng)?,
                StructuredOp::BBoxSubsetSafeRandomCrop(c) => {
                    sample.bbox_subset_safe_random_crop(c, &mut rng)?
                }
                StructuredOp::ConstrainedCoarseDropout(c) => {
                    sample.constrained_coarse_dropout(c, &mut rng)?
                }
                StructuredOp::SpatialDropout(c) => sample.spatial_dropout(c, &mut rng)?,
                StructuredOp::CopyAndPaste(c) => sample.copy_and_paste(c, &mut rng)?,
                StructuredOp::Mosaic(c) => sample.mosaic(c, &mut rng)?,
                StructuredOp::OverlayElements(c) => sample.overlay_elements(c, &mut rng)?,

                StructuredOp::GuidedCoarseDropout(c) => {
                    sample.guided_coarse_dropout(c, index, &mut rng)?
                }
                StructuredOp::MaskDropout(c) => sample.mask_dropout(c, &mut rng)?,
                StructuredOp::CropNonEmptyMask(c) => {
                    sample.crop_non_empty_mask_if_exists(c, &mut rng)?
                }
                StructuredOp::RandomCropNearBBox(c) => sample.random_crop_near_bbox(c, &mut rng)?,
                StructuredOp::BBoxSafeRandomCrop(c) => sample.bbox_safe_random_crop(c, &mut rng)?,
                StructuredOp::AtLeastOneBBoxRandomCrop(c) => {
                    sample.at_least_one_bbox_random_crop(c, &mut rng)?
                }
            };
        }
        sample.validate()?;
        Ok(sample)
    }
}
pub struct StructuredBatch {
    pub images: Tensor,
    pub samples: Vec<StructuredSample>,
}
pub struct StructuredLoader {
    pipeline: StructuredPipeline,
    indices: Vec<usize>,
    position: usize,
    failed: bool,
}
impl StructuredLoader {
    pub fn next_batch(&mut self) -> RivetResult<Option<StructuredBatch>> {
        if self.failed || self.position >= self.indices.len() {
            return Ok(None);
        }
        let end = self
            .position
            .saturating_add(self.pipeline.batch_size)
            .min(self.indices.len());
        if self.pipeline.drop_last && end - self.position < self.pipeline.batch_size {
            self.position = self.indices.len();
            return Ok(None);
        }
        let indices = &self.indices[self.position..end];
        self.position = end;
        let result = (|| -> RivetResult<StructuredBatch> {
            let samples = if self.pipeline.workers == 0 || indices.len() < 2 {
                indices
                    .iter()
                    .map(|&i| self.pipeline.apply(self.pipeline.samples[i].clone(), i))
                    .collect::<RivetResult<Vec<_>>>()?
            } else {
                let chunk = indices
                    .len()
                    .div_ceil(self.pipeline.workers.min(indices.len()));
                std::thread::scope(|scope| {
                    let handles = indices
                        .chunks(chunk)
                        .map(|ids| {
                            let pipeline = &self.pipeline;
                            scope.spawn(move || {
                                ids.iter()
                                    .map(|&i| pipeline.apply(pipeline.samples[i].clone(), i))
                                    .collect::<RivetResult<Vec<_>>>()
                            })
                        })
                        .collect::<Vec<_>>();
                    let mut values = Vec::with_capacity(indices.len());
                    for handle in handles {
                        values.extend(
                            handle
                                .join()
                                .map_err(|_| invalid_argument("structured worker panicked"))??,
                        );
                    }
                    Ok::<_, crate::errors::VisionError>(values)
                })?
            };
            if samples
                .iter()
                .any(|s| s.axis_order != samples[0].axis_order)
            {
                return Err(invalid_shape("structured batch layouts must match"));
            }
            let refs: Vec<_> = samples.iter().map(|s| &s.image).collect();
            let images = Tensor::stack(&refs, 0)?;
            Ok(StructuredBatch { images, samples })
        })();
        match result {
            Ok(batch) => Ok(Some(batch)),
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }
}
