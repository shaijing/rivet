//! Executable vision nodes. Both the fused CPU path and the DAG executor use
//! these same statically typed kernels; no compatibility execution plan exists.
use super::op::{
    BatchConfig, BatchKernel, CompiledSampleOp, PipelineImageState, SampleContext, SourceOp,
};
use crate::errors::{RivetResult, invalid_pipeline};
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageBatch, ImageSample};
use rivet_core::Tensor;
use rivet_data::random::RandomContext;
use rivet_exec::physical::{ExecutableGraph, ExecutionValue, Morsel, PhysicalOperator};
use rivet_exec::runtime::{RuntimeError, RuntimeResult};
use rivet_plan::ValueGranularity;
use std::sync::Arc;

/// Compile-time diagnostics only. Executable kernels live on physical nodes.
#[derive(Clone)]
pub struct ImageGraphInfo {
    pub input_state: PipelineImageState,
    /// State entering the first physical batch barrier (also for DAGs).
    pub pre_batch_state: PipelineImageState,
    pub output_state: PipelineImageState,
    pub(crate) sample_count: usize,
    pub(crate) batch_count: usize,
    pub(crate) first_sample: Option<&'static str>,
    pub(crate) first_batch: Option<&'static str>,
    pub(crate) batch_native: bool,
    #[cfg(test)]
    pub(crate) sample_ops: Vec<CompiledSampleOp>,
    #[cfg(test)]
    pub(crate) batch_ops: Vec<BatchKernel>,
}
impl ImageGraphInfo {
    pub fn sample_op_count(&self) -> usize {
        self.sample_count
    }
    pub fn batch_op_count(&self) -> usize {
        self.batch_count
    }
    pub fn first_sample_op_name(&self) -> Option<&'static str> {
        self.first_sample
    }
    pub fn first_batch_op_name(&self) -> Option<&'static str> {
        self.first_batch
    }
    pub fn can_use_batch_native(&self) -> bool {
        self.batch_native
    }
}

/// Typed operator payloads are shared by physical nodes and the linear
/// executor's cached traversal. The fast path avoids erased operator calls and
/// an extra enum/program wrapper for each sample transform.
pub(crate) struct SampleNode {
    pub(crate) op: CompiledSampleOp,
    pub(crate) random: RandomContext,
}
impl SampleNode {
    #[inline]
    pub(crate) fn execute(&self, sample: ImageSample, index: usize) -> RivetResult<ImageSample> {
        self.op
            .execute(sample, &SampleContext::with_random(index, self.random))
    }
}
pub(crate) struct BatchNode {
    pub(crate) op: BatchKernel,
    pub(crate) output_layout: ImageAxisOrder,
}
impl BatchNode {
    #[inline]
    pub(crate) fn execute(&self, mut batch: ImageBatch) -> RivetResult<ImageBatch> {
        batch.images = self.op.execute(batch.images)?;
        batch.axis_order = self.output_layout;
        Ok(batch)
    }
}

pub(crate) enum ImageKernel {
    Source(SourceOp),
    Sample(Arc<SampleNode>),
    Batch { axis_order: ImageAxisOrder },
    BatchOp(Arc<BatchNode>),
    Concat { axis: usize },
    Identity,
}

#[derive(Clone)]
enum ImageValues {
    Samples(Vec<ImageSample>),
    Batch(ImageBatch),
}

fn runtime_error(error: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Message(error.to_string())
}
fn take_values(morsel: &mut Morsel) -> RuntimeResult<ImageValues> {
    if morsel.values.len() != 1 {
        return Err(runtime_error("vision node requires one image envelope"));
    }
    let value = match morsel.values.pop().expect("checked envelope") {
        ExecutionValue::Encoded(value) | ExecutionValue::Decoded(value) => value,
        _ => return Err(runtime_error("invalid vision envelope")),
    };
    let value = value
        .downcast::<ImageValues>()
        .map_err(|_| runtime_error("invalid vision value type"))?;
    Ok(Arc::try_unwrap(value).unwrap_or_else(|shared| (*shared).clone()))
}
fn put_values(morsel: &mut Morsel, values: ImageValues) {
    match &values {
        ImageValues::Samples(samples) => {
            morsel.bytes = samples
                .iter()
                .map(|sample| match sample {
                    ImageSample::Encoded(s) => s.image.len(),
                    ImageSample::Decoded(s) => s.image.logical_bytes(),
                })
                .sum();
            morsel.granularity = ValueGranularity::Sample;
            morsel.shape = None;
            morsel.dtype = None;
        }
        ImageValues::Batch(batch) => {
            morsel.bytes = batch
                .images
                .logical_bytes()
                .saturating_add(batch.labels.logical_bytes());
            morsel.shape = Some(batch.images.dims().to_vec());
            morsel.dtype = Some(batch.images.dtype());
            morsel.granularity = ValueGranularity::Batch;
        }
    }
    morsel.residency = rivet_exec::physical::MorselResidency::Host;
    let encoded = matches!(&values, ImageValues::Samples(samples) if matches!(samples.first(), Some(ImageSample::Encoded(_))));
    let value = Arc::new(values);
    morsel.values = vec![if encoded {
        ExecutionValue::Encoded(value)
    } else {
        ExecutionValue::Decoded(value)
    }];
}
fn stack(samples: Vec<ImageSample>) -> RivetResult<ImageBatch> {
    let mut builder = crate::batch::ImageBatchBuilder::with_capacity(samples.len());
    for sample in samples {
        builder.push(sample.into_decoded()?)?;
    }
    builder.finish()
}

impl PhysicalOperator for ImageKernel {
    fn name(&self) -> &str {
        match self {
            Self::Source(_) => "ImageRead",
            Self::Sample(node) => node.op.name(),
            Self::Batch { .. } => "ImageStack",
            Self::BatchOp(node) => node.op.name(),
            Self::Concat { .. } => "ImageConcat",
            Self::Identity => "ImageIdentity",
        }
    }
    fn execute(&self, mut morsel: Morsel) -> RuntimeResult<Morsel> {
        let values = match self {
            Self::Source(source) => {
                let samples = source
                    .get_many(&morsel.source_indices)
                    .map_err(runtime_error)?;
                if samples.len() != morsel.source_indices.len() {
                    return Err(runtime_error(
                        "source sample count does not match requested indices",
                    ));
                }
                ImageValues::Samples(samples)
            }
            Self::Identity => return Ok(morsel),
            Self::Sample(node) => {
                let ImageValues::Samples(samples) = take_values(&mut morsel)? else {
                    return Err(runtime_error("sample kernel received a batch"));
                };
                if samples.len() != morsel.source_indices.len() {
                    return Err(runtime_error("sample count does not match source indices"));
                }
                ImageValues::Samples(
                    samples
                        .into_iter()
                        .zip(&morsel.source_indices)
                        .map(|(sample, index)| node.execute(sample, *index))
                        .collect::<RivetResult<Vec<_>>>()
                        .map_err(runtime_error)?,
                )
            }
            Self::Batch { axis_order } => match take_values(&mut morsel)? {
                ImageValues::Samples(samples) => {
                    let mut batch = stack(samples).map_err(runtime_error)?;
                    batch.axis_order = *axis_order;
                    ImageValues::Batch(batch)
                }
                batch => batch,
            },
            Self::BatchOp(node) => {
                match take_values(&mut morsel)? {
                    ImageValues::Batch(batch) => {
                        ImageValues::Batch(node.execute(batch).map_err(runtime_error)?)
                    }
                    ImageValues::Samples(samples) => {
                        // A DAG may share this value before its explicit batch
                        // barrier. Preserve sample granularity on that edge.
                        let mut output = Vec::with_capacity(samples.len());
                        for sample in samples {
                            let mut sample = sample.into_decoded().map_err(runtime_error)?;
                            let mut images = sample.image.unsqueeze(0).map_err(runtime_error)?;
                            images = node.op.execute(images).map_err(runtime_error)?;
                            sample.image = images.squeeze(0).map_err(runtime_error)?;
                            output.push(ImageSample::Decoded(sample));
                        }
                        ImageValues::Samples(output)
                    }
                }
            }
            Self::Concat { .. } => {
                return Err(runtime_error("ImageConcat requires multiple inputs"));
            }
        };
        put_values(&mut morsel, values);
        Ok(morsel)
    }
    fn execute_inputs(&self, mut inputs: Vec<Morsel>) -> RuntimeResult<Morsel> {
        let Self::Concat { axis } = self else {
            if inputs.len() != 1 {
                return Err(runtime_error("unary vision node received multiple inputs"));
            }
            return self.execute(inputs.pop().expect("checked input"));
        };
        if inputs.len() < 2 {
            return Err(runtime_error("ImageConcat requires at least two inputs"));
        }
        let mut first = inputs.remove(0);
        if inputs.iter().any(|input| {
            input.sequence_id != first.sequence_id
                || input.source_indices != first.source_indices
                || input.granularity != first.granularity
                || input.residency != first.residency
        }) {
            return Err(runtime_error(
                "ImageConcat inputs refer to different samples or residency",
            ));
        }
        let mut values = vec![take_values(&mut first)?];
        for mut input in inputs {
            values.push(take_values(&mut input)?);
        }
        let merged = match &values[0] {
            ImageValues::Samples(samples) => {
                let mut output = Vec::with_capacity(samples.len());
                for position in 0..samples.len() {
                    let branches = values
                        .iter()
                        .map(|value| match value {
                            ImageValues::Samples(samples) => samples
                                .get(position)
                                .cloned()
                                .ok_or_else(|| {
                                    invalid_pipeline("ImageConcat sample count mismatch")
                                })?
                                .into_decoded(),
                            _ => Err(invalid_pipeline("ImageConcat granularity mismatch")),
                        })
                        .collect::<RivetResult<Vec<_>>>()
                        .map_err(runtime_error)?;
                    if branches
                        .iter()
                        .any(|sample| sample.label != branches[0].label)
                    {
                        return Err(runtime_error("ImageConcat labels must match"));
                    }
                    let tensors = branches
                        .iter()
                        .map(|sample| &sample.image)
                        .collect::<Vec<_>>();
                    output.push(ImageSample::Decoded(DecodedSample {
                        image: Tensor::cat(&tensors, *axis).map_err(runtime_error)?,
                        label: branches[0].label,
                    }));
                }
                ImageValues::Samples(output)
            }
            ImageValues::Batch(first_batch) => {
                let labels = first_batch.labels.to_vec::<i64>().map_err(runtime_error)?;
                let batches = values
                    .iter()
                    .map(|value| match value {
                        ImageValues::Batch(batch) => Ok(batch),
                        _ => Err(runtime_error("ImageConcat granularity mismatch")),
                    })
                    .collect::<RuntimeResult<Vec<_>>>()?;
                if batches
                    .iter()
                    .any(|batch| batch.axis_order != first_batch.axis_order)
                {
                    return Err(runtime_error("ImageConcat layouts must match"));
                }
                for batch in &batches[1..] {
                    if batch.labels.to_vec::<i64>().map_err(runtime_error)? != labels {
                        return Err(runtime_error("ImageConcat labels must match"));
                    }
                }
                ImageValues::Batch(ImageBatch {
                    images: Tensor::cat(
                        &batches
                            .iter()
                            .map(|batch| &batch.images)
                            .collect::<Vec<_>>(),
                        axis + 1,
                    )
                    .map_err(runtime_error)?,
                    labels: first_batch.labels.clone(),
                    axis_order: first_batch.axis_order,
                })
            }
        };
        put_values(&mut first, merged);
        Ok(first)
    }
}

pub(crate) struct ImageExecutionGraph {
    pub(crate) executable: Arc<ExecutableGraph>,
    pub(crate) info: ImageGraphInfo,
    pub(crate) source: SourceOp,
    pub(crate) batch: BatchConfig,
    pub(crate) sample_nodes: Vec<Arc<SampleNode>>,
    pub(crate) batch_nodes: Vec<Arc<BatchNode>>,
    pub(crate) linear: bool,
    pub(crate) explanation: String,
}
impl ImageExecutionGraph {
    pub(crate) fn apply_sample_ops(
        &self,
        mut sample: ImageSample,
        index: usize,
    ) -> RivetResult<DecodedSample> {
        for node in &self.sample_nodes {
            sample = node.execute(sample, index)?;
        }
        sample.into_decoded()
    }
    pub(crate) fn decode_sample(
        &self,
        mut sample: ImageSample,
        index: usize,
    ) -> RivetResult<ImageSample> {
        for node in &self.sample_nodes {
            if node.op.is_decode() {
                sample = node.execute(sample, index)?;
            }
        }
        Ok(sample)
    }
    pub(crate) fn apply_sample_transforms(
        &self,
        mut sample: ImageSample,
        index: usize,
    ) -> RivetResult<DecodedSample> {
        for node in &self.sample_nodes {
            if !node.op.is_decode() {
                sample = node.execute(sample, index)?;
            }
        }
        sample.into_decoded()
    }
    pub(crate) fn apply_batch_ops(&self, mut batch: ImageBatch) -> RivetResult<ImageBatch> {
        for node in &self.batch_nodes {
            batch = node.execute(batch)?;
        }
        Ok(batch)
    }
    pub(crate) fn output_batch(mut morsel: Morsel) -> RivetResult<ImageBatch> {
        match take_values(&mut morsel).map_err(|error| invalid_pipeline(error.to_string()))? {
            ImageValues::Batch(batch) => Ok(batch),
            _ => Err(invalid_pipeline("image graph sink must produce a batch")),
        }
    }
}
