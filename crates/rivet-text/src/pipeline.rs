use crate::errors::invalid;
use crate::{
    EncodeOptions, LabelMode, Override, PaddingParams, TextBatch, TextError, TextPropertyInference,
    TextResult, TextSample, TextTokenizer, TruncationParams,
};
use rivet_data::dataset::Dataset;
use rivet_data::dataset::memory::MemoryDataset;
use rivet_data::dataset::source::validate_indices;
use rivet_data::sampler::{IndexSampler, SamplerPlan, permute};
use std::sync::Arc;

/// A CPU text pipeline. The tokenizer owns its internal batch parallelism;
/// source batches are pulled synchronously, in sampler order, without a second
/// pool. Each batch is padded independently, so dynamic width may vary by batch.
#[derive(Clone)]
pub struct TextPipeline {
    pub(crate) source: Arc<dyn Dataset<Item = TextSample>>,
    pub(crate) source_identity: u64,
    pub(crate) tokenizer: TextTokenizer,
    pub(crate) options: EncodeOptions,
    pub(crate) batch_size: usize,
    pub(crate) drop_last: bool,
    pub(crate) indices: Option<Vec<usize>>,
    pub(crate) shuffle_seed: Option<u64>,
    pub(crate) epoch: u64,
    pub(crate) label_mode: LabelMode,
}
impl TextPipeline {
    pub fn new(source: Arc<dyn Dataset<Item = TextSample>>, tokenizer: TextTokenizer) -> Self {
        Self {
            source,
            source_identity: crate::tokenizer::resource_identity(),
            tokenizer,
            options: EncodeOptions::default(),
            batch_size: 32,
            drop_last: false,
            indices: None,
            shuffle_seed: None,
            epoch: 0,
            label_mode: LabelMode::Sample,
        }
    }
    pub fn from_samples(samples: Vec<TextSample>, tokenizer: TextTokenizer) -> Self {
        Self::new(Arc::new(MemoryDataset::new(samples)), tokenizer)
    }
    pub fn encode_options(mut self, options: EncodeOptions) -> Self {
        self.options = options;
        self
    }
    pub fn add_special_tokens(mut self, enabled: bool) -> Self {
        self.options.add_special_tokens = enabled;
        self
    }
    pub fn truncate(mut self, max_length: usize) -> Self {
        self.options.truncation = Override::With(TruncationParams {
            max_length,
            ..Default::default()
        });
        self
    }
    pub fn truncation(mut self, params: TruncationParams) -> Self {
        self.options.truncation = Override::With(params);
        self
    }
    pub fn no_truncation(mut self) -> Self {
        self.options.truncation = Override::Off;
        self
    }
    pub fn padding(mut self, params: PaddingParams) -> Self {
        self.options.padding = Override::With(params);
        self
    }
    pub fn no_padding(mut self) -> Self {
        self.options.padding = Override::Off;
        self
    }
    pub fn batch(mut self, size: usize, drop_last: bool) -> Self {
        self.batch_size = size;
        self.drop_last = drop_last;
        self
    }
    /// Preserve the supplied order and duplicate indices, before optional shuffle.
    pub fn indices(mut self, indices: Vec<usize>) -> Self {
        self.indices = Some(indices);
        self
    }
    pub fn shuffle(mut self, seed: u64) -> Self {
        self.shuffle_seed = Some(seed);
        self
    }
    pub fn epoch(mut self, epoch: u64) -> Self {
        self.epoch = epoch;
        self
    }
    pub fn labels(mut self, mode: LabelMode) -> Self {
        self.label_mode = mode;
        self
    }
    pub fn compile(self) -> TextResult<TextDataLoader> {
        if self.batch_size == 0 {
            return Err(invalid("batch size must be positive"));
        }
        self.tokenizer.validate_options(&self.options)?;
        let plan = self.to_logical_plan()?;
        plan.infer_properties(&TextPropertyInference)
            .map_err(|e| TextError::Plan(e.to_string()))?;
        // Sequential sources avoid an O(dataset length) index allocation.
        let sampler_plan = match (self.indices, self.shuffle_seed) {
            (Some(indices), seed) => {
                validate_indices(&indices, self.source.len())?;
                let indices = if let Some(seed) = seed {
                    permute(indices.len(), rivet_data::random::combine(seed, self.epoch))
                        .into_iter()
                        .map(|index| indices[index])
                        .collect()
                } else {
                    indices
                };
                SamplerPlan::Permutation { indices }
            }
            (None, Some(seed)) => SamplerPlan::Permutation {
                indices: permute(
                    self.source.len(),
                    rivet_data::random::combine(seed, self.epoch),
                ),
            },
            (None, None) => SamplerPlan::Sequential {
                start: 0,
                end: self.source.len(),
            },
        };
        let remaining = sampler_plan.len();
        Ok(TextDataLoader {
            source: self.source,
            tokenizer: self.tokenizer,
            options: self.options,
            sampler: IndexSampler::new(sampler_plan, 0),
            remaining,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            label_mode: self.label_mode,
            finished: false,
        })
    }
}
pub struct TextDataLoader {
    source: Arc<dyn Dataset<Item = TextSample>>,
    tokenizer: TextTokenizer,
    options: EncodeOptions,
    sampler: IndexSampler,
    remaining: usize,
    batch_size: usize,
    drop_last: bool,
    label_mode: LabelMode,
    finished: bool,
}
impl TextDataLoader {
    pub fn next_batch(&mut self) -> TextResult<Option<TextBatch>> {
        if self.finished {
            return Ok(None);
        }
        let Some(indices) = self.sampler.next_indices(self.batch_size) else {
            self.finished = true;
            self.remaining = 0;
            return Ok(None);
        };
        self.remaining -= indices.len();
        if self.drop_last && indices.len() < self.batch_size {
            self.finished = true;
            return Ok(None);
        }
        let result = (|| {
            let samples = self.source.get_many(&indices)?;
            if samples.len() != indices.len() {
                return Err(invalid(format!(
                    "source returned {} rows for {} indices",
                    samples.len(),
                    indices.len()
                )));
            }
            let encodings = self.tokenizer.encode_batch(&samples, &self.options)?;
            let labels = samples
                .iter()
                .map(|sample| sample.label)
                .collect::<Vec<_>>();
            TextBatch::collate(&encodings, &labels, indices.clone(), self.label_mode)
        })();
        match result {
            Ok(batch) => Ok(Some(batch)),
            Err(source) => {
                self.finished = true;
                self.remaining = 0;
                Err(TextError::Batch {
                    indices,
                    source: Box::new(source),
                })
            }
        }
    }
}
impl Iterator for TextDataLoader {
    type Item = TextResult<TextBatch>;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch().transpose()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let batches = if self.finished {
            0
        } else if self.drop_last {
            self.remaining / self.batch_size
        } else {
            self.remaining.div_ceil(self.batch_size)
        };
        // A failing batch produces one error and terminates iteration.
        (0, Some(batches))
    }
}
impl std::iter::FusedIterator for TextDataLoader {}
