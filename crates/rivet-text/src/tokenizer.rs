use std::path::Path;
use tokenizers::pipeline::{Input, Inputs, PipelineTokenizer};

use crate::errors::{TextResult, invalid, tokenizer_error};
use crate::{TextEncoding, TextSample};

pub use tokenizers::pipeline::{EncodeOptions, Override};
pub use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, TruncationDirection, TruncationParams,
    TruncationStrategy,
};

/// Shared, immutable tokenizer. Configuration is per request, so different text
/// pipelines can safely use the same tokenizer without mutating global options.
#[derive(Clone)]
pub struct TextTokenizer {
    inner: PipelineTokenizer,
    identity: u64,
}
impl TextTokenizer {
    /// Accept canonical v2 and legacy tokenizer.json files via the upstream
    /// idempotent conversion pass. No network or model download is performed.
    pub fn from_json(json: &str) -> TextResult<Self> {
        let canonical =
            tokenizers::canonicalize_str(json).map_err(|e| tokenizer_error("convert", e))?;
        let inner = tokenizers::from_json(&canonical).map_err(|e| tokenizer_error("load", e))?;
        Ok(Self::from_tokenizer(inner))
    }
    pub fn from_file(path: impl AsRef<Path>) -> TextResult<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }
    pub fn from_tokenizer(inner: PipelineTokenizer) -> Self {
        Self {
            inner,
            identity: resource_identity(),
        }
    }
    /// Process-local semantic resource identity; preserved by clone.
    pub fn identity(&self) -> u64 {
        self.identity
    }
    pub fn as_tokenizer(&self) -> &PipelineTokenizer {
        &self.inner
    }
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> TextResult<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| tokenizer_error("decode", e))
    }
    pub fn encode(&self, text: &str, options: &EncodeOptions) -> TextResult<TextEncoding> {
        Ok(self
            .encode_batch(&[TextSample::new(text)], options)?
            .remove(0))
    }
    pub fn encode_pair(
        &self,
        text: &str,
        pair: &str,
        options: &EncodeOptions,
    ) -> TextResult<TextEncoding> {
        Ok(self
            .encode_batch(&[TextSample::pair(text, pair)], options)?
            .remove(0))
    }
    pub fn padding<'a>(&'a self, options: &'a EncodeOptions) -> Option<&'a PaddingParams> {
        self.inner.resolve_padding(&options.padding)
    }
    pub fn truncation<'a>(&'a self, options: &'a EncodeOptions) -> Option<&'a TruncationParams> {
        self.inner.resolve_truncation(&options.truncation)
    }
    pub fn validate_options(&self, options: &EncodeOptions) -> TextResult<()> {
        if let Some(truncation) = self.truncation(options) {
            if truncation.max_length == 0 {
                return Err(invalid("truncation max_length must be positive"));
            }
            if truncation.stride != 0 {
                return Err(invalid(
                    "tokenizers 1.0 RC does not expose overflow windows; nonzero stride is unsupported",
                ));
            }
        }
        if let Some(padding) = self.padding(options) {
            if padding.pad_type_id > u8::MAX as u32 {
                return Err(invalid("padding type id must fit in u8"));
            }
            if padding.pad_to_multiple_of == Some(0) {
                return Err(invalid("padding multiple must be positive"));
            }
            if matches!(padding.strategy, PaddingStrategy::Fixed(0)) {
                return Err(invalid("fixed padding length must be positive"));
            }
            if self.inner.token_to_id(&padding.pad_token) != Some(padding.pad_id) {
                return Err(invalid(
                    "padding token and id must identify the same token in the vocabulary",
                ));
            }
            if let PaddingStrategy::Fixed(length) = padding.strategy {
                padded_length(length, padding.pad_to_multiple_of)?;
            }
        }
        Ok(())
    }
    /// Encode in upstream batch mode and restore input order with wait(). Padding
    /// is performed after all encodings arrive, so batch-longest has batch scope.
    pub fn encode_batch(
        &self,
        samples: &[TextSample],
        options: &EncodeOptions,
    ) -> TextResult<Vec<TextEncoding>> {
        self.validate_options(options)?;
        for sample in samples {
            if let Some(truncation) = self.truncation(options) {
                if truncation.strategy == TruncationStrategy::OnlySecond
                    && sample.text_pair.is_none()
                {
                    return Err(invalid("OnlySecond truncation requires paired text"));
                }
                let processor = self.inner.get_post_processor();
                let template = if sample.text_pair.is_some() {
                    &processor.pair
                } else {
                    &processor.single
                };
                if options.add_special_tokens && truncation.max_length < template.n_special() {
                    return Err(invalid(
                        "truncation max_length cannot fit the post-processor's special tokens",
                    ));
                }
            }
        }
        let inputs = samples
            .iter()
            .map(|sample| match &sample.text_pair {
                Some(pair) => Input::Pair(sample.text.clone(), pair.clone()),
                None => Input::Single(sample.text.clone()),
            })
            .collect();
        // Do not ask upstream to allocate padding before checked target sizing.
        let mut unpadded = options.clone();
        unpadded.padding = Override::Off;
        let encodings = self
            .inner
            .encode(Inputs::Batch(inputs), &unpadded)
            .wait()
            .map_err(|e| tokenizer_error("encode", e))?;
        let mut result = encodings
            .into_iter()
            .map(|encoding| {
                let len = encoding.len();
                TextEncoding {
                    input_ids: encoding.ids().iter().map(|id| id.id()).collect(),
                    attention_mask: encoding
                        .attention_mask()
                        .map_or_else(|| vec![1; len], <[u8]>::to_vec),
                    token_type_ids: encoding
                        .type_ids()
                        .map_or_else(|| vec![0; len], <[u8]>::to_vec),
                }
            })
            .collect::<Vec<_>>();
        if let Some(padding) = self.padding(options) {
            let length = match padding.strategy {
                PaddingStrategy::Fixed(length) => length,
                PaddingStrategy::BatchLongest => {
                    result.iter().map(TextEncoding::len).max().unwrap_or(0)
                }
            };
            let target = padded_length(length, padding.pad_to_multiple_of)?;
            for encoding in &mut result {
                encoding.pad(
                    target,
                    padding.pad_id,
                    padding.pad_type_id as u8,
                    padding.direction == PaddingDirection::Left,
                )?;
            }
        }
        Ok(result)
    }
}
pub(crate) fn padded_length(length: usize, multiple: Option<usize>) -> TextResult<usize> {
    match multiple {
        Some(0) => Err(invalid("padding multiple must be positive")),
        Some(multiple) if !length.is_multiple_of(multiple) => length
            .checked_add(multiple - length % multiple)
            .ok_or_else(|| invalid("padded sequence length overflow")),
        _ => Ok(length),
    }
}

pub(crate) fn resource_identity() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
        value.checked_add(1)
    })
    .expect("text resource identity exhausted")
}
