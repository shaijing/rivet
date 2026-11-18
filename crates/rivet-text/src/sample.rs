use crate::errors::{TextError, TextResult, invalid};

/// One source row. Text is preserved verbatim for the tokenizer's normalizer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextSample {
    pub text: String,
    pub text_pair: Option<String>,
    pub label: Option<i64>,
}
impl TextSample {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            text_pair: None,
            label: None,
        }
    }
    pub fn pair(text: impl Into<String>, pair: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            text_pair: Some(pair.into()),
            label: None,
        }
    }
    pub fn with_label(mut self, label: i64) -> Self {
        self.label = Some(label);
        self
    }
}

/// Model-ready token fields. Upstream 1.0 RC does not expose offsets or overflow
/// windows; no synthetic offsets or silently dropped overflow windows are supplied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextEncoding {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub token_type_ids: Vec<u8>,
}
impl TextEncoding {
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }
    pub fn sequence_length(&self) -> usize {
        self.attention_mask
            .iter()
            .filter(|&&mask| mask == 1)
            .count()
    }
    pub fn validate(&self) -> TextResult<()> {
        if self.attention_mask.len() != self.len() || self.token_type_ids.len() != self.len() {
            return Err(invalid(
                "token ids, attention mask and type ids must have equal lengths",
            ));
        }
        if self.attention_mask.iter().any(|&mask| mask > 1) {
            return Err(invalid("attention masks must contain only 0 or 1"));
        }
        Ok(())
    }
    pub(crate) fn pad(
        &mut self,
        target: usize,
        id: u32,
        type_id: u8,
        left: bool,
    ) -> TextResult<()> {
        if target < self.len() {
            return Err(invalid(
                "fixed padding length is shorter than an encoding; configure truncation explicitly",
            ));
        }
        let count = target - self.len();
        self.input_ids
            .try_reserve(count)
            .map_err(|e| TextError::Allocation(e.to_string()))?;
        self.attention_mask
            .try_reserve(count)
            .map_err(|e| TextError::Allocation(e.to_string()))?;
        self.token_type_ids
            .try_reserve(count)
            .map_err(|e| TextError::Allocation(e.to_string()))?;
        self.input_ids.resize(target, id);
        self.attention_mask.resize(target, 0);
        self.token_type_ids.resize(target, type_id);
        if left {
            self.input_ids.rotate_right(count);
            self.attention_mask.rotate_right(count);
            self.token_type_ids.rotate_right(count);
        }
        Ok(())
    }
}
