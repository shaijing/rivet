use crate::TextEncoding;
use crate::errors::{TextResult, invalid};
use rivet_core::{Device, Tensor};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LabelMode {
    /// Preserve all-or-none per-sample class labels; mixed batches are rejected.
    #[default]
    Sample,
    /// Explicitly omit source labels.
    None,
    /// Unshifted token targets; the model performs the causal shift. Padding
    /// targets use ignore_index, determined by attention mask, never by token id.
    CausalLm { ignore_index: i64 },
}

#[derive(Clone, Debug)]
pub struct TextBatch {
    pub input_ids: Tensor,
    pub attention_mask: Tensor,
    pub token_type_ids: Tensor,
    pub position_ids: Tensor,
    pub labels: Option<Tensor>,
    pub lengths: Vec<usize>,
    pub source_indices: Vec<usize>,
}
impl TextBatch {
    /// Collate rectangular encodings into CPU i64 tensors. Ragged input is an
    /// error: callers must choose a padding policy rather than infer a pad id.
    pub fn collate(
        encodings: &[TextEncoding],
        labels: &[Option<i64>],
        source_indices: Vec<usize>,
        mode: LabelMode,
    ) -> TextResult<Self> {
        let rows = encodings.len();
        let width = encodings.first().map_or(0, TextEncoding::len);
        if labels.len() != rows || source_indices.len() != rows {
            return Err(invalid(
                "encodings, labels and source indices must have equal row counts",
            ));
        }
        rows.checked_mul(width)
            .ok_or_else(|| invalid("text batch shape overflow"))?;
        for encoding in encodings {
            encoding.validate()?;
            if encoding.len() != width {
                return Err(invalid(
                    "ragged text batch; configure dynamic or fixed padding",
                ));
            }
        }
        let labeled_rows = labels.iter().filter(|label| label.is_some()).count();
        if mode == LabelMode::Sample && labeled_rows != 0 && labeled_rows != rows {
            return Err(invalid(
                "a classification batch must have labels on every sample or none",
            ));
        }
        if matches!(mode, LabelMode::CausalLm { .. }) && labeled_rows != 0 {
            return Err(invalid(
                "causal LM collation requires unlabeled source samples",
            ));
        }
        let shape = [rows, width];
        let device = &Device::Cpu;
        // Typed writers fill final aligned storage without flattened staging Vecs.
        let input_ids = Tensor::from_exact_writer(shape, device, |out| {
            for encoding in encodings {
                for &id in &encoding.input_ids {
                    out.write_next(i64::from(id))?;
                }
            }
            Ok(())
        })?;
        let attention_mask = Tensor::from_exact_writer(shape, device, |out| {
            for encoding in encodings {
                for &mask in &encoding.attention_mask {
                    out.write_next(i64::from(mask))?;
                }
            }
            Ok(())
        })?;
        let token_type_ids = Tensor::from_exact_writer(shape, device, |out| {
            for encoding in encodings {
                for &id in &encoding.token_type_ids {
                    out.write_next(i64::from(id))?;
                }
            }
            Ok(())
        })?;
        let position_ids = Tensor::from_exact_writer(shape, device, |out| {
            for encoding in encodings {
                let mut position = 0i64;
                for &mask in &encoding.attention_mask {
                    if mask == 0 {
                        out.write_next(0i64)?;
                    } else {
                        out.write_next(position)?;
                        position += 1;
                    }
                }
            }
            Ok(())
        })?;
        let labels = match mode {
            LabelMode::None => None,
            LabelMode::Sample => {
                if labeled_rows == 0 {
                    None
                } else {
                    Some(Tensor::from_exact_iter(
                        labels.iter().map(|label| label.expect("validated labels")),
                        [rows],
                        device,
                    )?)
                }
            }
            LabelMode::CausalLm { ignore_index } => {
                Some(Tensor::from_exact_writer(shape, device, |out| {
                    for encoding in encodings {
                        for (&id, &mask) in encoding.input_ids.iter().zip(&encoding.attention_mask)
                        {
                            out.write_next(if mask == 0 {
                                ignore_index
                            } else {
                                i64::from(id)
                            })?;
                        }
                    }
                    Ok(())
                })?)
            }
        };
        Ok(Self {
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            labels,
            lengths: encodings
                .iter()
                .map(TextEncoding::sequence_length)
                .collect(),
            source_indices,
        })
    }
}
