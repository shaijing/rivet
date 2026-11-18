use crate::errors::invalid;
use crate::{TextEncoding, TextResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackingOptions {
    pub block_size: usize,
    /// Append this id after each document, even when it already ends with it.
    /// None concatenates raw token streams without inserting a separator.
    pub separator_id: Option<u32>,
    pub drop_last: bool,
}
/// Continuous causal-LM stream packing: remove padding by attention mask,
/// append explicit document separators, and emit fixed token blocks. Attention
/// across document boundaries is intentionally allowed; this is NOT segmented
/// or block-diagonal attention. The optional final short block remains unpadded.
pub fn pack_encodings(
    encodings: &[TextEncoding],
    options: PackingOptions,
) -> TextResult<Vec<TextEncoding>> {
    if options.block_size == 0 {
        return Err(invalid("packing block_size must be positive"));
    }
    for encoding in encodings {
        encoding.validate()?;
    }
    let mut blocks = Vec::new();
    let mut current = TextEncoding {
        input_ids: Vec::new(),
        attention_mask: Vec::new(),
        token_type_ids: Vec::new(),
    };
    let emit =
        |current: &mut TextEncoding, blocks: &mut Vec<TextEncoding>, id: u32, type_id: u8| {
            current.input_ids.push(id);
            current.attention_mask.push(1);
            current.token_type_ids.push(type_id);
            if current.len() == options.block_size {
                blocks.push(std::mem::replace(
                    current,
                    TextEncoding {
                        input_ids: Vec::new(),
                        attention_mask: Vec::new(),
                        token_type_ids: Vec::new(),
                    },
                ));
            }
        };
    for encoding in encodings {
        for ((&id, &mask), &type_id) in encoding
            .input_ids
            .iter()
            .zip(&encoding.attention_mask)
            .zip(&encoding.token_type_ids)
        {
            if mask == 1 {
                emit(&mut current, &mut blocks, id, type_id);
            }
        }
        if let Some(id) = options.separator_id {
            emit(&mut current, &mut blocks, id, 0);
        }
    }
    if !options.drop_last && !current.is_empty() {
        blocks.push(current);
    }
    Ok(blocks)
}
