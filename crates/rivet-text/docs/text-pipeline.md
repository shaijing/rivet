# Text preprocessing

`rivet-text` uses the locked `tokenizers 1.0.0-rc.4` inference engine. The local
reference checkout uses newer type names; this implementation uses the actual
RC APIs (`PipelineTokenizer`, `EncodeOptions`, and `EncodeHandle::wait`).
Normalization, pre-tokenization, model tokenization and special-token processing
come from the selected tokenizer JSON. Input strings are not manually split or
lowercased by Rivet.

## Classification and paired inputs

```rust,no_run
use rivet_text::{PaddingParams, TextPipeline, TextSample, TextTokenizer};

let tokenizer = TextTokenizer::from_file("tokenizer.json")?;
let pad_id = tokenizer.token_to_id("[PAD]").expect("select the model's pad token");
let source = vec![
    TextSample::new("a review").with_label(1),
    TextSample::pair("a question", "a passage").with_label(0),
];
let pipeline = TextPipeline::from_samples(source, tokenizer)
    .truncate(128)
    .padding(PaddingParams { pad_id, ..Default::default() })
    .shuffle(71)
    .epoch(0)
    .batch(16, false);

for batch in pipeline.compile()? {
    let batch = batch?;
    // CPU i64 tensors: input_ids, attention_mask, token_type_ids, position_ids.
    // labels: [batch] when all samples have class labels; otherwise None.
    // source_indices preserves attribution after selection and shuffle.
}
# Ok::<(), rivet_text::TextError>(())
```

Default request options inherit truncation/padding from the tokenizer file.
Builder methods can override or disable them per pipeline; cloned tokenizers
share immutable model state, not mutable request options. `TextTokenizer::encode`,
`encode_pair`, `encode_batch`, and `decode` are also usable without a loader.

`PaddingStrategy::BatchLongest` pads each batch to its own longest sequence.
`Fixed(n)` requires sequences no longer than the padding target, with optional
rounding to `pad_to_multiple_of`. Padding does not truncate; choose truncation
explicitly. Left and right padding are supported. Rectangular inputs need no
padding, but ragged collation without an explicit pad policy returns an error.
Padding token/id mismatches, zero multiples, type IDs outside u8, and length
arithmetic overflow are rejected.

Truncation is delegated before adding special tokens and supports left/right,
longest-first and pair-only strategies. Max length must fit the selected special
frame. Nonzero stride is rejected: the current upstream inference API does not
return overflow windows. Token offsets are likewise not supplied by this API.

## Arrow IPC streams

```rust,no_run
use std::sync::Arc;
use rivet_text::{ArrowTextDataset, TextColumns, TextPipeline, TextTokenizer};
let dataset = ArrowTextDataset::open_ipc_stream("sst5-train.arrow", TextColumns {
    text: "text".into(), label: Some("label".into()), text_pair: None,
})?;
let tokenizer = TextTokenizer::from_file("tokenizer.json")?;
let loader = TextPipeline::new(Arc::new(dataset), tokenizer).batch(32, false).compile()?;
# Ok::<(), rivet_text::TextError>(())
```

The source accepts `Utf8`/`LargeUtf8` text and optional paired-text columns, plus
an optional `Int64` class label. Nulls return errors with column and source-row
context. `from_batches` accepts existing `RecordBatch` values; `from_ipc_stream`
and `open_ipc_stream` eagerly retain all IPC stream batches. Requested rows
become owned strings. This is not a bounded-memory streaming reader. IPC file
format readers can feed `from_batches` through Arrow's `FileReader` separately.

Any `rivet_data::dataset::Dataset<Item = TextSample>` can back a pipeline.
Requests preserve order and repeated indices. `.indices(...)` selects rows
before optional epoch-seeded shuffle. Dropped partial batches are not fetched.
A source, tokenization or collation failure produces one error with source
indices and terminates the loader. Source batches are pulled synchronously;
upstream tokenizers manages its own parallel encoding. `wait()` restores input
order after parallel completion.

## Causal LM labels and packing

Use `.labels(LabelMode::CausalLm { ignore_index: -100 })` with unlabeled samples
to produce **unshifted** `[batch, sequence]` token targets; the model performs the
causal shift. Only attention-mask zeros become ignored targets. An EOS token
used as the pad ID remains trainable wherever attention is one. `position_ids`
count active tokens from zero, including with left padding. Mixed labeled and
unlabeled classification rows are rejected; `LabelMode::None` explicitly omits
sample labels.

`pack_encodings(&encodings, PackingOptions { block_size, separator_id,
drop_last })` removes masked padding, appends the configured separator after
every input document, and emits consecutive token blocks. Separator insertion
is explicit and may duplicate an existing EOS. A retained final short block is
unpadded. This is a continuous LM stream that allows attention across document
boundaries; it is not segmented attention. Feed rectangular packed blocks to
`TextBatch::collate` with `LabelMode::CausalLm`. Packing is an eager standalone
operation; classification labels are not reassigned to packed blocks.

## Planning and validation

`to_logical_plan()` exposes a source → tokenize → collate → sink snapshot.
`TextPropertyInference` reports raw text, variable-length token encodings, and
contiguous i64 batch tensors. Padding/label options, source selection, shuffle
seed and epoch are retained as semantic metadata; process-local resource IDs
keep distinct sources and tokenizer models distinct. Payloads contain no runtime
tokenizer handles or dataset buffers. `compile()` validates properties before
constructing the fixed CPU pull loader; execution of arbitrary edited DAGs is
not implemented.

Tensor collation uses typed writers directly into final aligned allocations.
The tests use a tiny checked-in tokenizer, so validation and the example require
no model downloads:

```bash
cargo test -j 12 -p rivet-text --lib --test workflows
cargo run -j 12 -p rivet-text --example preprocess_text
# Optional: pass a compatible tokenizer JSON after --.
```
