use arrow::array::{ArrayRef, Int64Array, LargeStringArray, StringArray};
use arrow::record_batch::RecordBatch;
use arrow_ipc::writer::StreamWriter;
use rivet_core::DType;
use rivet_data::dataset::Dataset;
use rivet_data::errors::DataResult;
use rivet_text::*;
use std::io::Cursor;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

const JSON: &str = include_str!("fixtures/wordlevel.json");
fn tokenizer() -> TextTokenizer {
    TextTokenizer::from_json(JSON).unwrap()
}
fn padding() -> PaddingParams {
    PaddingParams::default()
}
fn sample(text: &str) -> TextSample {
    TextSample::new(text)
}
fn encoding(ids: &[u32], masks: &[u8]) -> TextEncoding {
    TextEncoding {
        input_ids: ids.to_vec(),
        attention_mask: masks.to_vec(),
        token_type_ids: vec![0; ids.len()],
    }
}

#[test]
fn loads_legacy_json_and_runs_normalization_model_and_special_processing() {
    let tokenizer = tokenizer();
    assert_eq!(
        tokenizer
            .encode("HELLO Cafe\u{301} 中 absent", &EncodeOptions::default())
            .unwrap()
            .input_ids,
        [2, 4, 8, 11, 1, 3]
    );
    assert_eq!(
        tokenizer.decode(&[2, 4, 5, 3], true).unwrap(),
        "hello world"
    );
    assert!(
        tokenizer
            .decode(&[2, 4, 3], false)
            .unwrap()
            .contains("[CLS]")
    );
    assert_eq!(tokenizer.token_to_id("hello"), Some(4));
}
#[test]
fn file_loading_and_canonical_json_have_the_same_output() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wordlevel.json");
    let from_file = TextTokenizer::from_file(path).unwrap();
    let canonical = tokenizers::canonicalize_str(JSON).unwrap();
    let from_canonical = TextTokenizer::from_json(&canonical).unwrap();
    assert_eq!(
        from_file
            .encode("hello world", &EncodeOptions::default())
            .unwrap(),
        from_canonical
            .encode("hello world", &EncodeOptions::default())
            .unwrap()
    );
    assert!(TextTokenizer::from_json("invalid").is_err());
    assert!(TextTokenizer::from_file("/this/tokenizer/does/not/exist.json").is_err());
}
#[test]
fn pair_encoding_preserves_sequence_type_ids_and_special_tokens() {
    let result = tokenizer()
        .encode_pair("hello", "world", &EncodeOptions::default())
        .unwrap();
    assert_eq!(result.input_ids, [2, 4, 3, 5, 3]);
    assert_eq!(result.token_type_ids, [0, 0, 0, 1, 1]);
    assert_eq!(result.attention_mask, [1; 5]);
    let bare = tokenizer()
        .encode_pair("hello", "world", &EncodeOptions::no_specials())
        .unwrap();
    assert_eq!(bare.input_ids, [4, 5]);
    assert_eq!(bare.token_type_ids, [0, 1]);
}
#[test]
fn truncation_preserves_special_tokens_and_validates_unsupported_options() {
    let mut options = EncodeOptions {
        truncation: Override::With(TruncationParams {
            max_length: 3,
            ..Default::default()
        }),
        ..Default::default()
    };
    assert_eq!(
        tokenizer()
            .encode("hello world", &options)
            .unwrap()
            .input_ids,
        [2, 4, 3]
    );
    if let Override::With(truncation) = &mut options.truncation {
        truncation.direction = TruncationDirection::Left;
    }
    assert_eq!(
        tokenizer()
            .encode("hello world", &options)
            .unwrap()
            .input_ids,
        [2, 5, 3]
    );
    options.truncation = Override::With(TruncationParams {
        max_length: 2,
        ..Default::default()
    });
    assert_eq!(
        tokenizer().encode("hello", &options).unwrap().input_ids,
        [2, 3]
    );
    for (length, stride) in [(0, 0), (1, 0), (4, 1)] {
        options.truncation = Override::With(TruncationParams {
            max_length: length,
            stride,
            ..Default::default()
        });
        assert!(tokenizer().encode("hello", &options).is_err());
    }
    options.truncation = Override::With(TruncationParams {
        max_length: 6,
        strategy: TruncationStrategy::OnlySecond,
        ..Default::default()
    });
    assert!(tokenizer().encode("hello", &options).is_err());
    assert_eq!(
        tokenizer()
            .encode_pair("hello world", "good bad", &options)
            .unwrap()
            .input_ids
            .len(),
        6
    );
}
#[test]
fn dynamic_and_fixed_padding_support_left_right_and_multiples() {
    let samples = [sample("hello world"), sample("good")];
    let mut options = EncodeOptions {
        padding: Override::With(padding()),
        ..Default::default()
    };
    let result = tokenizer().encode_batch(&samples, &options).unwrap();
    assert_eq!(result[1].input_ids, [2, 6, 3, 0]);
    assert_eq!(result[1].attention_mask, [1, 1, 1, 0]);
    options.padding = Override::With(PaddingParams {
        strategy: PaddingStrategy::Fixed(5),
        direction: PaddingDirection::Left,
        pad_to_multiple_of: Some(4),
        ..padding()
    });
    let result = tokenizer().encode_batch(&samples, &options).unwrap();
    assert_eq!(result[1].input_ids, [0, 0, 0, 0, 0, 2, 6, 3]);
    assert_eq!(result[1].attention_mask, [0, 0, 0, 0, 0, 1, 1, 1]);
    assert_eq!(result[1].sequence_length(), 3);
}
#[test]
fn padding_never_silently_truncates_and_rejects_invalid_configuration() {
    for params in [
        PaddingParams {
            strategy: PaddingStrategy::Fixed(2),
            ..padding()
        },
        PaddingParams {
            strategy: PaddingStrategy::Fixed(0),
            ..padding()
        },
        PaddingParams {
            pad_to_multiple_of: Some(0),
            ..padding()
        },
        PaddingParams {
            pad_type_id: 256,
            ..padding()
        },
        PaddingParams {
            pad_id: 999,
            ..padding()
        },
        PaddingParams {
            strategy: PaddingStrategy::Fixed(usize::MAX),
            pad_to_multiple_of: Some(2),
            ..padding()
        },
    ] {
        assert!(
            tokenizer()
                .encode(
                    "hello",
                    &EncodeOptions {
                        padding: Override::With(params),
                        ..Default::default()
                    }
                )
                .is_err()
        );
    }
}
#[test]
fn inherited_configuration_can_be_overridden_without_mutating_the_tokenizer() {
    let json = JSON.replace("\"padding\": null", "\"padding\": {\"strategy\":{\"Fixed\":6},\"direction\":\"Right\",\"pad_to_multiple_of\":null,\"pad_id\":0,\"pad_type_id\":0,\"pad_token\":\"[PAD]\"}");
    let tokenizer = TextTokenizer::from_json(&json).unwrap();
    assert_eq!(
        tokenizer
            .encode("hello", &EncodeOptions::default())
            .unwrap()
            .len(),
        6
    );
    assert_eq!(
        tokenizer
            .encode(
                "hello",
                &EncodeOptions {
                    padding: Override::Off,
                    ..Default::default()
                }
            )
            .unwrap()
            .len(),
        3
    );
    assert_eq!(
        tokenizer
            .encode("hello", &EncodeOptions::default())
            .unwrap()
            .len(),
        6
    );
}
#[test]
fn upstream_parallel_batch_completion_is_restored_to_input_order() {
    let samples = (0..80)
        .map(|i| {
            sample(&if i % 2 == 0 {
                "hello ".repeat(256)
            } else {
                "world ".repeat(128)
            })
        })
        .collect::<Vec<_>>();
    let result = tokenizer()
        .encode_batch(&samples, &EncodeOptions::default())
        .unwrap();
    assert_eq!(result.len(), samples.len());
    for (i, encoding) in result.iter().enumerate() {
        assert_eq!(encoding.input_ids[1], if i % 2 == 0 { 4 } else { 5 });
        assert_eq!(encoding.len(), if i % 2 == 0 { 258 } else { 130 });
    }
}
#[test]
fn shared_tokenizer_request_options_are_thread_safe() {
    let tokenizer = tokenizer();
    std::thread::scope(|scope| {
        let bare = scope.spawn(|| {
            tokenizer
                .encode("hello", &EncodeOptions::no_specials())
                .unwrap()
        });
        let special = scope.spawn(|| {
            tokenizer
                .encode("hello", &EncodeOptions::default())
                .unwrap()
        });
        assert_eq!(bare.join().unwrap().input_ids, [4]);
        assert_eq!(special.join().unwrap().input_ids, [2, 4, 3]);
    });
}
#[test]
fn classification_batches_produce_i64_tensors_and_keep_labels_and_indices() {
    let samples = vec![
        sample("hello world").with_label(4),
        sample("good").with_label(2),
        sample("bad").with_label(0),
    ];
    let mut loader = TextPipeline::from_samples(samples, tokenizer())
        .padding(padding())
        .indices(vec![2, 0, 2])
        .batch(2, false)
        .compile()
        .unwrap();
    let batch = loader.next().unwrap().unwrap();
    assert_eq!(batch.source_indices, [2, 0]);
    assert_eq!(batch.input_ids.dims(), [2, 4]);
    assert_eq!(batch.input_ids.dtype(), DType::I64);
    assert_eq!(
        batch.input_ids.to_vec::<i64>().unwrap(),
        [2, 7, 3, 0, 2, 4, 5, 3]
    );
    assert_eq!(batch.labels.unwrap().to_vec::<i64>().unwrap(), [0, 4]);
    assert_eq!(batch.lengths, [3, 4]);
    assert_eq!(loader.next().unwrap().unwrap().source_indices, [2]);
    assert!(loader.next().is_none());
}
#[test]
fn causal_labels_mask_padding_even_when_pad_id_is_a_real_eos_id() {
    let batch =
        TextPipeline::from_samples(vec![sample("hello"), sample("hello world")], tokenizer())
            .padding(PaddingParams {
                pad_id: 3,
                pad_token: "[SEP]".into(),
                direction: PaddingDirection::Left,
                ..padding()
            })
            .labels(LabelMode::CausalLm { ignore_index: -100 })
            .batch(2, false)
            .compile()
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
    assert_eq!(
        batch.input_ids.to_vec::<i64>().unwrap(),
        [3, 2, 4, 3, 2, 4, 5, 3]
    );
    assert_eq!(
        batch.labels.unwrap().to_vec::<i64>().unwrap(),
        [-100, 2, 4, 3, 2, 4, 5, 3]
    );
    assert_eq!(
        batch.position_ids.to_vec::<i64>().unwrap(),
        [0, 0, 1, 2, 0, 1, 2, 3]
    );
}
#[test]
fn ragged_and_mixed_label_batches_error_once_with_source_indices() {
    for samples in [
        vec![sample("hello"), sample("hello world")],
        vec![sample("hello").with_label(1), sample("world")],
    ] {
        let mut loader = TextPipeline::from_samples(samples, tokenizer())
            .batch(2, false)
            .compile()
            .unwrap();
        let error = loader.next().unwrap().unwrap_err().to_string();
        assert!(error.contains("[0, 1]"));
        assert!(loader.next().is_none());
        assert!(loader.next_batch().unwrap().is_none());
    }
    assert!(
        TextPipeline::from_samples(vec![sample("hello").with_label(1)], tokenizer())
            .labels(LabelMode::CausalLm { ignore_index: -100 })
            .compile()
            .unwrap()
            .next()
            .unwrap()
            .is_err()
    );
}
#[test]
fn seeded_shuffle_is_reproducible_and_epoch_changes_the_order() {
    fn order(epoch: u64) -> Vec<usize> {
        TextPipeline::from_samples((0..20).map(|_| sample("hello")).collect(), tokenizer())
            .shuffle(17)
            .epoch(epoch)
            .batch(3, false)
            .compile()
            .unwrap()
            .flat_map(|b| b.unwrap().source_indices)
            .collect()
    }
    assert_eq!(order(0), order(0));
    assert_ne!(order(0), order(1));
    let mut sorted = order(1);
    sorted.sort_unstable();
    assert_eq!(sorted, (0..20).collect::<Vec<_>>());
}
struct CountingSource {
    calls: AtomicUsize,
    wrong_count: bool,
}
impl Dataset for CountingSource {
    type Item = TextSample;
    fn len(&self) -> usize {
        3
    }
    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<TextSample>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(indices
            .iter()
            .skip(usize::from(self.wrong_count))
            .map(|_| sample("hello"))
            .collect())
    }
}
#[test]
fn drop_last_does_not_fetch_discarded_rows_and_source_failures_are_terminal() {
    let source = Arc::new(CountingSource {
        calls: AtomicUsize::new(0),
        wrong_count: false,
    });
    let mut loader = TextPipeline::new(source.clone(), tokenizer())
        .batch(2, true)
        .compile()
        .unwrap();
    assert!(loader.next().unwrap().is_ok());
    assert!(loader.next().is_none());
    assert_eq!(source.calls.load(Ordering::Relaxed), 1);
    let source = Arc::new(CountingSource {
        calls: AtomicUsize::new(0),
        wrong_count: true,
    });
    let mut loader = TextPipeline::new(source, tokenizer())
        .batch(2, false)
        .compile()
        .unwrap();
    assert!(
        loader
            .next()
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("source returned")
    );
    assert!(loader.next().is_none());
}
#[test]
fn empty_and_invalid_pipeline_configuration_are_checked() {
    let mut empty = TextPipeline::from_samples(vec![], tokenizer())
        .compile()
        .unwrap();
    assert!(empty.next().is_none());
    assert!(
        TextPipeline::from_samples(vec![], tokenizer())
            .batch(0, false)
            .compile()
            .is_err()
    );
    assert!(
        TextPipeline::from_samples(vec![sample("hello")], tokenizer())
            .indices(vec![1])
            .compile()
            .is_err()
    );
    let batch = TextBatch::collate(&[], &[], vec![], LabelMode::Sample).unwrap();
    assert_eq!(batch.input_ids.dims(), [0, 0]);
    let empty = tokenizer()
        .encode("", &EncodeOptions::no_specials())
        .unwrap();
    assert!(empty.is_empty());
}
fn arrow_batch(texts: &[Option<&str>], labels: &[Option<i64>]) -> RecordBatch {
    RecordBatch::try_from_iter(vec![
        (
            "text",
            Arc::new(StringArray::from(texts.to_vec())) as ArrayRef,
        ),
        (
            "label",
            Arc::new(Int64Array::from(labels.to_vec())) as ArrayRef,
        ),
    ])
    .unwrap()
}
fn columns() -> TextColumns {
    TextColumns {
        label: Some("label".into()),
        ..Default::default()
    }
}
#[test]
fn arrow_sources_preserve_order_duplicates_empty_batches_and_labels() {
    let a = arrow_batch(&[], &[]);
    let b = arrow_batch(&[Some("hello"), Some("world")], &[Some(1), Some(2)]);
    let c = arrow_batch(&[Some("good")], &[Some(3)]);
    let dataset = ArrowTextDataset::from_batches(vec![a, b, c], columns()).unwrap();
    let rows = dataset.get_many(&[2, 0, 2]).unwrap();
    assert_eq!(
        rows.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(),
        ["good", "hello", "good"]
    );
    assert_eq!(
        rows.iter().map(|r| r.label).collect::<Vec<_>>(),
        [Some(3), Some(1), Some(3)]
    );
    assert!(dataset.get_many(&[]).unwrap().is_empty());
    assert!(dataset.get_many(&[3]).is_err());
    let batch = TextPipeline::new(Arc::new(dataset), tokenizer())
        .batch(3, false)
        .compile()
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(batch.labels.unwrap().to_vec::<i64>().unwrap(), [1, 2, 3]);
}
#[test]
fn arrow_large_strings_pairs_nulls_and_schema_errors_are_explicit() {
    let batch = RecordBatch::try_from_iter(vec![
        (
            "text",
            Arc::new(LargeStringArray::from(vec!["hello"])) as ArrayRef,
        ),
        (
            "pair",
            Arc::new(StringArray::from(vec!["world"])) as ArrayRef,
        ),
    ])
    .unwrap();
    let dataset = ArrowTextDataset::from_batches(
        vec![batch],
        TextColumns {
            text_pair: Some("pair".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        dataset.get_many(&[0]).unwrap()[0].text_pair.as_deref(),
        Some("world")
    );
    for batch in [
        arrow_batch(&[None], &[Some(1)]),
        arrow_batch(&[Some("hello")], &[None]),
    ] {
        let dataset = ArrowTextDataset::from_batches(vec![batch], columns()).unwrap();
        assert!(
            dataset
                .get_many(&[0])
                .unwrap_err()
                .to_string()
                .contains("source row 0")
        );
    }
    let batch = arrow_batch(&[Some("hello")], &[Some(1)]);
    assert!(
        ArrowTextDataset::from_batches(
            vec![batch],
            TextColumns {
                text: "missing".into(),
                ..columns()
            }
        )
        .is_err()
    );
}
#[test]
fn arrow_ipc_stream_to_model_batch_works_without_external_files() {
    let batch = arrow_batch(&[Some("hello"), Some("world")], &[Some(1), Some(0)]);
    let mut bytes = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut bytes, &batch.schema()).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }
    let source = ArrowTextDataset::from_ipc_stream(Cursor::new(bytes), columns()).unwrap();
    let batch = TextPipeline::new(Arc::new(source), tokenizer())
        .batch(2, false)
        .compile()
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(batch.input_ids.dims(), [2, 3]);
}
#[test]
fn packing_removes_only_padding_and_preserves_order_and_explicit_boundaries() {
    let a = encoding(&[0, 2, 4, 3], &[0, 1, 1, 1]);
    let b = encoding(&[2, 5, 3, 0], &[1, 1, 1, 0]);
    let options = PackingOptions {
        block_size: 4,
        separator_id: Some(3),
        drop_last: false,
    };
    let blocks = pack_encodings(&[a, b], options).unwrap();
    assert_eq!(
        blocks
            .iter()
            .map(|b| b.input_ids.clone())
            .collect::<Vec<_>>(),
        [vec![2, 4, 3, 3], vec![2, 5, 3, 3]]
    );
    assert!(blocks.iter().all(|b| b.attention_mask == [1; 4]));
    let tail = pack_encodings(
        &[encoding(&[4, 5, 6], &[1, 1, 1])],
        PackingOptions {
            block_size: 2,
            separator_id: None,
            drop_last: false,
        },
    )
    .unwrap();
    assert_eq!(tail[1].input_ids, [6]);
    let dropped = pack_encodings(
        &[encoding(&[4, 5, 6], &[1, 1, 1])],
        PackingOptions {
            block_size: 2,
            separator_id: None,
            drop_last: true,
        },
    )
    .unwrap();
    assert_eq!(dropped.len(), 1);
    assert!(
        pack_encodings(
            &[],
            PackingOptions {
                block_size: 0,
                ..options
            }
        )
        .is_err()
    );
}
#[test]
fn malformed_encodings_are_rejected_by_collation_and_packing() {
    for bad in [
        TextEncoding {
            input_ids: vec![4],
            attention_mask: vec![],
            token_type_ids: vec![0],
        },
        encoding(&[4], &[2]),
    ] {
        assert!(
            TextBatch::collate(
                std::slice::from_ref(&bad),
                &[None],
                vec![0],
                LabelMode::Sample
            )
            .is_err()
        );
        assert!(
            pack_encodings(
                &[bad],
                PackingOptions {
                    block_size: 2,
                    separator_id: None,
                    drop_last: false
                }
            )
            .is_err()
        );
    }
}
#[test]
fn logical_plan_reports_batch_shape_dtype_and_current_revision() {
    let pipeline = TextPipeline::from_samples(vec![sample("hello")], tokenizer())
        .truncate(4)
        .padding(PaddingParams {
            strategy: PaddingStrategy::Fixed(4),
            ..padding()
        })
        .batch(2, true);
    let plan = pipeline.to_logical_plan().unwrap();
    let facts = plan.infer_properties(&TextPropertyInference).unwrap();
    let output = facts.get(plan.root().unwrap()).unwrap();
    assert_eq!(output.dtype, Some(rivet_plan::DataType::I64));
    assert_eq!(
        output.granularity,
        Some(rivet_plan::ValueGranularity::Batch)
    );
    assert_eq!(
        output.shape,
        Some(rivet_plan::ValueShape(vec![
            rivet_plan::ShapeDim::Known(2),
            rivet_plan::ShapeDim::Known(4)
        ]))
    );
    assert!(facts.is_valid_for(&plan));
    assert!(pipeline.explain().unwrap().contains("Tokenize"));
}

#[test]
fn common_wordpiece_bpe_and_unigram_models_use_the_selected_model() {
    let cases = [
        (
            r###"{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,"vocab":{"[UNK]":0,"play":1,"##ing":2}}"###,
            "playing",
            vec![1, 2],
        ),
        (
            r#"{"type":"BPE","byte_level":false,"vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]}"#,
            "ab",
            vec![2],
        ),
        (
            r#"{"type":"Unigram","unk_id":0,"byte_fallback":false,"vocab":[["<unk>",0.0],["a",-1.0],["ab",-0.5]]}"#,
            "ab",
            vec![2],
        ),
    ];
    for (model, text, expected) in cases {
        let json = format!(
            r#"{{"version":"2.0","added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{model}}}"#
        );
        assert_eq!(
            TextTokenizer::from_json(&json)
                .unwrap()
                .encode(text, &EncodeOptions::default())
                .unwrap()
                .input_ids,
            expected
        );
    }
}
#[test]
fn empty_ipc_stream_still_validates_schema() {
    use arrow::datatypes::{DataType, Field, Schema};
    let schema = Schema::new(vec![Field::new("text", DataType::Int64, false)]);
    let mut bytes = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut bytes, &schema).unwrap();
        writer.finish().unwrap();
    }
    assert!(
        ArrowTextDataset::from_ipc_stream(Cursor::new(bytes), TextColumns::default())
            .unwrap_err()
            .to_string()
            .contains("Utf8")
    );
}
#[test]
fn empty_batches_and_empty_sequences_remain_well_defined() {
    let options = EncodeOptions {
        padding: Override::With(padding()),
        ..Default::default()
    };
    assert!(tokenizer().encode_batch(&[], &options).unwrap().is_empty());
    let encodings = tokenizer()
        .encode_batch(
            &[sample(""), sample("")],
            &EncodeOptions {
                add_special_tokens: false,
                ..options
            },
        )
        .unwrap();
    let batch = TextBatch::collate(
        &encodings,
        &[None, None],
        vec![0, 1],
        LabelMode::CausalLm { ignore_index: -100 },
    )
    .unwrap();
    assert_eq!(batch.input_ids.dims(), [2, 0]);
    assert_eq!(batch.labels.unwrap().dims(), [2, 0]);
}
#[test]
fn plans_keep_resource_identity_sampling_options_and_labels() {
    let tokenizer = tokenizer();
    assert_eq!(tokenizer.identity(), tokenizer.clone().identity());
    let pipeline = TextPipeline::from_samples(vec![sample("hello")], tokenizer.clone())
        .indices(vec![0, 0])
        .shuffle(11)
        .epoch(7)
        .labels(LabelMode::None);
    let plan = pipeline.to_logical_plan().unwrap();
    let other = pipeline.clone().to_logical_plan().unwrap();
    let ops = plan
        .nodes()
        .map(|(_, n)| n.payload_as::<TextPlanOp>().unwrap().clone())
        .collect::<Vec<_>>();
    assert_eq!(
        ops,
        other
            .nodes()
            .map(|(_, n)| n.payload_as::<TextPlanOp>().unwrap().clone())
            .collect::<Vec<_>>()
    );
    assert!(
        matches!(&ops[0],TextPlanOp::Source { indices:Some(indices),shuffle_seed:Some(11),epoch:7,.. } if indices==&[0,0])
    );
    assert!(matches!(
        &ops[2],
        TextPlanOp::Collate {
            labels: LabelMode::None,
            ..
        }
    ));
    let different = TextPipeline::from_samples(vec![sample("world")], tokenizer)
        .to_logical_plan()
        .unwrap();
    assert_ne!(
        ops[0],
        *different
            .nodes()
            .next()
            .unwrap()
            .1
            .payload_as::<TextPlanOp>()
            .unwrap()
    );
}
#[test]
fn packed_blocks_collate_into_continuous_causal_targets() {
    let blocks = pack_encodings(
        &[encoding(&[4, 5], &[1, 1]), encoding(&[6], &[1])],
        PackingOptions {
            block_size: 2,
            separator_id: Some(3),
            drop_last: true,
        },
    )
    .unwrap();
    let batch = TextBatch::collate(
        &blocks,
        &vec![None; blocks.len()],
        (0..blocks.len()).collect(),
        LabelMode::CausalLm { ignore_index: -100 },
    )
    .unwrap();
    assert_eq!(batch.input_ids.to_vec::<i64>().unwrap(), [4, 5, 3, 6]);
    assert_eq!(batch.labels.unwrap().to_vec::<i64>().unwrap(), [4, 5, 3, 6]);
}

#[test]
fn text_noise_covers_robustness_modes_and_edge_cases() {
    use rivet_data::random::RandomStream;

    let apply = |name: &str, ratio, input: &str| {
        TextNoise::new(name.parse().unwrap(), ratio)
            .unwrap()
            .apply(input, &mut RandomStream::from_seed(42))
            .unwrap()
    };
    assert_eq!(apply("clean", 0.0, " A\tB "), " A\tB ");
    assert_eq!(apply("char_delete", 1.0, "é中🦀"), "");
    assert_eq!(apply("char_delete", 0.0, "é"), "");
    assert_eq!(apply("char_swap", 0.0, "é中"), "中é");
    let inserted = apply("char_insert", 0.0, "");
    assert_eq!(inserted.len(), 1);
    assert!(inserted.as_bytes()[0].is_ascii_lowercase());
    let substituted = apply("char_substitute", 1.0, "A🙂");
    assert!(substituted.starts_with(['q', 's', 'z']));
    assert!(substituted.ends_with('🙂'));

    assert_eq!(apply("word_delete", 1.0, "  only  "), "  only  ");
    assert_eq!(
        apply("word_delete", 1.0, "one two three")
            .split_whitespace()
            .count(),
        1
    );
    assert_eq!(apply("word_swap", 0.0, "one\t two"), "two one");
    assert!(
        [
            "the", "a", "is", "it", "and", "to", "of", "in", "for", "with"
        ]
        .contains(&apply("word_insert", 0.0, "").as_str())
    );
    assert!(["[PAD]", "[UNK]", "[MASK]"].contains(&apply("pad_noise", 0.0, "").as_str()));

    assert_eq!(
        apply("truncate_50", 0.0, "one two three four five"),
        "one two"
    );
    assert_eq!(apply("truncate_80", 0.0, "one two three four five"), "one");
    assert_eq!(apply("truncate_80", 1.0, "   "), "");
    assert_eq!(apply("uppercase", 1.0, "Straße"), "STRASSE");
    assert_eq!(apply("lowercase", 0.0, "ΟΣ"), "ος");
    assert_eq!(apply("spelling", 1.0, "THE\tlike good!"), "teh lik e good!");
    assert_eq!(apply("spelling", 0.0, "THE\tlike"), "THE like");
    assert!("unknown".parse::<TextNoiseKind>().is_err());
    for ratio in [-0.1, 1.1, f64::NAN] {
        assert!(TextNoise::new(TextNoiseKind::CharInsert, ratio).is_err());
    }
}

#[test]
fn text_pipeline_applies_noise_before_tokenization_and_keys_it_by_source_index() {
    let source = vec![
        TextSample::pair("hello world", "world").with_label(7),
        TextSample::pair("world hello", "hello").with_label(8),
    ];
    let pipeline = TextPipeline::from_samples(source, tokenizer())
        .noise(TextNoise::new(TextNoiseKind::WordSwap, 0.1).unwrap(), 42);
    let plan = pipeline.explain().unwrap();
    assert!(plan.find("TextNoise").unwrap() < plan.find("Tokenize").unwrap());
    let batch = pipeline
        .clone()
        .indices(vec![1, 0, 1])
        .batch(3, false)
        .compile()
        .unwrap()
        .next_batch()
        .unwrap()
        .unwrap();
    assert_eq!(batch.source_indices, [1, 0, 1]);
    assert_eq!(
        batch.input_ids.to_vec::<i64>().unwrap(),
        [2, 4, 5, 3, 4, 3, 2, 5, 4, 3, 5, 3, 2, 4, 5, 3, 4, 3]
    );
    assert_eq!(batch.labels.unwrap().to_vec::<i64>().unwrap(), [8, 7, 8]);

    let noise = TextNoise::new(TextNoiseKind::WordInsert, 0.5).unwrap();
    let pipeline = pipeline.noise(noise, 42);
    let rows = |indices, size| {
        pipeline
            .clone()
            .indices(indices)
            .batch(size, false)
            .compile()
            .unwrap()
            .flat_map(|batch| batch.unwrap().input_ids.to_vec::<i64>().unwrap())
            .collect::<Vec<_>>()
    };
    let first = rows(vec![0], 1);
    let second = rows(vec![1], 1);
    assert_eq!(
        rows(vec![1, 0, 1], 3),
        [second.as_slice(), first.as_slice(), second.as_slice()].concat()
    );
}

#[test]
fn text_file_samples_lines_paragraphs_and_documents() {
    let raw = "甲\r\n\r\nbeta\rgamma\n\n\n\nlast\r\n";
    for (sample_by, keep_linebreaks, expected) in [
        (
            TextSampleBy::Line,
            false,
            vec!["甲", "", "beta", "gamma", "", "", "", "last"],
        ),
        (
            TextSampleBy::Line,
            true,
            vec![
                "甲\n", "\n", "beta\n", "gamma\n", "\n", "\n", "\n", "last\n",
            ],
        ),
        (
            TextSampleBy::Paragraph,
            false,
            vec!["甲", "beta\ngamma", "last\n"],
        ),
        (
            TextSampleBy::Document,
            false,
            vec!["甲\n\nbeta\ngamma\n\n\n\nlast\n"],
        ),
    ] {
        let source = TextFileDataset::from_reader(
            raw.as_bytes(),
            TextFileOptions {
                sample_by,
                keep_linebreaks,
                ..Default::default()
            },
        )
        .unwrap();
        let expected: Vec<_> = expected.into_iter().map(sample).collect();
        assert_eq!(source.len(), expected.len());
        let indices: Vec<_> = (0..source.len()).collect();
        assert_eq!(source.get_many(&indices).unwrap(), expected);
        let last = source.len() - 1;
        assert_eq!(
            source.get_many(&[last, 0, last]).unwrap(),
            [
                expected[last].clone(),
                expected[0].clone(),
                expected[last].clone()
            ]
        );
        assert!(source.get_many(&[source.len()]).is_err());
    }
}

#[test]
fn text_file_handles_empty_input_and_utf8_errors() {
    for (sample_by, expected_len) in [
        (TextSampleBy::Line, 0),
        (TextSampleBy::Paragraph, 0),
        (TextSampleBy::Document, 1),
    ] {
        let source = TextFileDataset::from_reader(
            &[][..],
            TextFileOptions {
                sample_by,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(source.len(), expected_len);
    }

    let bytes: &[u8] = b"hello\xff\r\nworld\xe2\x82";
    assert!(matches!(
        TextFileDataset::from_reader(bytes, TextFileOptions::default()),
        Err(TextError::Io(error)) if error.kind() == std::io::ErrorKind::InvalidData
    ));
    for (encoding_errors, expected) in [
        (TextEncodingErrors::Replace, ["hello�", "world�"]),
        (TextEncodingErrors::Ignore, ["hello", "world"]),
    ] {
        let source = TextFileDataset::from_reader(
            bytes,
            TextFileOptions {
                encoding_errors,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(source.get_many(&[0, 1]).unwrap(), expected.map(sample));
    }
}

#[test]
fn text_files_preserve_file_boundaries_and_feed_the_tokenizer() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!("rivet-text-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let first = directory.join("first.txt");
    let second = directory.join("second.txt");
    std::fs::write(&first, "hello").unwrap();
    std::fs::write(&second, "world\r\nhello\r").unwrap();
    let source = TextFileDataset::open_files([&first, &second], TextFileOptions::default());
    std::fs::remove_dir_all(directory).unwrap();

    let batch = TextPipeline::new(Arc::new(source.unwrap()), tokenizer())
        .indices(vec![2, 1, 0])
        .batch(3, false)
        .compile()
        .unwrap()
        .next_batch()
        .unwrap()
        .unwrap();
    assert_eq!(batch.source_indices, [2, 1, 0]);
    assert_eq!(
        batch.input_ids.to_vec::<i64>().unwrap(),
        [2, 4, 3, 2, 5, 3, 2, 4, 3]
    );
}

#[test]
fn text_file_decodes_before_normalizing_and_sampling() {
    let utf16 = |encode: fn(u16) -> [u8; 2]| {
        "é\r\n中🙂\r"
            .encode_utf16()
            .flat_map(encode)
            .collect::<Vec<_>>()
    };
    let le = utf16(u16::to_le_bytes);
    let be = utf16(u16::to_be_bytes);
    for (encoding, bytes) in [
        ("utf-8-sig", "\u{feff}é\r\n中🙂\r".as_bytes().to_vec()),
        ("UTF_16", [&[0xff, 0xfe], le.as_slice()].concat()),
        ("utf-16", [&[0xfe, 0xff], be.as_slice()].concat()),
        ("utf-16-le", le),
        ("utf-16-be", be),
    ] {
        let source = TextFileDataset::from_reader(
            bytes.as_slice(),
            TextFileOptions {
                encoding: encoding.parse().unwrap(),
                keep_linebreaks: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(source.len(), 2);
        assert_eq!(
            source.get_many(&[0, 1]).unwrap(),
            ["é\n", "中🙂\n"].map(sample)
        );
    }

    let latin1 = TextFileDataset::from_reader(
        &b"caf\xe9\r\n"[..],
        TextFileOptions {
            encoding: "iso-8859-1".parse().unwrap(),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(latin1.get_many(&[0]).unwrap(), [sample("café")]);
    let utf8 = TextFileDataset::from_reader("\u{feff}first".as_bytes(), TextFileOptions::default())
        .unwrap();
    assert_eq!(utf8.get_many(&[0]).unwrap(), [sample("\u{feff}first")]);
    assert!("unknown-encoding".parse::<TextFileEncoding>().is_err());
}

#[test]
fn text_file_applies_error_policy_to_ascii_and_utf16() {
    use TextFileEncoding::*;

    for (encoding, bytes, replaced, ignored) in [
        (Ascii, &b"a\xff\x80b"[..], "a��b", "ab"),
        (Utf16Le, &b"a\0\0\xd8b"[..], "a�", "a"),
        (Utf16Be, &b"\xdc\0\0b!"[..], "�b�", "b"),
    ] {
        let read = |encoding_errors| {
            TextFileDataset::from_reader(
                bytes,
                TextFileOptions {
                    sample_by: TextSampleBy::Document,
                    encoding,
                    encoding_errors,
                    ..Default::default()
                },
            )
        };
        assert!(matches!(
            read(TextEncodingErrors::Strict),
            Err(TextError::Io(error)) if error.kind() == std::io::ErrorKind::InvalidData
        ));
        assert_eq!(
            read(TextEncodingErrors::Replace)
                .unwrap()
                .get_many(&[0])
                .unwrap(),
            [sample(replaced)]
        );
        assert_eq!(
            read(TextEncodingErrors::Ignore)
                .unwrap()
                .get_many(&[0])
                .unwrap(),
            [sample(ignored)]
        );
    }
    assert!(
        TextFileDataset::from_reader(
            &b"a\0"[..],
            TextFileOptions {
                encoding: Utf16,
                encoding_errors: TextEncodingErrors::Ignore,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn record_datasets_load_json_csv_tsv_excel_and_tar_into_the_text_pipeline() {
    use std::io::Write;

    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory =
        std::env::temp_dir().join(format!("rivet-records-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    struct Scratch(std::path::PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let scratch = Scratch(directory);
    let write_file = |name, bytes: &[u8]| {
        let path = scratch.0.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    };
    let json = write_file(
        "reviews.json",
        br#"[{"body":"hello","pair":"world","class":3}]"#,
    );
    let jsonl = write_file(
        "reviews.jsonl",
        b"{\"body\":\"world\",\"pair\":\"hello\",\"class\":4}\n",
    );
    let csv = write_file(
        "reviews.csv",
        b"pair,class,body\r\nworld,5,\"hello,\nworld\"\r\n",
    );
    let tsv = write_file("reviews.tsv", b"body\tpair\tclass\nworld\thello\t6\n");
    let excel = write_file("reviews.xlsx", include_bytes!("fixtures/records.xlsx"));
    let options = TextRecordsOptions {
        columns: TextColumns {
            text: "body".into(),
            text_pair: Some("pair".into()),
            label: Some("class".into()),
        },
        ..Default::default()
    };
    let default_sheet = TextRecordsDataset::open(&excel, options.clone()).unwrap();
    assert_eq!(
        default_sheet.get_many(&[0]).unwrap(),
        [TextSample::pair("hello", "文").with_label(1)]
    );
    let bad = write_file(
        "bad.jsonl",
        br#"{"body":"hello","pair":"world","class":null}"#,
    );
    assert!(
        TextRecordsDataset::open(&bad, options.clone())
            .unwrap_err()
            .to_string()
            .contains("column \"class\": expected an int64")
    );
    let mut options = options;
    options.sheet = Some("Reviews".into());

    let mut archive = tar::Builder::new(Vec::new());
    for (name, data) in [
        ("nested/raw.txt", &b"hello\r\nworld\n"[..]),
        (
            "nested/rows.jsonl",
            &b"{\"body\":\"hello\",\"pair\":\"world\",\"class\":7}\n"[..],
        ),
        ("ignore.bin", &b"ignored"[..]),
    ] {
        let mut header = tar::Header::new_gnu();
        header.set_mode(0o644);
        header.set_size(data.len() as u64);
        archive.append_data(&mut header, name, data).unwrap();
    }
    let archive = archive.into_inner().unwrap();
    let tar = write_file("data.tar", &archive);
    let mut compressed = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    compressed.write_all(&archive).unwrap();
    let mut compressed = compressed.finish().unwrap();
    let tgz = write_file("data.tar.gz", &compressed);
    let checksum = compressed.len() - 8;
    compressed[checksum] ^= 1;
    let corrupt = write_file("corrupt.tgz", &compressed);
    assert!(TextRecordsDataset::open(corrupt, options.clone()).is_err());
    let tar_rows = TextRecordsDataset::open(tar, options.clone()).unwrap();
    let gzip_rows = TextRecordsDataset::open(tgz, options.clone()).unwrap();
    let expected = [
        sample("hello"),
        sample("world"),
        TextSample::pair("hello", "world").with_label(7),
    ];
    assert_eq!(tar_rows.get_many(&[0, 1, 2]).unwrap(), expected);
    assert_eq!(gzip_rows.get_many(&[0, 1, 2]).unwrap(), expected);

    let structured =
        TextRecordsDataset::open_files([json, jsonl, csv, tsv, excel], options).unwrap();
    assert_eq!(
        structured.get_many(&[4, 2, 4]).unwrap(),
        [
            TextSample::pair("world", "文").with_label(2),
            TextSample::pair("hello,\nworld", "world").with_label(5),
            TextSample::pair("world", "文").with_label(2),
        ]
    );
    let batch = TextPipeline::new(Arc::new(structured), tokenizer())
        .indices(vec![1, 0])
        .batch(2, false)
        .compile()
        .unwrap()
        .next_batch()
        .unwrap()
        .unwrap();
    assert_eq!(batch.source_indices, [1, 0]);
    assert_eq!(batch.labels.unwrap().to_vec::<i64>().unwrap(), [4, 3]);
}
