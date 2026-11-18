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
