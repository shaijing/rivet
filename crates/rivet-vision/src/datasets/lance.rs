mod dataset;
mod loader;
mod row;
mod schema;

pub use dataset::LanceImageDataset;
pub use loader::load_lance_image_dataset;
#[cfg(test)]
pub(super) use schema::validate_schema;

#[cfg(test)]
mod tests {
    use super::{LanceImageDataset, validate_schema};
    use crate::sample::image::ImageSample;
    use arrow::array::{BinaryArray, Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatchIterator;
    use rivet_data::dataset::Dataset;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TMP: AtomicUsize = AtomicUsize::new(0);

    fn temp_lance_path() -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "rivet-lance-test-{}-{}",
            std::process::id(),
            NEXT_TMP.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_fixture(path: &std::path::Path) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("image", DataType::Binary, false),
            Field::new("label", DataType::Int32, false),
            Field::new("ignored", DataType::Utf8, false),
        ]));
        let image: BinaryArray = [Some(&b"zero"[..]), Some(&b"one"[..]), Some(&b"two"[..])]
            .into_iter()
            .collect();
        let label = Int32Array::from(vec![0, 1, 2]);
        let ignored = arrow::array::StringArray::from(vec!["a", "b", "c"]);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(image), Arc::new(label), Arc::new(ignored)],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(lance::Dataset::write(reader, path.to_str().unwrap(), None))
            .unwrap();
    }

    #[test]
    fn reads_ordered_duplicate_rows_with_one_batch_request() {
        let path = temp_lance_path();
        write_fixture(&path);

        let dataset = LanceImageDataset::open_default(&path).unwrap();
        assert_eq!(dataset.len(), 3);
        let samples = dataset.get_many(&[2, 0, 2]).unwrap();
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].image.as_slice(), b"two");
        assert_eq!(samples[1].image.as_slice(), b"zero");
        assert_eq!(samples[2].image.as_slice(), b"two");
        assert_eq!(
            samples
                .iter()
                .map(|sample| sample.label)
                .collect::<Vec<_>>(),
            [2, 0, 2]
        );

        assert!(dataset.get_many(&[]).unwrap().is_empty());
        assert!(dataset.get_many(&[3]).is_err());
        assert_eq!(dataset.get(1).unwrap().label, 1);

        std::fs::remove_dir_all(path).ok();
    }

    #[test]
    fn encoded_cache_matches_the_lazy_source() {
        let path = temp_lance_path();
        write_fixture(&path);

        let source = crate::source::ImageSource::from_encoded(Arc::new(
            LanceImageDataset::open_default(&path).unwrap(),
        ));
        let expected = source.get_many(&[2, 0, 2]).unwrap();
        let cached = source.cache_encoded(2).unwrap();
        let actual = cached.get_many(&[2, 0, 2]).unwrap();

        assert_eq!(
            actual
                .iter()
                .map(|sample| match sample {
                    ImageSample::Encoded(sample) => sample.label,
                    ImageSample::Decoded(_) => panic!("expected encoded sample"),
                })
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|sample| match sample {
                    ImageSample::Encoded(sample) => sample.label,
                    ImageSample::Decoded(_) => panic!("expected encoded sample"),
                })
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual
                .iter()
                .map(|sample| match sample {
                    ImageSample::Encoded(sample) => sample.image.as_slice(),
                    ImageSample::Decoded(_) => panic!("expected encoded sample"),
                })
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|sample| match sample {
                    ImageSample::Encoded(sample) => sample.image.as_slice(),
                    ImageSample::Decoded(_) => panic!("expected encoded sample"),
                })
                .collect::<Vec<_>>()
        );

        std::fs::remove_dir_all(path).ok();
    }

    #[test]
    fn rejects_non_native_image_or_label_schema() {
        // Schema validation is intentionally kept in a small helper test so
        // invalid datasets fail during construction, before get_many.
        let schema = Schema::new(vec![
            Field::new("image", DataType::Utf8, false),
            Field::new("label", DataType::Int32, false),
        ]);
        let error = validate_schema(&schema, "image", "label").unwrap_err();
        assert!(error.to_string().contains("binary"));
    }

    #[test]
    fn accepts_huggingface_struct_image_as_compatibility_input() {
        let schema = Schema::new(vec![
            Field::new(
                "img",
                DataType::Struct(
                    vec![Arc::new(Field::new("bytes", DataType::Binary, true))].into(),
                ),
                true,
            ),
            Field::new("label", DataType::Int64, false),
        ]);
        validate_schema(&schema, "img", "label").unwrap();
    }
}
