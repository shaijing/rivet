use crate::errors::{RivetResult, invalid_argument};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use std::{collections::BTreeMap, path::Path, sync::Arc};

pub(super) fn native_schema(keep_path: bool) -> SchemaRef {
    let mut fields = vec![
        Field::new("image", DataType::Binary, false),
        Field::new("label", DataType::Int32, false),
    ];
    if keep_path {
        fields.push(Field::new("path", DataType::Utf8, true));
    }
    Arc::new(Schema::new(fields))
}

pub(super) fn validate_split_name(name: &str) -> RivetResult<()> {
    let path = Path::new(name);
    if name.is_empty()
        || path.is_absolute()
        || path.components().count() != 1
        || path.file_name().and_then(|value| value.to_str()) != Some(name)
    {
        return Err(invalid_argument(format!(
            "invalid split name {name:?}; split names must be one relative path component"
        )));
    }
    Ok(())
}

#[cfg(feature = "lance")]
pub(super) fn local_bundle_name(output: &Path) -> &str {
    // Default local Lance outputs use <dataset>/lance.
    let path = if output.file_name().is_some_and(|name| name == "lance") {
        output.parent().unwrap_or(output)
    } else {
        output
    };
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("rivet-dataset")
}

pub(super) fn write_image_manifest(
    output_root: &Path,
    rows_by_split: &BTreeMap<String, usize>,
    keep_path: bool,
    source_format: &str,
    classes: Option<&[String]>,
    extension: &str,
    local_name: Option<&str>,
) -> RivetResult<()> {
    let mut features = serde_json::json!({
        "image": {
            "type": "image",
            "column": "image",
            "representation": "encoded"
        },
        "label": {
            "type": "class_label",
            "column": "label",
            "dtype": "int32"
        }
    });
    if let Some(classes) = classes {
        features["label"]["names"] = serde_json::json!(classes);
    }
    if keep_path {
        features["path"] = serde_json::json!({
            "type": "text",
            "column": "path"
        });
    }

    let splits = rows_by_split
        .iter()
        .map(|(name, rows)| {
            (
                name.clone(),
                serde_json::json!({
                    "uri": format!("{name}.{extension}"),
                    "num_rows": rows
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let dataset_name = local_name
        .or_else(|| {
            output_root
                .file_name()
                .and_then(|value| value.to_str())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or("rivet-dataset");
    let mut manifest = serde_json::json!({
        "format_version": 2,
        "dataset": {"name": dataset_name, "modality": "image"},
        "features": features,
        "splits": splits,
        "created_by": {
            "tool": "rivet",
            "version": env!("CARGO_PKG_VERSION"),
            "source_format": source_format
        }
    });
    if local_name.is_some() {
        manifest["extensions"] =
            serde_json::json!({(rivet_data::dataset::DATASET_ORIGIN_KEY): "local"});
    }
    let contents = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| invalid_argument(format!("serialize dataset manifest: {error}")))?;
    std::fs::write(output_root.join("dataset.json"), contents)
        .map_err(|error| invalid_argument(format!("write dataset manifest: {error}")))?;
    Ok(())
}
