use crate::errors::{RivetResult, invalid_argument};
use arrow::datatypes::SchemaRef;
use arrow_array::RecordBatch;
use lance::Dataset;
use lance::dataset::ProjectionRequest;
use std::path::Path;
use std::sync::Arc;
use tokio::runtime::{Builder, Runtime};

/// A synchronous handle to an asynchronously opened Lance dataset.
struct LanceReader {
    runtime: Runtime,
    dataset: Dataset,
}

/// Modality-neutral Lance table access.
///
/// `take` receives all row indices for one logical Rivet batch and performs
/// exactly one Lance `Dataset::take` call. The row order and duplicates are
/// left to Lance's take operation, which is the native random-access path.
pub struct LanceTable {
    reader: Arc<LanceReader>,
    lance_schema: lance::datatypes::Schema,
    arrow_schema: SchemaRef,
    len: usize,
}

impl LanceTable {
    pub fn open(path: impl AsRef<Path>) -> RivetResult<Self> {
        let path = path.as_ref();
        let uri = path.to_str().ok_or_else(|| {
            invalid_argument(format!(
                "Lance dataset path is not valid UTF-8: {}",
                path.display()
            ))
        })?;

        // One runtime is retained for the lifetime of the table. In
        // particular, do not construct a runtime inside `take`.
        let runtime = Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        let (dataset, len) = runtime.block_on(async {
            let dataset = Dataset::open(uri).await?;
            let len = dataset.count_rows(None).await?;
            Ok::<_, lance::Error>((dataset, len))
        })?;

        let schema = dataset.schema().clone();
        let arrow_schema = Arc::new(arrow::datatypes::Schema::from(&schema));
        Ok(Self {
            reader: Arc::new(LanceReader { runtime, dataset }),
            lance_schema: schema,
            arrow_schema,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.arrow_schema
    }

    /// Project and fetch one batch of rows with one Lance call.
    pub fn take(&self, indices: &[usize], columns: &[&str]) -> RivetResult<RecordBatch> {
        let indices = indices
            .iter()
            .map(|&index| {
                u64::try_from(index).map_err(|_| {
                    invalid_argument(format!("row index {index} does not fit in a Lance index"))
                })
            })
            .collect::<RivetResult<Vec<_>>>()?;
        let projection =
            ProjectionRequest::from_columns(columns.iter().copied(), &self.lance_schema);

        self.reader
            .runtime
            .block_on(self.reader.dataset.take(&indices, projection))
            .map_err(Into::into)
    }
}
