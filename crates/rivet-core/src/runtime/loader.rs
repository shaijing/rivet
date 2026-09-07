use crate::batch::ImageBatchBuilder;
use crate::errors::{RivetError, RivetResult};
use crate::pipeline::op::ExecutionPlan;
use crate::runtime::pool::WorkerPool;
use crate::runtime::worker::WorkItem;
use crate::sample::image::{DecodedSample, ImageBatch};
use crate::sampler::IndexSampler;
use std::sync::Arc;

enum LoaderExecutor {
    /// `num_workers = 0`: direct synchronous execution, no channel overhead.
    Inline,
    /// Persistent worker pool executing samples in parallel.
    Workers(WorkerPool),
}

pub struct ImageDataLoader {
    pub plan: Arc<ExecutionPlan>,
    pub sampler: IndexSampler,
    executor: LoaderExecutor,
}

impl ImageDataLoader {
    pub(crate) fn new(
        plan: Arc<ExecutionPlan>,
        sampler: IndexSampler,
        num_workers: usize,
    ) -> RivetResult<Self> {
        let executor = if num_workers == 0 {
            LoaderExecutor::Inline
        } else {
            LoaderExecutor::Workers(WorkerPool::new(
                Arc::clone(&plan),
                num_workers,
                plan.batch.size,
            )?)
        };

        Ok(Self {
            plan,
            sampler,
            executor,
        })
    }

    pub fn next_batch(&mut self) -> RivetResult<Option<ImageBatch>> {
        let Self {
            plan,
            sampler,
            executor,
        } = self;

        match executor {
            LoaderExecutor::Inline => next_batch_inline(plan, sampler),
            LoaderExecutor::Workers(pool) => next_batch_workers(plan, sampler, pool),
        }
    }
}

/// Coordinator-side index selection shared by both executors.
fn take_indices(
    plan: &ExecutionPlan,
    sampler: &mut IndexSampler,
) -> RivetResult<Option<Vec<usize>>> {
    let Some(indices) = sampler.next_indices(plan.batch.size) else {
        return Ok(None);
    };

    if plan.batch.drop_last && indices.len() < plan.batch.size {
        return Ok(None);
    }

    Ok(Some(indices))
}

fn next_batch_inline(
    plan: &ExecutionPlan,
    sampler: &mut IndexSampler,
) -> RivetResult<Option<ImageBatch>> {
    let Some(indices) = take_indices(plan, sampler)? else {
        return Ok(None);
    };

    let mut batch = ImageBatchBuilder::with_capacity(indices.len());

    for index in indices {
        let encoded = plan.source.get(index)?;
        let decoded = plan.apply_ops(encoded, index)?;
        batch.push(decoded)?;
    }

    Ok(Some(batch.finish()))
}

fn next_batch_workers(
    plan: &ExecutionPlan,
    sampler: &mut IndexSampler,
    pool: &WorkerPool,
) -> RivetResult<Option<ImageBatch>> {
    let Some(indices) = take_indices(plan, sampler)? else {
        return Ok(None);
    };

    for (sequence, index) in indices.iter().copied().enumerate() {
        pool.submit(WorkItem { sequence, index })?;
    }

    // Reconstruct sampler order: results may arrive in any completion
    // order, so store by sequence and refuse to continue until every
    // submitted sample answered (or the pool died).
    let mut samples: Vec<Option<DecodedSample>> =
        std::iter::repeat_with(|| None).take(indices.len()).collect();
    let mut completed = 0usize;

    while completed < samples.len() {
        let result = pool.recv()?;
        match result.result {
            Ok(sample) => {
                samples[result.sequence] = Some(sample);
                completed += 1;
            }
            Err(err) => return Err(err),
        }
    }

    let mut batch = ImageBatchBuilder::with_capacity(samples.len());
    for (sequence, sample) in samples.into_iter().enumerate() {
        let sample = sample.ok_or_else(|| {
            RivetError::Worker(format!(
                "no result received for sample at sequence {sequence}"
            ))
        })?;
        batch.push(sample)?;
    }

    Ok(Some(batch.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::source::Dataset;
    use crate::pipeline::ImagePipeline;
    use crate::sample::image::{EncodedImageSample, ImageBuffer};
    use arrow_buffer::Buffer;

    // 1x1 RGB PNG (red pixel), valid input for decode_image.
    const PNG_1X1: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90\x77\x53\xde\x00\x00\x00\x0c\x49\x44\x41\x54\x78\x9c\x63\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";

    struct SampleDataset {
        len: usize,
        err_at: Option<usize>,
        panic_at: Option<usize>,
    }

    impl Dataset for SampleDataset {
        type Item = EncodedImageSample;

        fn len(&self) -> usize {
            self.len
        }

        fn get(&self, index: usize) -> RivetResult<Self::Item> {
            if self.err_at == Some(index) {
                return Err(crate::errors::invalid_argument("boom"));
            }
            if self.panic_at == Some(index) {
                panic!("sample {index} panics");
            }
            Ok(EncodedImageSample {
                image: Buffer::from(PNG_1X1.to_vec()),
                label: index as i64,
            })
        }
    }

    fn pipeline(len: usize, workers: usize) -> ImagePipeline {
        ImagePipeline::new(Arc::new(SampleDataset {
            len,
            err_at: None,
            panic_at: None,
        }))
        .decode_image()
        .workers(workers)
    }

    fn drain(loader: &mut ImageDataLoader) -> Vec<ImageBatch> {
        let mut out = Vec::new();
        while let Some(batch) = loader.next_batch().unwrap() {
            out.push(batch);
        }
        out
    }

    fn assert_batches_equal(left: &[ImageBatch], right: &[ImageBatch]) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right) {
            assert_eq!(a.labels, b.labels);
            assert_eq!(a.shape, b.shape);
            match (&a.images, &b.images) {
                (ImageBuffer::U8(a), ImageBuffer::U8(b)) => assert_eq!(a, b),
                _ => panic!("expected u8 images"),
            }
        }
    }

    #[test]
    fn workers_match_inline_order_and_content() {
        let mut inline = pipeline(37, 0).batch(8, false).compile().unwrap();
        let mut pooled = pipeline(37, 4).batch(8, false).compile().unwrap();

        let inline = drain(&mut inline);
        let pooled = drain(&mut pooled);
        assert_batches_equal(&inline, &pooled);
        assert_eq!(inline.iter().map(|b| b.labels.len()).sum::<usize>(), 37);
    }

    #[test]
    fn workers_preserve_skip_take_and_drop_last() {
        let mut inline = pipeline(50, 0)
            .skip(2)
            .take(11)
            .batch(5, true)
            .compile()
            .unwrap();
        let mut pooled = pipeline(50, 3)
            .skip(2)
            .take(11)
            .batch(5, true)
            .compile()
            .unwrap();

        let inline = drain(&mut inline);
        let pooled = drain(&mut pooled);
        assert_batches_equal(&inline, &pooled);
        // 11 taken rows, drop_last keeps 5+5, second batch is the last.
        assert_eq!(inline.len(), 2);
        assert_eq!(inline[0].labels, vec![2, 3, 4, 5, 6]);
        assert_eq!(inline[1].labels, vec![7, 8, 9, 10, 11]);
    }

    #[test]
    fn sample_error_propagates_from_workers() {
        let dataset = Arc::new(SampleDataset {
            len: 20,
            err_at: Some(1),
            panic_at: None,
        });
        let mut loader = ImagePipeline::new(dataset)
            .decode_image()
            .workers(2)
            .batch(4, false)
            .compile()
            .unwrap();

        let err = match loader.next_batch() {
            Err(err) => err,
            Ok(_) => panic!("expected a worker error"),
        };
        assert!(err.to_string().contains("boom"), "got: {err}");
    }

    #[test]
    fn worker_panic_becomes_error_not_hang() {
        let dataset = Arc::new(SampleDataset {
            len: 20,
            err_at: None,
            panic_at: Some(3),
        });
        let mut loader = ImagePipeline::new(dataset)
            .decode_image()
            .workers(2)
            .batch(4, false)
            .compile()
            .unwrap();

        let err = match loader.next_batch() {
            Err(err) => err,
            Ok(_) => panic!("expected a worker panic error"),
        };
        assert!(err.to_string().contains("panicked"), "got: {err}");
    }

    #[test]
    fn drop_shuts_down_workers_without_panicking() {
        let mut loader = pipeline(10, 4).batch(4, false).compile().unwrap();
        assert!(loader.next_batch().unwrap().is_some());
        // Dropping the loader joins every worker; nothing should hang.
    }
}
