use crate::batch::ImageBatchBuilder;
use crate::errors::{RivetError, RivetResult};
use crate::pipeline::op::ExecutionPlan;
use crate::runtime::pool::WorkerPool;
use crate::runtime::worker::WorkItem;
use crate::sample::image::{DecodedSample, ImageBatch};
use crate::sampler::IndexSampler;
use std::collections::BTreeMap;
use std::sync::Arc;

enum LoaderExecutor {
    /// `num_workers = 0`: direct synchronous execution, no channel overhead.
    Inline,
    /// Persistent worker pool with bounded cross-batch prefetch.
    Workers(WorkerPool, PrefetchCoordinator),
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
        prefetch_batches: usize,
    ) -> RivetResult<Self> {
        let executor = if num_workers == 0 {
            LoaderExecutor::Inline
        } else {
            let prefetch = prefetch_batches.max(1);
            let pool = WorkerPool::new(
                Arc::clone(&plan),
                num_workers,
                plan.batch.size * prefetch,
            )?;
            LoaderExecutor::Workers(pool, PrefetchCoordinator::new(prefetch))
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
            LoaderExecutor::Workers(pool, coordinator) => {
                next_batch_workers(plan, sampler, pool, coordinator)
            }
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

/// One in-flight logical batch being filled by workers.
struct PendingBatch {
    samples: Vec<Option<DecodedSample>>,
    remaining: usize,
}

/// Tracks submitted-but-not-yet-delivered batches so `prefetch_batches`
/// batches can run concurrently while delivery stays strictly in sampler
/// order.
struct PrefetchCoordinator {
    prefetch: usize,
    /// Number of submitted batches not yet delivered to the caller.
    in_flight: usize,
    /// Sampler produced its last batch (or its tail was dropped).
    closed: bool,
    next_batch_id: u64,
    pending: BTreeMap<u64, PendingBatch>,
}

impl PrefetchCoordinator {
    fn new(prefetch: usize) -> Self {
        Self {
            prefetch,
            in_flight: 0,
            closed: false,
            next_batch_id: 0,
            pending: BTreeMap::new(),
        }
    }

    fn submit(
        &mut self,
        indices: Vec<usize>,
        pool: &WorkerPool,
    ) -> RivetResult<()> {
        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;

        let batch = PendingBatch {
            samples: std::iter::repeat_with(|| None).take(indices.len()).collect(),
            remaining: indices.len(),
        };
        self.pending.insert(batch_id, batch);
        self.in_flight += 1;

        for (position, index) in indices.into_iter().enumerate() {
            pool.submit(WorkItem {
                batch_id,
                position,
                index,
            })?;
        }
        Ok(())
    }

    fn record(&mut self, batch_id: u64, position: usize, sample: DecodedSample) -> RivetResult<()> {
        let batch = self
            .pending
            .get_mut(&batch_id)
            .ok_or_else(|| RivetError::Worker(format!("result for unknown batch {batch_id}")))?;
        if batch.samples[position].is_some() {
            return Err(RivetError::Worker(format!(
                "duplicate result for batch {batch_id} position {position}"
            )));
        }
        batch.samples[position] = Some(sample);
        batch.remaining -= 1;
        Ok(())
    }

    /// Deliver the next batch when it is complete; delivery order is
    /// strictly the submission order regardless of completion order.
    fn deliver_ready(&mut self, batch_id: u64) -> RivetResult<Option<ImageBatch>> {
        let Some(batch) = self.pending.get(&batch_id) else {
            return Ok(None);
        };
        if batch.remaining != 0 {
            return Ok(None);
        }
        let batch = match self.pending.remove(&batch_id) {
            Some(batch) => batch,
            None => {
                return Err(RivetError::Worker(format!(
                    "deliverable batch {batch_id} vanished"
                )));
            }
        };
        self.in_flight -= 1;

        let mut builder = ImageBatchBuilder::with_capacity(batch.samples.len());
        for sample in batch.samples.into_iter().flatten() {
            builder.push(sample)?;
        }
        Ok(Some(builder.finish()))
    }
}

fn next_batch_workers(
    plan: &ExecutionPlan,
    sampler: &mut IndexSampler,
    pool: &WorkerPool,
    coordinator: &mut PrefetchCoordinator,
) -> RivetResult<Option<ImageBatch>> {
    loop {
        // Top up the in-flight window from the sampler.
        while coordinator.in_flight < coordinator.prefetch && !coordinator.closed {
            match take_indices(plan, sampler)? {
                Some(indices) => coordinator.submit(indices, pool)?,
                None => coordinator.closed = true,
            }
        }

        if let Some(batch) =
            coordinator.deliver_ready(coordinator.next_batch_id - coordinator.in_flight as u64)?
        {
            return Ok(Some(batch));
        }

        if coordinator.closed && coordinator.in_flight == 0 {
            return Ok(None);
        }

        // Nothing deliverable yet: wait for the next worker result.
        let result = pool.recv()?;
        match result.result {
            Ok(sample) => coordinator.record(result.batch_id, result.position, sample)?,
            Err(err) => return Err(err),
        }
    }
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
    fn prefetch_configs_preserve_sampler_order() {
        let mut baseline = pipeline(101, 0)
            .skip(3)
            .take(93)
            .batch(10, true)
            .compile()
            .unwrap();
        let baseline = drain(&mut baseline);

        for (workers, prefetch) in [(1, 0), (2, 1), (2, 3), (4, 4), (8, 2)] {
            let mut loader = pipeline(101, workers)
                .skip(3)
                .take(93)
                .batch(10, true)
                .prefetch_batches(prefetch)
                .compile()
                .unwrap();
            let pooled = drain(&mut loader);
            assert_batches_equal(&baseline, &pooled);
        }
    }

    #[test]
    fn prefetch_stops_after_dropped_tail() {
        // 30..43 spans 13 taken rows; drop_last with size 5 yields 5+5 and
        // silently discards the 3-row tail even when batches run ahead.
        for (workers, prefetch) in [(2, 3), (4, 1)] {
            let mut loader = pipeline(50, workers)
                .skip(30)
                .take(13)
                .batch(5, true)
                .prefetch_batches(prefetch)
                .compile()
                .unwrap();
            let batches = drain(&mut loader);
            assert_eq!(batches.len(), 2, "workers={workers} prefetch={prefetch}");
            assert_eq!(batches[0].labels, vec![30, 31, 32, 33, 34]);
            assert_eq!(batches[1].labels, vec![35, 36, 37, 38, 39]);
        }
    }

    #[test]
    fn drop_shuts_down_workers_without_panicking() {
        let mut loader = pipeline(10, 4).batch(4, false).compile().unwrap();
        assert!(loader.next_batch().unwrap().is_some());
        // Dropping the loader joins every worker; nothing should hang.
    }
}
