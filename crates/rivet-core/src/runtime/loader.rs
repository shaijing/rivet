use crate::batch::ImageBatchBuilder;
use crate::errors::{RivetError, RivetResult, invalid_argument};
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
    /// A sample/worker error terminated the loader; iteration is over.
    Failed,
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
            // prefetch_batches counts *future* batches: the current batch
            // plus that many prepared ahead are in flight.
            let max_in_flight = prefetch_batches.saturating_add(1);
            // Reject absurd sizes before constructing any channel: a wrap
            // here would silently produce a broken bounded queue.
            let max_in_flight_samples = plan.batch.size.checked_mul(max_in_flight).ok_or_else(
                || invalid_argument("worker queue capacity overflow"),
            )?;
            let pool = WorkerPool::new(
                Arc::clone(&plan),
                num_workers,
                max_in_flight_samples,
            )?;
            LoaderExecutor::Workers(pool, PrefetchCoordinator::new(max_in_flight))
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
            LoaderExecutor::Inline => match next_batch_inline(plan, sampler) {
                Ok(batch) => Ok(batch),
                Err(err) => {
                    *executor = LoaderExecutor::Failed;
                    Err(err)
                }
            },
            LoaderExecutor::Workers(pool, coordinator) => {
                match next_batch_workers(plan, sampler, pool, coordinator) {
                    Ok(batch) => Ok(batch),
                    Err(err) => {
                        // Terminal: a failed sample leaves its pending slot
                        // unfilled forever (and inline has already advanced
                        // the sampler), so the loader must not be used
                        // again instead of waiting for results that will
                        // never come or silently skipping a failed batch.
                        *executor = LoaderExecutor::Failed;
                        Err(err)
                    }
                }
            }
            LoaderExecutor::Failed => Err(RivetError::Worker(
                "loader is in failed state after a previous iteration error".to_string(),
            )),
        }
    }
    /// Convenience wrapper for `(&mut self).into_iter()`.
    pub fn iter(&mut self) -> ImageDataLoaderIter<'_> {
        ImageDataLoaderIter {
            loader: self,
            done: false,
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

/// Tracks submitted-but-not-yet-delivered batches so several batches can
/// run concurrently while delivery stays strictly in sampler order.
struct PrefetchCoordinator {
    /// Maximum batches in flight (window control only).
    max_in_flight: usize,
    /// Number of submitted batches not yet delivered to the caller.
    in_flight: usize,
    /// Sampler produced its last batch (or its tail was dropped).
    closed: bool,
    /// Next id handed out when submitting a batch.
    next_batch_id: u64,
    /// Next batch that must be returned to the caller, in order.
    next_deliver_id: u64,
    pending: BTreeMap<u64, PendingBatch>,
}

impl PrefetchCoordinator {
    fn new(max_in_flight: usize) -> Self {
        Self {
            max_in_flight,
            in_flight: 0,
            closed: false,
            next_batch_id: 0,
            next_deliver_id: 0,
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
        let slot = batch.samples.get_mut(position).ok_or_else(|| {
            RivetError::Worker(format!(
                "invalid result position {position} for batch {batch_id}"
            ))
        })?;
        if slot.is_some() {
            return Err(RivetError::Worker(format!(
                "duplicate result for batch {batch_id} position {position}"
            )));
        }
        *slot = Some(sample);
        batch.remaining -= 1;
        Ok(())
    }

    /// Deliver the next batch when it is complete; delivery order is
    /// strictly the submission order regardless of completion order.
    fn deliver_ready(&mut self) -> RivetResult<Option<ImageBatch>> {
        let batch_id = self.next_deliver_id;
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
        self.next_deliver_id += 1;

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
        while coordinator.in_flight < coordinator.max_in_flight && !coordinator.closed {
            match take_indices(plan, sampler)? {
                Some(indices) => coordinator.submit(indices, pool)?,
                None => coordinator.closed = true,
            }
        }

        if let Some(batch) = coordinator.deliver_ready()? {
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

/// Fallible iterator over an [`ImageDataLoader`]'s batches.
///
/// Item is `RivetResult<ImageBatch>` so errors are reported once and then
/// the iterator terminates: after EOF or a single error, every further
/// `next()` returns `None`. This matches the loader's terminal failed-state
/// model instead of yielding endless error items.
pub struct ImageDataLoaderIter<'a> {
    loader: &'a mut ImageDataLoader,
    done: bool,
}

impl Iterator for ImageDataLoaderIter<'_> {
    type Item = RivetResult<ImageBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        match self.loader.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(err) => {
                self.done = true;
                Some(Err(err))
            }
        }
    }
}

impl<'a> IntoIterator for &'a mut ImageDataLoader {
    type Item = RivetResult<ImageBatch>;
    type IntoIter = ImageDataLoaderIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        ImageDataLoaderIter {
            loader: self,
            done: false,
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
        /// Simulated per-sample latency for rows 0..8.
        slow_first_batch_ms: u64,
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
            if self.slow_first_batch_ms > 0 && index < 8 {
                std::thread::sleep(std::time::Duration::from_millis(
                    self.slow_first_batch_ms,
                ));
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
            slow_first_batch_ms: 0,
        }))
        .decode_image()
        .workers(workers)
    }

    fn drain(loader: &mut ImageDataLoader) -> Vec<ImageBatch> {
        loader
            .into_iter()
            .collect::<RivetResult<Vec<_>>>()
            .unwrap()
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
            slow_first_batch_ms: 0,
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
            slow_first_batch_ms: 0,
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

    /// A sample error must put the loader into a terminal failed state:
    /// the next call errors immediately instead of waiting forever for the
    /// result that will never arrive.
    #[test]
    fn next_batch_after_worker_error_returns_terminal_error() {
        let dataset = Arc::new(SampleDataset {
            len: 20,
            err_at: Some(1),
            panic_at: None,
            slow_first_batch_ms: 0,
        });
        let mut loader = ImagePipeline::new(dataset)
            .decode_image()
            .workers(2)
            .prefetch_batches(2)
            .batch(4, false)
            .compile()
            .unwrap();

        let first = match loader.next_batch() {
            Err(err) => err,
            Ok(_) => panic!("expected the sample error"),
        };
        assert!(first.to_string().contains("boom"), "got: {first}");

        let second = match loader.next_batch() {
            Err(err) => err,
            Ok(_) => panic!("expected the terminal failed-state error"),
        };
        assert!(
            second.to_string().contains("failed state"),
            "got: {second}"
        );
    }

    /// Deep prefetch with early drop must not deadlock: workers blocked on
    /// a full result queue are released by the result-receiver disconnect.
    #[test]
    fn drop_with_full_prefetch_does_not_hang() {
        let mut loader = pipeline(1000, 4)
            .batch(8, false)
            .prefetch_batches(8)
            .compile()
            .unwrap();
        let _ = loader.next_batch().unwrap();
        drop(loader);
    }

    /// Even when batch 1 completes before the slower batch 0, delivery must
    /// return batch 0 first.
    #[test]
    fn slow_earlier_batch_preserves_cross_batch_order() {
        let dataset = Arc::new(SampleDataset {
            len: 20,
            err_at: None,
            panic_at: None,
            slow_first_batch_ms: 30,
        });
        let mut loader = ImagePipeline::new(dataset)
            .decode_image()
            .workers(3)
            .prefetch_batches(2)
            .batch(4, false)
            .compile()
            .unwrap();

        let mut seen = Vec::new();
        while let Some(batch) = loader.next_batch().unwrap() {
            seen.extend(batch.labels);
        }
        assert_eq!(seen, (0..20).collect::<Vec<i64>>());
    }

    /// Protocol violations must surface as worker errors, not panics.
    #[test]
    fn record_rejects_out_of_range_and_duplicate_positions() {
        let mut coordinator = PrefetchCoordinator::new(2);
        coordinator.pending.insert(
            0,
            PendingBatch {
                samples: std::iter::repeat_with(|| None).take(2).collect(),
                remaining: 2,
            },
        );
        let make_sample = |label: i64| DecodedSample {
            image: ImageBuffer::U8(vec![0; 3]),
            width: 1,
            height: 1,
            channels: 3,
            label,
            layout: crate::sample::image::ImageLayout::Hwc,
        };

        let err = match coordinator.record(0, 5, make_sample(0)) {
            Err(err) => err,
            Ok(()) => panic!("expected an out-of-range error"),
        };
        assert!(err.to_string().contains("invalid result position"), "got: {err}");

        coordinator.record(0, 1, make_sample(1)).unwrap();
        let err = match coordinator.record(0, 1, make_sample(2)) {
            Err(err) => err,
            Ok(()) => panic!("expected a duplicate error"),
        };
        assert!(err.to_string().contains("duplicate"), "got: {err}");
    }

    /// Extreme batch/prefetch sizes must fail at compile with an argument
    /// error before any channel is constructed (no panic, no allocation).
    #[test]
    fn worker_queue_capacity_overflow_is_rejected() {
        let dataset = Arc::new(SampleDataset {
            len: 10,
            err_at: None,
            panic_at: None,
            slow_first_batch_ms: 0,
        });
        let result = ImagePipeline::new(dataset)
            .decode_image()
            .workers(4)
            .prefetch_batches(usize::MAX)
            .batch(usize::MAX, false)
            .compile();
        let err = match result {
            Err(err) => err,
            Ok(_) => panic!("expected a capacity overflow error"),
        };
        assert!(err.to_string().contains("overflow"), "got: {err}");
    }

    /// Inline errors must be terminal too: worker count must not change the
    /// loader's error semantics.
    #[test]
    fn inline_error_is_terminal_too() {
        let dataset = Arc::new(SampleDataset {
            len: 20,
            err_at: Some(1),
            panic_at: None,
            slow_first_batch_ms: 0,
        });
        let mut loader = ImagePipeline::new(dataset)
            .decode_image()
            .workers(0)
            .batch(4, false)
            .compile()
            .unwrap();

        let first = match loader.next_batch() {
            Err(err) => err,
            Ok(_) => panic!("expected the sample error"),
        };
        assert!(first.to_string().contains("boom"), "got: {first}");

        let second = match loader.next_batch() {
            Err(err) => err,
            Ok(_) => panic!("expected the terminal failed-state error"),
        };
        assert!(
            second.to_string().contains("failed state"),
            "got: {second}"
        );
    }

    #[test]
    fn iterator_stops_at_eof() {
        let mut loader = pipeline(10, 0).batch(4, false).compile().unwrap();

        let batches = (&mut loader)
            .into_iter()
            .collect::<RivetResult<Vec<_>>>()
            .unwrap();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].labels, vec![0, 1, 2, 3]);

        let mut iter = (&mut loader).into_iter();
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }

    #[test]
    fn iterator_yields_error_once_then_stops() {
        for workers in [0, 3] {
            let dataset = Arc::new(SampleDataset {
                len: 20,
                err_at: Some(1),
                panic_at: None,
                slow_first_batch_ms: 0,
            });
            let mut loader = ImagePipeline::new(dataset)
                .decode_image()
                .workers(workers)
                .batch(4, false)
                .compile()
                .unwrap();

            let mut iter = (&mut loader).into_iter();
            let first = iter.next().expect("must yield the failure");
            let err = match first {
                Err(err) => err,
                Ok(_) => panic!("expected an error item"),
            };
            assert!(err.to_string().contains("boom"), "got: {err}");

            // The error is reported exactly once; the iterator then ends.
            assert!(iter.next().is_none());
            assert!(iter.next().is_none());
        }
    }

    #[test]
    fn iterator_matches_inline_and_workers() {
        let mut inline = pipeline(37, 0)
            .skip(2)
            .take(30)
            .batch(8, true)
            .compile()
            .unwrap();
        let mut workers = pipeline(37, 3)
            .skip(2)
            .take(30)
            .batch(8, true)
            .prefetch_batches(2)
            .compile()
            .unwrap();

        let inline = drain(&mut inline);
        let workers = drain(&mut workers);
        assert_batches_equal(&inline, &workers);
    }

    #[test]
    fn iter_method_equals_into_iter() {
        let mut via_iter_loader = pipeline(10, 2).batch(4, false).compile().unwrap();
        let mut via_into_loader = pipeline(10, 2).batch(4, false).compile().unwrap();

        let via_iter = via_iter_loader
            .iter()
            .collect::<RivetResult<Vec<_>>>()
            .unwrap();
        let via_into = (&mut via_into_loader)
            .into_iter()
            .collect::<RivetResult<Vec<_>>>()
            .unwrap();
        assert_batches_equal(&via_iter, &via_into);
    }
}
