use super::batch::next_batch_inline;
use super::scheduler::{next_batch_workers, validate_worker_capacity};
use super::{ImagePrefetchCoordinator, ImageWorkerPool, runtime_error};
use crate::errors::{RivetError, RivetResult};
use crate::pipeline::op::ExecutionPlan;
use crate::sample::image::ImageBatch;
use crate::sampler::IndexSampler;
use std::sync::Arc;

enum LoaderExecutor {
    /// Direct synchronous execution, either because workers are disabled or
    /// because a dense decoded source can read a complete batch natively.
    Inline,
    /// Persistent worker pool with bounded cross-batch prefetch.
    Workers(ImageWorkerPool, ImagePrefetchCoordinator),
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
        let executor = if num_workers == 0 || plan.can_use_batch_native() {
            LoaderExecutor::Inline
        } else {
            // prefetch_batches counts *future* batches: the current batch
            // plus that many prepared ahead are in flight.
            let max_in_flight = prefetch_batches.saturating_add(1);
            // Reject absurd sizes before constructing any channel: a wrap
            // here would silently produce a broken bounded queue.
            let max_in_flight_samples = validate_worker_capacity(&plan, max_in_flight)?;
            let worker_plan = Arc::clone(&plan);
            let pool = ImageWorkerPool::new(
                num_workers,
                max_in_flight_samples,
                move |sample, index| worker_plan.apply_sample_ops(sample, index),
                move |worker_id, index| {
                    RivetError::Worker(format!(
                        "worker {worker_id} panicked while processing sample {index}"
                    ))
                },
            )
            .map_err(runtime_error)?;
            LoaderExecutor::Workers(pool, ImagePrefetchCoordinator::new(max_in_flight))
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
    use crate::pipeline::ImagePipeline;
    use crate::sample::image::EncodedImageSample;
    use arrow_buffer::Buffer;
    use rivet_core::DType;
    use rivet_data::dataset::Dataset;
    use std::sync::atomic::{AtomicUsize, Ordering};

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

        fn get_many(&self, indices: &[usize]) -> rivet_data::DataResult<Vec<Self::Item>> {
            indices.iter().map(|&index| self.get_one(index)).collect()
        }
    }

    impl SampleDataset {
        fn get_one(&self, index: usize) -> rivet_data::DataResult<EncodedImageSample> {
            if self.err_at == Some(index) {
                return Err(rivet_data::errors::invalid_argument("boom"));
            }
            if self.panic_at == Some(index) {
                panic!("sample {index} panics");
            }
            if self.slow_first_batch_ms > 0 && index < 8 {
                std::thread::sleep(std::time::Duration::from_millis(self.slow_first_batch_ms));
            }
            Ok(EncodedImageSample {
                image: Buffer::from(PNG_1X1.to_vec()),
                label: index as i64,
            })
        }
    }

    struct CountingDataset {
        len: usize,
        calls: Arc<AtomicUsize>,
    }

    impl Dataset for CountingDataset {
        type Item = EncodedImageSample;

        fn len(&self) -> usize {
            self.len
        }

        fn get_many(&self, indices: &[usize]) -> rivet_data::DataResult<Vec<Self::Item>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            indices
                .iter()
                .map(|&index| {
                    Ok(EncodedImageSample {
                        image: Buffer::from(PNG_1X1.to_vec()),
                        label: index as i64,
                    })
                })
                .collect()
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

    #[test]
    fn source_fetch_scales_with_logical_batches() {
        for workers in [0, 3] {
            let calls = Arc::new(AtomicUsize::new(0));
            let dataset = Arc::new(CountingDataset {
                len: 32,
                calls: Arc::clone(&calls),
            });
            let mut loader = ImagePipeline::new(dataset)
                .decode_image()
                .workers(workers)
                .batch(8, false)
                .compile()
                .unwrap();

            let batches = drain(&mut loader);
            assert_eq!(batches.len(), 4);
            assert_eq!(calls.load(Ordering::Relaxed), 4, "workers={workers}");
        }
    }

    fn drain(loader: &mut ImageDataLoader) -> Vec<ImageBatch> {
        loader.into_iter().collect::<RivetResult<Vec<_>>>().unwrap()
    }

    fn assert_batches_equal(left: &[ImageBatch], right: &[ImageBatch]) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right) {
            assert_eq!(labels(a), labels(b));
            assert_eq!(a.images.dims(), b.images.dims());
            assert_eq!(a.images.dtype(), b.images.dtype());
            match a.images.dtype() {
                DType::U8 => assert_eq!(
                    a.images.to_vec::<u8>().unwrap(),
                    b.images.to_vec::<u8>().unwrap()
                ),
                DType::F32 => assert_eq!(
                    a.images.to_vec::<f32>().unwrap(),
                    b.images.to_vec::<f32>().unwrap()
                ),
                dtype => panic!("unexpected test dtype: {dtype:?}"),
            }
        }
    }

    fn labels(batch: &ImageBatch) -> Vec<i64> {
        batch.labels.to_vec::<i64>().unwrap()
    }

    #[test]
    fn workers_match_inline_order_and_content() {
        let mut inline = pipeline(37, 0).batch(8, false).compile().unwrap();
        let mut pooled = pipeline(37, 4).batch(8, false).compile().unwrap();

        let inline = drain(&mut inline);
        let pooled = drain(&mut pooled);
        assert_batches_equal(&inline, &pooled);
        assert_eq!(inline.iter().map(|b| b.labels.dims()[0]).sum::<usize>(), 37);
    }

    #[test]
    fn workers_match_inline_for_batch_stage() {
        let mut inline = pipeline(17, 0)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .batch(4, false)
            .compile()
            .unwrap();
        let mut pooled = pipeline(17, 3)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .hwc_to_chw()
            .batch(4, false)
            .compile()
            .unwrap();

        let inline = drain(&mut inline);
        let pooled = drain(&mut pooled);
        assert_batches_equal(&inline, &pooled);
        assert_eq!(inline[0].images.dims()[1], 3);
        assert_eq!(inline[0].images.dtype(), rivet_core::DType::F32);
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
        assert_eq!(labels(&inline[0]), vec![2, 3, 4, 5, 6]);
        assert_eq!(labels(&inline[1]), vec![7, 8, 9, 10, 11]);
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
            assert_eq!(labels(&batches[0]), vec![30, 31, 32, 33, 34]);
            assert_eq!(labels(&batches[1]), vec![35, 36, 37, 38, 39]);
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
        assert!(second.to_string().contains("failed state"), "got: {second}");
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
            seen.extend(labels(&batch));
        }
        assert_eq!(seen, (0..20).collect::<Vec<i64>>());
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
        assert!(second.to_string().contains("failed state"), "got: {second}");
    }

    #[test]
    fn iterator_stops_at_eof() {
        let mut loader = pipeline(10, 0).batch(4, false).compile().unwrap();

        let batches = (&mut loader)
            .into_iter()
            .collect::<RivetResult<Vec<_>>>()
            .unwrap();
        assert_eq!(batches.len(), 3);
        assert_eq!(labels(&batches[0]), vec![0, 1, 2, 3]);

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

    #[test]
    fn shuffle_is_deterministic_across_workers() {
        let mut loader_a = pipeline(50, 0)
            .shuffle(7)
            .batch(8, false)
            .compile()
            .unwrap();
        let mut loader_b = pipeline(50, 4)
            .shuffle(7)
            .batch(8, false)
            .prefetch_batches(2)
            .compile()
            .unwrap();

        let a = drain(&mut loader_a);
        let b = drain(&mut loader_b);
        assert_batches_equal(&a, &b);

        let mut seen = Vec::new();
        for batch in &a {
            seen.extend(labels(batch));
        }
        assert_eq!(
            seen.iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            50
        );

        // A different seed changes the order.
        let mut loader_c = pipeline(50, 0)
            .shuffle(8)
            .batch(8, false)
            .compile()
            .unwrap();
        let c = drain(&mut loader_c);
        let c_labels: Vec<i64> = c.iter().flat_map(labels).collect();
        assert_ne!(c_labels, seen, "different seeds must reorder");
    }

    #[test]
    fn shuffle_applies_after_skip_and_take() {
        // Window is rows 10..=29 after skip/take; the seed only permutes
        // that window.
        let mut loader = pipeline(40, 0)
            .skip(10)
            .take(20)
            .shuffle(3)
            .batch(5, false)
            .compile()
            .unwrap();
        let batches = drain(&mut loader);
        let mut seen: Vec<i64> = batches.iter().flat_map(labels).collect();
        seen.sort_unstable();
        assert_eq!(seen, (10..30).collect::<Vec<i64>>(), "window preserved");
    }
}
