use super::pool::WorkerPool;
use super::{RuntimeResult, runtime_error};
use std::collections::BTreeMap;

/// One in-flight logical batch being filled by workers.
pub struct PendingBatch<R> {
    batch_id: u64,
    samples: Vec<Option<R>>,
    remaining: usize,
}

/// Consumes the completed slots of a pending batch without rebuilding them
/// into a second `Vec<R>`.
pub struct PendingSamples<R> {
    batch_id: u64,
    next_position: usize,
    samples: std::vec::IntoIter<Option<R>>,
}

impl<R> PendingBatch<R> {
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn into_results(self) -> PendingSamples<R> {
        PendingSamples {
            batch_id: self.batch_id,
            next_position: 0,
            samples: self.samples.into_iter(),
        }
    }
}

impl<R> Iterator for PendingSamples<R> {
    type Item = RuntimeResult<R>;

    fn next(&mut self) -> Option<Self::Item> {
        let position = self.next_position;
        let sample = self.samples.next()?;
        self.next_position += 1;
        Some(sample.ok_or_else(|| {
            runtime_error(format!(
                "missing result position {position} for batch {}",
                self.batch_id
            ))
        }))
    }
}

/// Tracks submitted-but-not-yet-delivered batches so several batches can run
/// concurrently while delivery stays strictly in submission order.
pub struct PrefetchCoordinator<R> {
    /// Maximum batches in flight (window control only).
    pub max_in_flight: usize,
    /// Number of submitted batches not yet delivered to the caller.
    pub in_flight: usize,
    /// Sampler produced its last batch (or its tail was dropped).
    pub closed: bool,
    next_batch_id: u64,
    next_deliver_id: u64,
    pending: BTreeMap<u64, PendingBatch<R>>,
}

impl<R> PrefetchCoordinator<R> {
    pub fn new(max_in_flight: usize) -> Self {
        Self {
            max_in_flight,
            in_flight: 0,
            closed: false,
            next_batch_id: 0,
            next_deliver_id: 0,
            pending: BTreeMap::new(),
        }
    }

    pub fn submit<T, E>(
        &mut self,
        indices: Vec<usize>,
        payloads: Vec<T>,
        pool: &WorkerPool<T, R, E>,
    ) -> RuntimeResult<()>
    where
        T: Send + 'static,
        R: Send + 'static,
        E: Send + 'static,
    {
        if indices.len() != payloads.len() {
            return Err(runtime_error(format!(
                "source returned {} payloads for {} indices",
                payloads.len(),
                indices.len()
            )));
        }
        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;

        self.pending.insert(
            batch_id,
            PendingBatch {
                batch_id,
                samples: std::iter::repeat_with(|| None)
                    .take(indices.len())
                    .collect(),
                remaining: indices.len(),
            },
        );
        self.in_flight += 1;

        for (position, (sample_index, payload)) in indices.into_iter().zip(payloads).enumerate() {
            pool.submit(super::WorkItem {
                batch_id,
                position,
                sample_index,
                payload,
            })?;
        }
        Ok(())
    }

    pub fn record(&mut self, batch_id: u64, position: usize, sample: R) -> RuntimeResult<()> {
        let batch = self
            .pending
            .get_mut(&batch_id)
            .ok_or_else(|| runtime_error(format!("result for unknown batch {batch_id}")))?;
        let slot = batch.samples.get_mut(position).ok_or_else(|| {
            runtime_error(format!(
                "invalid result position {position} for batch {batch_id}"
            ))
        })?;
        if slot.is_some() {
            return Err(runtime_error(format!(
                "duplicate result for batch {batch_id} position {position}"
            )));
        }
        *slot = Some(sample);
        batch.remaining -= 1;
        Ok(())
    }

    /// Take the next completed batch in submission order.
    pub fn take_ready(&mut self) -> RuntimeResult<Option<PendingBatch<R>>> {
        let batch_id = self.next_deliver_id;
        let Some(batch) = self.pending.get(&batch_id) else {
            return Ok(None);
        };
        if batch.remaining != 0 {
            return Ok(None);
        }
        let batch = self
            .pending
            .remove(&batch_id)
            .ok_or_else(|| runtime_error(format!("deliverable batch {batch_id} vanished")))?;
        self.in_flight -= 1;
        self.next_deliver_id += 1;

        Ok(Some(batch))
    }
}

#[cfg(test)]
mod tests {
    use super::{PendingBatch, PrefetchCoordinator};

    #[test]
    fn record_rejects_out_of_range_and_duplicate_positions() {
        let mut coordinator = PrefetchCoordinator::<usize>::new(2);
        coordinator.pending.insert(
            0,
            PendingBatch {
                batch_id: 0,
                samples: std::iter::repeat_with(|| None).take(2).collect(),
                remaining: 2,
            },
        );

        let err = coordinator.record(0, 5, 1).unwrap_err();
        assert!(err.to_string().contains("invalid result position"));

        coordinator.record(0, 1, 2).unwrap();
        let err = coordinator.record(0, 1, 3).unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn ready_batches_are_delivered_in_submission_order() {
        let mut coordinator = PrefetchCoordinator::<usize>::new(2);
        coordinator.pending.insert(
            0,
            PendingBatch {
                batch_id: 0,
                samples: vec![Some(10), Some(11)],
                remaining: 0,
            },
        );
        coordinator.pending.insert(
            1,
            PendingBatch {
                batch_id: 1,
                samples: vec![Some(20)],
                remaining: 0,
            },
        );
        coordinator.in_flight = 2;

        let first = coordinator.take_ready().unwrap().unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(
            first
                .into_results()
                .collect::<super::RuntimeResult<Vec<_>>>()
                .unwrap(),
            vec![10, 11]
        );
        let second = coordinator.take_ready().unwrap().unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(
            second
                .into_results()
                .collect::<super::RuntimeResult<Vec<_>>>()
                .unwrap(),
            vec![20]
        );
        assert!(coordinator.take_ready().unwrap().is_none());
    }
}
