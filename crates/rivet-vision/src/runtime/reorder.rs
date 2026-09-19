use super::batch::finish_samples;
use crate::errors::{RivetError, RivetResult};
use crate::runtime::pool::WorkerPool;
use crate::runtime::worker::WorkItem;
use crate::sample::image::{DecodedSample, ImageBatch, ImageSample};
use std::collections::BTreeMap;

/// One in-flight logical batch being filled by workers.
pub(super) struct PendingBatch {
    pub(super) samples: Vec<Option<DecodedSample>>,
    pub(super) remaining: usize,
}

/// Tracks submitted-but-not-yet-delivered batches so several batches can
/// run concurrently while delivery stays strictly in sampler order.
pub(super) struct PrefetchCoordinator {
    /// Maximum batches in flight (window control only).
    pub(super) max_in_flight: usize,
    /// Number of submitted batches not yet delivered to the caller.
    pub(super) in_flight: usize,
    /// Sampler produced its last batch (or its tail was dropped).
    pub(super) closed: bool,
    /// Next id handed out when submitting a batch.
    pub(super) next_batch_id: u64,
    /// Next batch that must be returned to the caller, in order.
    pub(super) next_deliver_id: u64,
    pub(super) pending: BTreeMap<u64, PendingBatch>,
}

impl PrefetchCoordinator {
    pub(super) fn new(max_in_flight: usize) -> Self {
        Self {
            max_in_flight,
            in_flight: 0,
            closed: false,
            next_batch_id: 0,
            next_deliver_id: 0,
            pending: BTreeMap::new(),
        }
    }

    pub(super) fn submit(
        &mut self,
        indices: Vec<usize>,
        samples: Vec<ImageSample>,
        pool: &WorkerPool,
    ) -> RivetResult<()> {
        if indices.len() != samples.len() {
            return Err(RivetError::Worker(format!(
                "source returned {} samples for {} indices",
                samples.len(),
                indices.len()
            )));
        }
        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;

        let batch = PendingBatch {
            samples: std::iter::repeat_with(|| None)
                .take(indices.len())
                .collect(),
            remaining: indices.len(),
        };
        self.pending.insert(batch_id, batch);
        self.in_flight += 1;

        for (position, (index, sample)) in indices.into_iter().zip(samples).enumerate() {
            pool.submit(WorkItem {
                batch_id,
                position,
                index,
                sample,
            })?;
        }
        Ok(())
    }

    pub(super) fn record(
        &mut self,
        batch_id: u64,
        position: usize,
        sample: DecodedSample,
    ) -> RivetResult<()> {
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
    pub(super) fn deliver_ready(&mut self) -> RivetResult<Option<ImageBatch>> {
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

        let capacity = batch.samples.len();
        Ok(Some(finish_samples(
            batch.samples.into_iter().flatten(),
            capacity,
        )?))
    }
}
