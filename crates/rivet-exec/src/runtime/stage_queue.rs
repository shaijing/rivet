use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use super::{RuntimeResult, runtime_error};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageQueueLimits {
    pub max_items: usize,
    pub max_bytes: usize,
}

impl StageQueueLimits {
    pub fn validate(self) -> RuntimeResult<Self> {
        if self.max_items == 0 {
            return Err(runtime_error(
                "stage queue max_items must be greater than 0",
            ));
        }
        if self.max_bytes == 0 {
            return Err(runtime_error(
                "stage queue max_bytes must be greater than 0",
            ));
        }
        Ok(self)
    }
}

#[derive(Debug)]
pub struct StageMessage<T> {
    pub sequence_id: u64,
    pub bytes: usize,
    pub value: T,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageQueueSnapshot {
    pub queued_items: usize,
    pub queued_bytes: usize,
    pub peak_items: usize,
    pub peak_bytes: usize,
    pub sends: u64,
    pub receives: u64,
    pub producer_wait: Duration,
    pub consumer_wait: Duration,
}

struct QueueState<T> {
    items: VecDeque<StageMessage<T>>,
    bytes: usize,
    peak_items: usize,
    peak_bytes: usize,
    sends: u64,
    receives: u64,
    producer_wait: Duration,
    consumer_wait: Duration,
    closed: bool,
    cancelled: bool,
}

struct QueueInner<T> {
    limits: StageQueueLimits,
    state: Mutex<QueueState<T>>,
    readable: Condvar,
    writable: Condvar,
}

/// A FIFO edge between persistent execution stages.
///
/// Both limits are enforced before an item is retained. An item larger than
/// `max_bytes` is rejected so the configured byte bound remains strict.
pub struct BoundedStageQueue<T> {
    inner: Arc<QueueInner<T>>,
}

impl<T> Clone for BoundedStageQueue<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> BoundedStageQueue<T> {
    pub fn new(limits: StageQueueLimits) -> RuntimeResult<Self> {
        let limits = limits.validate()?;
        Ok(Self {
            inner: Arc::new(QueueInner {
                limits,
                state: Mutex::new(QueueState {
                    items: VecDeque::new(),
                    bytes: 0,
                    peak_items: 0,
                    peak_bytes: 0,
                    sends: 0,
                    receives: 0,
                    producer_wait: Duration::ZERO,
                    consumer_wait: Duration::ZERO,
                    closed: false,
                    cancelled: false,
                }),
                readable: Condvar::new(),
                writable: Condvar::new(),
            }),
        })
    }

    pub fn limits(&self) -> StageQueueLimits {
        self.inner.limits
    }

    /// Enqueue an item, blocking while either queue budget is exhausted.
    /// Returns time spent waiting for capacity.
    pub fn send(&self, sequence_id: u64, value: T, bytes: usize) -> RuntimeResult<Duration> {
        if bytes > self.inner.limits.max_bytes {
            return Err(runtime_error(format!(
                "stage item {sequence_id} is {bytes} bytes, exceeding queue max_bytes {}",
                self.inner.limits.max_bytes
            )));
        }

        let mut state = self.inner.state.lock().expect("stage queue poisoned");
        let mut waited = Duration::ZERO;
        let mut wait_started: Option<Instant> = None;
        loop {
            if state.closed || state.cancelled {
                return Err(runtime_error("stage queue is closed"));
            }
            let has_item_capacity = state.items.len() < self.inner.limits.max_items;
            let has_byte_capacity = state
                .bytes
                .checked_add(bytes)
                .is_some_and(|total| total <= self.inner.limits.max_bytes);
            if has_item_capacity && has_byte_capacity {
                if let Some(started) = wait_started.take() {
                    waited += started.elapsed();
                    state.producer_wait += started.elapsed();
                }
                state.bytes += bytes;
                state.sends += 1;
                state.items.push_back(StageMessage {
                    sequence_id,
                    bytes,
                    value,
                });
                state.peak_items = state.peak_items.max(state.items.len());
                state.peak_bytes = state.peak_bytes.max(state.bytes);
                self.inner.readable.notify_one();
                return Ok(waited);
            }
            let started = wait_started.get_or_insert_with(Instant::now);
            let (next, _) = self
                .inner
                .writable
                .wait_timeout(state, Duration::from_millis(10))
                .expect("stage queue poisoned while waiting for capacity");
            state = next;
            if state.closed || state.cancelled {
                waited += started.elapsed();
                state.producer_wait += started.elapsed();
                return Err(runtime_error("stage queue closed while producer waited"));
            }
        }
    }

    /// Dequeue the next item and return time spent waiting for data. `None`
    /// means the queue was closed and drained, or cancelled.
    pub fn recv(&self) -> RuntimeResult<(Option<StageMessage<T>>, Duration)> {
        let mut state = self.inner.state.lock().expect("stage queue poisoned");
        let mut waited = Duration::ZERO;
        let mut wait_started: Option<Instant> = None;
        loop {
            if state.cancelled {
                return Ok((None, waited));
            }
            if let Some(item) = state.items.pop_front() {
                if let Some(started) = wait_started.take() {
                    waited += started.elapsed();
                    state.consumer_wait += started.elapsed();
                }
                state.bytes -= item.bytes;
                state.receives += 1;
                self.inner.writable.notify_all();
                return Ok((Some(item), waited));
            }
            if state.closed {
                return Ok((None, waited));
            }
            let started = wait_started.get_or_insert_with(Instant::now);
            let (next, _) = self
                .inner
                .readable
                .wait_timeout(state, Duration::from_millis(10))
                .expect("stage queue poisoned while waiting for data");
            state = next;
            if state.cancelled {
                waited += started.elapsed();
                state.consumer_wait += started.elapsed();
                return Ok((None, waited));
            }
        }
    }

    /// Stop accepting new work while allowing the consumer to drain queued
    /// items before `recv` returns `None`.
    pub fn close(&self) {
        let mut state = self.inner.state.lock().expect("stage queue poisoned");
        state.closed = true;
        self.inner.readable.notify_all();
        self.inner.writable.notify_all();
    }

    /// Drop queued values and wake blocked producers/consumers. Used during
    /// pipeline cancellation and early loader drop.
    pub fn cancel(&self) {
        let mut state = self.inner.state.lock().expect("stage queue poisoned");
        state.cancelled = true;
        state.closed = true;
        state.items.clear();
        state.bytes = 0;
        self.inner.readable.notify_all();
        self.inner.writable.notify_all();
    }

    pub fn snapshot(&self) -> StageQueueSnapshot {
        let state = self.inner.state.lock().expect("stage queue poisoned");
        StageQueueSnapshot {
            queued_items: state.items.len(),
            queued_bytes: state.bytes,
            peak_items: state.peak_items,
            peak_bytes: state.peak_bytes,
            sends: state.sends,
            receives: state.receives,
            producer_wait: state.producer_wait,
            consumer_wait: state.consumer_wait,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BoundedStageQueue, StageQueueLimits};
    use std::sync::mpsc;
    use std::time::Duration;

    fn limits(items: usize, bytes: usize) -> StageQueueLimits {
        StageQueueLimits {
            max_items: items,
            max_bytes: bytes,
        }
    }

    #[test]
    fn queue_preserves_sequence_and_enforces_item_and_byte_budgets() {
        let queue = BoundedStageQueue::new(limits(1, 4)).unwrap();
        queue.send(4, "abcd", 4).unwrap();
        let (received, _) = queue.recv().unwrap();
        let received = received.unwrap();
        assert_eq!(received.sequence_id, 4);
        assert_eq!(received.value, "abcd");
        assert_eq!(received.bytes, 4);
        assert_eq!(queue.snapshot().peak_items, 1);
        assert_eq!(queue.snapshot().peak_bytes, 4);
        assert!(
            queue
                .send(5, "oversized", 5)
                .unwrap_err()
                .to_string()
                .contains("max_bytes")
        );
    }

    #[test]
    fn producer_waits_for_capacity_then_preserves_fifo_order() {
        let queue = BoundedStageQueue::new(limits(1, 8)).unwrap();
        queue.send(0, 0usize, 8).unwrap();
        let producer_queue = queue.clone();
        let (done_tx, done_rx) = mpsc::channel();
        let producer = std::thread::spawn(move || {
            let waited = producer_queue.send(1, 1usize, 8).unwrap();
            done_tx.send(waited).unwrap();
        });

        assert!(done_rx.recv_timeout(Duration::from_millis(30)).is_err());
        assert_eq!(queue.snapshot().queued_items, 1);
        assert_eq!(queue.recv().unwrap().0.unwrap().sequence_id, 0);
        assert!(done_rx.recv_timeout(Duration::from_secs(1)).unwrap() > Duration::ZERO);
        assert_eq!(queue.recv().unwrap().0.unwrap().sequence_id, 1);
        producer.join().unwrap();
    }

    #[test]
    fn cancellation_unblocks_a_waiting_producer() {
        let queue = BoundedStageQueue::new(limits(1, 1)).unwrap();
        queue.send(0, (), 1).unwrap();
        let producer_queue = queue.clone();
        let (done_tx, done_rx) = mpsc::channel();
        let producer = std::thread::spawn(move || {
            done_tx
                .send(producer_queue.send(1, (), 1).is_err())
                .unwrap();
        });
        assert!(done_rx.recv_timeout(Duration::from_millis(30)).is_err());
        queue.cancel();
        assert!(done_rx.recv_timeout(Duration::from_secs(1)).unwrap());
        producer.join().unwrap();
        assert!(queue.recv().unwrap().0.is_none());
    }

    #[test]
    fn close_drains_items_then_reports_end_of_stream() {
        let queue = BoundedStageQueue::new(limits(2, 2)).unwrap();
        queue.send(9, 17, 1).unwrap();
        queue.close();
        assert_eq!(queue.recv().unwrap().0.unwrap().sequence_id, 9);
        assert!(queue.recv().unwrap().0.is_none());
        assert!(queue.send(10, 18, 1).is_err());
    }
}
