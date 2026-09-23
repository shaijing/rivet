use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use rivet_data::sampler::IndexSampler;

use crate::physical::{
    ExecutionLane, KernelStage, PhysNodeId, PhysicalGraph, PhysicalNodeKind, PhysicalNodeSpec,
    PhysicalProfiler, TransferKind,
};

use super::stages::{PersistentStageGraph, StageNodes};
use super::{RuntimeError, RuntimeResult, StageQueueLimits, runtime_error};

/// Domain callbacks used by the physical pipeline executor. The executor owns
/// sampling order, source-to-worker dispatch, prefetch, result ordering, and
/// terminal failure state. The adapter owns only domain values and kernels.
pub trait PhysicalPipelineAdapter: Send + Sync + 'static {
    type Sample: Send + 'static;
    type Output: Send + 'static;
    type Batch: Send + 'static;
    type BatchBuilder;
    type Error: Send + 'static;

    fn batch_size(&self) -> usize;
    fn drop_last(&self) -> bool;
    fn is_batch_native(&self) -> bool;
    fn fetch_samples(&self, indices: &[usize]) -> Result<Vec<Self::Sample>, Self::Error>;
    /// Decode or normalize source representation before CPU semantic ops.
    /// The default is identity for adapters whose source is already decoded.
    fn decode_sample(
        &self,
        sample: Self::Sample,
        _sample_index: usize,
    ) -> Result<Self::Sample, Self::Error> {
        Ok(sample)
    }
    fn process_sample(
        &self,
        sample: Self::Sample,
        sample_index: usize,
    ) -> Result<Self::Output, Self::Error>;
    fn worker_panic_error(&self, worker_id: usize, sample_index: usize) -> Self::Error;

    fn sample_bytes(&self, _sample: &Self::Sample) -> usize {
        0
    }

    fn output_bytes(&self, _output: &Self::Output) -> usize {
        0
    }

    fn batch_bytes(&self, _batch: &Self::Batch) -> usize {
        0
    }

    /// Batch-native sources can bypass per-sample materialization and the
    /// sample kernel stage. Return `None` only when `is_batch_native()` is
    /// false.
    fn fetch_batch(&self, _indices: &[usize]) -> Result<Option<Self::Batch>, Self::Error> {
        Ok(None)
    }

    fn batch_builder(&self, capacity: usize) -> Self::BatchBuilder;
    fn push_batch_sample(
        &self,
        builder: &mut Self::BatchBuilder,
        sample: Self::Output,
    ) -> Result<(), Self::Error>;
    fn finish_batch(&self, builder: Self::BatchBuilder) -> Result<Self::Batch, Self::Error>;
    fn apply_batch(&self, batch: Self::Batch) -> Result<Self::Batch, Self::Error>;

    /// Execute a batch operation with the logical source indices that formed
    /// this batch. Stochastic batch-native adapters use these indices to keep
    /// augmentation parameters independent of worker scheduling and batching.
    fn apply_batch_with_indices(
        &self,
        batch: Self::Batch,
        _indices: &[usize],
    ) -> Result<Self::Batch, Self::Error> {
        self.apply_batch(batch)
    }

    /// Execute one explicitly planned residency transition. Adapters that do
    /// not support transfer nodes fail closed instead of hiding a copy in a
    /// kernel or sink callback.
    fn transfer_batch(
        &self,
        _batch: Self::Batch,
        _kind: TransferKind,
        _target: ExecutionLane,
    ) -> Result<Self::Batch, PipelineError<Self::Error>> {
        Err(PipelineError::Runtime(runtime_error(
            "physical graph contains a transfer node unsupported by this adapter",
        )))
    }

    /// Execute a batch-stage kernel placed on a device lane after any explicit
    /// transfer node. The default fails closed for adapters without device
    /// kernels.
    fn apply_device_batch(
        &self,
        _batch: Self::Batch,
        _target: ExecutionLane,
    ) -> Result<Self::Batch, PipelineError<Self::Error>> {
        Err(PipelineError::Runtime(runtime_error(
            "physical graph contains a device batch kernel unsupported by this adapter",
        )))
    }

    /// Device-kernel counterpart of [`Self::apply_batch_with_indices`].
    fn apply_device_batch_with_indices(
        &self,
        batch: Self::Batch,
        target: ExecutionLane,
        _indices: &[usize],
    ) -> Result<Self::Batch, PipelineError<Self::Error>> {
        self.apply_device_batch(batch, target)
    }
}

#[derive(Debug)]
pub enum PipelineError<E> {
    Runtime(RuntimeError),
    Domain(E),
}

impl<E: fmt::Display> fmt::Display for PipelineError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(f),
            Self::Domain(error) => error.fmt(f),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for PipelineError<E> {}

/// Persistent physical pipeline runtime used by domain adapters such as vision.
/// Stage queues bound retained work, while sequence ids preserve sampler order.
pub struct PhysicalPipelineExecutor<P: PhysicalPipelineAdapter> {
    adapter: Arc<P>,
    max_in_flight_batches: usize,
    graph: PhysicalGraph,
    sampler_node: Option<PhysNodeId>,
    stages: PersistentStageGraph<P>,
    profiler: PhysicalProfiler,
    num_workers: usize,
    next_sequence_id: u64,
    next_output_sequence_id: u64,
    in_flight_batches: usize,
    source_exhausted: bool,
    failed: bool,
}

impl<P: PhysicalPipelineAdapter> PhysicalPipelineExecutor<P> {
    pub fn new(
        adapter: Arc<P>,
        num_workers: usize,
        prefetch_batches: usize,
    ) -> RuntimeResult<Self> {
        let graph = default_graph(adapter.is_batch_native()).map_err(graph_runtime_error)?;
        Self::with_graph(adapter, num_workers, prefetch_batches, graph)
    }

    /// Create a runtime from the physical graph lowered by a domain plugin.
    pub fn with_graph(
        adapter: Arc<P>,
        num_workers: usize,
        prefetch_batches: usize,
        graph: PhysicalGraph,
    ) -> RuntimeResult<Self> {
        Self::with_graph_limits(
            adapter,
            num_workers,
            prefetch_batches,
            512 * 1024 * 1024,
            graph,
        )
    }

    pub fn with_graph_limits(
        adapter: Arc<P>,
        num_workers: usize,
        prefetch_batches: usize,
        stage_queue_max_bytes: usize,
        mut graph: PhysicalGraph,
    ) -> RuntimeResult<Self> {
        let batch_size = adapter.batch_size();
        if batch_size == 0 {
            return Err(runtime_error(
                "physical pipeline batch size must be greater than 0",
            ));
        }

        graph.validate().map_err(graph_runtime_error)?;
        let order = graph.execution_order().map_err(graph_runtime_error)?;
        let mut positions = vec![0usize; graph.nodes().len()];
        for (position, id) in order.iter().copied().enumerate() {
            positions[id.index()] = position;
        }
        let source_node = find_node(&graph, |kind| kind == PhysicalNodeKind::Source)
            .ok_or_else(|| runtime_error("physical pipeline graph has no source node"))?;
        let sampler_node = find_node(&graph, |kind| kind == PhysicalNodeKind::Sampler);
        let decode_node = find_node(&graph, |kind| {
            kind == PhysicalNodeKind::Kernel(KernelStage::Decode)
        });
        let sample_node = find_node(&graph, |kind| {
            kind == PhysicalNodeKind::Kernel(KernelStage::Sample)
        });
        if decode_node
            .is_some_and(|decode| positions[decode.index()] <= positions[source_node.index()])
        {
            return Err(runtime_error("decode stage must follow source reads"));
        }
        if sample_node.is_some_and(|sample| {
            let previous = decode_node.unwrap_or(source_node);
            positions[sample.index()] <= positions[previous.index()]
        }) {
            return Err(runtime_error(
                "sample kernel must follow source/decode stages",
            ));
        }
        let batch_node = find_node(&graph, |kind| kind == PhysicalNodeKind::Batch);
        let batch_node =
            batch_node.ok_or_else(|| runtime_error("physical pipeline graph has no batch node"))?;
        if positions[batch_node.index()] <= positions[source_node.index()] {
            return Err(runtime_error(
                "physical batch barrier must follow source reads",
            ));
        }
        if sample_node
            .is_some_and(|sample| positions[sample.index()] >= positions[batch_node.index()])
        {
            return Err(runtime_error(
                "sample kernel must precede the physical batch barrier",
            ));
        }
        let batch_kernel = find_node(&graph, |kind| {
            kind == PhysicalNodeKind::Kernel(KernelStage::Batch)
        });
        if batch_kernel
            .is_some_and(|kernel| positions[kernel.index()] <= positions[batch_node.index()])
        {
            return Err(runtime_error(
                "batch kernel must follow the physical batch barrier",
            ));
        }
        let root = graph.root().map_err(graph_runtime_error)?;
        if graph.node(root).map_err(graph_runtime_error)?.kind != PhysicalNodeKind::Sink {
            return Err(runtime_error("physical pipeline graph root must be a sink"));
        }
        let mut transfer_nodes = Vec::new();
        for id in &order {
            let node = graph.node(*id).map_err(graph_runtime_error)?;
            if let PhysicalNodeKind::Transfer(kind) = node.kind {
                let target = node.transfer_target.ok_or_else(|| {
                    runtime_error(format!("transfer node p{} has no target lane", id.index()))
                })?;
                transfer_nodes.push((*id, kind, target));
            }
        }
        if transfer_nodes.len() > 1 {
            return Err(runtime_error(
                "the initial physical pipeline supports one transfer boundary",
            ));
        }
        let batch_kernel_node = batch_kernel.unwrap_or(if transfer_nodes.is_empty() {
            root
        } else {
            batch_node
        });
        let device_batch_kernel = batch_kernel.is_some_and(|kernel| {
            matches!(
                graph.node(kernel).map(|node| node.lane),
                Ok(ExecutionLane::Device { .. })
            )
        });
        if let Some((transfer, _, _)) = transfer_nodes.first() {
            let expected_consumer = if device_batch_kernel {
                batch_kernel_node
            } else {
                root
            };
            let consumer_inputs = graph
                .node(expected_consumer)
                .map_err(graph_runtime_error)?
                .inputs
                .as_slice();
            let is_after_batch = positions[transfer.index()] > positions[batch_node.index()];
            let is_before_consumer =
                positions[transfer.index()] < positions[expected_consumer.index()];
            if !is_after_batch
                || !is_before_consumer
                || consumer_inputs != std::slice::from_ref(transfer)
            {
                return Err(runtime_error(
                    "the initial physical pipeline requires its transfer to feed the device batch kernel or sink directly",
                ));
            }
        }
        if !device_batch_kernel
            && transfer_nodes.iter().any(|(transfer, _, _)| {
                positions[transfer.index()] <= positions[batch_kernel_node.index()]
            })
        {
            return Err(runtime_error(
                "CPU batch kernels must precede the explicit transfer boundary",
            ));
        }

        let max_in_flight_batches = prefetch_batches
            .checked_add(1)
            .ok_or_else(|| runtime_error("stage graph batch window overflow"))?;
        let worker_capacity = batch_size;
        let queue_limits = StageQueueLimits {
            max_items: max_in_flight_batches,
            max_bytes: stage_queue_max_bytes,
        }
        .validate()?;
        let batch_kernel_node = batch_kernel.unwrap_or(if transfer_nodes.is_empty() {
            root
        } else {
            batch_node
        });
        let decode_node = decode_node.unwrap_or(source_node);
        let device_target = graph
            .node(batch_kernel_node)
            .map_err(graph_runtime_error)?
            .lane;
        let nodes = StageNodes {
            source: source_node,
            decode: decode_node,
            sample: sample_node,
            batch: batch_node,
            batch_kernel: batch_kernel_node,
            sink: root,
            device_batch_kernel,
            device_target,
            transfer_nodes,
        };
        let profiler = PhysicalProfiler::default();
        let inter_sample_workers = if adapter.is_batch_native() {
            0
        } else {
            num_workers
        };
        let stages = PersistentStageGraph::new(
            Arc::clone(&adapter),
            inter_sample_workers,
            worker_capacity,
            queue_limits,
            nodes,
            profiler.clone(),
        )?;

        Ok(Self {
            adapter,
            max_in_flight_batches,
            graph,
            sampler_node,
            stages,
            profiler,
            num_workers,
            next_sequence_id: 0,
            next_output_sequence_id: 0,
            in_flight_batches: 0,
            source_exhausted: false,
            failed: false,
        })
    }

    pub fn physical_explain(&mut self) -> Result<String, RuntimeError> {
        let mut explanation = self.graph.explain().map_err(graph_runtime_error)?;
        explanation.push_str(&format!(
            "Runtime parallelism: inter_sample_workers={} intra_op=backend_managed\n",
            self.num_workers
        ));
        explanation.push_str(&self.stages.queue_explain());
        Ok(explanation)
    }

    pub fn profiler(&self) -> PhysicalProfiler {
        self.profiler.clone()
    }

    pub fn next_batch(
        &mut self,
        sampler: &mut IndexSampler,
    ) -> Result<Option<P::Batch>, PipelineError<P::Error>> {
        if self.failed {
            return Err(PipelineError::Runtime(runtime_error(
                "loader is in failed state after a previous iteration error",
            )));
        }

        while self.in_flight_batches < self.max_in_flight_batches && !self.source_exhausted {
            let started = Instant::now();
            let Some(indices) = sampler.next_indices(self.adapter.batch_size()) else {
                self.source_exhausted = true;
                self.stages.close_requests();
                break;
            };
            if self.adapter.drop_last() && indices.len() < self.adapter.batch_size() {
                self.source_exhausted = true;
                self.stages.close_requests();
                break;
            }
            let sequence_id = self.next_sequence_id;
            let bytes = indices.len().saturating_mul(std::mem::size_of::<usize>());
            if let Some(node) = self.sampler_node {
                self.profiler.record(node, started.elapsed(), 0, bytes);
            }
            let waited = match self.stages.submit(sequence_id, indices) {
                Ok(waited) => waited,
                Err(error) => {
                    self.failed = true;
                    self.stages.cancel_upstream();
                    return Err(PipelineError::Runtime(error));
                }
            };
            if let Some(node) = self.sampler_node {
                self.profiler.record_wait(node, waited);
            }
            self.next_sequence_id = self.next_sequence_id.checked_add(1).ok_or_else(|| {
                self.failed = true;
                PipelineError::Runtime(runtime_error("stage sequence id overflow"))
            })?;
            self.in_flight_batches += 1;
        }

        if self.in_flight_batches == 0 && self.source_exhausted {
            return Ok(None);
        }

        let (message, waited) = self.stages.recv_output().map_err(PipelineError::Runtime)?;
        self.profiler.record_wait(
            self.graph
                .root()
                .map_err(graph_runtime_error)
                .map_err(PipelineError::Runtime)?,
            waited,
        );
        let Some(message) = message else {
            self.failed = true;
            self.stages.cancel_upstream();
            return if self.in_flight_batches == 0 && self.source_exhausted {
                Ok(None)
            } else {
                Err(PipelineError::Runtime(runtime_error(
                    "stage graph ended before all submitted sequences reached the sink",
                )))
            };
        };
        if message.sequence_id != self.next_output_sequence_id {
            self.failed = true;
            self.stages.cancel_upstream();
            return Err(PipelineError::Runtime(runtime_error(format!(
                "stage graph delivered sequence {}, expected {}",
                message.sequence_id, self.next_output_sequence_id
            ))));
        }
        self.next_output_sequence_id = match self.next_output_sequence_id.checked_add(1) {
            Some(sequence_id) => sequence_id,
            None => {
                self.failed = true;
                self.stages.cancel_upstream();
                return Err(PipelineError::Runtime(runtime_error(
                    "output sequence id overflow",
                )));
            }
        };
        self.in_flight_batches = self.in_flight_batches.saturating_sub(1);
        match message.value {
            Ok(batch) => Ok(Some(batch)),
            Err(error) => {
                self.failed = true;
                self.stages.cancel_upstream();
                Err(error)
            }
        }
    }

    pub fn max_in_flight_batches(&self) -> usize {
        self.max_in_flight_batches
    }

    pub fn stage_queue_limits(&self) -> StageQueueLimits {
        self.stages.limits()
    }
}

fn graph_runtime_error(error: impl fmt::Display) -> RuntimeError {
    runtime_error(format!("invalid physical pipeline graph: {error}"))
}

fn default_graph(batch_native: bool) -> Result<PhysicalGraph, crate::physical::PhysicalGraphError> {
    let mut graph = PhysicalGraph::new();
    let sampler = graph.add_node(
        None,
        PhysicalNodeSpec::new(PhysicalNodeKind::Sampler, ExecutionLane::Cpu),
        [],
    );
    let source = graph.add_node(
        None,
        PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Io),
        [sampler],
    );
    let decode = graph.add_node(
        None,
        PhysicalNodeSpec::new(
            PhysicalNodeKind::Kernel(KernelStage::Decode),
            ExecutionLane::Cpu,
        ),
        [source],
    );
    let sample_input = if batch_native {
        decode
    } else {
        graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Kernel(KernelStage::Sample),
                ExecutionLane::Cpu,
            ),
            [decode],
        )
    };
    let batch = graph.add_node(
        None,
        PhysicalNodeSpec::new(PhysicalNodeKind::Batch, ExecutionLane::Cpu),
        [sample_input],
    );
    let batch_kernel = graph.add_node(
        None,
        PhysicalNodeSpec::new(
            PhysicalNodeKind::Kernel(KernelStage::Batch),
            ExecutionLane::Cpu,
        ),
        [batch],
    );
    let sink = graph.add_node(
        None,
        PhysicalNodeSpec::new(PhysicalNodeKind::Sink, ExecutionLane::Cpu),
        [batch_kernel],
    );
    graph.set_root(sink)?;
    graph.validate()?;
    Ok(graph)
}

fn find_node(
    graph: &PhysicalGraph,
    predicate: impl Fn(PhysicalNodeKind) -> bool,
) -> Option<PhysNodeId> {
    graph
        .nodes()
        .iter()
        .find(|node| predicate(node.kind))
        .map(|node| node.id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical::TransferKind;
    use rivet_data::sampler::SamplerPlan;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    #[derive(Default)]
    struct OverlapProbe {
        device_started: std::sync::atomic::AtomicBool,
        source_during_device: AtomicUsize,
    }

    struct Adapter {
        calls: AtomicUsize,
        transfers: AtomicUsize,
        fail_transfer: bool,
        overlap_probe: Option<Arc<OverlapProbe>>,
    }

    impl PhysicalPipelineAdapter for Adapter {
        type Sample = usize;
        type Output = usize;
        type Batch = Vec<usize>;
        type BatchBuilder = Vec<usize>;
        type Error = String;

        fn batch_size(&self) -> usize {
            3
        }
        fn drop_last(&self) -> bool {
            false
        }
        fn is_batch_native(&self) -> bool {
            false
        }
        fn fetch_samples(&self, indices: &[usize]) -> Result<Vec<usize>, String> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if let Some(probe) = &self.overlap_probe {
                thread::sleep(Duration::from_millis(30));
                if probe.device_started.load(Ordering::Relaxed) {
                    probe.source_during_device.fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok(indices.to_vec())
        }
        fn process_sample(&self, sample: usize, index: usize) -> Result<usize, String> {
            Ok(sample + index)
        }
        fn worker_panic_error(&self, worker: usize, index: usize) -> String {
            format!("worker {worker} panicked at {index}")
        }
        fn batch_builder(&self, capacity: usize) -> Vec<usize> {
            Vec::with_capacity(capacity)
        }
        fn push_batch_sample(&self, builder: &mut Vec<usize>, sample: usize) -> Result<(), String> {
            builder.push(sample);
            Ok(())
        }
        fn finish_batch(&self, builder: Vec<usize>) -> Result<Vec<usize>, String> {
            Ok(builder)
        }
        fn batch_bytes(&self, batch: &Vec<usize>) -> usize {
            batch.len() * std::mem::size_of::<usize>()
        }
        fn sample_bytes(&self, _sample: &usize) -> usize {
            std::mem::size_of::<usize>()
        }
        fn output_bytes(&self, _output: &usize) -> usize {
            std::mem::size_of::<usize>()
        }
        fn apply_batch(&self, batch: Vec<usize>) -> Result<Vec<usize>, String> {
            Ok(batch)
        }
        fn transfer_batch(
            &self,
            batch: Vec<usize>,
            kind: TransferKind,
            target: ExecutionLane,
        ) -> Result<Vec<usize>, PipelineError<String>> {
            assert_eq!(kind, TransferKind::HostToDevice);
            assert_eq!(target, ExecutionLane::Device { ordinal: 0 });
            if self.fail_transfer {
                return Err(PipelineError::Domain("transfer failed".to_owned()));
            }
            if let Some(probe) = &self.overlap_probe {
                probe.device_started.store(true, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(120));
            }
            self.transfers.fetch_add(1, Ordering::Relaxed);
            Ok(batch)
        }
    }

    fn h2d_graph() -> (PhysicalGraph, PhysNodeId) {
        let mut graph = PhysicalGraph::new();
        let sampler = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Sampler, ExecutionLane::Cpu),
            [],
        );
        let source = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Io),
            [sampler],
        );
        let sample = graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Kernel(KernelStage::Sample),
                ExecutionLane::Cpu,
            ),
            [source],
        );
        let batch = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Batch, ExecutionLane::Cpu),
            [sample],
        );
        let batch_kernel = graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Kernel(KernelStage::Batch),
                ExecutionLane::Cpu,
            ),
            [batch],
        );
        let transfer = graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Transfer(TransferKind::HostToDevice),
                ExecutionLane::Transfer,
            )
            .with_transfer_target(ExecutionLane::Device { ordinal: 0 }),
            [batch_kernel],
        );
        let sink = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Sink, ExecutionLane::Device { ordinal: 0 }),
            [transfer],
        );
        graph.set_root(sink).unwrap();
        (graph, transfer)
    }

    #[test]
    fn physical_runtime_preserves_order_for_inline_and_worker_lanes() {
        for workers in [0, 3] {
            let adapter = Arc::new(Adapter {
                calls: AtomicUsize::new(0),
                transfers: AtomicUsize::new(0),
                fail_transfer: false,
                overlap_probe: None,
            });
            let mut executor = PhysicalPipelineExecutor::new(adapter, workers, 2).unwrap();
            let mut sampler = IndexSampler::new(
                SamplerPlan::Permutation {
                    indices: vec![2, 0, 1, 4, 3],
                },
                0,
            );
            let mut batches = Vec::new();
            while let Some(batch) = executor.next_batch(&mut sampler).unwrap() {
                batches.push(batch);
            }
            assert_eq!(batches, vec![vec![4, 0, 2], vec![8, 6]]);
            assert!(
                executor
                    .physical_explain()
                    .unwrap()
                    .contains("SampleKernel")
            );
        }
    }

    #[test]
    fn persistent_stages_overlap_source_with_device_work_and_keep_sequence_order() {
        let probe = Arc::new(OverlapProbe::default());
        let adapter = Arc::new(Adapter {
            calls: AtomicUsize::new(0),
            transfers: AtomicUsize::new(0),
            fail_transfer: false,
            overlap_probe: Some(Arc::clone(&probe)),
        });
        let (graph, transfer) = h2d_graph();
        let sink = graph.root().unwrap();
        let mut executor = PhysicalPipelineExecutor::with_graph(adapter, 3, 2, graph).unwrap();
        assert_eq!(executor.stage_queue_limits().max_items, 3);
        assert!(executor.stage_queue_limits().max_bytes > 0);

        let mut sampler = IndexSampler::new(SamplerPlan::Sequential { start: 0, end: 9 }, 0);
        let mut batches = Vec::new();
        while let Some(batch) = executor.next_batch(&mut sampler).unwrap() {
            batches.push(batch);
        }
        assert_eq!(batches, [vec![0, 2, 4], vec![6, 8, 10], vec![12, 14, 16]]);
        assert!(probe.source_during_device.load(Ordering::Relaxed) > 0);

        let profile = executor.profiler().snapshot();
        assert!(profile[&transfer].elapsed >= Duration::from_millis(120));
        assert!(profile[&sink].wait_elapsed > Duration::ZERO);
        assert!(profile[&sink].wait_events > 0);
        assert!(profile.values().any(|node| node.executions > 0));
        let explanation = executor.physical_explain().unwrap();
        assert!(explanation.contains("Runtime stage queues:"));
        assert!(explanation.contains("inter_sample_workers=3"));
        assert!(explanation.contains("Sampler->Source: max_items=3"));
        assert!(explanation.contains("CPU->Transfer: max_items=3"));
        assert!(explanation.contains("Transfer->Device: max_items=3"));
        assert!(explanation.contains("max_items=3"));
        assert!(explanation.contains("max_bytes="));
    }

    #[test]
    fn worker_batch_prefetch_limit_is_explicit_and_checked() {
        let adapter = Arc::new(Adapter {
            calls: AtomicUsize::new(0),
            transfers: AtomicUsize::new(0),
            fail_transfer: false,
            overlap_probe: None,
        });
        let executor = PhysicalPipelineExecutor::new(adapter, 2, 4).unwrap();
        assert_eq!(executor.max_in_flight_batches(), 5);
    }

    #[test]
    fn source_payload_over_byte_budget_is_reported_and_terminal() {
        let adapter = Arc::new(Adapter {
            calls: AtomicUsize::new(0),
            transfers: AtomicUsize::new(0),
            fail_transfer: false,
            overlap_probe: None,
        });
        let graph = default_graph(false).unwrap();
        let mut executor =
            PhysicalPipelineExecutor::with_graph_limits(adapter, 0, 0, 24, graph).unwrap();
        let mut sampler = IndexSampler::new(SamplerPlan::Sequential { start: 0, end: 3 }, 0);
        assert!(matches!(
            executor.next_batch(&mut sampler),
            Err(PipelineError::Runtime(RuntimeError::Message(message)))
                if message.contains("source batch is 48 bytes")
        ));
        assert!(matches!(
            executor.next_batch(&mut sampler),
            Err(PipelineError::Runtime(RuntimeError::Message(message)))
                if message.contains("failed state")
        ));
    }

    #[test]
    fn physical_runtime_executes_and_profiles_explicit_transfer() {
        let adapter = Arc::new(Adapter {
            calls: AtomicUsize::new(0),
            transfers: AtomicUsize::new(0),
            fail_transfer: false,
            overlap_probe: None,
        });
        let (graph, transfer) = h2d_graph();

        let mut executor =
            PhysicalPipelineExecutor::with_graph(adapter.clone(), 0, 0, graph).unwrap();
        let mut sampler = IndexSampler::new(SamplerPlan::Sequential { start: 0, end: 2 }, 0);
        let output = executor.next_batch(&mut sampler).unwrap().unwrap();
        assert_eq!(output, [0, 2]);
        assert_eq!(adapter.transfers.load(Ordering::Relaxed), 1);
        let profile = executor.profiler().snapshot();
        assert_eq!(profile[&transfer].executions, 1);
        assert_eq!(
            profile[&transfer].input_bytes,
            2 * std::mem::size_of::<usize>() as u64
        );
        assert_eq!(
            profile[&transfer].output_bytes,
            profile[&transfer].input_bytes
        );
    }

    #[test]
    fn transfer_failure_is_reported_and_makes_the_executor_terminal() {
        let adapter = Arc::new(Adapter {
            calls: AtomicUsize::new(0),
            transfers: AtomicUsize::new(0),
            fail_transfer: true,
            overlap_probe: None,
        });
        let (graph, _) = h2d_graph();
        let mut executor = PhysicalPipelineExecutor::with_graph(adapter, 0, 0, graph).unwrap();
        let mut sampler = IndexSampler::new(SamplerPlan::Sequential { start: 0, end: 2 }, 0);
        assert!(matches!(
            executor.next_batch(&mut sampler),
            Err(PipelineError::Domain(error)) if error == "transfer failed"
        ));
        assert!(matches!(
            executor.next_batch(&mut sampler),
            Err(PipelineError::Runtime(RuntimeError::Message(message)))
                if message.contains("failed state")
        ));
    }
}
