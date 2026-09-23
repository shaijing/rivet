use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use rivet_data::sampler::IndexSampler;

use crate::physical::{
    ExecutionLane, KernelStage, PhysNodeId, PhysicalGraph, PhysicalNodeKind, PhysicalNodeSpec,
    PhysicalProfiler, TransferKind,
};

use super::{PrefetchCoordinator, RuntimeError, RuntimeResult, WorkerPool, runtime_error};

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

enum ExecutorState<P: PhysicalPipelineAdapter> {
    Inline,
    Workers(
        WorkerPool<P::Sample, P::Output, P::Error>,
        PrefetchCoordinator<P::Output>,
    ),
    Failed,
}

/// CPU physical pipeline runtime used by domain adapters such as vision.
///
/// Its physical graph is the executable stage contract. The executor owns all
/// generic scheduling and worker lifetimes; callbacks retain domain behavior.
pub struct PhysicalPipelineExecutor<P: PhysicalPipelineAdapter> {
    adapter: Arc<P>,
    state: ExecutorState<P>,
    max_in_flight_batches: usize,
    graph: PhysicalGraph,
    sampler_node: Option<PhysNodeId>,
    source_node: PhysNodeId,
    sample_node: Option<PhysNodeId>,
    batch_node: PhysNodeId,
    batch_kernel_node: PhysNodeId,
    device_batch_kernel: bool,
    transfer_nodes: Vec<(PhysNodeId, TransferKind, ExecutionLane)>,
    profiler: PhysicalProfiler,
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
        let sample_node = find_node(&graph, |kind| {
            kind == PhysicalNodeKind::Kernel(KernelStage::Sample)
        });
        if sample_node
            .is_some_and(|sample| positions[sample.index()] <= positions[source_node.index()])
        {
            return Err(runtime_error("sample kernel must follow source reads"));
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

        let state = if num_workers == 0 || adapter.is_batch_native() {
            ExecutorState::Inline
        } else {
            // The current batch plus all configured future batches are
            // allowed in flight. Check before channel allocation.
            let max_in_flight_batches = prefetch_batches.saturating_add(1);
            let max_samples = batch_size
                .checked_mul(max_in_flight_batches)
                .ok_or_else(|| runtime_error("worker queue capacity overflow"))?;
            let worker_adapter = Arc::clone(&adapter);
            let profiler = PhysicalProfiler::default();
            let worker_profiler = profiler.clone();
            // A domain may compile an empty sample stage (for example, only
            // converting an already-decoded source sample into the typed
            // runtime value). Attribute its work to Source until the planner
            // materializes an explicit no-op/sample-conversion kernel node.
            let sample_node_id = sample_node.unwrap_or(source_node);
            let pool = WorkerPool::new(
                num_workers,
                max_samples,
                move |sample, index| {
                    let started = Instant::now();
                    let input_bytes = worker_adapter.sample_bytes(&sample);
                    let result = worker_adapter.process_sample(sample, index);
                    let output_bytes = result
                        .as_ref()
                        .map(|output| worker_adapter.output_bytes(output))
                        .unwrap_or(0);
                    worker_profiler.record(
                        sample_node_id,
                        started.elapsed(),
                        input_bytes,
                        output_bytes,
                    );
                    result
                },
                {
                    let adapter = Arc::clone(&adapter);
                    move |worker, index| adapter.worker_panic_error(worker, index)
                },
            )?;
            // Worker stage timing is recorded from worker threads in the
            // closure above. Share this same profiler for the rest of runtime.
            return Ok(Self {
                adapter,
                state: ExecutorState::Workers(
                    pool,
                    PrefetchCoordinator::new(max_in_flight_batches),
                ),
                max_in_flight_batches,
                graph,
                sampler_node,
                source_node,
                sample_node,
                batch_node,
                batch_kernel_node,
                device_batch_kernel,
                transfer_nodes,
                profiler,
            });
        };

        Ok(Self {
            adapter,
            state,
            max_in_flight_batches: 1,
            graph,
            sampler_node,
            source_node,
            sample_node,
            batch_node,
            batch_kernel_node,
            device_batch_kernel,
            transfer_nodes,
            profiler: PhysicalProfiler::default(),
        })
    }

    pub fn physical_explain(&mut self) -> Result<String, RuntimeError> {
        self.graph.explain().map_err(graph_runtime_error)
    }

    pub fn profiler(&self) -> PhysicalProfiler {
        self.profiler.clone()
    }

    pub fn next_batch(
        &mut self,
        sampler: &mut IndexSampler,
    ) -> Result<Option<P::Batch>, PipelineError<P::Error>> {
        let state = std::mem::replace(&mut self.state, ExecutorState::Failed);
        let (result, next_state) = match state {
            ExecutorState::Inline => {
                let result = self.next_batch_inline(sampler);
                let next_state = if result.is_err() {
                    ExecutorState::Failed
                } else {
                    ExecutorState::Inline
                };
                (result, next_state)
            }
            ExecutorState::Workers(pool, mut coordinator) => {
                let result = self.next_batch_workers(sampler, &pool, &mut coordinator);
                let next_state = if result.is_err() {
                    ExecutorState::Failed
                } else {
                    ExecutorState::Workers(pool, coordinator)
                };
                (result, next_state)
            }
            ExecutorState::Failed => (
                Err(PipelineError::Runtime(runtime_error(
                    "loader is in failed state after a previous iteration error",
                ))),
                ExecutorState::Failed,
            ),
        };
        self.state = next_state;
        result
    }

    fn next_indices(&mut self, sampler: &mut IndexSampler) -> Option<Vec<usize>> {
        let started = Instant::now();
        let indices = sampler.next_indices(self.adapter.batch_size())?;
        let output = if self.adapter.drop_last() && indices.len() < self.adapter.batch_size() {
            None
        } else {
            Some(indices)
        };
        if let Some(node) = self.sampler_node {
            let bytes = output
                .as_ref()
                .map(|indices| indices.len().saturating_mul(std::mem::size_of::<usize>()))
                .unwrap_or(0);
            self.profiler.record(node, started.elapsed(), 0, bytes);
        }
        output
    }

    fn next_batch_inline(
        &mut self,
        sampler: &mut IndexSampler,
    ) -> Result<Option<P::Batch>, PipelineError<P::Error>> {
        let Some(indices) = self.next_indices(sampler) else {
            return Ok(None);
        };
        if self.adapter.is_batch_native() {
            let started = Instant::now();
            let batch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.adapter.fetch_batch(&indices)
            }))
            .map_err(|_| {
                PipelineError::Runtime(runtime_error(
                    "dataset get_batch panicked while fetching a batch",
                ))
            })?
            .map_err(PipelineError::Domain)?
            .ok_or_else(|| {
                PipelineError::Runtime(runtime_error("batch-native source capability disappeared"))
            })?;
            let bytes = self.adapter.batch_bytes(&batch);
            self.profiler
                .record(self.source_node, started.elapsed(), 0, bytes);
            return self.apply_batch(batch, &indices).map(Some);
        }

        let samples = self.fetch_samples(&indices)?;
        let capacity = samples.len();
        let mut builder = self.adapter.batch_builder(capacity);
        for (&index, sample) in indices.iter().zip(samples) {
            let started = Instant::now();
            let input_bytes = self.adapter.sample_bytes(&sample);
            let output = self
                .adapter
                .process_sample(sample, index)
                .map_err(PipelineError::Domain)?;
            if let Some(sample_node) = self.sample_node {
                self.profiler.record(
                    sample_node,
                    started.elapsed(),
                    input_bytes,
                    self.adapter.output_bytes(&output),
                );
            }
            self.adapter
                .push_batch_sample(&mut builder, output)
                .map_err(PipelineError::Domain)?;
        }
        let started = Instant::now();
        let batch = self
            .adapter
            .finish_batch(builder)
            .map_err(PipelineError::Domain)?;
        let output_bytes = self.adapter.batch_bytes(&batch);
        self.profiler
            .record(self.batch_node, started.elapsed(), 0, output_bytes);
        self.apply_batch(batch, &indices).map(Some)
    }

    fn next_batch_workers(
        &mut self,
        sampler: &mut IndexSampler,
        pool: &WorkerPool<P::Sample, P::Output, P::Error>,
        coordinator: &mut PrefetchCoordinator<P::Output>,
    ) -> Result<Option<P::Batch>, PipelineError<P::Error>> {
        loop {
            while coordinator.in_flight < coordinator.max_in_flight && !coordinator.closed {
                match self.next_indices(sampler) {
                    Some(indices) => {
                        let samples = self.fetch_samples(&indices)?;
                        coordinator
                            .submit(indices, samples, pool)
                            .map_err(PipelineError::Runtime)?;
                    }
                    None => coordinator.closed = true,
                }
            }

            if let Some(pending) = coordinator.take_ready().map_err(PipelineError::Runtime)? {
                let (batch_indices, pending_samples) = pending.into_parts();
                let capacity = batch_indices.len();
                let mut builder = self.adapter.batch_builder(capacity);
                for sample in pending_samples {
                    let sample = sample.map_err(PipelineError::Runtime)?;
                    self.adapter
                        .push_batch_sample(&mut builder, sample)
                        .map_err(PipelineError::Domain)?;
                }
                let started = Instant::now();
                let batch = self
                    .adapter
                    .finish_batch(builder)
                    .map_err(PipelineError::Domain)?;
                let output_bytes = self.adapter.batch_bytes(&batch);
                self.profiler
                    .record(self.batch_node, started.elapsed(), 0, output_bytes);
                return self.apply_batch(batch, &batch_indices).map(Some);
            }

            if coordinator.closed && coordinator.in_flight == 0 {
                return Ok(None);
            }

            let result = pool.recv().map_err(PipelineError::Runtime)?;
            match result.result {
                Ok(sample) => coordinator
                    .record(result.batch_id, result.position, sample)
                    .map_err(PipelineError::Runtime)?,
                Err(error) => return Err(PipelineError::Domain(error)),
            }
        }
    }

    fn fetch_samples(
        &mut self,
        indices: &[usize],
    ) -> Result<Vec<P::Sample>, PipelineError<P::Error>> {
        let started = Instant::now();
        let samples = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.adapter.fetch_samples(indices)
        }))
        .map_err(|_| {
            PipelineError::Runtime(runtime_error(
                "dataset get_many panicked while fetching a batch",
            ))
        })?
        .map_err(PipelineError::Domain)?;
        if samples.len() != indices.len() {
            return Err(PipelineError::Runtime(runtime_error(format!(
                "source returned {} samples for {} indices",
                samples.len(),
                indices.len()
            ))));
        }
        let bytes = samples
            .iter()
            .map(|sample| self.adapter.sample_bytes(sample))
            .sum();
        self.profiler
            .record(self.source_node, started.elapsed(), 0, bytes);
        Ok(samples)
    }

    fn apply_batch(
        &mut self,
        batch: P::Batch,
        indices: &[usize],
    ) -> Result<P::Batch, PipelineError<P::Error>> {
        let mut batch = batch;
        if self.device_batch_kernel {
            batch = self.transfer_before_device_kernel(batch)?;
            let started = Instant::now();
            let input_bytes = self.adapter.batch_bytes(&batch);
            let target = self
                .graph
                .node(self.batch_kernel_node)
                .map_err(graph_runtime_error)
                .map_err(PipelineError::Runtime)?
                .lane;
            batch = self
                .adapter
                .apply_device_batch_with_indices(batch, target, indices)?;
            self.profiler.record(
                self.batch_kernel_node,
                started.elapsed(),
                input_bytes,
                self.adapter.batch_bytes(&batch),
            );
        } else {
            let started = Instant::now();
            let input_bytes = self.adapter.batch_bytes(&batch);
            batch = self
                .adapter
                .apply_batch_with_indices(batch, indices)
                .map_err(PipelineError::Domain)?;
            self.profiler.record(
                self.batch_kernel_node,
                started.elapsed(),
                input_bytes,
                self.adapter.batch_bytes(&batch),
            );
            for (node, kind, target) in self.transfer_nodes.iter().copied() {
                let started = Instant::now();
                let input_bytes = self.adapter.batch_bytes(&batch);
                batch = self.adapter.transfer_batch(batch, kind, target)?;
                self.profiler.record(
                    node,
                    started.elapsed(),
                    input_bytes,
                    self.adapter.batch_bytes(&batch),
                );
            }
        }
        Ok(batch)
    }

    fn transfer_before_device_kernel(
        &mut self,
        mut batch: P::Batch,
    ) -> Result<P::Batch, PipelineError<P::Error>> {
        for (node, kind, target) in self.transfer_nodes.iter().copied() {
            let started = Instant::now();
            let input_bytes = self.adapter.batch_bytes(&batch);
            batch = self.adapter.transfer_batch(batch, kind, target)?;
            self.profiler.record(
                node,
                started.elapsed(),
                input_bytes,
                self.adapter.batch_bytes(&batch),
            );
        }
        Ok(batch)
    }

    pub fn max_in_flight_batches(&self) -> usize {
        self.max_in_flight_batches
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
    let sample_input = if batch_native {
        source
    } else {
        graph.add_node(
            None,
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Kernel(KernelStage::Sample),
                ExecutionLane::Cpu,
            ),
            [source],
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

    struct Adapter {
        calls: AtomicUsize,
        transfers: AtomicUsize,
        fail_transfer: bool,
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
    fn worker_batch_prefetch_limit_is_explicit_and_checked() {
        let adapter = Arc::new(Adapter {
            calls: AtomicUsize::new(0),
            transfers: AtomicUsize::new(0),
            fail_transfer: false,
        });
        let executor = PhysicalPipelineExecutor::new(adapter, 2, 4).unwrap();
        assert_eq!(executor.max_in_flight_batches(), 5);
    }

    #[test]
    fn physical_runtime_executes_and_profiles_explicit_transfer() {
        let adapter = Arc::new(Adapter {
            calls: AtomicUsize::new(0),
            transfers: AtomicUsize::new(0),
            fail_transfer: false,
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
