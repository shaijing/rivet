//! Bounded batch-morsel scheduling for executable DAGs. Nodes run in dependency
//! order within each morsel; independent morsels execute on persistent workers.
use super::{RuntimeResult, WorkItem, WorkerPool, runtime_error};
use crate::physical::{ExecutableGraph, Morsel, PhysicalProfiler};
use rivet_data::sampler::IndexSampler;
use std::collections::BTreeMap;
use std::sync::Arc;

pub struct DagPipelineExecutor {
    graph: Arc<ExecutableGraph>,
    profiler: PhysicalProfiler,
    workers: Option<WorkerPool<Morsel, Morsel, super::RuntimeError>>,
    ready: BTreeMap<u64, Morsel>,
    batch_size: usize,
    drop_last: bool,
    capacity: usize,
    memory_limits: crate::memory::GraphMemoryLimits,
    submitted: u64,
    delivered: u64,
    exhausted: bool,
    failed: bool,
}
impl DagPipelineExecutor {
    /// Compatibility constructor: the legacy byte argument limits ONE
    /// producer-reported logical output, not queues or total executor memory.
    pub fn new(
        graph: Arc<ExecutableGraph>,
        batch_size: usize,
        drop_last: bool,
        num_workers: usize,
        prefetch_batches: usize,
        max_value_logical_bytes: usize,
        profiling: bool,
    ) -> RuntimeResult<Self> {
        Self::with_memory_limits(
            graph,
            batch_size,
            drop_last,
            num_workers,
            prefetch_batches,
            crate::memory::GraphMemoryLimits {
                max_value_logical_bytes,
                ..Default::default()
            },
            profiling,
        )
    }

    /// Observed-live limits apply independently to each graph invocation;
    /// reorder buffers, other workers and caller outputs are outside this scope.
    pub fn with_memory_limits(
        graph: Arc<ExecutableGraph>,
        batch_size: usize,
        drop_last: bool,
        num_workers: usize,
        prefetch_batches: usize,
        memory_limits: crate::memory::GraphMemoryLimits,
        profiling: bool,
    ) -> RuntimeResult<Self> {
        if batch_size == 0 || memory_limits.max_value_logical_bytes == 0 {
            return Err(runtime_error(
                "DAG batch size and single-value logical byte limit must be positive",
            ));
        }
        let capacity = prefetch_batches
            .checked_add(1)
            .ok_or_else(|| runtime_error("DAG prefetch capacity overflow"))?;
        let profiler = if profiling {
            PhysicalProfiler::enabled(graph.graph().nodes().len())
        } else {
            PhysicalProfiler::default()
        };
        let workers = if num_workers == 0 {
            None
        } else {
            let graph = Arc::clone(&graph);
            let profiler = profiler.clone();
            Some(WorkerPool::new(
                num_workers,
                capacity,
                move |input, _| {
                    graph
                        .execute_with_memory_limits(input, &profiler, memory_limits)
                        .map_err(|error| runtime_error(error.to_string()))
                },
                |worker, index| {
                    runtime_error(format!(
                        "DAG worker {worker} panicked at source index {index}"
                    ))
                },
            )?)
        };
        Ok(Self {
            graph,
            profiler,
            workers,
            ready: BTreeMap::new(),
            batch_size,
            drop_last,
            capacity,
            memory_limits,
            submitted: 0,
            delivered: 0,
            exhausted: false,
            failed: false,
        })
    }
    pub fn memory_limits(&self) -> crate::memory::GraphMemoryLimits {
        self.memory_limits
    }
    pub fn memory_explain(&self) -> String {
        format!(
            "DAG memory limits: max_value_logical_bytes={} max_observed_live_backing_bytes={:?}; checks=post_allocation node_boundaries=per_invocation managed_executor_budget=unavailable; reorder_and_caller_outputs=excluded\n",
            self.memory_limits.max_value_logical_bytes,
            self.memory_limits.max_observed_live_backing_bytes
        )
    }
    pub fn profiler(&self) -> PhysicalProfiler {
        self.profiler.clone()
    }
    pub fn next_morsel(&mut self, sampler: &mut IndexSampler) -> RuntimeResult<Option<Morsel>> {
        if self.failed {
            return Ok(None);
        }
        let result = self.next_inner(sampler);
        if result.is_err() {
            self.failed = true;
            self.ready.clear();
            self.workers.take();
        }
        result
    }
    fn next_inner(&mut self, sampler: &mut IndexSampler) -> RuntimeResult<Option<Morsel>> {
        if self.workers.is_none() {
            if self.exhausted {
                return Ok(None);
            }
            let Some(input) = self.sample(sampler) else {
                return Ok(None);
            };
            let output = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.graph
                    .execute_with_memory_limits(input, &self.profiler, self.memory_limits)
            }))
            .map_err(|_| runtime_error("DAG inline operator panicked"))?
            .map_err(|error| runtime_error(error.to_string()))?;
            self.delivered += 1;
            return Ok(Some(output));
        }
        while !self.exhausted && self.submitted - self.delivered < self.capacity as u64 {
            let Some(input) = self.sample(sampler) else {
                break;
            };
            self.workers
                .as_ref()
                .expect("worker path")
                .submit(WorkItem {
                    batch_id: input.sequence_id,
                    position: 0,
                    sample_index: input.source_indices.first().copied().unwrap_or(0),
                    payload: input,
                })?;
        }
        if self.delivered == self.submitted {
            return Ok(None);
        }
        loop {
            if let Some(output) = self.ready.remove(&self.delivered) {
                self.delivered += 1;
                return Ok(Some(output));
            }
            let result = self.workers.as_ref().expect("worker path").recv()?;
            self.ready.insert(result.batch_id, result.result?);
        }
    }
    fn sample(&mut self, sampler: &mut IndexSampler) -> Option<Morsel> {
        let Some(indices) = sampler.next_indices(self.batch_size) else {
            self.exhausted = true;
            return None;
        };
        if self.drop_last && indices.len() < self.batch_size {
            self.exhausted = true;
            return None;
        }
        let input = Morsel::new(self.submitted, indices);
        self.submitted += 1;
        Some(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical::{
        ExecutionLane, PhysicalGraph, PhysicalNodeKind, PhysicalNodeSpec, PhysicalOperator,
    };
    use rivet_data::sampler::SamplerPlan;

    struct Read {
        panic_at: Option<usize>,
    }
    impl PhysicalOperator for Read {
        fn name(&self) -> &str {
            "Read"
        }
        fn execute(&self, mut input: Morsel) -> RuntimeResult<Morsel> {
            if self
                .panic_at
                .is_some_and(|index| input.source_indices.contains(&index))
            {
                panic!("read failure");
            }
            if input.sequence_id == 0 {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            input.bytes = input.source_indices.len() * 16;
            Ok(input)
        }
    }
    fn executor(
        workers: usize,
        drop_last: bool,
        max_bytes: usize,
        panic_at: Option<usize>,
    ) -> DagPipelineExecutor {
        let mut graph = PhysicalGraph::new();
        let root = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Io)
                .with_operator(Arc::new(Read { panic_at })),
            [],
        );
        graph.set_root(root).unwrap();
        DagPipelineExecutor::new(
            Arc::new(ExecutableGraph::new(graph).unwrap()),
            2,
            drop_last,
            workers,
            2,
            max_bytes,
            true,
        )
        .unwrap()
    }
    #[test]
    fn morsel_workers_preserve_order_and_drop_last() {
        for workers in [0, 3] {
            for drop_last in [false, true] {
                let mut executor = executor(workers, drop_last, 128, None);
                let mut sampler =
                    IndexSampler::new(SamplerPlan::Sequential { start: 0, end: 5 }, 0);
                let mut indices = Vec::new();
                while let Some(output) = executor.next_morsel(&mut sampler).unwrap() {
                    assert_eq!(output.sequence_id as usize, indices.len());
                    indices.push(output.source_indices);
                }
                assert_eq!(
                    indices,
                    if drop_last {
                        vec![vec![0, 1], vec![2, 3]]
                    } else {
                        vec![vec![0, 1], vec![2, 3], vec![4]]
                    }
                );
                assert!(
                    executor
                        .profiler()
                        .snapshot()
                        .values()
                        .any(|entry| entry.executions > 0)
                );
            }
        }
    }
    #[test]
    fn byte_limit_and_panics_make_morsel_execution_terminal() {
        for workers in [0, 2] {
            for (limit, panic_at) in [(8, None), (128, Some(0))] {
                let mut executor = executor(workers, false, limit, panic_at);
                let mut sampler =
                    IndexSampler::new(SamplerPlan::Sequential { start: 0, end: 20 }, 0);
                assert!(executor.next_morsel(&mut sampler).is_err());
                assert!(executor.next_morsel(&mut sampler).unwrap().is_none());
            }
        }
    }
    #[test]
    fn observed_limit_is_per_invocation_and_unknown_values_fail_terminally() {
        for workers in [0, 2] {
            let original = executor(workers, false, 128, None);
            let limits = crate::memory::GraphMemoryLimits {
                max_value_logical_bytes: 128,
                max_observed_live_backing_bytes: Some(1024),
            };
            let mut executor = DagPipelineExecutor::with_memory_limits(
                Arc::clone(&original.graph),
                2,
                false,
                workers,
                2,
                limits,
                true,
            )
            .unwrap();
            assert_eq!(executor.memory_limits(), limits);
            assert!(executor.memory_explain().contains("per_invocation"));
            assert!(
                executor
                    .memory_explain()
                    .contains("managed_executor_budget=unavailable")
            );
            let mut sampler = IndexSampler::new(SamplerPlan::Sequential { start: 0, end: 5 }, 0);
            assert!(
                executor
                    .next_morsel(&mut sampler)
                    .unwrap_err()
                    .to_string()
                    .contains("unknown backing capacity")
            );
            assert!(executor.next_morsel(&mut sampler).unwrap().is_none());
        }
    }
}
