//! Bounded batch-morsel scheduling for executable DAGs. Nodes run in dependency
//! order within each morsel; independent morsels execute on persistent workers.
use super::{RuntimeResult, WorkItem, WorkerPool, runtime_error};
use crate::memory::{ManagedMemoryBudget, ManagedMemorySnapshot, MemoryReservation};
use crate::physical::{ExecutableGraph, GraphScratch, Morsel, PhysicalProfiler};
use rivet_data::sampler::IndexSampler;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// Per-morsel managed footprint, including allocation while the morsel runs.
#[derive(Clone, Debug)]
pub struct DagMemoryAdmission {
    pub budget: ManagedMemoryBudget,
    pub bytes_per_morsel: usize,
}

struct AdmittedMorsel {
    value: Morsel,
    reservation: Option<MemoryReservation>,
}

pub struct DagPipelineExecutor {
    graph: Arc<ExecutableGraph>,
    profiler: PhysicalProfiler,
    workers: Option<WorkerPool<AdmittedMorsel, AdmittedMorsel, super::RuntimeError>>,
    inline_scratch: Option<GraphScratch>,
    ready: BTreeMap<u64, (AdmittedMorsel, Option<Instant>)>,
    admission: Option<DagMemoryAdmission>,
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
        Self::with_managed_admission(
            graph,
            batch_size,
            drop_last,
            num_workers,
            prefetch_batches,
            memory_limits,
            profiling,
            None,
        )
    }

    /// Admit whole morsels in sequence order using the declared footprint.
    /// The declaration covers managed allocation only. Callers must separately
    /// document allocations that their kernels cannot bound.
    pub fn with_managed_admission(
        graph: Arc<ExecutableGraph>,
        batch_size: usize,
        drop_last: bool,
        num_workers: usize,
        prefetch_batches: usize,
        memory_limits: crate::memory::GraphMemoryLimits,
        profiling: bool,
        admission: Option<DagMemoryAdmission>,
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
            PhysicalProfiler::enabled_for_dag(graph.graph().nodes().len())
        } else {
            PhysicalProfiler::default()
        };
        let workers = if num_workers == 0 {
            None
        } else {
            let graph = Arc::clone(&graph);
            let profiler = profiler.clone();
            Some(WorkerPool::with_worker(
                num_workers,
                capacity,
                move || {
                    let graph = Arc::clone(&graph);
                    let profiler = profiler.clone();
                    let mut scratch = graph.scratch();
                    move |mut input: AdmittedMorsel, _| {
                        let value = graph
                            .execute_reusing_with_budget(
                                input.value,
                                &profiler,
                                memory_limits,
                                &mut scratch,
                                input.reservation.as_mut(),
                            )
                            .map_err(|error| runtime_error(error.to_string()))?;
                        Ok(AdmittedMorsel {
                            value,
                            reservation: input.reservation,
                        })
                    }
                },
                |worker, index| {
                    runtime_error(format!(
                        "DAG worker {worker} panicked at source index {index}"
                    ))
                },
            )?)
        };
        let inline_scratch = (num_workers == 0).then(|| graph.scratch());
        Ok(Self {
            graph,
            profiler,
            workers,
            inline_scratch,
            admission,
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
    pub fn managed_memory_snapshot(&self) -> Option<ManagedMemorySnapshot> {
        self.admission
            .as_ref()
            .map(|admission| admission.budget.snapshot())
    }

    pub fn memory_explain(&self) -> String {
        let admission = self.admission.as_ref().map_or_else(
            || "disabled".to_owned(),
            |admission| {
                format!(
                    "{}B ({}B per morsel, queued through ordered delivery)",
                    admission.budget.snapshot().limit_bytes,
                    admission.bytes_per_morsel,
                )
            },
        );
        format!(
            "DAG memory limits: max_value_logical_bytes={} max_observed_live_backing_bytes={:?}; checks=post_allocation node_boundaries=per_invocation observed_reorder=excluded managed_executor_budget={admission}; caller_outputs=excluded\n",
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
            self.inline_scratch.take();
        }
        result
    }
    fn next_inner(&mut self, sampler: &mut IndexSampler) -> RuntimeResult<Option<Morsel>> {
        if self.workers.is_none() {
            if self.exhausted {
                return Ok(None);
            }
            let mut reservation = self.reserve_morsel()?.ok_or_else(|| {
                runtime_error("managed memory capacity is held by another executor")
            })?;
            let Some(input) = self.sample(sampler) else {
                self.inline_scratch.take();
                return Ok(None);
            };
            let output = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.graph.execute_reusing_with_budget(
                    input,
                    &self.profiler,
                    self.memory_limits,
                    self.inline_scratch.as_mut().expect("inline graph"),
                    reservation.as_mut(),
                )
            }))
            .map_err(|_| runtime_error("DAG inline operator panicked"))?
            .map_err(|error| runtime_error(error.to_string()))?;
            drop(reservation);
            self.delivered += 1;
            return Ok(Some(output));
        }
        while !self.exhausted && self.submitted - self.delivered < self.capacity as u64 {
            let Some(reservation) = self.reserve_morsel()? else {
                break;
            };
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
                    payload: AdmittedMorsel {
                        value: input,
                        reservation,
                    },
                })?;
        }
        if self.profiler.is_enabled() {
            self.profiler
                .observe_dag_depth(self.submitted - self.delivered, self.ready.len());
        }
        if self.delivered == self.submitted {
            return if !self.exhausted && self.admission.is_some() {
                Err(runtime_error(
                    "managed memory capacity is held by another executor",
                ))
            } else {
                self.workers.take();
                Ok(None)
            };
        }
        loop {
            if let Some((output, ready_at)) = self.ready.remove(&self.delivered) {
                if let Some(ready_at) = ready_at {
                    self.profiler.record_dag_wait(ready_at.elapsed(), true);
                }
                self.delivered += 1;
                return Ok(Some(output.value));
            }
            let workers = self.workers.as_ref().expect("worker path");
            let started =
                (self.profiler.is_enabled() && workers.result_queue_is_empty()).then(Instant::now);
            let result = workers.recv()?;
            if let Some(started) = started {
                self.profiler.record_dag_wait(started.elapsed(), false);
            }
            let output = result.result?;
            if result.batch_id == self.delivered {
                self.delivered += 1;
                return Ok(Some(output.value));
            }
            let ready_at = self.profiler.is_enabled().then(Instant::now);
            self.ready.insert(result.batch_id, (output, ready_at));
            if ready_at.is_some() {
                self.profiler
                    .observe_dag_depth(self.submitted - self.delivered, self.ready.len());
            }
        }
    }
    // Outer None asks the coordinator to drain existing work. Inner None
    // represents unbudgeted execution.
    fn reserve_morsel(&self) -> RuntimeResult<Option<Option<MemoryReservation>>> {
        let Some(admission) = &self.admission else {
            return Ok(Some(None));
        };
        admission
            .budget
            .try_reserve(admission.bytes_per_morsel)
            .map(|result| result.map(Some))
            .map_err(|error| runtime_error(error.to_string()))
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
    fn bounded_tensor_results_keep_reservations_while_reordered() {
        use crate::physical::ExecutionValue;
        use std::sync::{Mutex, mpsc};
        use std::time::Duration;

        struct GateFirst(Mutex<mpsc::Receiver<()>>);
        impl PhysicalOperator for GateFirst {
            fn name(&self) -> &str {
                "GateFirst"
            }
            fn managed_output_bytes(&self, _: &[Morsel]) -> RuntimeResult<Option<usize>> {
                Ok(Some(8))
            }
            fn execute(&self, mut morsel: Morsel) -> RuntimeResult<Morsel> {
                if morsel.sequence_id == 0 {
                    self.0
                        .lock()
                        .unwrap()
                        .recv_timeout(Duration::from_secs(5))
                        .unwrap();
                }
                let value = rivet_core::Tensor::from_slice(
                    &[morsel.sequence_id as u8; 8],
                    [8],
                    &rivet_core::Device::Cpu,
                )
                .unwrap();
                morsel.values.push(ExecutionValue::Tensor(value));
                Ok(morsel)
            }
        }

        let (release, gate) = mpsc::channel();
        let mut graph = PhysicalGraph::new();
        let root = graph.add_node(
            None,
            PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Io)
                .with_operator(Arc::new(GateFirst(Mutex::new(gate)))),
            [],
        );
        graph.set_root(root).unwrap();
        let budget = ManagedMemoryBudget::new(16);
        let mut executor = DagPipelineExecutor::with_managed_admission(
            Arc::new(ExecutableGraph::new(graph).unwrap()),
            2,
            false,
            2,
            6,
            Default::default(),
            true,
            Some(DagMemoryAdmission {
                budget: budget.clone(),
                bytes_per_morsel: 8,
            }),
        )
        .unwrap();
        let profiler = executor.profiler();
        let handle = std::thread::spawn(move || {
            let mut sampler = IndexSampler::new(SamplerPlan::Sequential { start: 0, end: 4 }, 0);
            let first = executor.next_morsel(&mut sampler).unwrap().unwrap();
            (executor, sampler, first)
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        while profiler
            .dag_scheduler_snapshot()
            .unwrap()
            .peak_reordered_morsels
            == 0
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        let reordered = profiler
            .dag_scheduler_snapshot()
            .unwrap()
            .peak_reordered_morsels;
        let held = budget.snapshot().reserved_bytes;
        release.send(()).unwrap();
        let (mut executor, mut sampler, first) = handle.join().unwrap();
        assert_eq!(reordered, 1);
        assert_eq!(held, 16);
        assert_eq!(first.sequence_id, 0);
        assert_eq!(budget.snapshot().reserved_bytes, 8);
        let second = executor.next_morsel(&mut sampler).unwrap().unwrap();
        assert_eq!(second.sequence_id, 1);
        assert_eq!(budget.snapshot().reserved_bytes, 0);
        assert_eq!(budget.snapshot().peak_reserved_bytes, 16);
        assert!(executor.next_morsel(&mut sampler).unwrap().is_none());
    }

    #[test]
    fn managed_admission_throttles_reorder_and_releases_on_error_or_drop() {
        for (limit, panic_at, take) in [
            (64, None, 6),
            (64, None, 1),
            (64, Some(2), 6),
            (16, None, 6),
        ] {
            let graph = executor(0, false, 128, panic_at).graph.clone();
            let budget = ManagedMemoryBudget::new(limit);
            let mut executor = DagPipelineExecutor::with_managed_admission(
                graph,
                2,
                false,
                4,
                6,
                Default::default(),
                true,
                Some(DagMemoryAdmission {
                    budget: budget.clone(),
                    bytes_per_morsel: 32,
                }),
            )
            .unwrap();
            let mut sampler = IndexSampler::new(SamplerPlan::Sequential { start: 0, end: 12 }, 0);
            let mut failed = false;
            for sequence in 0..take {
                match executor.next_morsel(&mut sampler) {
                    Ok(Some(output)) => assert_eq!(output.sequence_id, sequence),
                    other => {
                        assert!(other.is_err());
                        failed = true;
                        break;
                    }
                }
                // A slow consumer leaves at most one later result reserved.
                assert!(budget.snapshot().reserved_bytes <= 32);
            }
            assert_eq!(failed, panic_at.is_some() || limit < 32);
            drop(executor);
            let memory = budget.snapshot();
            assert_eq!(memory.reserved_bytes, 0);
            assert_eq!(memory.peak_reserved_bytes, if limit < 32 { 0 } else { 64 });
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
                    .contains("managed_executor_budget=disabled")
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
