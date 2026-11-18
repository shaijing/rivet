use super::*;
use crate::physical::{
    ExecutableGraph, ExecutionLane, ExecutionValue, Morsel, PhysicalGraph, PhysicalGraphError,
    PhysicalNodeKind, PhysicalNodeSpec, PhysicalOperator, PhysicalProfiler,
};
use crate::runtime::{BoundedStageQueue, RuntimeResult, StageQueueLimits};
use rivet_core::{Device, Tensor};
use std::sync::Arc;

fn tensor_morsel(tensor: Tensor) -> Morsel {
    let mut morsel = Morsel::new(0, vec![]);
    morsel.bytes = tensor.logical_bytes();
    morsel.values = vec![ExecutionValue::Tensor(tensor)];
    morsel
}
struct Allocate(usize);
impl PhysicalOperator for Allocate {
    fn name(&self) -> &str {
        "Allocate"
    }
    fn execute(&self, mut morsel: Morsel) -> RuntimeResult<Morsel> {
        let tensor = Tensor::from_vec(vec![0u8; self.0], [self.0], &Device::Cpu).unwrap();
        morsel.bytes = self.0;
        morsel.values = vec![ExecutionValue::Tensor(tensor)];
        Ok(morsel)
    }
}
struct Join;
impl PhysicalOperator for Join {
    fn name(&self) -> &str {
        "Join"
    }
    fn execute(&self, morsel: Morsel) -> RuntimeResult<Morsel> {
        Ok(morsel)
    }
    fn execute_inputs(&self, inputs: Vec<Morsel>) -> RuntimeResult<Morsel> {
        Ok(inputs[0].clone())
    }
}
fn diamond(fresh_branches: bool) -> ExecutableGraph {
    let mut graph = PhysicalGraph::new();
    let source = graph.add_node(
        None,
        PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Cpu)
            .with_operator(Arc::new(Allocate(8))),
        [],
    );
    let branches = (0..2)
        .map(|_| {
            let mut spec = PhysicalNodeSpec::new(PhysicalNodeKind::Cache, ExecutionLane::Cpu);
            if fresh_branches {
                spec = spec.with_operator(Arc::new(Allocate(8)));
            }
            graph.add_node(None, spec, [source])
        })
        .collect::<Vec<_>>();
    let root = graph.add_node(
        None,
        PhysicalNodeSpec::new(PhysicalNodeKind::Sink, ExecutionLane::Cpu)
            .with_operator(Arc::new(Join)),
        [branches[0], source, branches[1]],
    );
    graph.set_root(root).unwrap();
    ExecutableGraph::new(graph).unwrap()
}
#[test]
fn views_charge_full_storage_once_and_logical_regions_separately() {
    let base = Tensor::from_vec(vec![0u8; 64], [64], &Device::Cpu).unwrap();
    let a = tensor_morsel(base.narrow(0, 1, 2).unwrap());
    let b = tensor_morsel(base.narrow(0, 8, 2).unwrap());
    let snapshot = observe_live_morsels([&a, &b]);
    assert_eq!(snapshot.logical_tensor_bytes, 4);
    assert_eq!(snapshot.known_backing_bytes, 64);
    assert_eq!(snapshot.unique_backing_allocations, 1);
    assert!(snapshot.is_complete());
}
#[test]
fn diamond_fanout_does_not_double_count_shared_allocations() {
    let graph = diamond(false);
    let profiler = PhysicalProfiler::enabled(graph.graph().nodes().len());
    graph
        .execute_with_memory_limits(
            Morsel::new(0, vec![]),
            &profiler,
            GraphMemoryLimits {
                max_value_logical_bytes: 8,
                max_observed_live_backing_bytes: Some(8),
            },
        )
        .unwrap();
    let profile = profiler.memory_snapshot().unwrap();
    assert_eq!(profile.peak_known_backing_bytes_per_invocation, 8);
    assert_eq!(profile.incomplete_observations, 0);
    assert!(profile.has_complete_boundary_observations());
    assert!(profile.boundary_observations > 0);
}
#[test]
fn multiple_live_outputs_exceed_observed_limit_even_when_each_value_fits() {
    let graph = diamond(true);
    let profiler = PhysicalProfiler::enabled(graph.graph().nodes().len());
    let error = graph
        .execute_with_memory_limits(
            Morsel::new(0, vec![]),
            &profiler,
            GraphMemoryLimits {
                max_value_logical_bytes: 8,
                max_observed_live_backing_bytes: Some(16),
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        PhysicalGraphError::MemoryAccounting {
            source: MemoryAccountingError::ObservedLiveExceeded {
                bytes: 24,
                limit: 16
            },
            ..
        }
    ));
    let profile = profiler.memory_snapshot().unwrap();
    assert_eq!(profile.value_logical_limit_failures, 0);
    assert_eq!(profile.observed_live_limit_failures, 1);
    assert_eq!(profile.peak_known_backing_bytes_per_invocation, 24);
}
#[test]
fn single_value_failures_have_a_separate_error_and_profiler_counter() {
    let graph = diamond(false);
    let profiler = PhysicalProfiler::enabled(graph.graph().nodes().len());
    let error = graph
        .execute_with_limit(Morsel::new(0, vec![]), &profiler, 7)
        .unwrap_err();
    assert!(matches!(
        error,
        PhysicalGraphError::ValueLogicalLimit {
            bytes: 8,
            limit: 7,
            ..
        }
    ));
    assert!(error.to_string().contains("single-value"));
    let profile = profiler.memory_snapshot().unwrap();
    assert_eq!(profile.value_logical_limit_failures, 1);
    assert_eq!(profile.observed_live_limit_failures, 0);
}
#[test]
fn opaque_and_legacy_byte_only_values_have_unknown_capacity() {
    let mut a = Morsel::new(0, vec![]);
    a.values = vec![ExecutionValue::Decoded(Arc::new(vec![0u8; 16]))];
    let b = a.clone();
    let snapshot = observe_live_morsels([&a, &b]);
    assert_eq!(snapshot.known_backing_bytes, 0);
    assert_eq!(snapshot.unknown_values, 1);
    assert!(!snapshot.is_complete());
    assert!(matches!(
        snapshot.check_observed_limit(100),
        Err(MemoryAccountingError::Incomplete { .. })
    ));
    let mut legacy = Morsel::new(0, vec![]);
    legacy.bytes = 16;
    assert!(!observe_live_morsels([&legacy]).is_complete());
}
#[test]
fn observed_limit_fails_closed_on_unknown_graph_values() {
    struct Opaque;
    impl PhysicalOperator for Opaque {
        fn name(&self) -> &str {
            "Opaque"
        }
        fn execute(&self, mut morsel: Morsel) -> RuntimeResult<Morsel> {
            morsel.values = vec![ExecutionValue::Encoded(Arc::new(vec![0u8; 16]))];
            Ok(morsel)
        }
    }
    let mut graph = PhysicalGraph::new();
    let root = graph.add_node(
        None,
        PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Cpu)
            .with_operator(Arc::new(Opaque)),
        [],
    );
    graph.set_root(root).unwrap();
    let graph = ExecutableGraph::new(graph).unwrap();
    let profiler = PhysicalProfiler::enabled(1);
    let error = graph
        .execute_with_memory_limits(
            Morsel::new(0, vec![]),
            &profiler,
            GraphMemoryLimits {
                max_observed_live_backing_bytes: Some(100),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("unknown backing capacity"));
    assert_eq!(
        profiler.memory_snapshot().unwrap().incomplete_observations,
        1
    );
    assert_eq!(
        profiler
            .memory_snapshot()
            .unwrap()
            .observed_live_limit_failures,
        1
    );
}
#[test]
fn queue_weight_is_released_at_dequeue_while_payload_is_still_alive() {
    let value = Arc::new(vec![0u8; 16]);
    let queue = BoundedStageQueue::new(StageQueueLimits {
        max_items: 1,
        max_bytes: 16,
    })
    .unwrap();
    queue.send(0, Arc::clone(&value), 16).unwrap();
    let received = queue.recv().unwrap().0.unwrap();
    assert_eq!(queue.snapshot().queued_bytes, 0);
    assert_eq!(queue.snapshot().peak_bytes, 16);
    assert!(Arc::ptr_eq(&received.value, &value));
    assert!(
        queue
            .send(1, Arc::clone(&value), 17)
            .unwrap_err()
            .to_string()
            .contains("queued only")
    );
}
#[test]
fn disabled_profiling_keeps_accounting_absent_and_default_limits_compatible() {
    let profiler = PhysicalProfiler::default();
    assert!(profiler.memory_snapshot().is_none());
    diamond(true)
        .execute(Morsel::new(0, vec![]), &profiler)
        .unwrap();
    assert!(profiler.snapshot().is_empty());
}
#[test]
fn empty_storage_and_accounting_overflow_are_explicit() {
    let empty = tensor_morsel(Tensor::from_vec(Vec::<u8>::new(), [0], &Device::Cpu).unwrap());
    let snapshot = observe_live_morsels([&empty]);
    assert!(snapshot.is_complete());
    assert_eq!(snapshot.unique_backing_allocations, 0);
    let overflow = LiveMemorySnapshot {
        overflowed: true,
        ..Default::default()
    };
    assert_eq!(
        overflow.check_observed_limit(usize::MAX),
        Err(MemoryAccountingError::Overflow)
    );
}
