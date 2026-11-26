//! Run with cargo test --release -p rivet-exec --test dag_schedule_benchmark -- --ignored --nocapture.
use rivet_data::sampler::{IndexSampler, SamplerPlan};
use rivet_exec::physical::{
    ExecutableGraph, ExecutionLane, KernelStage, Morsel, PhysicalGraph, PhysicalNodeKind,
    PhysicalNodeSpec, PhysicalOperator,
};
use rivet_exec::runtime::{DagPipelineExecutor, RuntimeResult};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::time::Instant;

thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| {
            if let Some(n) = count.get() {
                count.set(Some(n + 1));
            }
        });
        // The wrapper leaves the allocation and layout unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

struct Join;

impl PhysicalOperator for Join {
    fn name(&self) -> &str {
        "Join"
    }

    fn execute(&self, input: Morsel) -> RuntimeResult<Morsel> {
        Ok(input)
    }

    fn execute_inputs(&self, mut inputs: Vec<Morsel>) -> RuntimeResult<Morsel> {
        Ok(inputs.swap_remove(0))
    }
}

fn graph() -> Arc<ExecutableGraph> {
    let mut graph = PhysicalGraph::new();
    let source = graph.add_node(
        None,
        PhysicalNodeSpec::new(PhysicalNodeKind::Source, ExecutionLane::Cpu),
        [],
    );
    let mut left = source;
    let mut right = source;
    for _ in 0..6 {
        let spec = || {
            PhysicalNodeSpec::new(
                PhysicalNodeKind::Kernel(KernelStage::Sample),
                ExecutionLane::Cpu,
            )
        };
        left = graph.add_node(None, spec(), [left]);
        right = graph.add_node(None, spec(), [right]);
    }
    let sink = graph.add_node(
        None,
        PhysicalNodeSpec::new(PhysicalNodeKind::Sink, ExecutionLane::Cpu)
            .with_operator(Arc::new(Join)),
        [right, left],
    );
    graph.set_root(sink).unwrap();
    Arc::new(ExecutableGraph::new(graph).unwrap())
}

fn drain(profiling: bool, rows: usize, count_allocations: bool) -> (f64, f64) {
    // Inline worker makes this thread's allocation counter cover the complete
    // lifecycle without taking a timestamp or locking from worker threads.
    let mut executor =
        DagPipelineExecutor::new(graph(), 1, false, 0, 1, usize::MAX, profiling).unwrap();
    let mut sampler = IndexSampler::new(
        SamplerPlan::Sequential {
            start: 0,
            end: rows,
        },
        0,
    );
    ALLOCATIONS.set(count_allocations.then_some(0));
    let started = Instant::now();
    for expected in 0..rows {
        let output = executor.next_morsel(&mut sampler).unwrap().unwrap();
        assert_eq!(output.source_indices, [expected]);
        std::hint::black_box(output);
    }
    let elapsed = started.elapsed().as_nanos() as f64 / rows as f64;
    let allocations = ALLOCATIONS.replace(None).unwrap_or_default() as f64 / rows as f64;
    assert!(executor.next_morsel(&mut sampler).unwrap().is_none());
    (elapsed, allocations)
}

#[test]
#[ignore = "release benchmark"]
fn cheap_dag_schedule() {
    for profiling in [false, true] {
        drain(profiling, 5_000, false);
        let (_, allocations) = drain(profiling, 20_000, true);
        let mut elapsed = (0..5)
            .map(|_| drain(profiling, 100_000, false).0)
            .collect::<Vec<_>>();
        elapsed.sort_by(f64::total_cmp);
        println!(
            "profiling={profiling} allocations_per_morsel={allocations:.2} median_ns_per_morsel={:.1} samples_ns={elapsed:?}",
            elapsed[2]
        );
    }
}
