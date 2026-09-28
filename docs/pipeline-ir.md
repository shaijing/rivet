# Executable pipeline IR

Image pipelines now compile directly from `LogicalPlan` to a graph of executable
physical operators. Compilation no longer reconstructs an `ImagePipeline` or
builds the former `ExecutionPlan`. `ImageDataLoader.info` contains compile-time
layout, dtype and kernel-count diagnostics; executable kernels belong to nodes.

`LogicalPlan::topological_order()` visits dependencies before consumers and
visits shared nodes once. `PhysicalOperator::execute_inputs()` receives input
ports in their declared order. Unary operators reject extra inputs by default;
a join implements this method explicitly. `ExecutableGraph` caches validation,
scheduling and edge use counts. Execution shares values at a fork and moves or
releases each retained result at its last use.

CPU linear image graphs retain the fused pull executor and typed kernel path.
DAGs use `DagPipelineExecutor`: persistent workers execute independent batch
morsels, each graph runs in dependency order, and results return in sampler
order. Prefetch bounds the number of outstanding morsels. The byte limit applies
to each DAG edge value; it is not a global allocation limit. Operator errors,
panics and oversized values terminate the executor. Independent branches within
one morsel currently run sequentially.

Linear placement retains its exact dynamic program. DAG placement uses a
deterministic topological heuristic, accounting for every incoming edge and
checking kernel requirements on every input. This heuristic does not guarantee
a globally optimal device assignment. Transfer descriptions identify both the
producer and consumer.

## Image branches

The Rust entry point is `ImagePipeline::compile_logical_plan(plan, start)`.
`ImageConcat { axis }` joins two or more image branches; `axis` is one of the
three image axes and excludes the batch dimension. It works before or after
batching. Inputs must have matching sample identities, dtype, layout and
non-concatenated extents. Batch joins also require matching labels.

```rust
use std::sync::Arc;
use rivet_plan::{LogicalNode, NodeKind};
use rivet_vision::api::{
    ImageConcat, ImageDataLoader, ImageOp, ImagePipeline, ImageSource, RivetResult,
};
use rivet_vision::pipeline::op::BatchConfig;

fn two_views(source: ImageSource) -> RivetResult<ImageDataLoader> {
    // A decoded image source; the builder supplies runtime and seed context.
    let mut plan = ImagePipeline::from_source(source)
        .workers(4)
        .seed(42)
        .to_logical_plan();
    let source = plan.nodes()
        .find(|(_, node)| node.kind() == NodeKind::Source)
        .unwrap().0;
    let inverted = plan.add_node(LogicalNode::new(
        NodeKind::Op, [source], Some(Arc::new(ImageOp::invert())),
    ));
    let views = plan.add_node(LogicalNode::new(
        NodeKind::Op, [source, inverted],
        Some(Arc::new(ImageConcat { axis: 0 })),
    ));
    let batch = plan.add_node(LogicalNode::new(
        NodeKind::Batch, [views],
        Some(Arc::new(BatchConfig::new(128, false))),
    ));
    let sink = plan.add_node(LogicalNode::new(NodeKind::Sink, [batch], None));
    plan.set_root(sink).unwrap();
    ImagePipeline::compile_logical_plan(plan, 0)
}
```

Vision graphs currently share one source and one sampler prefix. Every batch
barrier uses the same batch size and `drop_last` policy. Normalize, dtype
conversion and layout operations also support explicit batch inputs. DeviceCut
execution remains unavailable. The Python builder continues to expose linear
pipelines; graph editing and joins are currently Rust APIs.

See the [validation and performance record](perf/pipeline-dag.md) for targeted
tests and release CIFAR measurements.
