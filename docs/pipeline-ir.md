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

## Order-preserving semantic rewrites

The builder records image, index, batch and device operations in declaration
order. Its unoptimized IR therefore retains `Decode -> RandomCrop -> Take ->
Shuffle -> Batch`. The selection rewrite changes graph edges to move eligible
index operations toward the source, producing `Take -> Shuffle -> Decode ->
RandomCrop -> Batch`. Image transforms retain their relative order, and index
operations retain theirs. The optimizer does not execute source reads.

An index may cross a known image operation only when that operation preserves
independent one-to-one samples and their ordering, has no external side effects,
and receives sample-granularity input. Random operations also require a stable
semantic identity. Unknown operators, multiple inputs, shared producer edges,
explicit Batch, Cache and device boundaries stop the rewrite. The current
compiler supports one shared source sampler prefix: residual branch-local and
post-barrier selections fail with a node-specific capability error. They remain
in the IR at their declared positions rather than being silently hoisted.

Operator semantic facts grant permission to specific selection rules; they do
not imply that two image operations commute. Crop and Normalize are not swapped.
For example, zero padding before Normalize with mean/std 0.5 becomes -1; padding
with zero after Normalize would produce 0. Current Crop/RandomCrop dtype
requirements also prevent moving them after Normalize. Physical batch lifting
requires an explicit per-sample/stack equivalence declaration and only moves a
contiguous eligible unary CPU suffix immediately before stacking. The linear
typed executor is selected only when physical sample and batch ordering matches
its cached traversal.

Source sampling now composes operations in declaration order. `take(4).shuffle`
permutes the first four source rows; `shuffle.take(4)` selects the first four
rows of the full permutation. Multiple shuffles compose on the current sequence.
The first shuffle preserves the existing global-seed/epoch namespace; subsequent
shuffles derive namespaces from its base seed, shuffle ordinal and declared seed.
An explicit pipeline `.seed()` continues to override the first shuffle's seed.
Range arithmetic saturates instead of overflowing.

Random streams depend on global seed, epoch, original source sample identity and
the random operation's semantic identity. Identities are assigned before rewrites
and survive edge rewiring, arena compaction and builder round trips. Newly added
random nodes receive unused identities. Changing worker count or removing an
earlier random node therefore does not renumber surviving streams.

Selection uses **selected-sample error semantics**: image transforms are evaluated
for selected samples, so a decode or transform error in an excluded sample need
not be observed. Graph and configuration validation still happen at compilation.
`placement_explain()` reports successful rewrites and blocked decisions. This
preserves the existing selective source-read behavior while making the rewrite
and error contract explicit.

See [semantic rewrite validation](perf/semantic-ir-rewrite.md) for differential
tests and release throughput measurements.
