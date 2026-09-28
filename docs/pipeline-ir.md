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

## Exact sharing and local canonicalization

Common-subplan elimination (CSE) runs after source selection pushdown and local
canonicalization, before fusion. It shares identical deterministic unary image
nodes with the same ordered input node identities. Matching uses explicit
operation and parameter encodings, not formatted debug strings or approximate
numeric equality. Float parameters use their IEEE bit patterns, so signed zero
and distinct NaN payloads remain distinct; mean/std vector lengths are part of
the key too.

The current whitelist is Decode, Resize, Crop, CenterCrop, Pad, Flip, Normalize,
Layout and supported ConvertImageDtype operations. Every relevant parameter is
included: interpolation mode, crop coordinates, padding sides/mode/fill, flip
direction, normalization statistics, target layout and target dtype. CSE does
not combine consecutive operations or exchange their order. In particular,
Crop and Normalize remain in their declared order.

Sources, random operators, unknown/custom payloads, Batch, Cache, device
boundaries, joins and already fused groups are excluded. Separate source nodes
remain separate even if they carry the same source object. Equal parameters do
not make two RandomCrop operations equivalent: their stable semantic identities
can intentionally produce different views. Deterministic successors of distinct
random inputs also remain distinct.

Shared outputs preserve labels and sample identities. Consumers receive their
declared input ports, including repeated references to one shared result.
Arena compaction preserves random identities and invalidates annotations, which
are inferred again before fusion. Execution shares immutable backing storage
until the final consumer releases it. Sharing may extend a tensor's lifetime;
CSE currently has no cost model comparing recomputation with retained memory.
Normalize/Layout fusion requires the Normalize output to have exactly one
consumer edge. A shared Normalize remains shared rather than being fused into
one consumer and recomputed for the others; explanations report this blocked
fusion decision.

Adjacent Take/Skip chains are composed into at most one Skip followed by one
Take using saturating interval arithmetic. They are combined only through
single-consumer selection edges. Shuffle, image operators, batching, shared
branches and other intervening nodes stop composition; operations are never
composed across Shuffle. Source sampler ordering remains mandatory even when
this optional canonicalization is disabled.

Identity dtype/layout operations and full-image crops can be removed when
inferred properties prove they leave the value unchanged. Adjacent inverse
Layout views can additionally be removed when the input representation, rank
and axis order prove both permutations are valid. The inner layout must have
one consumer. The inverse pair restores the original shape, strides and
contiguity, including a strided input; this rule never crosses Normalize or a
materializing fusion. Unknown properties retain the operations and their runtime
checks.

## Numerical equivalence and execution choices

Logical image properties and rewrite legality are independent of worker count.
An identity conversion is no longer kept merely to act as a scheduling barrier.
The physical compiler separately chooses sample or batch kernels from actual
input granularity, preceding work and runtime worker configuration. Per-node
physical choices do not grant permission to reorder arbitrary image operations.

For workers greater than zero, an unshared terminal Normalize/Layout fusion
immediately before Batch/Sink can lower to sample HWC normalization followed by
stacking and a batch Layout view. The compiler applies this only to a linear
sample-work prefix without a prior batch implementation or device boundary.
This uses the specialized RGB normalization path and avoids an expensive
channel-major sample write. Generic and shared DAGs retain their fused writer;
the logical fusion and values do not depend on worker count.

Logical NormalizeToChw fusion reports contiguity as Unknown because equivalent
physical implementations can either write a contiguous result or defer the
layout as a strided view. Actual physical writers and tensors determine concrete
contiguity. Consumers must not treat the logical fusion name as proof that its
output is contiguous.

The default numerical policy preserves the rounding introduced by declared
operations. An explicit U8-to-F32 conversion followed by Normalize remains two
operations: bypassing the conversion can use a reassociated affine expression
and change low floating-point bits. That rewrite requires
`allow_float_reassociation: true` and a single consumer. This policy concerns
optimizer rewrites; it does not replace the established arithmetic of a directly
declared U8 Normalize kernel or promise identical floating-point behavior across
hardware platforms.

Normalize/Layout fusion preserves their order and output values. Specialized
three-channel HWC/NHWC normalization and NHWC-to-NCHW paths remain available.
Crop and CenterCrop return strided tensor views; compilation does not insert an
extra contiguous copy for those intermediate views. Consumers materialize when
required, such as stacking a batch or writing Normalize's output. Strided views
retain their source storage, so delaying a copy can also extend storage lifetime.
Further direct-output fusion requires a separate equivalence and profitability
rule; it is not implied by either CSE or view support.

## Optimization controls and explanations

`ImageOptimizationOptions` is available from `rivet_vision::api` and
`rivet_vision::pipeline`. Defaults enable CSE and the new selection/layout
canonicalizations, disable float reassociation, and omit per-pass snapshots.

```rust
use rivet_vision::api::ImageOptimizationOptions;

let options = ImageOptimizationOptions {
    common_subplan_elimination: false,
    record_snapshots: true,
    ..ImageOptimizationOptions::default()
};
// Explain without reading the dataset:
let explanation = pipeline.optimization_explain(options)?;
// Compile a builder with the same options:
let loader = pipeline.compile_with_options(options)?;
```

Edited DAGs use
`ImagePipeline::compile_logical_plan_with_options(plan, start, options)`.
The flags independently control CSE, adjacent selection composition, inverse
layout simplification and floating-point reassociation. Disabling them does not
turn off graph/configuration validation, source-index order handling, fundamental
identity elimination or existing fusion rules. Python's builder does not expose
these Rust optimization options yet.

`optimization_explain()` shows the declared graph, optimized graph, placement
and rule diagnostics. With `record_snapshots`, it also includes per-pass graph
snapshots. `LogicalPlan::explain()` shows an edited graph directly, and the
compiled loader's `physical_explain()` reports the executable graph/runtime path.
Snapshots are optional to avoid constructing repeated full graph strings during
normal compilation. Validation still runs after passes. Domain rewrites clear
invalid properties or update them after proving downstream facts unchanged;
inference passes reuse complete validated annotations. Execution preparation
expands eligible terminal fusion before final inference and placement, so the
compiler places the implementation graph once and reuses that result. Normal
execution also skips formatting placement explanations; explain APIs retain
their reports. The planar sample writer's setup is kept in a separate function
to limit its effect on common sample dispatch code generation.

## Standalone Python reordering checks

`Pipeline.explain()` exposes the declared logical graph, optimized logical graph,
placement and rewrite diagnostics through Python. It validates/optimizes a
separate inspection plan without executing image reads and does not cache a
compiled execution plan for subsequent `execute()` calls.

After building the Python extension, run the self-contained test script:

```sh
.venv/bin/python tests/test_ir_reordering.py --show-plans
# Optional worker/epoch configurations:
.venv/bin/python tests/test_ir_reordering.py --workers 0 4 --epochs 0 1 7
# The same tests are also discoverable by pytest:
.venv/bin/pytest -q tests/test_ir_reordering.py
```

The script generates 24 distinct RGB PNG samples in temporary Arrow IPC files,
including a dataset with a corrupt row. It requires no downloaded dataset and
returns a nonzero exit code on failures. It follows input edges from the graph
root rather than treating arena/display order as execution order. `IndexOp` is
a generic graph label; labels and pixel comparisons separately verify Take,
Skip and Shuffle ordering and parameters.

Twelve tests cover actual selection pushdown, comparison against an explicit
reordered pipeline and full-dataset random augmentation by source identity,
worker/epoch reproducibility, slice/shuffle barriers, adjacent slice composition,
inverse layout removal, padded Crop before Normalize, rejected Normalize before
Crop and post-Batch selections, excluded corrupt samples, and repeatable
inspection that leaves the declared pipeline intact. Pixel comparisons use exact
equality except the independent NumPy normalization reference, which allows
floating-point rounding tolerance while checking padding maps exactly to -1.

## Remaining work

The current rules are deliberately local and conservative. Future work includes
cost-aware sharing/fusion, automatic sharing of equivalent source scans,
branch-local and post-batch selection execution, metadata filtering and field
projection, incremental property inference/compilation caching, more direct-output
kernels, parallel scheduling within a morsel and a global retained-memory budget.
GPU/DeviceCut execution and a Python DAG-building API also remain separate work.
No general commutativity rule permits Crop/Normalize, resize chains or dtype
conversion chains to be exchanged or collapsed.
