# CSE and canonicalization validation

Branch: `feat/ir-cse-canonicalization`. Baseline: `main` at `ea23c74`.

The changes add exact deterministic common-subplan elimination, guarded
selection/layout canonicalization, worker-independent logical inference, and
separate physical kernel selection. Compilation skips optional graph snapshots
by default, reuses inferred properties and placement for unchanged graphs, and
updates selection consumer edges and properties incrementally. Structural graph
validation remains enabled.

## Semantics and execution checks

Differential execution tests cover workers 0 and 3, labels and source reads,
shared Decode/Resize/Crop chains, unequal operation parameters, independent
random branches, shared Normalize outputs, slice ordering and dtype barriers.
Two identical three-operation branches save six profiled node executions over
two morsels; two source calls still read eight selected samples. Random nodes
remain independent while their deterministic prefixes can share work.

Take/Skip tests exercise 1,728 compositions across different input lengths and
symbolic `usize::MAX` bounds. Layout tests cover inverse views on contiguous and
strided inputs, unknown ranks and shared consumers. Compiler tests compare
optimized/unoptimized pixels and labels, with explicit checks for the selected
sample/batch implementation. Strict default conversion/normalization tests
compare floating-point bits; reassociation is an explicit opt-in and can change
rounding. Crop and Normalize are never exchanged.

Unknown logical fusion contiguity allows a concrete contiguous writer to be
selected. Placement tests cover both linear and DAG graphs and still reject a
writer conflicting with a concrete strided output contract. Shared Normalize
outputs block fusion that would repeat normalization work.

## Release CIFAR benchmark

Three complete baseline runs and three final runs used release extensions and
the decoded CIFAR-10 cache in the same environment:

```sh
.venv/bin/python bench/compare_cifar10_torchvision_rivet_decoded.py \
  --torch-root /home/lingyu/.data/pytorch \
  --rivet-root /home/lingyu/.cache/rivet/datasets/cifar10 \
  --batch 128 --workers 4 --epochs 3
```

| Implementation | Three runs (img/s) | Mean (img/s) | Population stddev |
| --- | --- | --- | --- |
| Baseline | 744460, 737022, 746932 | 742805 | 4212 |
| After terminal fast-path restoration | 742141, 731380, 741870 | 738464 | 5010 |

This initial mean was 0.58% lower. Treating the difference as run variation was
insufficient: the alternating phase measurements below confirmed a remaining
runtime regression and tested a further fix.

An initial implementation forced worker sample Normalize/Layout fusion through
a channel-major CHW writer and measured 533936, 527681 and 525894 img/s, about
29% below baseline. A static three-channel specialization improved one run to
552845 img/s but did not resolve the regression. The final physical selector
retains the original HWC three-channel Normalize followed by stacking and a CHW
layout view for eligible terminal, unshared worker chains. Logical fusion remains
independent of worker count. Shared/nonterminal groups retain the typed fused
writer. Strided crop views are read directly into final output allocations;
the dedicated contiguous NHWC-to-NCHW path remains available.

## Residual regression investigation and fix

Release packages for main, the previous branch state, compiler-only changes and
the final dispatch change were loaded from separate directories using
`PYTHONPATH`. Four rounds alternated/reversed their order. Each run cached the
dataset once, warmed up five complete epochs and measured 30 complete epochs
with seed 123. The first batch's pixels and labels had the same SHA-256 digest
in all 16 runs. There were no concurrent builds during these measurements.

The values below are means of the four per-run medians. Remaining iteration
excludes startup, the first batch and loader cleanup.

| Release variant | Startup only (ms) | Remaining iteration (ms) | Total epoch (ms) | Median throughput (img/s) |
| --- | --- | --- | --- | --- |
| main | 0.07685 | 66.01324 | 66.61925 | 750543 |
| Previous branch state | 0.07562 | 66.98917 | 67.58519 | 739811 |
| Compiler changes only | 0.05794 | 66.43294 | 66.98870 | 746400 |
| Compiler changes + outlined sample fusion | 0.05771 | 66.04178 | 66.62244 | 750504 |

An earlier four-round, 40-epoch comparison also showed slower remaining
iteration (66.928 vs 66.443 ms), while startup was slightly faster. Thus repeated
compilation was not the primary cause of the remaining runtime regression.

The added `SampleNormalizeToChw` arm changed the common sample dispatcher code
generation even when the terminal pipeline used the original HWC kernel. Its
release symbol size grew from 2056 bytes on main to 4584 bytes on the previous
branch. Moving the planar writer's view setup into an `#[inline(never)]` typed
helper reduced the dispatcher to 3987 bytes and restored the measured common
pipeline throughput. The helper retains direct final allocation and existing
typed kernels. No new dynamic dispatch or per-element callback was introduced.
The A/B comparison supports a code-generation effect; without hardware
counters it does not identify instruction-cache misses versus register spills
or another specific microarchitectural cause.

Compilation was also improved independently: execution preparation now occurs
before the final inference/placement boundary, so expansion does not trigger a
second placement. Complete validated annotations skip repeated inference, and
normal execution does not format placement explanations. Semantic explain
still reports the worker-independent fused graph. Cache invalidation tests
cover unchanged graphs, appended nodes, same-size replacements and rejected
dtype changes. A shared DAG execution test exercises the outlined writer with
workers 0 and 3 and verifies pixels and sample granularity.

The standard reference command above was rerun with main and the final version
alternating across three pairs:

| Release variant | Three runs (img/s) | Mean (img/s) |
| --- | --- | --- |
| main | 746585, 745571, 748312 | 746823 |
| Final | 755440, 749511, 751920 | 752290 |

The final reference mean is 0.73% higher; the longer phase benchmark is within
0.01% of main. These measurements demonstrate restoration of this common path,
not a guaranteed speedup across workloads or hardware.

The phase benchmark is available for future investigations:

```sh
.venv/bin/python bench/measure_cifar10_ir_phases.py \
  --rivet-root /home/lingyu/.cache/rivet/datasets/cifar10 \
  --batch 128 --workers 4 --epochs 30 --warmup 5 --startup-repeats 100
```

It emits the loaded extension path, fixed-seed first-batch digest, phase medians
and individual epoch timings. Use isolated release packages and alternate run
order when comparing small performance differences.

## Final checks

- `cargo test -j 12 -p rivet-plan -p rivet-exec -p rivet-data -p rivet-vision
  --lib`: 16 plan, 29 execution, 22 data and 218 vision tests passed (285 total).
- Release editable extension built with maturin, using 12 jobs.
- `.venv/bin/pytest -q tests/test_arrow_pipeline.py`: 37 tests passed with the
  rebuilt release extension.
- Scoped library Clippy completed with existing repository warnings; this is
  not a clean `-D warnings` result.
- Changed Rust files pass rustfmt checks; `git diff --check` passed.

## Limits

The options are currently Rust APIs. CSE is a whitelist with exact parameter
bits and ordered input identity, not an algebraic equality solver or source-scan
deduplicator. It has no retained-memory/recomputation cost model. The terminal
physical expansion is a measured CPU heuristic, not a general cost optimizer.
Branch-local/post-batch selections, filter/projection pushdown, global memory
planning, parallel execution within a morsel and GPU execution remain future
work. Per-swap validation and some inference passes still traverse the full
graph.
