# Rivet Phase 3 rewrite and fusion evaluation

Measurement date: 2026-09-23. The implementation was evaluated in the worktree based on commit `50753b4`.

## Environment and command

| Item | Value |
| --- | --- |
| OS / kernel | Fedora 44, Linux 7.2.6, x86_64 |
| CPU | Intel Core Ultra 7 270K Plus, 24 cores |
| Rust | rustc 1.98.1 |
| Build | release, Criterion 0.8.2 |
| Input | CPU tensors; 16 × 64 × 64 × 3 U8 batch for normalization cases; 64 × 64 × 3 U8 sample for geometry cases |
| Criterion settings | 20 samples, 1 s warmup, 2 s measurement |

```bash
cargo bench -j 12 -p rivet-vision --bench phase3_fusions
```

The table gives Criterion's lower / point estimate / upper time estimate. Results are local microbenchmarks, not end-to-end loader throughput.

| Case | Time estimate |
| --- | ---: |
| Convert U8→F32 then Normalize | 1.0707 / 1.0713 / 1.0721 ms |
| Late promotion: Normalize U8 directly | 89.745 / 89.996 / 90.185 µs |
| Convert then Normalize then Layout | 1.0733 / 1.0746 / 1.0761 ms |
| Normalize U8 then HWC→CHW view | 88.410 / 88.417 / 88.425 µs |
| Fused NormalizeToChw | 88.465 / 88.482 / 88.503 µs |
| Crop then Resize reference | 49.576 / 49.590 / 49.609 µs |
| Flip then Normalize reference | 8.8617 / 8.9261 / 8.9710 µs |

The U8 direct path was about 11.9× faster than explicit Convert+Normalize for this input. Convert+Normalize+Layout was about 12.1× slower than the direct NormalizeToChw path. Normalize+Layout's dedicated fused kernel measured about 0.07% slower than U8 Normalize followed by a metadata-only layout view; this difference is immaterial at this scale, and fusion provides contiguous CHW storage. Crop/Resize and Flip/Normalize values are reference timings only because no fused kernels are registered.

## Rewrite legality

| Pattern | Decision | Preconditions / reason |
| --- | --- | --- |
| Convert(U8→F32) + Normalize | Apply late dtype promotion | Input must be a validated decoded U8 image. Convert must have exactly one consumer, the Normalize node, so a sibling consumer cannot lose its F32 values. The rewrite must not cross a worker sample-stage barrier; otherwise it could move Normalize from batch to worker execution. Normalize's U8 path computes the same `((u8 / 255) - mean) / std` values. Shape, axis order, sample identity, and random state do not change. |
| Convert(U8→F32) + Normalize + Layout(HWC→CHW) | Late promotion, then NormalizeToChw FusionGroup | Inherits the prior preconditions; additionally requires U8 HWC input, CHW output, a valid Normalize configuration, and a batch-stage path. Worker/sample-stage constraints can make the fusion illegal. Output is F32 CHW and contiguous. Legacy lowering parity is covered by a test. |
| Normalize + Layout(HWC→CHW) | Apply `NormalizeToChw` FusionGroup | Requires inferred U8 HWC input and the supported CHW result. The group retains both semantic ops for compatible lowering; placement requires the registered fused CPU capability. |
| Crop + Resize | Keep declared order; reject reorder/fusion | Crop coordinates refer to source pixels, while resize interpolation and border handling use the cropped extent. Reordering needs transformed coordinates and a kernel that preserves sampling semantics. A known full-image Crop is removed as an identity; other crops remain in their declared position. |
| Flip + Normalize | Keep declared order; reject fusion | Deterministic spatial Flip commutes mathematically with channel-wise affine Normalize for known U8 HWC data, but the current implementation crosses sample and batch stages. Moving it changes worker-stage execution, and no cross-stage kernel is registered. |
| Skip / Take / Shuffle + source | Push through the source sampler | Index ops are kept before image transforms and compiled into the sampler before reads. The pass validates this ordering and records it in optimizer diagnostics; it does not reorder transform nodes. |

All semantic changes clear stale property annotations and are followed by dead-node removal and property reinference. Late passes can request a bounded full restart with `PassResult::request_replan()`; the planner errors after eight unsuccessful replans.

## CIFAR-10 decoded release check

The Python extension was rebuilt with `maturin develop --release -j 12`, then the AGENTS.md benchmark command was run five times with batch 128, workers 4, and 3 epochs. Rivet steady throughput was **727,157, 717,152, 722,957, 721,885, and 746,993 img/s**: mean **727,229 img/s**, population standard deviation **10,382 img/s**, range **717,152–746,993 img/s**. Compared with the Phase 0 same-environment reference of 731,303 img/s, the mean is 0.56% lower; the observed ranges overlap, so this does not indicate a material regression. The wide current range includes one high run and should be interpreted across all five runs.

Reproduction:

```bash
maturin develop --release -j 12
for index in 1 2 3 4 5; do .venv/bin/python bench/compare_cifar10_torchvision_rivet_decoded.py --torch-root /data/datasets/pytorch --rivet-root /data/datasets/rivet/cifar10 --batch 128 --workers 4 --epochs 3; done
```

## Verification

- `cargo test -j 12 -p rivet-plan --lib`: 9 passed.
- `cargo test -j 12 -p rivet-vision --lib`: 150 passed.
- Legacy output comparison passed for Convert+Normalize and Convert+Normalize+Layout.
- Placement explain lists the `NormalizeToChw` physical candidate and the selected registered FusionGroup kernel.
- The generic planner contains no vision payload downcasts; Vision registers inference, rewrite passes, fusion rules, and physical candidate providers through its explicit plugin.
