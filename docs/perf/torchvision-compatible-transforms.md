# Compatible vision transform fast paths

## Scope and result invariants

These fast paths optimize memory access and allocation while retaining the
Pillow/torchvision numerical contract documented in
[image compatibility](../torchvision-compatibility.md). Public parameters and
random stream identities are unchanged.

- Continuous U8/F32 ranges honor `Layout::start_offset`; strided dtype and
  elementwise color inputs retain logical-order traversal.
- Native RGB kernels borrow continuous input slices. Strided RGB input is
  packed into a temporary aligned Tensor before the same kernel is applied.
- Brightness and contrast build 256-entry tables using the original float32
  blend expression, including its clamp and truncation. Contrast computes the
  same rounded grayscale mean before building its table.
- Autocontrast computes RGB extrema in one pass and applies the exact integer
  mapping through per-channel tables. Equalize retains the same histogram step
  and half-step rounding; continuous RGB histogram updates share one traversal.
- RGB sharpness addresses three source rows directly, preserving Pillow's
  bottom/center/top accumulation, row grouping, rounding and border pixels.
  Factor 1 still materializes output rather than exposing an input alias.
- RandomCrop draws y and then x, including singleton ranges. It copies only
  the intersecting source rows and fills the rest of the final allocation with
  zero. It does not create full padded/cropped intermediate images.
- HWC RGB flips and constant padding write directly to the final allocation.
  Existing CHW and other-channel backend/fallback paths remain available.
- Resize and Gaussian blur borrow input without changing interpolation
  weights, pass order, float reduction order or rounding boundaries. Normalize
  and the specialized NHWC-to-NCHW paths are unchanged.

No SIMD intrinsics, relaxed floating arithmetic or new unsafe code are needed.
Pixel output goes directly into the final aligned allocation using typed
writers or statically dispatched exact-size iterators.

## Verification

The Rust library suite passes 222 tests, including differential regressions for:

- nonzero cache offsets, transposed images and cropped strided views;
- direct crop versus the former padded image backend, including all-zero and
  full-padding windows, and equality of the subsequent RNG state;
- RGB histogram/sharpness fast paths versus the CHW generic implementation;
- direct flips/padding versus the retained CHW image backend, including
  fractional and clipped fill values;
- dtype conversion across contiguous and strided storage, including NaN,
  infinities and out-of-range float inputs.

The full Python suite passes 220 tests plus 28 subtests, including the 147
existing torchvision numerical comparisons. Release extension build:

```bash
maturin develop --release -j 12
cargo test -j 12 -p rivet-vision --lib --no-default-features
.venv/bin/python -m pytest -q tests
```

## CIFAR release throughput

Before this change, three complete same-environment runs of the AGENTS.md
command produced **735260, 734489, 721849 img/s**, mean **730533 img/s**.
Afterwards, five complete runs produced **995007, 991373, 981210, 994896,
997346 img/s**, mean **991966 img/s**: **+35.8%** over the immediate baseline.
The repository's older 731303 img/s reference is consistent with the immediate
baseline. This is the decoded pipeline with batch 128, four workers and three
epochs; it is not a throughput claim for every augmentation or dataset.

```bash
.venv/bin/python bench/compare_cifar10_torchvision_rivet_decoded.py \
  --torch-root ~/.data/pytorch \
  --rivet-root ~/.cache/rivet/datasets/cifar10 \
  --batch 128 --workers 4 --epochs 3
```

## Native per-image measurements

The same temporary single-thread release harness was used before and after,
with input allocation outside timing, warm input/output caches and three
60 ms rounds per operator (median shown). Inputs were continuous HWC RGB.
The baseline is `a94474e`. These measurements include result allocation and
destruction, and exclude loading, decoding and Python. They are per-operator
measurements, not end-to-end speedups.

224×224, microseconds per image:

| Operation | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| brightness | 144.978 | 30.633 | 4.73× |
| contrast | 172.162 | 52.735 | 3.26× |
| posterize | 282.594 | 2.825 | 100.03× |
| autocontrast | 1898.683 | 52.711 | 36.02× |
| equalize | 1501.998 | 53.417 | 28.12× |
| sharpness | 10039.123 | 648.428 | 15.48× |
| convert_f32 | 269.085 | 14.200 | 18.95× |
| flip | 81.462 | 24.574 | 3.31× |
| pad4 | 113.516 | 4.100 | 27.69× |
| random_crop_pad4 | 150.285 | 3.455 | 43.50× |
| resize_half | 156.570 | 151.527 | 1.03× |
| normalize | 68.430 | 68.061 | 1.01× |

A permanent Criterion target now covers 32, 224 and 512 pixel images with
nonzero storage offsets, to support repeatable future measurements. The 224
pixel group was also run successfully during this change:

```bash
cargo bench -j 12 -p rivet-vision --bench transforms \
  --no-default-features -- 'transforms/224' --noplot
```

The Criterion harness has different warmup/sampling and uses offset-backed
inputs, so its absolute timings should be compared within that harness.
Remaining strided/generic kernels and numerical-order-sensitive algorithm
changes require separate profiling and differential validation.
