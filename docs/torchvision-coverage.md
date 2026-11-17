# torchvision image API coverage

This inventory compares the local torchvision 0.29 image transforms with
Rivet's native Rust implementation. The baseline before this migration was
`70e7e27`. Python methods below only configure Rust operations: there are no
Python image-processing callbacks or runtime PyTorch dependencies.

The numerical target remains PIL/RGB uint8 unless a Tensor reference is stated.
See [numerical compatibility](torchvision-compatibility.md) for the distinction
between PIL and Tensor results. Similar names do not imply identical Python
object types, argument order, memory strides, or support for every dtype.

## Added in this migration

| torchvision function / transform | Native Rivet entry point | Notes |
| --- | --- | --- |
| adjust_saturation | `saturation(factor)` / `SaturationConfig` | Pillow L-image blend; nonnegative factor |
| adjust_gamma | `gamma(gamma, gain=1)` / `GammaConfig` | f64 PIL lookup table, uint8 clipping; negative finite gain yields black |
| ColorJitter saturation | `color_jitter(..., saturation=...)` / `ColorJitterConfig.with_saturation(...)` | All four components; existing three positional arguments retain brightness/contrast/hue order |
| affine | `affine(angle, translate, scale, shear, ..., center)` / `AffineConfig` | Deterministic f64 inverse matrix, custom center, nearest/bilinear/bicubic |
| erase | `erase(x, y, width, height, value)` / `EraseConfig` | uint8/float32, scalar or per-channel fill, HWC/CHW; uint8 fill truncates and wraps |
| resized_crop | `resized_crop(x, y, width, height, out_width, out_height, interpolation)` | Native crop + resize; dimensions remain width-first |
| RandomVerticalFlip | `random_vertical_flip(probability=0.5)` | Native RandomApply + vertical flip |
| RandomRotation | `random_rotation(degrees, ..., expand=False)` / `RandomRotationConfig` | Scalar symmetric range or explicit `[min,max]` |
| GaussianBlur sampled sigma | `random_gaussian_blur(kernel_size, sigma=[0.1,2.0])` | Positive sigma range, square odd kernel; deterministic `gaussian_blur` remains available |
| RandomInvert | `random_invert(probability=0.5)` | Native random wrapper |
| RandomPosterize | `random_posterize(bits, probability=0.5)` | Native random wrapper |
| RandomSolarize | `random_solarize(threshold, probability=0.5)` | Native random wrapper |
| RandomAdjustSharpness | `random_adjust_sharpness(factor, probability=0.5)` | Native random wrapper |
| RandomAutocontrast | `random_autocontrast(probability=0.5)` | Native random wrapper |
| RandomEqualize | `random_equalize(probability=0.5)` | Native random wrapper |
| AutoAugment | `auto_augment(policy="imagenet")` / `AutoAugmentConfig` | ImageNet, CIFAR10 and SVHN 25-pair learned policy tables |
| RandAugment | `rand_augment(num_ops=2, magnitude=9, num_magnitude_bins=31)` | torchvision operation space and signed magnitude bins |
| TrivialAugmentWide | `trivial_augment_wide(num_magnitude_bins=31)` | torchvision wide operation/magnitude space |
| AugMix | `augmix(severity=3, mixture_width=3, chain_depth=-1, alpha=1)` | Native Dirichlet sampling, Tensor-reference branches and float32 mixing |
| LinearTransformation | `linear_transform(matrix, mean)` / `LinearTransformationConfig` | Float32 whitening, CHW flatten order, row-vector multiplication by the supplied matrix; input must already be float32 |
| ToTensor numeric/layout equivalent | `to_tensor()` | Decoded HWC → float32 CHW; uint8 divided by 255; returns Rivet/NumPy data, not a torch.Tensor |
| PILToTensor numeric/layout equivalent | `pil_to_tensor()` | Decoded HWC → CHW, preserving dtype/range; does not accept or create a Python PIL object |
| BOX / HAMMING resize | `resize(..., interpolation="box" / "hamming")` | The same signed fixed-point resampler, with Pillow filter definitions |

Native methods are available on both `TransformSequence` and `ImagePipeline`,
and through Python `rivet.Transform` / `rivet.Pipeline`. Typed configs are
exported from `rivet_vision::transforms` and `rivet_vision::api`.

Random wrappers lower to existing Rust composition nodes. Policy algorithms
have distinct native IR IDs and random namespaces. Equal Rivet seeds reproduce
results across worker counts; they do not reproduce PyTorch's random sequence.

The existing ColorJitter constructor preserves its old random draws when the
new saturation range is zero. Rust struct literals for ColorJitter configs or
resolved parameters need the new saturation/order fields; prefer constructors.

## Already present before this migration

| Family | Existing support |
| --- | --- |
| Composition | Compose/TransformSequence, RandomApply, RandomChoice, RandomOrder |
| Spatial | Resize, Crop, CenterCrop, Pad (constant/edge/reflect/symmetric), horizontal/vertical flip, RandomCrop, RandomResizedCrop, RandomHorizontalFlip |
| Geometry | Fixed/arbitrary rotate, RandomAffine, Perspective, RandomPerspective, ElasticTransform |
| Color | Brightness, Contrast, Hue, Grayscale, RandomGrayscale, Sharpness, Invert, Posterize, Solarize, Autocontrast, Equalize |
| Representation | DecodeImage, Normalize, uint8↔float32 ConvertImageDtype, HWC↔CHW |
| Erasing/filtering | RandomErasing, deterministic GaussianBlur |
| Multi-view | FiveCropConfig and TenCropConfig return multiple samples in Rust; they are not Python pipeline methods because that pipeline currently yields one image per source sample |
| Metadata | Shape/dtype/layout are available from native Tensor and plan properties; this is not a separate pixel transform |

## Remaining gaps

The standard v1 algorithm families above are covered for supported image
configurations, but this is not a full clone of torchvision's Python API.

- `ToPILImage` and arbitrary Python `Lambda` are Python object/callback bridges,
  not native image kernels. They are not installed as runtime dependencies.
- Dimensions remain width-first and crop positions/padding are nonnegative.
  See the API completion section below for supported parameter overloads and
  the distinction between legacy shear maxima and explicit shear intervals.
- RGB uint8 is the main pixel-kernel contract. Generic torchvision Tensor
  batches, all integer/float dtypes, GPU/autograd and every PIL mode are not
  supported. LinearTransformation currently accepts float32 only. Tensor
  conversion aliases retain Rivet's view/stride conventions.
- v2-only image features remain: JPEG augmentation, GaussianNoise,
  RandomChannelPermutation, RandomPhotometricDistort, RandomResize,
  RandomShortestSize, RandomZoomOut and ScaleJitter. RGB decoding exists, but
  there is no standalone v2 RGB conversion transform for arbitrary input modes.
  JPEG parity additionally needs the corresponding encoder/subsampling contract;
  it is not approximated by an incompatible `image` encoder.
- MixUp/CutMix need batch and soft-label semantics beyond the current scalar
  class-label sample model. RandomIoUCrop and bounding-box/keypoint sanitation
  need structured annotation support. UniformTemporalSubsample is a video
  operation. Those are separate data-model extensions, not silently applied
  to RGB images alone.

## Existing API parameter completion

These additions are available on both `rivet.Transform` and `rivet.Pipeline`.
They extend existing operators; the v2-only image and data-model gaps above
remain separate work.

| API | Parameters and behavior |
| --- | --- |
| `random_choice(choices, p=None)` | Optional relative weights; finite, nonnegative, matching the branch count, with a finite positive sum. Weights are normalized once for compiled execution. Omitted weights preserve the original uniform sampler and random stream. |
| `resize(width, height=None, interpolation="bilinear", *, max_size=None)` | Omitting height scales the short edge while preserving aspect ratio. `max_size` caps the long edge and must exceed the requested short edge. Explicit width/height calls retain their original meaning. |
| `random_crop` | Scalar or per-side padding, `pad_if_needed`, fill and padding modes. Padding precedes random crop sampling. |
| `random_affine` | Scalar symmetric degrees or an explicit angle interval; explicit shear intervals via `shear_ranges`. The existing two-element `shear` remains symmetric per-axis maxima. |
| `arbitrary_rotate` / `random_rotation` | Optional `(x, y)` rotation center in image coordinates. |
| Geometry fill | Scalar or RGB-triplet fills for supported padding and warp operations. Warp triplets require three-channel uint8 images; padding also supports float32. Fill does not change interpolation or axis conventions. |
| `gaussian_blur` | Scalar or `(sigma_x, sigma_y)` sigma; scalar or `(width, height)` kernel size. Existing sigma-only calls remain supported. |
| `random_gaussian_blur` | Scalar or rectangular kernel; sigma is a fixed scalar or `[min, max]` sampling range, not a pair of axis sigmas. A sampled sigma is shared by both axes. |
| `random_erasing` / `random_erasing_with_options` | Scalar/channel fill or `value="random"` for independent standard-normal pixels. Float32 fills retain their values; uint8 conversion truncates and wraps modulo 256, as in the Tensor reference. |

For example:

```python
from rivet import vision as rv

pipeline = (
    rv.scan_image_folder("data/train")
    .decode_image()
    .resize(256, max_size=384)
    .random_crop(224, 224, padding=[4, 8, 4, 8], pad_if_needed=True)
    .random_choice(
        [rv.Transform().horizontal_flip(), rv.Transform().brightness(0.8)],
        p=[1, 3],
    )
    .random_affine([-10, 15], shear_ranges=[-5, 5, -2, 2])
    .gaussian_blur([0.8, 1.2], kernel_size=[3, 5])
    .to_tensor()
    .random_erasing(value="random")
    .batch(128)
)
```

Random operations remain deterministic for a Rivet seed/epoch across worker
counts. Their random streams do not match PyTorch's RNG sequence. Existing
RGB uint8 and layout restrictions still apply to individual operators.

## Numerical tests and reference generation

`tests/test_torch_image_migration.py` compares new functions to installed
Pillow/torchvision, checks probability-0/1 and fixed-range random limits,
validates bad parameters and checks reproducibility across workers. The older
`tests/test_torch_image_compat.py` remains unchanged.

The native policy tests use independently generated CPU/PIL fixtures stored at
`crates/rivet-vision/tests/fixtures/torchvision_augment.json`. The reference
contains every policy primitive, magnitude tables and a fixed nontrivial
AugMix mixture. PIL primitives are checked for exact bytes; Tensor-policy
floating results and the fixed AugMix mixture allow one uint8 level. Affine
float interpolation allows one level; float32 LinearTransformation uses float
tolerances. Nearest affine/rotation follows Pillow's 16.16 fixed-point path
where Pillow selects it, rather than relying only on ideal double coordinates.

```bash
# Regenerate fixtures only when intentionally updating the reference version.
.venv/bin/python bench/generate_torchvision_augment_reference.py
maturin develop --release -j 12
cargo test -j 12 -p rivet-vision --lib --no-default-features
.venv/bin/python -m pytest -q tests
```

The completed validation passed **228 Rust library tests**, **288 Python tests**
and **28 subtests**. The new migration file contributes 68 Python cases. The
CIFAR decoded regression command from `AGENTS.md` was run three times in
release mode (batch 128, workers 4, epochs 3), giving **1137122**, **1142782**
and **1127746 img/s**, arithmetic mean **1135883 img/s**. No common-pipeline
regression was observed relative to the preceding 991966 img/s reference.
This checks the existing decoded pipeline, not the cost of enabling each new
augmentation policy, and is not a claim that the migration itself caused the
throughput difference.

Example native pipeline:

```python
import rivet

train = (
    rivet.scan_image_folder("train")
    .decode_image()
    .random_resized_crop(224, 224)
    .random_vertical_flip(0.1)
    .color_jitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05)
    .rand_augment(num_ops=2, magnitude=9)
    .normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    .hwc_to_chw()
    .seed(42)
    .workers(4)
    .batch(128)
)
```

## API completion validation (2026-11-17)

The integrated release extension passed 406 Python tests plus 28 subtests and
248 `rivet-vision` library tests. Coverage includes parameter errors, PIL/Tensor
comparisons, custom rotation centers, sequential crop padding, weighted choice,
and seed/epoch reproducibility across worker counts.

Using the decoded CIFAR benchmark with batch 128, four workers and three epochs,
three pre-change runs averaged 1,130,686 img/s (population standard deviation
4,317); five post-change runs averaged 1,123,509 img/s (standard deviation
5,102), a -0.63% change. These are fresh same-environment measurements; the
older absolute reference is not substituted for the pre-change comparison.
