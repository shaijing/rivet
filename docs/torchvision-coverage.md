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
- Some arguments remain narrower: width-first dimensions; nonnegative crop
  positions/padding; scalar geometric fill; square Gaussian kernels; no custom
  rotation center; no Resize scalar short-edge/max-size overload; RandomCrop
  has constant symmetric padding and lacks `pad_if_needed`; RandomAffine still
  exposes symmetric angle/shear ranges. Standard random wrappers are available
  but not every arbitrary per-channel or random-valued RandomErasing fill.
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
