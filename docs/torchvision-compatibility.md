# torchvision image result compatibility

Rivet's Rust transforms use torchvision's PIL/RGB uint8 path as the numerical
reference. Gaussian blur and elastic transforms follow the Tensor path that
torchvision also uses for PIL input. The Rust runtime does not call Python,
Pillow, or PyTorch. Comparisons use lossless PNG fixtures to isolate transform
results from decoder differences.

## Sources

Implementation was checked against these local sources:

- `/home/lingyu/ws/pyWS/Pillow`, revision `8697c0801`:
  `src/libImaging/{Resample,Geometry,Blend,Convert,Filter}.c` and
  `src/PIL/{Image,ImageEnhance,ImageFilter,ImageOps}.py`.
- The project virtual environment's torchvision `0.29.0+cu130`:
  `transforms/{functional,_functional_pil,_functional_tensor,transforms}.py`.
- Local PyTorch `2.14.0+cu130` supplies the Tensor reference operations;
  installed Pillow is `12.3.0`.

These references describe the tested environment, not a promise of identical
results across all future torchvision releases or devices.

## Behavior

| Operation | Rust behavior and reference |
| --- | --- |
| Resize / resized crop | Pillow's normalized separable filters, signed 22-bit weights, horizontal then vertical u8 rounding. Nearest, bilinear, bicubic, Lanczos3, BOX and HAMMING are checked byte for byte, including upsampling, downsampling and single-pixel axes. |
| Brightness | Nonnegative multiplicative factor, Pillow float32 blend and truncation. |
| Contrast | Blend against the rounded mean of the Pillow L image, rather than a fixed mid-gray value. |
| Hue | Pillow RGB → HSV → RGB conversion, wrapping an integer hue offset derived from a factor in `[-0.5, 0.5]`. |
| Sharpness | Pillow SMOOTH kernel (`1,1,1; 1,5,1; 1,1,1`, divided by 13), unchanged border pixels, then blend by factor. |
| Grayscale / invert / posterize / solarize / autocontrast / equalize | Exact uint8 checks; posterize also accepts zero bits. Grayscale uses Pillow BT.601 rounding. |
| Crop / center crop | In-bounds crops retain tensor views. Out-of-bounds right/bottom crop regions are zero-padded. Oversized center crops pad with the extra odd pixel on the right/bottom; offsets use ties-to-even rounding. |
| FiveCrop / TenCrop | Center window uses the same ties-to-even positioning as CenterCrop. |
| Padding | Constant, edge, reflect and symmetric; reflected coordinate tables are computed before pixel loops. PIL-style repeated reflection supports padding wider than the image. |
| Fixed rotation | Positive angles are counter-clockwise. This convenience operation always expands, equivalent to torchvision rotate with `expand=True`. |
| Arbitrary rotation | Pillow reverse matrix, pixel centers, expansion bounds, interpolation and truncation; defaults to nearest with `expand=False`. |
| Perspective | Solve the inverse homography in f64, quantize its coefficients to f32 as torchvision does, then sample using Pillow coordinates and clipping. |
| Gaussian blur | Normalized float32 Gaussian kernel, reflection padding, ties-to-even uint8 rounding. Explicit odd kernel size is supported; differential checks allow 1 uint8 level for floating reduction differences. |
| ConvertImageDtype | U8 → F32 divides by 255; F32 → U8 scales by `256 - 1e-3`, truncates and wraps, rather than clamping and rounding. Nonfinite values convert to zero on this CPU implementation. |
| Normalize / layouts | Existing specialized RGB normalization and NHWC-to-NCHW kernels are retained; float comparisons use tolerances. |

## Random transforms

Random results are reproducible within Rivet, including across worker counts.
The RNG is not PyTorch's generator, so equal seeds do not produce equal images.
Alignment applies to the sampling policy and to execution for equal parameters:

- ColorJitter samples brightness/contrast factors around 1 and hue around 0,
  shuffles operation order, and skips disabled adjustments. This API covers
  brightness, contrast, saturation and hue. Saturation is an appended optional
  argument, so the earlier three positional arguments retain their order.
- RandomAffine uses rounded integer translation drawn from fractions of full
  image width/height and torchvision's rotation/scale/shear matrix composition.
- RandomPerspective draws integer corner points inward within torchvision's
  corner ranges, rather than perturbing corners in arbitrary directions.
- RandomResizedCrop uses ties-to-even size rounding and the centered fallback.
- RandomErasing interprets ratio as height/width, requires the rectangle to be
  smaller than the image, and preserves the original image/storage after ten
  failed attempts. Full-image erasure is no longer the failure fallback.
- RandomGrayscale defaults to three output channels for RGB input.
- ElasticTransform uses uniform noise, reflected Gaussian smoothing with
  torchvision's `int(8*sigma+1)` odd kernel, no maximum-amplitude renormalization,
  and pixel displacement `noise * alpha / 2`. Its Tensor sampler uses zero
  padding, a fill mask and ties-to-even output rounding. Rust golden tests
  cover nonzero displacement and the smoothing boundary independently of RNG.

## Migration

Existing applications must update calls that relied on old Rivet semantics:

- `brightness(value)` now takes a floating factor: `1.0` is identity, `0.0`
  is black. The old additive offset has no universal equivalent factor.
- `contrast(value)` now takes a nonnegative factor around the grayscale mean.
- `hue(factor)` uses fractions of a hue cycle, not degrees. `HueConfig.degrees`
  was renamed to `factor`; HSV quantization differs from the old hue matrix.
- `sharpness(amount)` now uses torchvision's factor: `1.0` is identity,
  `0.0` smooths, and values above 1 sharpen.
- ColorJitter brightness and hue arguments are now float ranges; e.g.
  `.color_jitter(0.2, 0.4, 0.1)`. Rust callers must use float literals.
- Positive rotations reverse the old clockwise convention. Negate old angles
  to retain the previous direction. ArbitraryRotate and RandomAffine default
  to nearest interpolation; ArbitraryRotate no longer expands by default.
- `.gaussian_blur(sigma, kernel_size=5)` selects an explicit odd kernel.
  Sigma-only calls use `2 * ceil(3*sigma) + 1`; this is a Rivet convenience
  because torchvision requires a kernel size. Sigma zero remains identity.
- `.pad(..., padding_mode="reflect")` and `.pad_with_sides(...)` expose the
  additional Rust padding modes; constant remains the default.

## Limits

This is result alignment for supported image configurations, not full
torchvision API compatibility. Rivet still uses width-first dimensions and
nonnegative crop coordinates/padding. Many color and resize kernels require
RGB uint8 HWC input; arbitrary Tensor batches/dtypes and all PIL modes are not
supported. A general Tensor-input torchvision pipeline may produce different
results from the PIL path. General source-decoder parity is outside these comparisons; the standalone
JPEG augmentation below defines its own encode/decode contract.

Perspective coefficients can differ at floating-point degeneracies because the
Rust linear solver is not LAPACK's `gels`; exact boundary ties are not guaranteed
across every quadrilateral. Gaussian blur and elastic sampling may have small
floating-point differences from PyTorch's CPU/GPU kernels. Reflection blur
padding must be smaller than image dimensions, as in PyTorch. Geometry supports
nearest/bilinear/bicubic; elastic sampling supports nearest/bilinear. Lanczos is
available for resize, not warp sampling. BOX and HAMMING are resize-only too.
Zero-sized image configurations are not supported.

## Extended parameters

Both `Transform` and `Pipeline` support weighted `random_choice`, short-edge
resize with a long-edge cap, configurable random-crop padding, rotation centers,
explicit affine angle/shear intervals, per-channel geometric fill, rectangular
Gaussian kernels with axis-specific sigma, and per-channel/random erasing.
See the [API parameter inventory](torchvision-coverage.md#existing-api-parameter-completion)
for signatures and an example.

The original width-first calling convention remains. In particular,
`resize(224, 192)` sets width 224 and height 192, while `resize(224)` scales the
short edge. Gaussian kernel and deterministic sigma pairs are ordered x/y.
Random blur's sigma pair is a sampling interval instead.

Legacy `random_affine(..., shear=[x, y])` continues to mean symmetric maxima
for each axis. Use `shear_ranges=[min_x, max_x]` or
`shear_ranges=[min_x, max_x, min_y, max_y]` for explicit torchvision-style
intervals. Scalar degrees remain symmetric; a two-element degrees sequence
specifies its bounds.

`random_choice(..., p=[...])` accepts relative nonnegative weights and
normalizes them. Omitting `p` retains the previous uniform sampler. Random
erasing uses independent standard-normal values for `value="random"`;
uint8 fill conversion truncates and wraps modulo 256, whereas float32 retains
the sampled or supplied values. Random reproducibility is defined by Rivet's
seed/epoch contract, not PyTorch's RNG sequence.

## Validation

Run the native kernels against the installed torchvision reference:

```bash
maturin develop --release -j 12
.venv/bin/python -m pytest -q tests/test_torch_image_compat.py \
  tests/test_arrow_pipeline.py tests/test_image_folder.py
cargo test -j 12 -p rivet-vision --lib --no-default-features
```

Use the CIFAR release benchmark in `AGENTS.md` to check the common decoded
pipeline; this throughput check does not benchmark every augmentation kernel.

The initial compatibility validation passed 216 Rust library tests and the full Python
suite (220 tests plus 28 subtests). Five complete release CIFAR runs during this
change measured `737277`, `714489`, `735307`, `730894` and `740055 img/s`:
mean `731604 img/s`, versus the same-environment reference mean `731303 img/s`
(approximately +0.04%). No meaningful common-pipeline regression was observed.


Subsequent semantics-preserving fast paths and their current validation and
throughput measurements are documented in
[compatible transform performance](perf/torchvision-compatible-transforms.md).


See the [torchvision API coverage inventory](torchvision-coverage.md) for the
subsequent migration of missing operators, policy algorithms, native Tensor
conversion equivalents, and remaining v2/data-model gaps.


## Multi-view evaluation

`five_crop` and `ten_crop` match torchvision's view order and half-to-even
center positioning for supported uint8/float32 HWC/CHW images. TenCrop defaults
to horizontal flipping and also supports `vertical_flip=True`. Rivet flattens
views into the image batch and repeats labels; it does not return Python tuples
of images. `batch(B)` counts source samples and `drop_last` preserves whole
view groups. See the [multi-view guide](torchvision-coverage.md#multi-view-pipelines)
for grouping metadata, random streams and current composition restrictions.


## v2 image contracts

The v2 augmentations use native image operations through both Python entry
points. RandomPhotometricDistort follows brightness → optional early contrast
→ saturation → hue → optional late contrast → optional channel permutation.
Each enabled adjustment samples an independent factor from its configured
interval, and contrast placement is an independent 50/50 draw. Its numerical
reference is the existing PIL/RGB uint8 color implementation.

RandomResize samples an integer short edge with an **exclusive** maximum, as
in torchvision's implementation (the upstream prose describes this interval
less precisely). RandomShortestSize samples a candidate uniformly and applies
the long-edge cap to a shared floating scale before truncating both dimensions.
ScaleJitter also truncates dimensions after sampling a shared scale. Resize
filters retain Rivet's PIL reference numerics. RandomZoomOut samples a shared
canvas ratio, truncates canvas dimensions, then samples horizontal and vertical
placements with `int(extra_space * uniform[0,1))`; it preserves the original
pixels and uses scalar or channel fill in the input dtype's units.

GaussianNoise follows the Tensor reference instead of PIL. It accepts uint8 or
float32 and uses normalized mean/sigma for both: uint8 multiplies them by 255,
truncates sampled noise to int16 before addition, and clips or wraps the final
uint8 result. Float32 adds noise directly and optionally clips to `[0,1]`.
The normal samples use Rivet's random stream; equal parameters do not imply
elementwise equality with PyTorch noise. Reproducibility, fixed-noise limits
and distribution properties are tested separately.

`rgb()` converts decoded channel modes independently of decoding. One or two
channels mean gray or gray-alpha and replicate the luminance; four mean RGBA
and discard alpha. Three channels are already RGB. This channel contract does
not infer palette indices, CMYK or color profiles from an untyped tensor.

`to_dtype("float32", scale=False)` casts uint8 to float32 without normalization.
`to_dtype("uint8", scale=False)` truncates floats and wraps integer overflow.
With `scale=True`, the existing image conversion applies (division by 255, or
multiplication by `256 - 1e-3` before uint8 conversion). The default is False,
matching v2 ToDtype; existing `convert_image_dtype` still scales by default.
Only uint8 and float32 are supported, and no structured annotation dtype map
is implied. Both conversions can execute after the batch barrier.


The JPEG augmentation uses bundled MozJPEG in its `set_fastest_defaults()`
libjpeg-v6 compatibility profile, rather than the `image` crate's JPEG encoder.
RGB encoding explicitly selects 4:2:0 chroma subsampling; grayscale keeps one
component. It uses baseline sequential coding with ordinary quality tables,
without optimized entropy coding or progressive scans. The same libjpeg codec
then decodes the result. Quality is sampled inclusively and uniformly from
integer bounds in `[1,100]`. Fixed qualities are compared with Pillow's
libjpeg JPEG round trip; these tests isolate augmentation from source decoding.
Codec versions/builds can still differ slightly, so pixel comparisons allow
one uint8 level. No Python encoder or decoder runs in the data pipeline.

Spatial resize variants and photometric distortion retain the existing uint8
HWC kernel contract. Permutation, RGB conversion, GaussianNoise and zoom-out
support uint8/float32 in HWC or CHW; JPEG supports uint8 HWC/CHW. Arbitrary
Tensor batch prefixes and PIL objects remain outside the native sample model.
A sampled scale that produces a zero or unrepresentable output dimension is
rejected rather than silently clamped. Random-size transforms need a fixed-size
crop or resize before stacking a batch.


## P3 batch, annotation and temporal references

MixUp and CutMix follow torchvision v2's previous-row pairing and shared Beta
coefficient. Images use float32 MixUp or uint8/float32 CutMix; labels become
float32 probability matrices and CutMix uses actual clipped area. Random streams
are Rivet-specific and deterministic for seed, epoch and output batch ordinal.
The terminal batch stage is not reordered into per-sample operations.

Structured geometry uses the existing PIL image kernels and shared resolved
parameters for targets. Bounding-box/keypoint differential tests use float32
coordinate tolerances; mask tests use exact categorical values. Affine and
perspective masks mirror float32 normalized Tensor-grid nearest selection,
while categorical int64 values are kept in integer storage. Expanded rotations
use the PIL image canvas, which can differ from Tensor annotation offsets.
Unsupported structured transforms fail explicitly. SSD crop bounds its outer
retries and returns identity on exhaustion, unlike torchvision's unbounded loop.

Temporal selection follows CPU float32 linspace including mirrored endpoint
rounding, rather than exact rational indices. Oversampling repeats frame indices
and timestamps together. All input/export and modality limits are documented
in the [structured and video guide](structured-vision.md).
