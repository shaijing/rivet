# Batch mixing, annotations and video sequences

P3 extends the image API with batch mixing and explicit native structured
samples. Existing classification source schemas and the ordinary single-view
execution path retain their behavior. All augmentation, geometry and temporal
selection execute in the native runtime; Python configures operations and
exports tensors. The supported entry points are available from `rivet` and
`rivet.vision`.

## MixUp and CutMix

```python
from rivet import vision as rv

loader = (
    rv.scan_image_folder("data/train")
    .decode_image()
    .random_crop(224, 224, pad_if_needed=True)
    .to_tensor()
    .seed(42)
    .workers(4)
    .batch(128)
    .mixup(num_classes=1000, alpha=0.8)
    .execute()
)
batch = next(loader)
# images: float32 [B,3,224,224]; labels: float32 [B,1000]
```

Use `.cutmix(num_classes=1000, alpha=1.0)` for CutMix. These are terminal batch
output operations: place them after an explicit `.batch()` and all image and
index operations. Multiple mixing operations may be chained, in declaration
order. The logical explanation lists the terminal output operations separately
from the image graph. They execute after the executor returns a complete batch,
with the GIL released, and never move across image transforms or batching.

One `Beta(alpha, alpha)` sample is shared by a batch. `alpha` must be finite
and positive; the sampler works in log space to handle extreme shape values.
Pairing is a roll by one along the first dimension: item `i` uses item `i-1`,
and item zero uses the last item. Shuffle the source when appropriate. MixUp
requires float32 images; it blends both pixels and labels without clipping.
CutMix accepts uint8 or float32, samples a rectangle around a random pixel,
clips the rectangle to the canvas, copies pixels from the paired image, and
adjusts the label coefficient using the **actual clipped area**.

Index labels become one-hot float32 labels. Class IDs must lie in
`[0,num_classes)`. Subsequent mixing accepts those soft labels. For direct
NumPy batches, the callable classes also accept preexisting float32 `[B,C]`
labels, with `num_classes` optional. Soft labels must be finite; their rows
are blended as supplied without automatic normalization:

```python
mixed = rv.MixUp(alpha=0.8, num_classes=10)(
    images, labels, layout="NCHW", seed=42, epoch=3, batch_index=0
)
cut = rv.CutMix()(mixed["images"], mixed["labels"], layout="NCHW")
```

Both classes return batch dictionaries. Images must be rank 4, NHWC or NCHW.
They do not mix annotations or video batches. Soft-label outputs additionally
carry `labels_shape` and `labels_dtype`; ordinary integer-label batch schemas
stay unchanged. NumPy and DLPack/Torch exports retain the float32 label shape
and storage ownership. `as_numpy=False` returns image bytes and nested label
lists, retaining the shape/dtype metadata. Callable classes accept
`dlpack=True` for public DLPack producers.

Randomness derives from seed, epoch, output batch ordinal and operator
position, independently of worker completion order. Batch size and source
ordering define batch membership and can change mixing results. A fresh loader
starts at batch ordinal zero. Tail batches use their actual size; `drop_last`
removes them before mixing. A singleton pairs with itself. Multi-view batches
mix their flattened rows, including adjacent views of the same source; grouping
metadata stays intact. These choices match the torchvision batch pairing and
sampling policy, without reproducing PyTorch's RNG sequence.

## Structured samples and batching

```python
sample = rv.AnnotatedSample(
    image,                     # uint8/float32 [H,W,C], or CHW with layout="CHW"
    boxes=[[10, 20, 80, 90]],  # float32 [N,4], XYXY edge coordinates
    labels=[3],                # int64 [N], per-instance classes
    masks=instance_masks,      # uint8/int64 [N,H,W]
    keypoints=[[[15, 25], [70, 80]]],  # float32 [N,K,2]
)

loader = (
    rv.StructuredPipeline([sample])
    .compose(rv.Transform().random_horizontal_flip())
    .random_iou_crop()
    .sanitize_bounding_boxes()
    .compose(rv.Transform().resize(224, 224).to_tensor())
    .seed(42)
    .workers(4)
    .batch(8)
    .execute()
)
for batch in loader:
    images = batch["images"]      # stacked NumPy batch
    targets = batch["targets"]    # one dictionary per image, ragged N
```

Fields are optional. Every instance field shares its leading count `N`,
including `N=0`. A two-dimensional `masks=[H,W]` array instead represents
semantic segmentation and is independent of the number of instances. Semantic
maps follow geometry but are not filtered when boxes are sanitized. Mixing
semantic and instance masks in the same field is not supported.

Construction copies NumPy inputs once, including strided/negative-stride
views, into owned native tensors. Boxes and keypoints are float32; masks keep
uint8 or int64 class IDs without passing through floating storage. Nonfinite
coordinates, mismatched instance counts and mismatched canvas sizes fail
explicitly. Input box format is `xyxy`, `xywh` or `cxcywh`. Rotated box formats
and keypoint visibility triplets are not supported.

`sample.apply(transform, seed=..., epoch=..., sample_index=...)` applies a
reusable `Transform` directly and returns another `AnnotatedSample`.
`sample.to_dict()` exports read-only arrays plus layout/canvas/format metadata.
A `StructuredPipeline` is a lazy pipeline over decoded in-memory samples;
Arrow/Lance annotation-column ingestion is not added by this change.

Batches stack only images; variable instance fields remain a list of target
dictionaries. Targets include `canvas_size` (height, width), `box_format`, and
per-field `shapes` and `dtypes`. `next_dlpack()` exports images and each target
field as public DLPack producers. `next_torch()` consumes those producers.
For `as_numpy=False`, images and target fields are bytes with their metadata.
Exports retain their owners after loaders and samples are destroyed.

`shuffle()`, `seed()`, `epoch()`, `workers()`, `batch()` and `drop_last` define
source ordering and batching. Random decisions depend on original source
identity and operator namespaces, so worker count and batch size do not change
per-source results. The structured runtime processes a bounded batch at a
time, without prefetching; workers parallelize that batch. It is separate from
the optimized classification DAG executor. As with ordinary pipelines, make
spatial sizes uniform before stacking. Exhaustion and errors are sticky.

## Shared geometry and target rules

The structured adapter resolves each random geometric operation **once**,
then applies the same resolved parameters to image, boxes, masks and keypoints.
It supports resize/short-edge resize, crop/center crop, pad, horizontal/vertical
flip, random crop/resized crop, affine/random affine, rotation/random rotation,
perspective/random perspective, random composition controls, and the v2 random
resize/shortest-size/scale-jitter/zoom-out operations.

Image kernels retain their existing dtype/layout/interpolation restrictions.
For example, resize and color kernels use the existing RGB uint8 path; floating
images can be flipped, cropped, padded, converted and normalized. CHW uint8
resize temporarily changes axis metadata before calling the HWC resize kernel.
Color-only transforms affect only the image. Decode, FiveCrop/TenCrop,
ElasticTransform and learned augmentation policies are rejected for structured
samples because their annotation/multi-output semantics are not defined here.
Unsupported operations are rejected even when nested in a random branch.

Boxes use continuous edge coordinates (`0..W`, `0..H`); a transformed box is
the axis-aligned envelope of its four transformed corners. Crops and warps
clamp boxes to the output canvas. Keypoint flips use `W-1-x` / `H-1-y`, resize
multiplies coordinates by spatial ratios, and crop/pad subtract/add offsets.
Keypoints may remain outside the canvas, matching torchvision's coordinate
behavior; no visibility flags are invented. Expanded rotations use the actual
PIL-style image canvas and its inverse map for all annotations. Exact expanded
coordinate offsets can differ from torchvision Tensor's canvas convention.

Masks always use nearest interpolation. Resize uses the Tensor nearest
index rule; affine/perspective use float32 normalized grids and ties-to-even
nearest pixel selection. Masks retain categorical int64 values, including IDs
larger than float32 can represent. Image interpolation remains configurable;
PIL image resampling and Tensor mask resampling can choose different pixels at
boundaries. Padding uses the requested mode for both image and masks; mask
constant fill is zero regardless of image fill. Coordinates use float32 output
with small floating tolerances; degenerate projective mappings fail explicitly.

## Box maintenance and SSD crop

```python
pipeline = (
    rv.StructuredPipeline(samples)
    .convert_bounding_box_format("xyxy")
    .clamp_bounding_boxes()
    .random_iou_crop(
        min_scale=0.3, max_scale=1.0,
        sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        trials=40, max_attempts=100,
    )
    .sanitize_bounding_boxes(min_size=1.0, min_area=1.0)
)
```

Conversion preserves instance order. Clamp clips box edges; sanitize keeps
boxes with both sides at least `min_size`, area at least `min_area`, and edges
inside the canvas. Both thresholds must be finite and at least one. It removes
the corresponding rows of labels, instance masks and keypoints in lockstep.
It does not silently clamp before checking validity.

RandomIoUCrop follows SSD's sampling policy: choose a minimum-IoU option,
sample crop width/height independently, check aspect ratio, require a box
center strictly inside the crop, and require the maximum IoU among centered
boxes to reach the threshold. Values >=1 select identity. Boxes whose centers
fall outside an accepted crop are marked zero; call sanitize to remove them.
Empty boxes retain the source sample. `max_attempts` bounds option retries and
returns the source unchanged on exhaustion; this intentional difference from
torchvision's unbounded outer loop makes impossible configurations terminate.

## Keypoint maintenance

Call `StructuredPipeline.clamp_keypoints()` after geometry to clamp coordinates
to pixel centers in `[0, W-1]` and `[0, H-1]`. To remove out-of-canvas points
instead, call `sanitize_keypoints()`. It retains an instance only if all its
keypoints are in `[0, W)` and `[0, H)`. Boxes, labels and instance masks follow
the retained rows. Semantic masks keep their original contents.

## Decoded video and temporal subsampling

```python
video = rv.VideoSample(
    frames, layout="THWC", label=7,
    timestamps=[0.00, 0.04, 0.08, 0.12, 0.16],
)
result = rv.UniformTemporalSubsample(8)(video).to_dict()
# frames and timestamps repeat together; label and layout are preserved

# Tensor-style batched prefixes, default [...,T,C,H,W]:
subsampled = rv.uniform_temporal_subsample(videos, 8)
# For [...,T,H,W,C], pass the appropriate temporal_dim explicitly.
```

VideoSample requires nonempty rank-4 uint8/float32 THWC or TCHW frames. Time is
axis zero. Optional timestamps are finite nondecreasing float64 `[T]` values;
they use the same sampled indices as frames. The label describes the entire
sequence. Repeated output timestamps are valid after oversampling.

Temporal sampling follows CPU torchvision's default float32 `torch.linspace`
index generation, then integer truncation: one requested frame selects frame
zero, multiple frames include both endpoints, and oversampling repeats frames.
The mirrored endpoint arithmetic and fused float32 rounding matter at integer
boundaries, so exact rational interpolation is not substituted. This follows
[PyTorch's range kernel](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/RangeFactoriesKernel.cpp).
Indices are clamped to the input extent for very large float32 index ranges.
The functional interface accepts arbitrary prefixes and an explicit positive
or negative temporal axis. It can return NumPy arrays or DLPack producers;
VideoSample exports the same way through `to_dict(dlpack=True)`. Selection is
deterministic and does not consume augmentation RNG state. No video file
codec, streaming decoder or frame-wise annotation synchronization is implied.
