"""P3 data-model contracts, with torchvision differential checks."""

from __future__ import annotations

import gc

import numpy as np
import pytest
from PIL import Image

import rivet


@pytest.fixture
def batch_images():
    return np.random.default_rng(12).random((4, 8, 10, 3), dtype=np.float32)


@pytest.mark.parametrize("chw", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_mixup_torch_differential(batch_images, chw, strided):
    torch = pytest.importorskip("torch")
    images = batch_images.transpose(0, 3, 1, 2) if chw else batch_images
    if strided:
        images = images[..., ::-1]
    labels = np.arange(4)
    batch = rivet.MixUp(alpha=0.7, num_classes=4)(
        images, labels, layout="NCHW" if chw else "NHWC", seed=42
    )
    lam = batch["labels"][0, 0]
    assert 0 < lam < 1
    x = torch.from_numpy(images.copy())
    expected = x.roll(1, 0).mul(1 - float(lam)).add(x.mul(float(lam)))
    np.testing.assert_allclose(batch["images"], expected.numpy(), atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(batch["labels"].sum(1), 1)
    assert batch["labels_shape"] == [4, 4]
    assert batch["labels_dtype"] == "float32"
    assert not batch["images"].flags.writeable


@pytest.mark.parametrize("chw", [False, True])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_cutmix_area_labels_and_pairing(chw, dtype):
    x = np.stack([np.full((16, 20, 3), i * 40, dtype=dtype) for i in range(4)])
    images = x.transpose(0, 3, 1, 2) if chw else x
    batch = rivet.CutMix(num_classes=4)(
        images, np.arange(4), layout="NCHW" if chw else "NHWC", seed=9
    )
    out = batch["images"].transpose(0, 2, 3, 1) if chw else batch["images"]
    changed = np.any(out[0] != x[0], axis=-1)
    assert changed.any()
    ys, xs = np.where(changed)
    assert changed.sum() == (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    lam = 1 - changed.sum() / (16 * 20)
    for i in range(4):
        prev = (i - 1) % 4
        np.testing.assert_array_equal(out[i][changed], x[prev][changed])
        np.testing.assert_array_equal(out[i][~changed], x[i][~changed])
        expected = np.eye(4, dtype=np.float32)[i] * lam + np.eye(4, dtype=np.float32)[
            prev
        ] * (1 - lam)
        np.testing.assert_allclose(batch["labels"][i], expected, atol=1e-7)


@pytest.mark.parametrize("cls", [rivet.MixUp, rivet.CutMix])
def test_soft_labels_singleton_and_chain(batch_images, cls):
    labels = np.eye(4, dtype=np.float32)
    a = cls()(batch_images, labels, seed=3)
    b = cls()(batch_images, labels, seed=3)
    np.testing.assert_array_equal(a["images"], b["images"])
    for name in ("images", "labels"):
        singleton = cls()(batch_images[:1], labels[:1], seed=17)[name]
        np.testing.assert_allclose(
            singleton, batch_images[:1] if name == "images" else labels[:1], atol=1e-7
        )
    result = rivet.MixUp()(a["images"], a["labels"], seed=4)
    np.testing.assert_allclose(result["labels"].sum(1), 1, atol=1e-7)
    assert not np.array_equal(
        a["labels"], cls()(batch_images, labels, seed=3, epoch=1)["labels"]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0},
        {"alpha": -1},
        {"alpha": float("nan")},
        {"num_classes": 0},
        {"num_classes": True},
    ],
)
def test_mix_invalid_config(kwargs):
    with pytest.raises((ValueError, TypeError, OverflowError)):
        rivet.MixUp(**kwargs)


@pytest.mark.parametrize(
    "labels,kwargs",
    [
        (np.arange(4), {}),
        (np.arange(4), {"num_classes": 3}),
        (np.arange(3), {"num_classes": 4}),
        (np.array([-1, 0, 1, 2]), {"num_classes": 4}),
        (np.eye(4, dtype=np.float32), {"num_classes": 3}),
        (np.ones((4, 0), dtype=np.float32), {}),
        (np.ones((4, 4, 1)), {}),
    ],
)
def test_mix_invalid_labels(batch_images, labels, kwargs):
    with pytest.raises((ValueError, RuntimeError)):
        rivet.MixUp(**kwargs)(batch_images, labels)


def test_mixup_rejects_uint8(batch_images):
    with pytest.raises(ValueError, match="float32"):
        rivet.MixUp(num_classes=4)((batch_images * 255).astype(np.uint8), np.arange(4))


@pytest.fixture
def folder(tmp_path):
    pixels = np.random.default_rng(71).integers(0, 256, (10, 12, 3), dtype=np.uint8)
    for c in range(3):
        root = tmp_path / str(c)
        root.mkdir()
        for i in range(3):
            Image.fromarray(np.roll(pixels, i + c, 0)).save(root / f"{i}.png")
    return tmp_path


@pytest.mark.parametrize("kind", ["mixup", "cutmix"])
@pytest.mark.parametrize("chw", [False, True])
def test_pipeline_batch_mix_workers_deliveries(folder, kind, chw):
    def pipeline(workers=0, epoch=0):
        p = (
            rivet.scan_image_folder(folder)
            .decode_image()
            .to_dtype("float32", scale=True)
            .seed(18)
            .epoch(epoch)
            .workers(workers)
        )
        if chw:
            p = p.hwc_to_chw()
        return getattr(p.batch(4), kind)(num_classes=3, alpha=0.8)

    a, b = list(pipeline().execute()), list(pipeline(2).execute())
    assert [v["images"].shape[0] for v in a] == [4, 4, 1]
    for x, y in zip(a, b, strict=True):
        for name in ("images", "labels"):
            np.testing.assert_array_equal(x[name], y[name])
    assert "Terminal batch output" in pipeline().explain()
    raw = next(pipeline().execute(as_numpy=False))
    np.testing.assert_array_equal(
        np.frombuffer(raw["images"], dtype=np.float32).reshape(raw["shape"]),
        a[0]["images"],
    )
    np.testing.assert_array_equal(raw["labels"], a[0]["labels"])
    torch = pytest.importorskip("torch")
    native = pipeline().execute().next_torch()
    np.testing.assert_array_equal(native["labels"].numpy(), a[0]["labels"])
    dl = pipeline().execute().next_dlpack()
    np.testing.assert_array_equal(
        torch.from_dlpack(dl["images"]).numpy(), a[0]["images"]
    )
    later = next(pipeline(epoch=1).execute())
    assert not np.array_equal(later["labels"], a[0]["labels"])


@pytest.mark.parametrize(
    "build",
    [
        lambda p: p.mixup(3).batch(4),
        lambda p: p.batch(4).mixup(3).to_dtype("float32"),
        lambda p: p.batch(4).mixup(3).take(2),
        lambda p: p.batch(4).cutmix(3).batch(2),
    ],
)
def test_batch_mixing_terminal_order(folder, build):
    with pytest.raises(ValueError, match="MixUp/CutMix"):
        build(rivet.scan_image_folder(folder).decode_image()).execute()


def test_batch_mix_drop_last_chain_and_sticky_error(folder):
    p = (
        rivet.scan_image_folder(folder)
        .decode_image()
        .to_dtype("float32", scale=True)
        .batch(4, drop_last=True)
        .mixup(3)
        .cutmix(3)
    )
    assert len(list(p.execute())) == 2
    bad = rivet.scan_image_folder(folder).decode_image().batch(4).mixup(3).execute()
    with pytest.raises(ValueError, match="float32"):
        next(bad)
    with pytest.raises(StopIteration):
        next(bad)


@pytest.fixture
def annotated():
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    masks = np.zeros((2, 10, 12), dtype=np.uint8)
    masks[0, 2:6, 1:5] = 1
    masks[1, 5:9, 7:11] = 2
    image[..., 0] = masks[0] * 80
    image[..., 1] = masks[1] * 60
    boxes = np.array([[1, 2, 5, 6], [7, 5, 11, 9]], np.float32)
    points = np.array([[[2, 3], [4, 5]], [[8, 6], [10, 8]]], np.float32)
    return rivet.AnnotatedSample(
        image, boxes=boxes, labels=[4, 9], masks=masks, keypoints=points
    )


def expected_torchvision(sample, transform):
    torch = pytest.importorskip("torch")
    from torchvision import tv_tensors as tv

    source = sample.to_dict()
    image = tv.Image(torch.from_numpy(source["image"].copy()).permute(2, 0, 1))
    boxes = tv.BoundingBoxes(
        torch.from_numpy(source["boxes"].copy()),
        format="XYXY",
        canvas_size=source["canvas_size"],
    )
    masks = tv.Mask(torch.from_numpy(source["masks"].copy()))
    points = tv.KeyPoints(
        torch.from_numpy(source["keypoints"].copy()), canvas_size=source["canvas_size"]
    )
    return transform(image, boxes, masks, points)


@pytest.mark.parametrize(
    "case",
    [
        "flip",
        "vflip",
        "resize",
        "crop",
        "pad",
        "center_crop",
        "affine",
        "rotation",
        "perspective",
    ],
)
def test_structured_geometry_torchvision(annotated, case):
    from torchvision.transforms import InterpolationMode as I
    from torchvision.transforms.v2 import functional as F

    native = rivet.Transform()
    if case == "flip":
        native = native.horizontal_flip()
        ref = lambda *args: tuple(F.horizontal_flip(v) for v in args)
    elif case == "vflip":
        native = native.vertical_flip()
        ref = lambda *args: tuple(F.vertical_flip(v) for v in args)
    elif case == "resize":
        native = native.resize(6, 5, "nearest")
        ref = lambda *args: tuple(
            F.resize(v, [5, 6], interpolation=I.NEAREST) for v in args
        )
    elif case == "crop":
        native = native.crop(2, 3, 8, 6)
        ref = lambda *args: tuple(F.crop(v, 3, 2, 6, 8) for v in args)
    elif case == "pad":
        native = native.pad_with_sides(2, 3, 4, 1, fill=17)
        ref = lambda *args: tuple(
            F.pad(v, [2, 3, 4, 1], fill=17 if i == 0 else 0) for i, v in enumerate(args)
        )
    elif case == "center_crop":
        native = native.center_crop(16, 14)
        ref = lambda *args: tuple(F.center_crop(v, [14, 16]) for v in args)
    elif case == "affine":
        native = native.affine(0, translate=[2, -1])
        ref = lambda *args: tuple(
            F.affine(v, 0, [2, -1], 1, [0, 0], interpolation=I.NEAREST) for v in args
        )
    elif case == "rotation":
        native = native.arbitrary_rotate(180)
        ref = lambda *args: tuple(
            F.rotate(v, 180, interpolation=I.NEAREST) for v in args
        )
    else:
        corners = [[0, 0], [11, 0], [11, 9], [0, 9]]
        end = [[1, 0], [10, 0], [10, 9], [1, 9]]
        native = native.perspective(corners, end, interpolation="nearest")
        ref = lambda *args: tuple(
            F.perspective(v, corners, end, interpolation=I.NEAREST) for v in args
        )
    expected = expected_torchvision(annotated, ref)
    out = annotated.apply(native).to_dict()
    # Native images use the existing PIL path; masks and coordinates use tensor semantics.
    if case != "resize":
        np.testing.assert_array_equal(
            out["image"].transpose(2, 0, 1), expected[0].numpy()
        )
    np.testing.assert_allclose(out["boxes"], expected[1].numpy(), atol=1e-4)
    np.testing.assert_array_equal(out["masks"], expected[2].numpy())
    np.testing.assert_allclose(out["keypoints"], expected[3].numpy(), atol=1e-4)


@pytest.mark.parametrize(
    "transform",
    [
        rivet.Transform().random_horizontal_flip(),
        rivet.Transform().random_crop(8, 8, padding=2),
        rivet.Transform().random_resized_crop(6, 6),
        rivet.Transform().random_affine(20, translate=[0.1, 0.2]),
        rivet.Transform().random_rotation([-30, 30]),
        rivet.Transform().random_perspective(0.4, 1),
        rivet.Transform().random_resize(6, 10).center_crop(6, 6),
        rivet.Transform().random_shortest_size([6, 8]).center_crop(6, 6),
        rivet.Transform().scale_jitter(10, 8, [0.8, 1.2]).center_crop(6, 6),
        rivet.Transform().random_zoom_out(probability=1).center_crop(24, 24),
        rivet.Transform().pixel_dropout(0.3, mask_drop_value=0),
        rivet.Transform().xy_masking((1, 2), (1, 2), (2, 3), (1, 2), fill_mask=0),
        rivet.Transform().grid_mask(fill_mask=0),
        rivet.Transform().random_apply(1, rivet.Transform().horizontal_flip()),
    ],
)
def test_structured_random_worker_and_batch_reproducibility(annotated, transform):
    source = rivet.StructuredPipeline([annotated] * 7).compose(transform).seed(17)
    a = list(source.batch(2).execute())
    b = list(source.workers(3).batch(3).execute())

    def flatten(values):
        return [
            (img, target)
            for batch in values
            for img, target in zip(batch["images"], batch["targets"], strict=True)
        ]

    for (x, t), (y, u) in zip(flatten(a), flatten(b), strict=True):
        np.testing.assert_array_equal(x, y)
        for name in t["shapes"]:
            np.testing.assert_array_equal(t[name], u[name])


@pytest.mark.parametrize("format", ["xyxy", "xywh", "cxcywh"])
def test_box_format_roundtrip_and_clamp(annotated, format):
    import torch
    from torchvision import tv_tensors as tv
    from torchvision.transforms.v2 import functional as F

    pipeline = rivet.StructuredPipeline([annotated]).convert_bounding_box_format(format)
    actual = next(pipeline.execute())["targets"][0]
    expected = F.convert_bounding_box_format(
        tv.BoundingBoxes(
            torch.tensor(annotated.to_dict()["boxes"]),
            format="XYXY",
            canvas_size=(10, 12),
        ),
        new_format=format.upper(),
    )
    np.testing.assert_array_equal(actual["boxes"], expected.numpy())
    restored = next(pipeline.convert_bounding_box_format("xyxy").execute())["targets"][
        0
    ]
    np.testing.assert_array_equal(restored["boxes"], annotated.to_dict()["boxes"])
    invalid = rivet.AnnotatedSample(
        np.zeros((5, 6, 3), np.uint8), boxes=[[-2, -3, 9, 8]]
    )
    clipped = next(
        rivet.StructuredPipeline([invalid]).clamp_bounding_boxes().execute()
    )["targets"][0]
    np.testing.assert_array_equal(clipped["boxes"], [[0, 0, 6, 5]])


def test_sanitize_filters_all_instance_fields(annotated):
    p = (
        rivet.StructuredPipeline([annotated])
        .compose(rivet.Transform().crop(0, 0, 6, 6))
        .sanitize_bounding_boxes()
    )
    out = next(p.execute())["targets"][0]
    assert out["boxes"].shape == (1, 4)
    np.testing.assert_array_equal(out["labels"], [4])
    assert out["masks"].shape == (1, 6, 6)
    assert out["keypoints"].shape == (1, 2, 2)
    empty = next(p.sanitize_bounding_boxes(min_area=999).execute())["targets"][0]
    assert empty["boxes"].shape == (0, 4)
    assert empty["masks"].shape == (0, 6, 6)
    assert empty["labels"].shape == (0,)


@pytest.mark.parametrize("semantic_mask", [False, True])
def test_keypoint_maintenance_matches_torchvision(semantic_mask):
    torch = pytest.importorskip("torch")
    from torchvision.transforms.v2 import functional as F

    image = np.zeros((3, 5, 3), dtype=np.uint8)
    points = np.array(
        [[[0, 0], [4.5, 2.5]], [[1, 1], [5, 2]], [[-0.5, 2], [2, 3]]],
        dtype=np.float32,
    )
    boxes = np.array([[0, 0, 5, 3], [1, 1, 4, 3], [0, 1, 3, 3]], dtype=np.float32)
    labels = np.array([10, 20, 30], dtype=np.int64)
    masks = np.arange(45, dtype=np.uint8).reshape(3, 3, 5)
    if semantic_mask:
        masks = masks[0]
    sample = rivet.AnnotatedSample(
        image, keypoints=points, boxes=boxes, labels=labels, masks=masks
    )
    pipeline = rivet.StructuredPipeline([sample])
    reference = torch.from_numpy(points)
    expected_clamped = F.clamp_keypoints(reference, canvas_size=(3, 5)).numpy()
    clamped = next(pipeline.clamp_keypoints().execute())["targets"][0]
    np.testing.assert_array_equal(clamped["keypoints"], expected_clamped)
    np.testing.assert_array_equal(clamped["labels"], labels)

    expected_points, keep = F.sanitize_keypoints(reference, canvas_size=(3, 5))
    keep = keep.numpy()
    sanitized = next(pipeline.sanitize_keypoints().execute())["targets"][0]
    np.testing.assert_array_equal(sanitized["keypoints"], expected_points.numpy())
    np.testing.assert_array_equal(sanitized["labels"], labels[keep])
    np.testing.assert_array_equal(sanitized["boxes"], boxes[keep])
    np.testing.assert_array_equal(
        sanitized["masks"], masks if semantic_mask else masks[keep]
    )


@pytest.mark.parametrize("options", [[1.0], [0.0], [0.5], [0.99]])
def test_iou_crop_bounded_valid_and_sanitize(annotated, options):
    p = (
        rivet.StructuredPipeline([annotated] * 4)
        .random_iou_crop(
            min_scale=0.5,
            max_scale=0.5,
            sampler_options=options,
            trials=10,
            max_attempts=5,
        )
        .seed(8)
    )
    a, b = list(p.execute()), list(p.workers(2).execute())
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x["images"], y["images"])
        target = x["targets"][0]
        _h, w = target["canvas_size"]
        assert np.all(target["boxes"][:, [0, 2]] >= 0)
        assert np.all(target["boxes"][:, [0, 2]] <= w)
        if options == [1.0] or options == [0.99]:
            np.testing.assert_array_equal(x["images"][0], annotated.to_dict()["image"])
    cleaned = list(p.sanitize_bounding_boxes().execute())
    for item in cleaned:
        t = item["targets"][0]
        assert (
            len(t["labels"])
            == len(t["boxes"])
            == len(t["masks"])
            == len(t["keypoints"])
        )


def test_structured_exports_lifetime_and_tail(annotated):
    p = (
        rivet.StructuredPipeline([annotated] * 5)
        .compose(rivet.Transform().to_tensor())
        .batch(2)
        .workers(2)
    )
    a = list(p.execute())
    assert [v["images"].shape[0] for v in a] == [2, 2, 1]
    assert len(list(p.batch(2, drop_last=True).execute())) == 2
    raw = next(p.execute(as_numpy=False))
    np.testing.assert_array_equal(
        np.frombuffer(raw["images"], np.float32).reshape(raw["shape"]), a[0]["images"]
    )
    target = raw["targets"][0]
    np.testing.assert_array_equal(
        np.frombuffer(target["boxes"], np.float32).reshape(target["shapes"]["boxes"]),
        a[0]["targets"][0]["boxes"],
    )
    torch = pytest.importorskip("torch")
    handles = p.execute().next_dlpack()
    del p
    gc.collect()
    np.testing.assert_array_equal(
        torch.from_dlpack(handles["images"]).numpy(), a[0]["images"]
    )
    np.testing.assert_array_equal(
        torch.from_dlpack(handles["targets"][0]["masks"]).numpy(),
        a[0]["targets"][0]["masks"],
    )
    with pytest.raises(ValueError):
        a[0]["targets"][0]["boxes"][0, 0] = 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"boxes": [[1, 2, 3, float("nan")]]},
        {"boxes": [[0, 0, 2, 2]], "labels": [1, 2]},
        {"boxes": [[0, 0, 2, 2]], "masks": np.zeros((1, 2, 2), np.uint8)},
        {"boxes": [[0, 0, 2, 2]], "keypoints": np.zeros((1, 3, 3))},
        {"labels": [1.5]},
        {"box_format": "invalid"},
    ],
)
def test_structured_validation(kwargs):
    with pytest.raises((ValueError, TypeError)):
        rivet.AnnotatedSample(np.zeros((10, 12, 3), np.uint8), **kwargs)


@pytest.mark.parametrize(
    "transform",
    [
        rivet.Transform().five_crop(2, 2),
        rivet.Transform().auto_augment(),
        rivet.Transform().decode_image(),
    ],
)
def test_structured_unsupported_rejected(annotated, transform):
    with pytest.raises(ValueError, match="unsupported"):
        rivet.StructuredPipeline([annotated]).compose(transform).execute()


@pytest.mark.parametrize("count", [1, 2, 3, 5, 8, 11])
@pytest.mark.parametrize("layout", ["THWC", "TCHW"])
def test_video_sample_torchvision_indices_and_timestamps(count, layout):
    torch = pytest.importorskip("torch")
    from torchvision.transforms.v2 import functional as F

    frames = np.arange(5 * 3 * 4 * 3, dtype=np.uint8).reshape(5, 3, 4, 3)
    if layout == "TCHW":
        frames = frames.transpose(0, 3, 1, 2)
    video = rivet.VideoSample(
        frames, layout=layout, timestamps=np.arange(5) * 0.2, label=7
    )
    out = rivet.UniformTemporalSubsample(count)(video).to_dict()
    source = torch.from_numpy(frames.copy())
    if layout == "THWC":
        source = source.permute(0, 3, 1, 2)
    expected = F.uniform_temporal_subsample(source, count)
    if layout == "THWC":
        expected = expected.permute(0, 2, 3, 1)
    np.testing.assert_array_equal(out["frames"], expected.numpy())
    ids = torch.linspace(0, 4, count).long().numpy()
    np.testing.assert_array_equal(out["timestamps"], ids * 0.2)
    assert out["label"] == 7
    assert out["layout"] == layout


@pytest.mark.parametrize("axis", [0, 1, -4])
def test_temporal_function_batched_strided_and_dlpack(axis):
    torch = pytest.importorskip("torch")
    x = np.arange(2 * 5 * 3 * 4 * 6, dtype=np.float32).reshape(2, 5, 3, 4, 6)[..., ::-1]
    normalized = axis if axis >= 0 else x.ndim + axis
    out = rivet.uniform_temporal_subsample(x, 9, temporal_dim=axis)
    ids = torch.linspace(0, x.shape[normalized] - 1, 9).long()
    expected = torch.index_select(torch.from_numpy(x.copy()), normalized, ids)
    np.testing.assert_array_equal(out, expected.numpy())
    handle = rivet.uniform_temporal_subsample(x, 9, temporal_dim=axis, dlpack=True)
    np.testing.assert_array_equal(torch.from_dlpack(handle).numpy(), expected.numpy())


@pytest.mark.parametrize(
    "frames,kwargs",
    [
        (np.zeros((0, 2, 2, 3), np.uint8), {}),
        (np.zeros((4, 2, 2), np.uint8), {}),
        (np.zeros((4, 2, 2, 3), np.float64), {}),
        (np.zeros((4, 2, 2, 3), np.uint8), {"timestamps": [0, 1]}),
        (np.zeros((4, 2, 2, 3), np.uint8), {"timestamps": [0, 2, 1, 3]}),
        (np.zeros((4, 2, 2, 3), np.uint8), {"layout": "NCHW"}),
    ],
)
def test_video_validation(frames, kwargs):
    with pytest.raises(ValueError):
        rivet.VideoSample(frames, **kwargs)


def test_temporal_invalid_and_lifetime():
    x = np.arange(3 * 2 * 2 * 1, dtype=np.uint8).reshape(3, 2, 2, 1)
    for n in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            rivet.uniform_temporal_subsample(x, n)
    with pytest.raises(ValueError):
        rivet.uniform_temporal_subsample(x, 2, temporal_dim=4)
    out = (
        rivet.VideoSample(x, timestamps=[0, 1, 2])
        .uniform_temporal_subsample(5)
        .to_dict()
    )
    del x
    gc.collect()
    assert out["frames"].shape == (5, 2, 2, 1)
    assert not out["frames"].flags.writeable


def test_temporal_float32_linspace_boundary_matches_torchvision():
    import torch
    from torchvision.transforms.v2 import functional as F

    # Halfway comes from endpoint-minus-step and lands just below 1.0,
    # unlike exact integer-rational interpolation.
    x = np.arange(3, dtype=np.uint8).reshape(3, 1, 1, 1)
    actual = rivet.uniform_temporal_subsample(x, 15)
    expected = F.uniform_temporal_subsample(torch.from_numpy(x), 15)
    np.testing.assert_array_equal(actual, expected.numpy())


@pytest.mark.parametrize("alpha", [1e-300, 0.01, 1.0, 100.0, 1e300])
def test_mix_extreme_beta_coefficients_remain_finite(batch_images, alpha):
    batch = rivet.MixUp(alpha=alpha, num_classes=4)(batch_images, np.arange(4))
    assert np.isfinite(batch["labels"]).all()
    np.testing.assert_allclose(batch["labels"].sum(1), 1, atol=1e-7)


@pytest.mark.parametrize("shear", [0, 30, 45])
def test_structured_mask_nearest_ties_and_int64_classes(shear):
    import torch
    from torchvision import tv_tensors as tv
    from torchvision.transforms.v2 import functional as F

    # Large class IDs must never pass through float pixel storage.
    masks = (np.arange(80, dtype=np.int64) + 2**40).reshape(1, 8, 10)
    sample = rivet.AnnotatedSample(np.zeros((8, 10, 3), np.uint8), masks=masks)
    actual = sample.apply(rivet.Transform().affine(0, shear=[shear, 0])).to_dict()[
        "masks"
    ]
    expected = F.affine(tv.Mask(torch.from_numpy(masks)), 0, [0, 0], 1, [shear, 0])
    # torchvision currently casts int64 masks through float32 internally;
    # compare the source class indices instead to assert nearest positions.
    indices = np.arange(80, dtype=np.int64).reshape(1, 8, 10)
    expected_indices = F.affine(
        tv.Mask(torch.from_numpy(indices)), 0, [0, 0], 1, [shear, 0]
    ).numpy()
    occupancy = F.affine(
        tv.Mask(torch.ones_like(torch.from_numpy(indices))), 0, [0, 0], 1, [shear, 0]
    ).numpy()
    expected = np.where(occupancy, expected_indices + 2**40, 0)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("chw", [False, True])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_structured_flip_crop_pad_strides_and_dtype(chw, dtype):
    image = np.arange(6 * 8 * 3, dtype=dtype).reshape(6, 8, 3)[:, ::-1]
    if chw:
        image = image.transpose(2, 0, 1)
    mask = np.arange(6 * 8, dtype=np.int64).reshape(1, 6, 8)[:, :, ::-1]
    sample = rivet.AnnotatedSample(image, layout="CHW" if chw else "HWC", masks=mask)
    transform = (
        rivet.Transform()
        .horizontal_flip()
        .crop(1, 2, 5, 3)
        .pad_with_sides(1, 1, 2, 0, fill=7)
    )
    actual = sample.apply(transform).to_dict()
    hwc = image.transpose(1, 2, 0) if chw else image
    expected = np.pad(
        hwc[:, ::-1][2:5, 1:6], ((1, 0), (1, 2), (0, 0)), constant_values=7
    )
    np.testing.assert_array_equal(
        actual["image"], expected.transpose(2, 0, 1) if chw else expected
    )
    np.testing.assert_array_equal(
        actual["masks"], np.pad(mask[:, :, ::-1][:, 2:5, 1:6], ((0, 0), (1, 0), (1, 2)))
    )


def test_structured_empty_instances_and_empty_source():
    sample = rivet.AnnotatedSample(
        np.zeros((4, 5, 3), np.uint8),
        boxes=[],
        labels=[],
        masks=np.zeros((0, 4, 5), np.uint8),
        keypoints=np.zeros((0, 2, 2)),
    )
    result = next(
        rivet.StructuredPipeline([sample])
        .compose(rivet.Transform().horizontal_flip())
        .random_iou_crop()
        .sanitize_bounding_boxes()
        .execute()
    )["targets"][0]
    assert result["boxes"].shape == (0, 4)
    assert result["masks"].shape == (0, 4, 5)
    assert list(rivet.StructuredPipeline([]).batch(2).execute()) == []


def test_structured_box_rank_is_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="shape"):
        rivet.AnnotatedSample(np.zeros((4, 5, 3), np.uint8), boxes=np.zeros((2, 2, 2)))


def test_structured_next_torch_shuffle_epoch_and_sticky_error(annotated):
    import torch

    samples = [
        rivet.AnnotatedSample(
            np.full((4, 5, 3), i, np.uint8), boxes=[[0, 0, 2, 2]], labels=[i]
        )
        for i in range(8)
    ]
    p = rivet.StructuredPipeline(samples).shuffle().seed(9).batch(3)
    a, b = list(p.execute()), list(p.workers(2).execute())
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x["images"], y["images"])
    order = [int(t["labels"][0]) for batch in a for t in batch["targets"]]
    other = [
        int(t["labels"][0]) for batch in p.epoch(1).execute() for t in batch["targets"]
    ]
    assert sorted(order) == sorted(other) == list(range(8))
    assert order != other
    batch = p.execute().next_torch()
    assert isinstance(batch["images"], torch.Tensor)
    assert isinstance(batch["targets"][0]["boxes"], torch.Tensor)
    bad = (
        rivet.StructuredPipeline([annotated])
        .compose(rivet.Transform().random_crop(999, 999))
        .execute()
    )
    with pytest.raises(ValueError):
        next(bad)
    with pytest.raises(StopIteration):
        next(bad)


def test_semantic_segmentation_map_is_not_filtered_with_instances():
    image = np.zeros((6, 8, 3), np.uint8)
    mask = np.arange(48, dtype=np.int64).reshape(6, 8)
    sample = rivet.AnnotatedSample(
        image, boxes=[[1, 1, 4, 4], [7, 4, 8, 6]], labels=[2, 5], masks=mask
    )
    transform = rivet.Transform().horizontal_flip().crop(1, 1, 5, 4)
    batch = next(
        rivet.StructuredPipeline([sample])
        .compose(transform)
        .sanitize_bounding_boxes()
        .execute()
    )
    target = batch["targets"][0]
    assert len(target["boxes"]) == 1
    np.testing.assert_array_equal(target["masks"], mask[:, ::-1][1:5, 1:6])


@pytest.mark.parametrize("method", ["random_choice", "random_order"])
def test_structured_random_control_nested_streams_reproduce(annotated, method):
    choices = [
        rivet.Transform().random_crop(8, 8).brightness(0.8),
        rivet.Transform().random_horizontal_flip().center_crop(8, 8),
    ]
    transform = (
        rivet.Transform().random_choice(choices)
        if method == "random_choice"
        else rivet.Transform().random_order(
            rivet.Transform().random_horizontal_flip().brightness(0.8).random_crop(8, 8)
        )
    )
    p = rivet.StructuredPipeline([annotated] * 4).compose(transform).seed(3).batch(2)
    for x, y in zip(p.execute(), p.workers(2).execute(), strict=True):
        np.testing.assert_array_equal(x["images"], y["images"])
        for t, u in zip(x["targets"], y["targets"], strict=True):
            np.testing.assert_array_equal(t["boxes"], u["boxes"])


def test_iou_crop_greater_than_one_is_identity_sentinel(annotated):
    p = rivet.StructuredPipeline([annotated]).random_iou_crop(sampler_options=[2.0])
    actual = next(p.execute())
    np.testing.assert_array_equal(actual["images"][0], annotated.to_dict()["image"])


@pytest.mark.parametrize("erosion", [0.0, 1.0])
def test_at_least_one_bbox_crop_transforms_aligned_fields(erosion):
    y, x = np.indices((12, 12), dtype=np.uint8)
    image = np.stack((x, y, np.zeros_like(x)), axis=-1)
    boxes = np.array([[1, 1, 3, 3], [9, 9, 11, 11]], dtype=np.float32)
    points = np.array([[[2, 2]], [[10, 10]]], dtype=np.float32)
    masks = np.stack((np.ones_like(x), np.full_like(x, 2)))
    sample = rivet.AnnotatedSample(
        image, boxes=boxes, labels=[0, 1], masks=masks, keypoints=points
    )
    pipeline = (
        rivet.StructuredPipeline([sample] * 6)
        .at_least_one_bbox_random_crop(5, 5, erosion_factor=erosion)
        .seed(13)
        .batch(2)
    )
    outputs = list(pipeline.execute())
    threaded = list(pipeline.workers(2).execute())
    for batch, parallel in zip(outputs, threaded, strict=True):
        np.testing.assert_array_equal(batch["images"], parallel["images"])
        for cropped, target in zip(batch["images"], batch["targets"], strict=True):
            cx, cy = cropped[0, 0, :2].astype(int)
            assert cropped.shape == (5, 5, 3)
            np.testing.assert_array_equal(cropped, image[cy : cy + 5, cx : cx + 5])
            [label] = target["labels"]
            translated = boxes[[label]] - [cx, cy, cx, cy]
            expected_box = np.clip(translated, 0, 5)
            if erosion == 0:
                np.testing.assert_array_equal(expected_box, translated)
            np.testing.assert_array_equal(target["boxes"], expected_box)
            np.testing.assert_array_equal(
                target["masks"], masks[[label], cy : cy + 5, cx : cx + 5]
            )
            np.testing.assert_array_equal(
                target["keypoints"], points[[label]] - [cx, cy]
            )


def test_at_least_one_bbox_crop_falls_back_without_boxes():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    mask = image[..., 0].copy()
    source = rivet.StructuredPipeline([rivet.AnnotatedSample(image, masks=mask)])
    [batch] = source.at_least_one_bbox_random_crop(4, 6).execute()
    assert batch["images"].shape == (1, 6, 4, 3)
    np.testing.assert_array_equal(
        batch["images"][0, ..., 0], batch["targets"][0]["masks"]
    )
    with pytest.raises(ValueError):
        next(iter(source.at_least_one_bbox_random_crop(9, 4).execute()))
    with pytest.raises(ValueError):
        source.at_least_one_bbox_random_crop(4, 4, erosion_factor=1.1)


@pytest.mark.parametrize("layout", ["HWC", "CHW"])
@pytest.mark.parametrize("mask_dtype,mask_fill", [(np.uint8, 255), (np.int64, -1)])
def test_letterbox_maps_structured_targets(layout, mask_dtype, mask_fill):
    image = np.full((3, 5, 3), 20, dtype=np.uint8)
    mask = np.arange(15, dtype=mask_dtype).reshape(1, 3, 5)
    sample = rivet.AnnotatedSample(
        image if layout == "HWC" else image.transpose(2, 0, 1),
        boxes=[[1, 1, 4, 3]],
        keypoints=[[[2, 1]]],
        labels=[7],
        masks=mask,
        layout=layout,
    )
    result = sample.apply(
        rivet.Transform().letterbox(
            10, 9, interpolation="nearest", fill=31, mask_fill=mask_fill
        )
    ).to_dict()
    pixels = result["image"] if layout == "HWC" else result["image"].transpose(1, 2, 0)
    assert pixels.shape == (9, 10, 3)
    assert np.all(pixels[1:7] == 20)
    assert np.all(pixels[:1] == 31) and np.all(pixels[7:] == 31)
    assert result["masks"].dtype == mask_dtype
    np.testing.assert_array_equal(
        result["masks"][:, 1:7], mask.repeat(2, axis=1).repeat(2, axis=2)
    )
    assert np.all(result["masks"][:, :1] == mask_fill)
    assert np.all(result["masks"][:, 7:] == mask_fill)
    np.testing.assert_array_equal(result["boxes"], [[2, 3, 8, 7]])
    np.testing.assert_array_equal(result["keypoints"], [[[4, 3]]])
    np.testing.assert_array_equal(result["labels"], [7])


def test_pixel_dropout_synchronizes_masks_and_instance_fields():
    image = np.full((7, 9, 3), 120, dtype=np.uint8)
    mask = np.full((7, 9), 3, dtype=np.int64)
    sample = rivet.AnnotatedSample(
        image, boxes=[[1, 2, 5, 6]], labels=[9], masks=mask, keypoints=[[[2, 3]]]
    )
    transform = rivet.Transform().pixel_dropout(0.5, mask_drop_value=-1)
    result = sample.apply(transform).to_dict()
    dropped = result["image"][..., 0] == 0
    assert dropped.any() and not dropped.all()
    np.testing.assert_array_equal(result["masks"], np.where(dropped, -1, 3))
    np.testing.assert_array_equal(result["boxes"], [[1, 2, 5, 6]])
    np.testing.assert_array_equal(result["keypoints"], [[[2, 3]]])
    np.testing.assert_array_equal(result["labels"], [9])
    all_dropped = sample.apply(
        rivet.Transform().pixel_dropout(1, mask_drop_value=-1)
    ).to_dict()
    np.testing.assert_array_equal(all_dropped["masks"], -1)
    assert all_dropped["boxes"].shape == (0, 4)
    assert all_dropped["labels"].shape == (0,)
    assert all_dropped["keypoints"].shape == (0, 1, 2)


def test_pixel_dropout_per_channel_and_empty_instance_masks():
    image = np.full((7, 9, 3), 120, dtype=np.uint8)
    sample = rivet.AnnotatedSample(
        image, boxes=np.zeros((0, 4), dtype=np.float32), masks=np.zeros((0, 7, 9), dtype=np.uint8)
    )
    result = sample.apply(rivet.Transform().pixel_dropout(0.5, per_channel=True)).to_dict()
    dropped = result["image"] == 0
    assert np.any(dropped[..., 0] != dropped[..., 1])
    assert result["masks"].shape == (0, 7, 9)


def test_channel_dropout_preserves_structured_targets(annotated):
    original = annotated.to_dict()
    transformed = annotated.apply(rivet.Transform().channel_dropout([0], 1, 23)).to_dict()
    np.testing.assert_array_equal(transformed["image"][..., 0], 23)
    np.testing.assert_array_equal(transformed["image"][..., 1:], original["image"][..., 1:])
    for key in ["boxes", "labels", "keypoints", "masks"]:
        np.testing.assert_array_equal(transformed[key], original[key])


def test_xy_masking_updates_mask_and_removes_fully_occluded_instances(annotated):
    result = annotated.apply(
        rivet.Transform().xy_masking((1, 1), (0, 0), (1.0, 1.0), fill=23, fill_mask=0)
    ).to_dict()
    np.testing.assert_array_equal(result["image"], 23)
    assert result["boxes"].shape == (0, 4)
    assert result["labels"].shape == (0,)
    assert result["masks"].shape == (0, 10, 12)
    assert result["keypoints"].shape == (0, 2, 2)


def test_xy_masking_semantic_fill_matches_visible_strips():
    image = np.full((6, 8, 3), 120, dtype=np.uint8)
    mask = np.full((6, 8), 3, dtype=np.int64)
    sample = rivet.AnnotatedSample(image, masks=mask)
    result = sample.apply(
        rivet.Transform().xy_masking((1, 1), (1, 1), (2, 2), (1, 1), fill_mask=-1)
    ).to_dict()
    covered = result["image"][..., 0] == 0
    assert 0 < covered.sum() < covered.size
    np.testing.assert_array_equal(result["masks"], np.where(covered, -1, 3))


def test_bbox_safe_random_crop_preserves_all_visible_targets():
    h, w = 13, 17
    y, x = np.indices((h, w), dtype=np.uint8)
    image = np.stack((x, y, x + y), axis=-1)
    boxes = np.array([[3.2, 2.5, 6, 5.5], [9, 7, 13.5, 10]], dtype=np.float32)
    points = np.array([[[4, 4]], [[11, 8]]], dtype=np.float32)
    masks = np.stack(((x + y) % 3, (x * y) % 7)).astype(np.uint8)
    sample = rivet.AnnotatedSample(image, boxes=boxes, labels=[11, 22], masks=masks, keypoints=points)
    pipeline = rivet.StructuredPipeline([sample] * 5).bbox_safe_random_crop().seed(23)
    serial = list(pipeline.execute())
    parallel = list(pipeline.workers(3).execute())
    for left_batch, right_batch in zip(serial, parallel, strict=True):
        out = left_batch["images"][0]
        np.testing.assert_array_equal(out, right_batch["images"][0])
        left, top = map(int, out[0, 0, :2])
        oh, ow = out.shape[:2]
        assert left <= 3 and top <= 2 and left + ow >= 14 and top + oh >= 10
        target = left_batch["targets"][0]
        np.testing.assert_allclose(target["boxes"], boxes - [left, top, left, top], atol=1e-6)
        np.testing.assert_array_equal(target["keypoints"], points - [left, top])
        np.testing.assert_array_equal(target["labels"], [11, 22])
        np.testing.assert_array_equal(target["masks"], masks[:, top:top + oh, left:left + ow])


def test_random_sized_bbox_safe_crop_full_image_and_empty_fallback(annotated):
    image = annotated.to_dict()["image"]
    empty = rivet.AnnotatedSample(image, boxes=np.zeros((0, 4), np.float32))
    preserved = next(iter(rivet.StructuredPipeline([empty]).bbox_safe_random_crop().execute()))
    np.testing.assert_array_equal(preserved["images"][0], image)
    assert preserved["targets"][0]["boxes"].shape == (0, 4)
    full = rivet.AnnotatedSample(
        image, boxes=[[0, 0, 12, 10]], labels=[5],
        masks=np.ones((1, 10, 12), np.uint8), keypoints=[[[6, 4]]],
    )
    pipeline = rivet.StructuredPipeline([full] * 3).random_sized_bbox_safe_crop(6, 5)
    result = next(iter(pipeline.workers(3).batch(3).execute()))
    assert result["images"].shape == (3, 5, 6, 3)
    target = result["targets"][0]
    np.testing.assert_array_equal(target["boxes"], [[0, 0, 6, 5]])
    np.testing.assert_array_equal(target["labels"], [5])
    np.testing.assert_array_equal(target["masks"], np.ones((1, 5, 6), np.uint8))
    np.testing.assert_array_equal(target["keypoints"], [[[3, 2]]])
    with pytest.raises(ValueError):
        rivet.StructuredPipeline([full]).bbox_safe_random_crop(1.1)


def test_longest_max_size_scales_structured_fields(annotated):
    original = annotated.to_dict()
    result = annotated.apply(rivet.Transform().longest_max_size(6, interpolation="nearest")).to_dict()
    assert result["image"].shape == (5, 6, 3)
    np.testing.assert_array_equal(result["boxes"], original["boxes"] / 2)
    np.testing.assert_array_equal(result["labels"], original["labels"])
    np.testing.assert_array_equal(result["keypoints"], original["keypoints"] / 2)
    np.testing.assert_array_equal(result["masks"], original["masks"][:, ::2, ::2])


def test_random_scale_scales_image_and_structured_fields(annotated):
    original = annotated.to_dict()
    result = annotated.apply(
        rivet.Transform().random_scale({"x": (-0.5, -0.5), "y": (0, 0)}, "nearest")
    ).to_dict()
    assert result["image"].shape == (10, 6, 3)
    np.testing.assert_array_equal(result["boxes"], original["boxes"] * [0.5, 1, 0.5, 1])
    np.testing.assert_array_equal(result["keypoints"], original["keypoints"] * [0.5, 1])
    np.testing.assert_array_equal(result["labels"], original["labels"])
    np.testing.assert_array_equal(result["masks"], original["masks"][:, :, ::2])
def test_pad_if_needed_keeps_structured_annotations_aligned(annotated):
    original = annotated.to_dict()
    result = annotated.apply(
        rivet.Transform().pad_if_needed(12, 14, position="bottom_right", fill_mask=7)
    ).to_dict()
    assert result["image"].shape == (12, 14, 3)
    np.testing.assert_array_equal(result["image"][2:, 2:], original["image"])
    np.testing.assert_array_equal(result["boxes"], original["boxes"] + 2)
    np.testing.assert_array_equal(result["keypoints"], original["keypoints"] + 2)
    np.testing.assert_array_equal(result["labels"], original["labels"])
    np.testing.assert_array_equal(
        result["masks"], np.pad(original["masks"], ((0, 0), (2, 0), (2, 0)), constant_values=7)
    )


def test_grid_mask_semantic_fill_uses_rotated_pixels():
    image = np.full((7, 9, 3), 120, dtype=np.uint8)
    mask = np.full((7, 9), 3, dtype=np.int64)
    sample = rivet.AnnotatedSample(image, masks=mask)
    result = sample.apply(
        rivet.Transform().grid_mask((2, 2), (0.25, 0.25), (0.3, 0.3), fill_mask=-1)
    ).to_dict()
    covered = result["image"][..., 0] == 0
    assert 0 < covered.sum() < covered.size
    np.testing.assert_array_equal(result["masks"], np.where(covered, -1, 3))


def test_bbox_subset_safe_crop_selects_intact_box_and_synchronizes():
    h, w = 12, 24
    y, x = np.indices((h, w), dtype=np.uint8)
    image = np.stack((x, y, x + y), axis=-1)
    original = np.array([[1, 2, 4, 6], [20, 5, 23, 9]], dtype=np.float32)
    points = np.array([[[2, 3]], [[21, 7]]], dtype=np.float32)
    masks = np.stack([(x < 4) & (y < 6), (x >= 20) & (y >= 5)]).astype(np.uint8)
    sample = rivet.AnnotatedSample(image, boxes=original, labels=[0, 1], masks=masks, keypoints=points)
    pipeline = rivet.StructuredPipeline([sample] * 9).bbox_subset_safe_random_crop(
        (0.5, 0.5), aspect_ratio_range=(1, 1)
    ).seed(31)
    seen = set()
    for a, b in zip(pipeline.execute(), pipeline.workers(3).execute(), strict=True):
        actual = a["images"][0]
        np.testing.assert_array_equal(actual, b["images"][0])
        top, left = int(actual[0, 0, 1]), int(actual[0, 0, 0])
        oh, ow = actual.shape[:2]
        assert oh == ow
        target = a["targets"][0]
        labels = target["labels"]
        np.testing.assert_array_equal(labels, b["targets"][0]["labels"])
        assert labels.shape == (1,)
        seen.add(int(labels[0]))
        np.testing.assert_array_equal(target["boxes"], original[labels] - [left, top, left, top])
        np.testing.assert_array_equal(target["masks"], masks[labels, top:top + oh, left:left + ow])
        np.testing.assert_array_equal(target["keypoints"], points[labels] - [left, top])
    assert seen == {0, 1}
    fallback = rivet.StructuredPipeline([sample]).bbox_subset_safe_random_crop(
        (1, 1), aspect_ratio_range=(30, 31)
    )
    np.testing.assert_array_equal(next(iter(fallback.execute()))["images"][0], image)
    with pytest.raises(ValueError):
        rivet.StructuredPipeline([sample]).bbox_subset_safe_random_crop((0, 1))


@pytest.mark.parametrize("instance", [False, True])
def test_crop_non_empty_mask_selects_eligible_corner(instance):
    y, x = np.indices((9, 11), dtype=np.uint8)
    image = np.stack((x, y, x + y), axis=-1)
    mask = np.zeros((9, 11), dtype=np.int64)
    mask[8, 10] = 7
    mask[0, 0] = 5
    if instance:
        ignored = np.zeros_like(mask)
        ignored[0, 0] = 42
        mask = np.stack([ignored, mask])
    sample = rivet.AnnotatedSample(image, masks=mask)
    pipeline = rivet.StructuredPipeline([sample] * 4).crop_non_empty_mask_if_exists(
        4, 3, ignore_values=[5], ignore_channels=[0] if instance else [],
    ).seed(41)
    serial = next(iter(pipeline.batch(4).execute()))
    parallel = next(iter(pipeline.workers(3).batch(4).execute()))
    np.testing.assert_array_equal(serial["images"], parallel["images"])
    for actual, target in zip(serial["images"], serial["targets"], strict=True):
        np.testing.assert_array_equal(actual, image[6:9, 7:11])
        np.testing.assert_array_equal(target["masks"], mask[..., 6:9, 7:11])


def test_crop_non_empty_mask_missing_and_ignored_fallback():
    y, x = np.indices((6, 8), dtype=np.uint8)
    image = np.stack((x, y, x + y), axis=-1)
    missing = rivet.AnnotatedSample(image)
    ignored = rivet.AnnotatedSample(image, masks=np.full((6, 8), 4, np.uint8))

    def crop(sample):
        return next(iter(rivet.StructuredPipeline([sample]).seed(13)
            .crop_non_empty_mask_if_exists(4, 4, ignore_values=[4]).execute()))

    a = crop(missing)
    b = crop(ignored)
    assert a["images"].shape == (1, 4, 4, 3)
    np.testing.assert_array_equal(a["images"], b["images"])
    np.testing.assert_array_equal(b["targets"][0]["masks"], 4)


def test_random_crop_near_bbox_fixed_and_missing_reference(annotated):
    original = annotated.to_dict()
    pipeline = rivet.StructuredPipeline([annotated]).random_crop_near_bbox((0, 0), box_index=1)
    batch = next(iter(pipeline.execute()))
    np.testing.assert_array_equal(batch["images"][0], original["image"][5:9, 7:11])
    target = batch["targets"][0]
    np.testing.assert_array_equal(target["boxes"], [[0, 0, 4, 4]])
    np.testing.assert_array_equal(target["labels"], [9])
    np.testing.assert_array_equal(target["keypoints"], original["keypoints"][1:] - [7, 5])
    np.testing.assert_array_equal(target["masks"], original["masks"][1:, 5:9, 7:11])
    empty = rivet.AnnotatedSample(original["image"])
    missing = rivet.StructuredPipeline([empty]).random_crop_near_bbox()
    np.testing.assert_array_equal(next(iter(missing.execute()))["images"][0], original["image"])
    explicit = rivet.StructuredPipeline([empty]).random_crop_near_bbox(
        (0, 0), reference_box=(-1, 1, 3, 5)
    )
    np.testing.assert_array_equal(next(iter(explicit.execute()))["images"][0], original["image"][1:5, :3])


def test_random_crop_near_bbox_seed_across_workers(annotated):
    pipeline = rivet.StructuredPipeline([annotated] * 7).random_crop_near_bbox((0.4, 0.4)).seed(31)
    for a, b in zip(pipeline.execute(), pipeline.workers(3).execute(), strict=True):
        np.testing.assert_array_equal(a["images"], b["images"])
        for key in ("boxes", "labels", "keypoints", "masks"):
            np.testing.assert_array_equal(a["targets"][0][key], b["targets"][0][key])


def test_random_crop_from_borders_aligns_structured_targets():
    y, x = np.indices((10, 12), dtype=np.uint8)
    image = np.stack([x, y, x+y], axis=-1)
    sample = rivet.AnnotatedSample(
        image, boxes=[[6, 5, 12, 10]], masks=np.ones((1, 10, 12), np.uint8),
        labels=[3], keypoints=[[[9, 7]]],
    )
    transform = rivet.Transform().random_crop_from_borders(0.5, 0, 0.5, 0)
    pipeline = rivet.StructuredPipeline([sample] * 5).compose(transform).seed(23)
    for a, b in zip(pipeline.execute(), pipeline.workers(3).execute(), strict=True):
        np.testing.assert_array_equal(a["images"], b["images"])
        out = a["images"][0]
        left, top = map(int, out[0, 0, :2])
        target = a["targets"][0]
        np.testing.assert_array_equal(target["boxes"], [[6-left, 5-top, 12-left, 10-top]])
        np.testing.assert_array_equal(target["keypoints"], [[[9-left, 7-top]]])
        np.testing.assert_array_equal(target["labels"], [3])
        assert target["masks"].shape == (1, *out.shape[:2])


def test_random_sized_crop_resizes_structured_fields(annotated):
    original = annotated.to_dict()
    transform = rivet.Transform().random_sized_crop((10, 10), (5, 6), 1.2, "nearest")
    pipeline = rivet.StructuredPipeline([annotated] * 3).compose(transform).seed(33).batch(3)
    result = next(iter(pipeline.workers(3).execute()))
    assert result["images"].shape == (3, 5, 6, 3)
    target = result["targets"][0]
    np.testing.assert_array_equal(target["boxes"], original["boxes"] / 2)
    np.testing.assert_array_equal(target["labels"], original["labels"])
    np.testing.assert_array_equal(target["keypoints"], original["keypoints"] / 2)
    np.testing.assert_array_equal(target["masks"], original["masks"][:, ::2, ::2])


def test_crop_and_pad_synchronizes_structured_annotations(annotated):
    original = annotated.to_dict()
    result = annotated.apply(
        rivet.Transform().crop_and_pad(px=(-2, 3, 1, -5), keep_size=False, fill_mask=7)
    ).to_dict()
    np.testing.assert_array_equal(result["boxes"], [[2, 3, 6, 7]])
    np.testing.assert_array_equal(result["labels"], [9])
    np.testing.assert_array_equal(result["keypoints"], original["keypoints"][1:] - [5, 2])
    np.testing.assert_array_equal(
        result["masks"], np.pad(original["masks"][1:, 2:, 5:], ((0, 0), (0, 1), (0, 3)), constant_values=7)
    )
    assert result["image"].shape == (9, 10, 3)


def test_transpose_swaps_all_spatial_targets(annotated):
    original = annotated.to_dict()
    result = annotated.apply(rivet.Transform().transpose()).to_dict()
    np.testing.assert_array_equal(result["image"], original["image"].transpose(1, 0, 2))
    np.testing.assert_array_equal(result["boxes"], original["boxes"][:, [1, 0, 3, 2]])
    np.testing.assert_array_equal(result["keypoints"], original["keypoints"][..., ::-1])
    np.testing.assert_array_equal(result["masks"], original["masks"].transpose(0, 2, 1))
    np.testing.assert_array_equal(result["labels"], original["labels"])


def test_random_rotate90_structured_targets(annotated):
    original = annotated.to_dict()
    pipeline = rivet.StructuredPipeline([annotated] * 9).compose(rivet.Transform().random_rotate90()).seed(4)
    seen = set()
    for batch, parallel in zip(pipeline.execute(), pipeline.workers(3).execute(), strict=True):
        img = batch["images"][0]
        np.testing.assert_array_equal(img, parallel["images"][0])
        k = next(i for i in range(4) if np.array_equal(img, np.rot90(original["image"], i)))
        seen.add(k)
        target = batch["targets"][0]
        boxes = original["boxes"].copy()
        points = original["keypoints"].copy()
        h, w = original["image"].shape[:2]
        for _ in range(k):
            boxes = np.stack([boxes[:, 1], w - boxes[:, 2], boxes[:, 3], w - boxes[:, 0]], axis=-1)
            points = np.stack([points[..., 1], w - 1 - points[..., 0]], axis=-1)
            h, w = w, h
        np.testing.assert_array_equal(target["masks"], np.rot90(original["masks"], k, axes=(1, 2)))
        np.testing.assert_array_equal(target["boxes"], boxes)
        np.testing.assert_array_equal(target["keypoints"], points)
        np.testing.assert_array_equal(target["labels"], original["labels"])
    assert seen == {0, 1, 2, 3}


@pytest.mark.parametrize("element", ["e", "r90", "r180", "r270", "v", "h", "t", "hvt"])
def test_square_symmetry_fixed_elements_and_annotations(annotated, element):
    original = annotated.to_dict()
    reverse = {"r90": "r270", "r270": "r90"}.get(element, element)
    transform = rivet.Transform().d4(element)
    result = annotated.apply(transform).to_dict()
    img = original["image"]
    if element.startswith("r"):
        expected = np.rot90(img, int(element[1:]) // 90)
    elif element == "v":
        expected = img[:, ::-1]
    elif element == "h":
        expected = img[::-1]
    elif element == "t":
        expected = img.transpose(1, 0, 2)
    elif element == "hvt":
        expected = img[::-1, ::-1].transpose(1, 0, 2)
    else:
        expected = img
    np.testing.assert_array_equal(result["image"], expected)
    recovered = annotated.apply(transform.square_symmetry(reverse)).to_dict()
    for key in ("image", "boxes", "keypoints", "masks", "labels"):
        np.testing.assert_array_equal(recovered[key], original[key])


def test_safe_rotate_aligns_constant_fill_and_targets():
    y, x = np.indices((8, 12), dtype=np.uint8)
    mask = x + 12 * y + 1
    image = np.repeat(mask[..., None], 3, axis=-1)
    sample = rivet.AnnotatedSample(
        image, masks=mask, boxes=[[0, 0, 12, 8]], keypoints=[[[5.5, 3.5]]], labels=[4]
    )
    transform = rivet.Transform().safe_rotate((30, 30), interpolation="nearest", fill=99, fill_mask=99)
    transformed = sample.apply(transform).to_dict()
    assert transformed["image"].shape == image.shape
    np.testing.assert_array_equal(transformed["image"][..., 0], transformed["masks"])
    assert np.any(transformed["masks"] == 99)
    np.testing.assert_array_equal(transformed["labels"], [4])
    np.testing.assert_allclose(transformed["keypoints"], [[[5.5, 3.5]]], atol=1e-6)
    l, t, r, b = transformed["boxes"][0]
    assert 0 <= l < r <= 12 and 0 <= t < b <= 8


def test_mask_dropout_keeps_instance_annotations_in_sync(annotated):
    original = annotated.to_dict()
    pipeline = rivet.StructuredPipeline([annotated] * 6).mask_dropout((1, 1), fill=23).seed(11)
    for a, b in zip(pipeline.execute(), pipeline.workers(3).execute(), strict=True):
        np.testing.assert_array_equal(a["images"], b["images"])
        target = a["targets"][0]
        keep = 0 if target["labels"][0] == 4 else 1
        covered = original["masks"][1 - keep] != 0
        np.testing.assert_array_equal(a["images"][0][covered], 23)
        np.testing.assert_array_equal(a["images"][0][~covered], original["image"][~covered])
        for key in ["boxes", "labels", "keypoints", "masks"]:
            np.testing.assert_array_equal(target[key], original[key][keep:keep + 1])


def test_mask_dropout_semantic_ids_and_empty_fallback():
    image = np.full((5, 7, 3), 80, np.uint8)
    mask = np.zeros((5, 7), np.int64)
    mask[0, 0] = mask[4, 6] = 32
    mask[2, 3] = -2
    sample = rivet.AnnotatedSample(image, masks=mask)
    pipeline = rivet.StructuredPipeline([sample]).mask_dropout((2, 2), fill_mask=-9)
    batch = next(iter(pipeline.execute()))
    np.testing.assert_array_equal(batch["images"][0], np.where(mask[..., None] != 0, 0, image))
    np.testing.assert_array_equal(batch["targets"][0]["masks"], np.where(mask != 0, -9, 0))
    empty = rivet.AnnotatedSample(image, masks=np.zeros_like(mask))
    unchanged = next(iter(rivet.StructuredPipeline([empty]).mask_dropout().execute()))
    np.testing.assert_array_equal(unchanged["images"][0], image)


def test_constrained_dropout_box_labels_and_worker_seed(annotated):
    original = annotated.to_dict()
    pipeline = rivet.StructuredPipeline([annotated] * 6).constrained_coarse_dropout(
        (1, 1), (0.5, 0.5), (0.5, 0.5), bbox_labels=[9], fill=23, fill_mask=0
    ).seed(17)
    for a, b in zip(pipeline.execute(), pipeline.workers(3).execute(), strict=True):
        image = a["images"][0]
        np.testing.assert_array_equal(image, b["images"][0])
        covered = np.all(image == 23, axis=-1)
        assert covered.sum() == 4 and covered[5:9, 7:11].sum() == 4
        t = a["targets"][0]
        np.testing.assert_array_equal(t["labels"], original["labels"])
        np.testing.assert_array_equal(t["boxes"], original["boxes"])
        np.testing.assert_array_equal(t["keypoints"], original["keypoints"])
        np.testing.assert_array_equal(t["masks"], np.where(covered, 0, original["masks"]))
    whole = rivet.StructuredPipeline([annotated]).constrained_coarse_dropout(
        (1, 1), (1, 1), (1, 1), bbox_labels=[9], fill=23
    )
    np.testing.assert_array_equal(next(iter(whole.execute()))["targets"][0]["labels"], [4])


def test_constrained_dropout_semantic_components_and_empty():
    image = np.full((7, 9, 3), 80, np.uint8)
    mask = np.zeros((7, 9), np.int64)
    mask[1, 2] = mask[4, 6] = 5
    mask[3, 3] = 9
    sample = rivet.AnnotatedSample(image, masks=mask)
    result = next(iter(rivet.StructuredPipeline([sample]).constrained_coarse_dropout(
        mask_indices=[5], fill=23, fill_mask=-1
    ).execute()))
    np.testing.assert_array_equal(result["images"][0], np.where(mask[..., None] == 5, 23, image))
    np.testing.assert_array_equal(result["targets"][0]["masks"], np.where(mask == 5, -1, mask))
    unmatched = next(iter(rivet.StructuredPipeline([sample]).constrained_coarse_dropout(mask_indices=[99]).execute()))
    np.testing.assert_array_equal(unmatched["images"][0], image)


@pytest.mark.parametrize("mode", ["coarse", "grid"])
def test_structured_spatial_dropout_mask_and_visibility(mode):
    image = np.full((8, 8, 3), 120, dtype=np.uint8)
    mask = np.full((8, 8), 6, dtype=np.int64)
    sample = rivet.AnnotatedSample(
        image, boxes=[[0, 0, 8, 8]], labels=[5], masks=mask, keypoints=[[[2, 3]]]
    )

    def pipeline(visibility):
        p = rivet.StructuredPipeline([sample] * 4).seed(17)
        kwargs = dict(fill=23, fill_mask=-1, min_visibility=visibility)
        if mode == "coarse":
            return p.coarse_dropout((1, 1), hole_height=(4, 4), hole_width=(4, 4), **kwargs)
        return p.grid_dropout(4, ratio=0.5, **kwargs)

    p = pipeline(0.75)
    for a, b in zip(p.execute(), p.workers(3).execute(), strict=True):
        np.testing.assert_array_equal(a["images"], b["images"])
        covered = a["images"][0][..., 0] == 23
        assert covered.sum() == 16
        t = a["targets"][0]
        np.testing.assert_array_equal(t["masks"], np.where(covered, -1, 6))
        np.testing.assert_array_equal(t["labels"], [5])
        np.testing.assert_array_equal(t["keypoints"], [[[2, 3]]])
    removed = next(iter(pipeline(0.8).execute()))["targets"][0]
    assert removed["boxes"].shape == (0, 4)
    assert removed["labels"].shape == (0,)
    assert removed["keypoints"].shape == (0, 1, 2)


@pytest.mark.parametrize("instance", [False, True])
def test_morphological_masks_use_independent_numeric_planes(instance):
    image = np.zeros((5, 7, 3), np.uint8)
    mask = np.zeros((5, 7), np.int64)
    mask[0, 0] = 9
    expected = np.zeros_like(mask)
    expected[0, :2] = expected[1, 0] = 9
    if instance:
        mask = np.stack([mask, np.zeros_like(mask)])
        expected = np.stack([expected, np.zeros_like(expected)])
    sample = rivet.AnnotatedSample(image, masks=mask)
    t = sample.apply(rivet.Transform().morphological(3, "dilation")).to_dict()
    np.testing.assert_array_equal(t["masks"], expected)


def test_optical_distortion_shares_pixel_map_and_inverts_keypoints():
    h, w = 9, 11
    yy, xx = np.indices((h, w))
    mask = (xx + w * yy + 1).astype(np.uint8)
    image = np.repeat(mask[..., None], 3, axis=-1)
    sample = rivet.AnnotatedSample(
        image, masks=mask, boxes=[[2, 2, 9, 7]], keypoints=[[[6, 4]]], labels=[4]
    )
    result = sample.apply(rivet.Transform().optical_distortion(
        (0.4, 0.4), "fisheye", interpolation="nearest", fill=200, fill_mask=200
    )).to_dict()
    f = 1 + 0.4 * ((xx - 5)**2 + (yy - 4)**2) / ((w*w+h*h) / 4)
    sx, sy = (xx - 5) * f + 5, (yy - 4) * f + 4
    ix, iy = np.floor(sx + 0.5).astype(int), np.floor(sy + 0.5).astype(int)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    expected = np.full((h, w), 200, dtype=np.uint8)
    expected[valid] = mask[iy[valid], ix[valid]]
    np.testing.assert_array_equal(result["masks"], expected)
    np.testing.assert_array_equal(result["image"][..., 0], expected)
    inside = (sx+0.5 >= 2) & (sx+0.5 < 9) & (sy+0.5 >= 2) & (sy+0.5 < 7)
    y, x = np.where(inside)
    np.testing.assert_array_equal(result["boxes"], [[x.min(), y.min(), x.max()+1, y.max()+1]])
    np.testing.assert_array_equal(result["labels"], [4])
    x, y = result["keypoints"][0, 0]
    assert abs(y - 4) < 1e-6
    assert abs((x - 5) * (1 + 0.4 * (x - 5)**2 / ((w*w+h*h) / 4)) - 1) < 1e-6


def test_grid_distortion_shared_map_and_empty_instance_visibility():
    h,w=7,9
    y,x=np.indices((h,w),dtype=np.uint8)
    mask=y*w+x+1
    img=np.repeat(mask[...,None],3,axis=-1)
    sample=rivet.AnnotatedSample(img,masks=mask,boxes=[[0,0,3,3],[7,6,7.4,7]],labels=[4,7],keypoints=[[[2,2]],[[8,6]]])
    result=sample.apply(rivet.Transform().grid_distortion(
        1,(0.5,0.5),normalized=False,interpolation="nearest",fill=200,fill_mask=200
    )).to_dict()
    sx,sy=np.floor(x*1.5+0.5).astype(int),np.floor(y*1.5+0.5).astype(int)
    valid=(sx<w)&(sy<h)
    expected=np.full_like(mask,200)
    expected[valid]=mask[sy[valid],sx[valid]]
    np.testing.assert_array_equal(result["image"][...,0],expected)
    np.testing.assert_array_equal(result["masks"],expected)
    np.testing.assert_array_equal(result["boxes"],[[0,0,2,2]])
    np.testing.assert_array_equal(result["labels"],[4])
    np.testing.assert_allclose(result["keypoints"],[[[4/3,4/3]]],atol=1e-6)


def test_piecewise_affine_uses_one_map_for_semantic_mask_and_image():
    mask = np.arange(1, 100, dtype=np.uint8).reshape(9, 11)
    image = np.repeat(mask[..., None], 3, axis=-1)
    sample = rivet.AnnotatedSample(image, masks=mask, boxes=[[2, 2, 9, 7]], labels=[3], keypoints=[[[5, 4]]])
    t = rivet.Transform().piecewise_affine(
        (1, 1), absolute_scale=True, interpolation="nearest", fill=200, fill_mask=200
    )
    result = sample.apply(t).to_dict()
    np.testing.assert_array_equal(result["image"][..., 0], result["masks"])
    assert np.isfinite(result["keypoints"]).all()
    np.testing.assert_array_equal(result["labels"], [3])


def test_grid_elastic_deform_uses_shared_remap_for_targets():
    mask = np.arange(1, 100, dtype=np.uint8).reshape(9, 11)
    image = np.repeat(mask[..., None], 3, axis=-1)
    sample = rivet.AnnotatedSample(image, masks=mask, boxes=[[0, 0, 11, 9]], labels=[3], keypoints=[[[0, 0], [6, 5]]])
    transform = rivet.Transform().grid_elastic_deform((2, 2), 10, interpolation="nearest")
    result = sample.apply(transform).to_dict()
    np.testing.assert_array_equal(result["image"][..., 0], result["masks"])
    np.testing.assert_array_equal(result["boxes"], [[0, 0, 11, 9]])
    np.testing.assert_allclose(result["keypoints"][0, 0], [0, 0], atol=1e-6)
    assert np.isfinite(result["keypoints"]).all()
    np.testing.assert_array_equal(result["labels"], [3])


def test_thin_plate_spline_matches_nearest_masks_and_preserves_degenerate_image():
    mask = np.arange(1, 100, dtype=np.uint8).reshape(9, 11)
    image = np.repeat(mask[..., None], 3, axis=-1)
    t = rivet.Transform().thin_plate_spline(interpolation="nearest", fill=200, fill_mask=200)
    sample = rivet.AnnotatedSample(image, masks=mask, boxes=[[2, 2, 9, 7]], labels=[4], keypoints=[[[5, 4]]])
    result = sample.apply(t).to_dict()
    np.testing.assert_array_equal(result["image"][..., 0], result["masks"])
    np.testing.assert_array_equal(result["labels"], [4])
    assert np.isfinite(result["keypoints"]).all()
    one_row = rivet.AnnotatedSample(image[:1], masks=mask[:1])
    np.testing.assert_array_equal(one_row.apply(t).to_dict()["image"], image[:1])


def test_pixel_spread_shared_integer_map_and_keypoint_policy():
    h, w = 9, 11
    mask = np.arange(1, h*w+1, dtype=np.uint8).reshape(h,w)
    image = np.repeat(mask[..., None], 3, axis=-1)
    sample = rivet.AnnotatedSample(
        image, masks=mask, boxes=[[0, 0, w, h]], labels=[2], keypoints=[[[5, 4]]]
    )
    for epoch in range(4):
        pipeline = rivet.StructuredPipeline([sample]).compose(rivet.Transform().pixel_spread(2)).seed(23).epoch(epoch)
        result = next(iter(pipeline.execute()))
        target = result["targets"][0]
        np.testing.assert_array_equal(result["images"][0][..., 0], target["masks"])
        y, x = np.indices((h,w))
        src = target["masks"].astype(int) - 1
        assert np.max(abs(src % w - x)) <= 2
        assert np.max(abs(src // w - y)) <= 2
        ys, xs = np.where(target["masks"] == mask[4,5])
        if len(xs):
            i = np.argmin((xs-5)**2+(ys-4)**2)
            expected = [xs[i], ys[i]]
        else:
            expected = [5,4]
        np.testing.assert_array_equal(target["keypoints"], [[expected]])
        np.testing.assert_array_equal(target["boxes"], [[0, 0, w, h]])
        np.testing.assert_array_equal(target["labels"], [2])


def test_random_grid_shuffle_moves_fragments_and_aligned_targets():
    mask = np.arange(1, 25, dtype=np.int64).reshape(4, 6)
    image = np.repeat(mask.astype(np.uint8)[..., None], 3, axis=-1)
    sample = rivet.AnnotatedSample(image, masks=mask, boxes=[[1.5, 0.5, 4.5, 1.5]],
                                  labels=[3], keypoints=[[[1.25, 1]]])
    pipeline = rivet.StructuredPipeline([sample] * 5).compose(rivet.Transform().random_grid_shuffle((2, 2))).seed(31)
    for a, b in zip(pipeline.execute(), pipeline.workers(3).execute(), strict=True):
        t = a["targets"][0]
        actual = a["images"][0]
        np.testing.assert_array_equal(actual, b["images"][0])
        np.testing.assert_array_equal(actual[..., 0], t["masks"])
        y0, x0 = np.argwhere(t["masks"] == 1)[0]
        y1, x1 = np.argwhere(t["masks"] == 4)[0]
        expect = [[min(x0 + 1.5, x1), min(y0, y1) + 0.5,
                   max(x0 + 3, x1 + 1.5), max(y0, y1) + 1.5]]
        np.testing.assert_array_equal(t["boxes"], expect)
        np.testing.assert_array_equal(t["keypoints"], [[[x0 + 1.25, y0 + 1]]])
        np.testing.assert_array_equal(t["labels"], [3])


def test_water_refraction_matches_integer_mask_sampling():
    mask = np.arange(1, 101, dtype=np.uint8).reshape(10, 10)
    image = np.repeat(mask[..., None], 3, axis=-1)
    sample = rivet.AnnotatedSample(image, masks=mask, boxes=[[2, 2, 8, 8]], labels=[6], keypoints=[[[5, 5]]])
    transform = rivet.Transform().water_refraction(
        (0.3, 0.3), (1, 1), (1, 1), interpolation="nearest", fill=200, fill_mask=200
    )
    result = sample.apply(transform).to_dict()
    np.testing.assert_array_equal(result["image"][..., 0], result["masks"])
    np.testing.assert_array_equal(result["labels"], [6])
    assert np.isfinite(result["keypoints"]).all()


def test_structured_elastic_transform_shares_mask_and_worker_seed():
    h, w = 9, 11
    mask = np.arange(1, h*w+1, dtype=np.uint8).reshape(h,w)
    image = np.repeat(mask[..., None], 3, axis=-1)
    sample = rivet.AnnotatedSample(
        image, masks=mask, boxes=[[2, 2, 8, 8]], labels=[5], keypoints=[[[5, 4]]],
    )
    transform = rivet.Transform().elastic_transform(5, 0, interpolation="nearest")
    pipeline = rivet.StructuredPipeline([sample] * 5).compose(transform).seed(31)
    for a, b in zip(pipeline.execute(), pipeline.workers(3).execute(), strict=True):
        result = a["images"][0]
        np.testing.assert_array_equal(result, b["images"][0])
        assert not np.array_equal(result, image)
        t = a["targets"][0]
        np.testing.assert_array_equal(result[..., 0], t["masks"])
        np.testing.assert_array_equal(t["labels"], [5])
        assert np.isfinite(t["keypoints"]).all()
        l, top, r, bottom = t["boxes"][0]
        assert 0 <= l < r <= w and 0 <= top < bottom <= h
    identity = sample.apply(rivet.Transform().elastic_transform(0, 0)).to_dict()
    for key, value in sample.to_dict().items():
        if key in {"image", "masks", "boxes", "labels", "keypoints"}:
            np.testing.assert_array_equal(identity[key], value)
