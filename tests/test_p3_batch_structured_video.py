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
    np.testing.assert_array_equal(sanitized["masks"], masks if semantic_mask else masks[keep])


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
        rivet.Transform().elastic_transform(1, 1),
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
