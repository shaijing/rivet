"""P2 v2 augmentation contracts and pipeline/Transform parity."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import rivet


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.random.default_rng(19).integers(0, 256, (12, 18, 3), dtype=np.uint8)
    for index in range(6):
        Image.fromarray(np.roll(pixels, index, axis=0)).save(folder / f"{index}.png")
    return tmp_path, pixels


def output(root, build, *, transform=False, workers=0, epoch=0, seed=712):
    pipeline = rivet.scan_image_folder(root).decode_image().seed(seed).epoch(epoch)
    pipeline = pipeline.workers(workers)
    pipeline = (
        pipeline.compose(build(rivet.Transform())) if transform else build(pipeline)
    )
    return [batch["images"][0] for batch in pipeline.batch(1).execute()]


@pytest.mark.parametrize("transform", [False, True])
def test_channel_permutation(source, transform):
    root, pixels = source
    actual = output(root, lambda p: p.permute_channels([2, 0, 1]), transform=transform)
    np.testing.assert_array_equal(actual[0], pixels[..., [2, 0, 1]])
    random = output(root, lambda p: p.random_channel_permutation(), transform=transform)
    for index, image in enumerate(random):
        np.testing.assert_array_equal(
            np.sort(image, axis=-1), np.sort(np.roll(pixels, index, axis=0), axis=-1)
        )


@pytest.mark.parametrize("transform", [False, True])
@pytest.mark.parametrize(
    "build,shape",
    [
        (lambda p: p.random_resize(8, 9, "nearest"), (8, 12, 3)),
        (lambda p: p.random_shortest_size(8, interpolation="nearest"), (8, 12, 3)),
        (lambda p: p.random_shortest_size([8], 10, "nearest"), (6, 10, 3)),
        (lambda p: p.random_shortest_size([8], 6, "nearest"), (4, 6, 3)),
        (lambda p: p.scale_jitter(9, 12, [1, 1], "nearest"), (6, 9, 3)),
        (lambda p: p.random_zoom_out(side_range=[2, 2], probability=1), (24, 36, 3)),
    ],
)
def test_geometry_shapes(source, transform, build, shape):
    assert output(source[0], build, transform=transform)[0].shape == shape


@pytest.mark.parametrize("transform", [False, True])
def test_fixed_resize_matches_torchvision(source, transform):
    F = pytest.importorskip("torchvision.transforms.v2.functional")
    root, pixels = source
    actual = output(root, lambda p: p.random_resize(8, 9), transform=transform)[0]
    expected = np.asarray(F.resize(Image.fromarray(pixels), [8]))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("transform", [False, True])
def test_probability_zero_and_rgb_identity(source, transform):
    root, pixels = source
    actual = output(
        root,
        lambda p: (
            p.random_photometric_distort(probability=0)
            .random_zoom_out(probability=0)
            .rgb()
        ),
        transform=transform,
    )[0]
    np.testing.assert_array_equal(actual, pixels)


@pytest.mark.parametrize("transform", [False, True])
@pytest.mark.parametrize("scale", [False, True])
def test_to_dtype_and_roundtrip(source, transform, scale):
    root, pixels = source
    actual = output(root, lambda p: p.to_dtype("float32", scale), transform=transform)[
        0
    ]
    expected = pixels.astype(np.float32) / (255 if scale else 1)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
    restored = output(
        root,
        lambda p: p.to_dtype("float32", scale).to_dtype("uint8", scale),
        transform=transform,
    )[0]
    np.testing.assert_array_equal(restored, pixels)


@pytest.mark.parametrize("transform", [False, True])
def test_gaussian_noise_fixed_shift_and_clip(source, transform):
    root, pixels = source
    for clip in (False, True):
        actual = output(
            root,
            lambda p, clip=clip: p.to_dtype("float32", scale=True).gaussian_noise(
                0.25, 0, clip
            ),
            transform=transform,
        )[0]
        expected = pixels.astype(np.float32) / 255 + 0.25
        if clip:
            expected = np.clip(expected, 0, 1)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    actual = output(root, lambda p: p.gaussian_noise(0.25, 0), transform=transform)[0]
    expected = np.clip(pixels.astype(np.int16) + 63, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("transform", [False, True])
def test_shot_noise_float_pipeline_and_seed(source, transform):
    root, _ = source

    def build(p):
        return p.to_tensor().shot_noise()

    single_worker = output(root, build, transform=transform)
    multi_worker = output(root, build, transform=transform, workers=3)
    np.testing.assert_array_equal(single_worker, multi_worker)
    assert single_worker[0].dtype == np.float32
    assert np.all((np.asarray(single_worker) >= 0) & (np.asarray(single_worker) <= 1))
    zero = output(root, lambda p: p.to_tensor().shot_noise(0.0), transform=transform)
    np.testing.assert_array_equal(zero, output(root, lambda p: p.to_tensor()))
    assert not np.array_equal(single_worker[0], zero[0])
    with pytest.raises(ValueError, match="ShotNoise intensity"):
        output(root, lambda p: p.shot_noise(-0.1), transform=transform)
    with pytest.raises(ValueError, match="ShotNoise requires float32 input"):
        output(root, lambda p: p.shot_noise(), transform=transform)


@pytest.mark.parametrize("transform", [False, True])
def test_jpeg_quality_and_dtype(source, transform):
    root, pixels = source
    low = output(root, lambda p: p.jpeg(10), transform=transform)[0]
    high = output(root, lambda p: p.jpeg([100, 100]), transform=transform)[0]
    assert low.shape == high.shape == pixels.shape
    assert low.dtype == high.dtype == np.uint8
    assert not np.array_equal(low, high)
    with pytest.raises((ValueError, RuntimeError)):
        output(root, lambda p: p.to_dtype("float32").jpeg(80), transform=transform)


@pytest.mark.parametrize(
    "build",
    [
        lambda p: p.permute_channels([]),
        lambda p: p.permute_channels([0, 0, 1]),
        lambda p: p.random_photometric_distort(brightness=[-1, 1]),
        lambda p: p.random_photometric_distort(hue=[-0.6, 0.5]),
        lambda p: p.random_photometric_distort(probability=1.1),
        lambda p: p.random_resize(8, 8),
        lambda p: p.random_resize(0, 8),
        lambda p: p.random_shortest_size([]),
        lambda p: p.random_shortest_size([0]),
        lambda p: p.scale_jitter(0, 4),
        lambda p: p.scale_jitter(4, 4, [2, 1]),
        lambda p: p.random_zoom_out(side_range=[0.5, 2]),
        lambda p: p.random_zoom_out(fill=[float("nan")]),
        lambda p: p.random_zoom_out(probability=-0.1),
        lambda p: p.gaussian_noise(sigma=-0.1),
        lambda p: p.gaussian_noise(mean=float("nan")),
        lambda p: p.jpeg([80]),
        lambda p: p.jpeg([80, 20]),
        lambda p: p.jpeg(0),
        lambda p: p.to_dtype("float64"),
    ],
)
@pytest.mark.parametrize("transform", [False, True])
def test_invalid_parameters(source, build, transform):
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        output(source[0], build, transform=transform)


@pytest.mark.parametrize(
    "build",
    [
        lambda p: p.random_channel_permutation(),
        lambda p: p.random_photometric_distort(),
        lambda p: p.random_resize(7, 13),
        lambda p: p.random_shortest_size([7, 9, 12]),
        lambda p: p.scale_jitter(18, 12),
        lambda p: p.random_zoom_out(probability=1),
        lambda p: p.to_dtype("float32", scale=True).gaussian_noise(),
        lambda p: p.jpeg([10, 95]),
    ],
)
def test_random_reproducibility_workers_and_composition(source, build):
    root, _ = source
    first = output(root, build)
    second = output(root, build, workers=2)
    composed = output(root, build, transform=True, workers=2)
    for a, b, c in zip(first, second, composed, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)
    later = output(root, build, epoch=1)
    assert any(not np.array_equal(a, b) for a, b in zip(first, later, strict=True))


@pytest.mark.parametrize("transform", [False, True])
def test_zoom_out_preserves_image_and_channel_fill(source, transform):
    root, pixels = source
    fill = np.array([7, 31, 89], dtype=np.uint8)
    actual = output(
        root,
        lambda p: p.random_zoom_out(
            fill=fill.tolist(), side_range=[2, 2], probability=1
        ),
        transform=transform,
    )[0]
    matches = [
        (y, x)
        for y in range(13)
        for x in range(19)
        if np.array_equal(actual[y : y + 12, x : x + 18], pixels)
    ]
    assert len(matches) == 1
    y, x = matches[0]
    mask = np.ones(actual.shape[:2], dtype=bool)
    mask[y : y + 12, x : x + 18] = False
    np.testing.assert_array_equal(actual[mask], np.tile(fill, (mask.sum(), 1)))


@pytest.mark.parametrize("transform", [False, True])
def test_gaussian_noise_uint8_wrap(source, transform):
    root, pixels = source
    actual = output(
        root, lambda p: p.gaussian_noise(0.25, 0, clip=False), transform=transform
    )[0]
    np.testing.assert_array_equal(
        actual, (pixels.astype(np.int16) + 63).astype(np.uint8)
    )


@pytest.mark.parametrize("transform", [False, True])
def test_channels_after_chw(source, transform):
    root, pixels = source
    actual = output(
        root,
        lambda p: p.pil_to_tensor().permute_channels([2, 0, 1]),
        transform=transform,
    )[0]
    np.testing.assert_array_equal(actual, pixels[..., [2, 0, 1]].transpose(2, 0, 1))


@pytest.mark.parametrize("quality", [1, 5, 10, 20, 24, 25, 30, 49, 80, 100])
def test_jpeg_matches_pillow_libjpeg(source, quality):
    import io

    root, pixels = source
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="JPEG", quality=quality, subsampling=2)
    encoded.seek(0)
    expected = np.asarray(Image.open(encoded))
    actual = output(root, lambda p: p.jpeg(quality))[0]
    np.testing.assert_allclose(
        actual.astype(np.int16), expected.astype(np.int16), atol=1
    )


@pytest.mark.parametrize("transform", [False, True])
@pytest.mark.parametrize("chw", [False, True])
def test_rgb_from_grayscale(source, transform, chw):
    root, _ = source

    def grayscale(p):
        p = p.grayscale(1)
        return p.pil_to_tensor() if chw else p

    gray = output(root, grayscale, transform=transform)[0]
    actual = output(root, lambda p: grayscale(p).rgb(), transform=transform)[0]
    expected = np.repeat(gray, 3, axis=0 if chw else -1)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("chw", [False, True])
def test_to_dtype_after_batch(source, scale, chw):
    root, pixels = source
    pipeline = rivet.scan_image_folder(root).decode_image()
    if chw:
        pipeline = pipeline.pil_to_tensor()
    actual = next(pipeline.batch(2).to_dtype("float32", scale=scale).execute())[
        "images"
    ]
    expected = np.stack([pixels, np.roll(pixels, 1, axis=0)]).astype(np.float32)
    if chw:
        expected = expected.transpose(0, 3, 1, 2)
    if scale:
        expected /= 255
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("transform", [False, True])
def test_photometric_fixed_factors_match_pillow(tmp_path, transform):
    from PIL import ImageEnhance

    gray = np.arange(12 * 18, dtype=np.uint8).reshape(12, 18)
    pixels = np.repeat(gray[..., None], 3, axis=-1)
    folder = tmp_path / "class"
    folder.mkdir()
    image = Image.fromarray(pixels)
    image.save(folder / "sample.png")
    actual = output(
        tmp_path,
        lambda p: p.random_photometric_distort(
            brightness=[0.7, 0.7],
            contrast=[1.3, 1.3],
            saturation=[1, 1],
            hue=[0, 0],
            probability=1,
        ),
        transform=transform,
    )[0]
    expected = np.asarray(
        ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(0.7)).enhance(1.3)
    )
    np.testing.assert_array_equal(actual, expected)


def test_gaussian_noise_distribution_and_independence(source):
    root, pixels = source
    images = output(
        root,
        lambda p: p.to_dtype("float32", scale=True).gaussian_noise(
            0.02, 0.1, clip=False
        ),
    )
    noise = np.concatenate(
        [
            (image - np.roll(pixels, index, axis=0).astype(np.float32) / 255).ravel()
            for index, image in enumerate(images)
        ]
    )
    assert abs(noise.mean() - 0.02) < 0.006
    assert abs(noise.std() - 0.1) < 0.006
    assert abs(np.corrcoef(noise[:-1], noise[1:])[0, 1]) < 0.06
    assert np.unique(noise).size > noise.size * 0.9


def test_random_resize_excludes_upper_bound(source):
    root, _ = source
    sizes = {
        image.shape[0]
        for seed in range(10)
        for image in output(root, lambda p: p.random_resize(7, 10), seed=seed)
    }
    assert sizes == {7, 8, 9}


@pytest.mark.parametrize("chw", [False, True])
def test_jpeg_grayscale_and_layout(source, chw):
    import io

    root, _ = source
    gray = output(root, lambda p: p.grayscale(1))[0]
    encoded = io.BytesIO()
    Image.fromarray(gray[..., 0]).save(encoded, format="JPEG", quality=80)
    encoded.seek(0)
    expected = np.asarray(Image.open(encoded))[..., None]

    def build(p):
        p = p.grayscale(1)
        if chw:
            p = p.pil_to_tensor()
        return p.jpeg(80)

    actual = output(root, build)[0]
    if chw:
        actual = actual.transpose(1, 2, 0)
    np.testing.assert_allclose(
        actual.astype(np.int16), expected.astype(np.int16), atol=1
    )


@pytest.mark.parametrize("mean", [-37.75, 300.5])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("chw", [False, True])
def test_to_dtype_unscaled_wrap_matches_torchvision(source, mean, batched, chw):
    torch = pytest.importorskip("torch")
    F = pytest.importorskip("torchvision.transforms.v2.functional")
    root, pixels = source
    pipeline = (
        rivet.scan_image_folder(root)
        .decode_image()
        .to_dtype("float32", scale=False)
        .gaussian_noise(mean=mean, sigma=0, clip=False)
    )
    if chw:
        pipeline = pipeline.pil_to_tensor()
    if batched:
        pipeline = pipeline.batch(2).to_dtype("uint8", scale=False)
    else:
        pipeline = pipeline.to_dtype("uint8", scale=False).batch(2)
    actual = next(pipeline.execute())["images"]
    values = np.stack([pixels, np.roll(pixels, 1, axis=0)]).astype(np.float32) + mean
    if chw:
        values = values.transpose(0, 3, 1, 2)
    expected = F.to_dtype(
        torch.from_numpy(values), dtype=torch.uint8, scale=False
    ).numpy()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "quality", [True, False, 50.0, [True, 80], [20, False], [20.0, 80], [20, 80.0]]
)
@pytest.mark.parametrize("transform", [False, True])
def test_jpeg_rejects_noninteger_quality(source, quality, transform):
    with pytest.raises(TypeError):
        output(source[0], lambda p: p.jpeg(quality), transform=transform)


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (13, 19)])
@pytest.mark.parametrize("quality", [1, 19, 25, 80, 100])
def test_jpeg_tiny_odd_images_match_pillow(tmp_path, shape, quality):
    import io

    pixels = np.random.default_rng(46).integers(0, 256, (*shape, 3), dtype=np.uint8)
    folder = tmp_path / "class"
    folder.mkdir()
    Image.fromarray(pixels).save(folder / "sample.png")
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="JPEG", quality=quality, subsampling=2)
    encoded.seek(0)
    expected = np.asarray(Image.open(encoded))
    actual = output(tmp_path, lambda p: p.jpeg(quality))[0]
    np.testing.assert_allclose(
        actual.astype(np.int16), expected.astype(np.int16), atol=1
    )


def test_jpeg_low_quality_integer_quantization_ties(tmp_path):
    """Quality 9 exercises IJG's integer-scale half-up quantization tables."""
    import io

    pixels = np.random.default_rng(0).integers(0, 256, (37, 41, 3), dtype=np.uint8)
    folder = tmp_path / "class"
    folder.mkdir()
    Image.fromarray(pixels).save(folder / "a.png")
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="JPEG", quality=9, subsampling=2)
    encoded.seek(0)
    expected = np.asarray(Image.open(encoded))
    actual = output(tmp_path, lambda p: p.jpeg(9))[0]
    np.testing.assert_allclose(
        actual.astype(np.int16), expected.astype(np.int16), atol=1
    )


@pytest.mark.parametrize("transform", [False, True])
def test_downscale_restores_original_shape(source, transform):
    root, _ = source
    actual = output(
        root,
        lambda p: p.downscale((0.5, 0.5), up_interpolation="bilinear"),
        transform=transform,
    )
    expected = output(
        root, lambda p: p.resize(9, 6, "nearest").resize(18, 12, "bilinear")
    )
    for image, reference in zip(actual, expected, strict=True):
        assert image.dtype == np.uint8
        np.testing.assert_array_equal(image, reference)


def test_salt_and_pepper_synchronizes_channels_and_layouts(source):
    root, pixels = source

    def aug(p):
        return p.salt_and_pepper(1.0, salt_ratio=0.5)

    hwc = output(root, aug, transform=True, workers=2)[0]
    chw = output(root, lambda p: aug(p.to_dtype("float32", True).hwc_to_chw()))[0]
    assert chw.dtype == np.float32
    assert set(np.unique(hwc)) == {0, 255}
    np.testing.assert_array_equal(hwc, np.broadcast_to(hwc[..., :1], hwc.shape))
    np.testing.assert_array_equal(chw.transpose(1, 2, 0), hwc / 255)
    np.testing.assert_array_equal(
        output(root, lambda p: p.salt_and_pepper(0))[0], pixels
    )


def test_grid_dropout_is_regular_and_worker_independent(source):
    root, pixels = source

    def augment(p):
        return p.to_dtype("float32", True).grid_dropout(6, ratio=0.5, fill=-1)

    actual = output(root, augment, workers=0)
    parallel = output(root, augment, transform=True, workers=2)
    for index, (image, reference) in enumerate(zip(actual, parallel, strict=True)):
        np.testing.assert_array_equal(image, reference)
        mask = image[..., 0] == -1
        np.testing.assert_array_equal(
            image == -1, np.broadcast_to(mask[..., None], image.shape)
        )
        unit = mask[:6, :6]
        assert unit.sum() == 9
        ys, xs = np.nonzero(unit)
        assert ys.max() - ys.min() == xs.max() - xs.min() == 2
        np.testing.assert_array_equal(mask, np.tile(unit, (2, 3)))
        original = np.roll(pixels, index, axis=0).astype(np.float32) / 255
        np.testing.assert_allclose(image[~mask], original[~mask], rtol=1e-7)


@pytest.mark.parametrize(
    "build",
    [
        lambda p: p.downscale((0.0, 0.5)),
        lambda p: p.downscale((0.5, 1.1)),
        lambda p: p.salt_and_pepper(float("nan")),
        lambda p: p.salt_and_pepper(0.5, salt_ratio=-0.1),
        lambda p: p.grid_dropout(1),
        lambda p: p.grid_dropout(4, ratio=1.0),
    ],
)
def test_new_augmentations_reject_invalid_parameters(build):
    with pytest.raises(ValueError):
        build(rivet.Transform())


@pytest.mark.parametrize(
    "transform",
    [
        rivet.Transform().downscale((0.5, 0.5)),
        rivet.Transform().salt_and_pepper(1),
        rivet.Transform().grid_dropout(4),
        rivet.Transform().random_tone_curve(0.25),
        rivet.Transform().clahe(grid=(2, 2)),
        rivet.Transform().chromatic_aberration((0.3, 0.3), (-0.3, -0.3)),
    ],
)
def test_degradations_preserve_structured_targets(source, transform):
    _, pixels = source
    sample = rivet.AnnotatedSample(
        pixels,
        boxes=[[1, 1, 6, 6]],
        labels=[2],
        masks=np.ones(pixels.shape[:2], dtype=np.uint8),
        keypoints=[[[2, 2]]],
    )
    original = sample.to_dict()
    result = sample.apply(transform, seed=19).to_dict()
    assert result["image"].shape == pixels.shape
    assert not np.array_equal(result["image"], pixels)
    for name in ("boxes", "labels", "masks", "keypoints"):
        np.testing.assert_array_equal(result[name], original[name])


@pytest.mark.parametrize("transform", [False, True])
@pytest.mark.parametrize("angle,axis", [(0, 1), (90, 0)])
def test_motion_blur_direction_and_reflection(source, transform, angle, axis):
    root, pixels = source
    result = output(
        root,
        lambda p: p.motion_blur((3, 3), angle_range=(angle, angle)),
        transform=transform,
        workers=2,
    )[0]
    padded = np.pad(
        pixels.astype(np.uint16),
        ((1, 1), (0, 0), (0, 0)) if axis == 0 else ((0, 0), (1, 1), (0, 0)),
        mode="reflect",
    )
    taps = [
        np.take(padded, range(index, index + pixels.shape[axis]), axis=axis)
        for index in range(3)
    ]
    np.testing.assert_array_equal(result, (sum(taps) + 1) // 3)


@pytest.mark.parametrize("transform", [False, True])
def test_unsharp_mask_matches_thresholded_gaussian_residual(source, transform):
    root, pixels = source
    smooth = output(root, lambda p: p.gaussian_blur(1.0))[0]
    detail = pixels.astype(np.int16) - smooth.astype(np.int16)
    expected = np.where(np.abs(detail) > 20, np.clip(pixels + detail, 0, 255), pixels)
    actual = output(root, lambda p: p.unsharp_mask(threshold=20), transform=transform)[
        0
    ]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        output(root, lambda p: p.unsharp_mask(amount=0), transform=transform)[0],
        pixels,
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda p: p.motion_blur((2, 7)),
        lambda p: p.motion_blur(angle_range=(100, 10)),
        lambda p: p.unsharp_mask(sigma=0),
        lambda p: p.unsharp_mask(amount=float("nan")),
    ],
)
def test_new_filters_validate_parameters(build):
    with pytest.raises(ValueError):
        build(rivet.Transform())


@pytest.mark.parametrize("transform", [False, True])
def test_defocus_uses_reflected_disk(source, transform):
    root, pixels = source
    padded = np.pad(pixels.astype(np.uint16), ((1, 1), (1, 1), (0, 0)), mode="reflect")
    expected = (
        padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + 2
    ) // 5
    actual = output(root, lambda p: p.defocus((1, 1)), transform=transform)[0]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        output(root, lambda p: p.defocus((0, 0)), transform=transform)[0], pixels
    )


@pytest.mark.parametrize("transform", [False, True])
def test_letterbox_restores_fixed_canvas(source, transform):
    root, _ = source
    actual = output(
        root,
        lambda p: p.letterbox(9, 9, interpolation="nearest", fill=173),
        transform=transform,
    )[0]
    reduced = output(root, lambda p: p.resize(9, 6, "nearest"))[0]
    assert actual.shape == (9, 9, 3)
    np.testing.assert_array_equal(actual[1:7], reduced)
    assert np.all(actual[:1] == 173)
    assert np.all(actual[7:] == 173)


def test_coarse_dropout_layout_overlap_and_reproducibility(source):
    root, pixels = source

    def augment(p):
        return p.coarse_dropout((2, 2), hole_height=(2, 2), hole_width=(3, 3), fill=-1)

    hwc = output(root, lambda p: augment(p.to_dtype("float32", True)), workers=2)
    chw = output(
        root,
        lambda p: augment(p.to_dtype("float32", True).hwc_to_chw()),
        transform=True,
    )
    for index, (image, reference) in enumerate(zip(hwc, chw, strict=True)):
        np.testing.assert_array_equal(image, reference.transpose(1, 2, 0))
        mask = image[..., 0] == -1
        assert 6 <= mask.sum() <= 12
        assert np.all(image[mask] == -1)
        np.testing.assert_allclose(
            image[~mask], np.roll(pixels, index, axis=0)[~mask] / 255, rtol=1e-7
        )
    unchanged = output(root, lambda p: p.coarse_dropout((0, 0)))[0]
    np.testing.assert_array_equal(unchanged, pixels)
    covered = output(
        root,
        lambda p: p.coarse_dropout((1, 1), hole_height=(99, 99), hole_width=(99, 99)),
    )[0]
    assert np.all(covered == 0)


@pytest.mark.parametrize("transform", [False, True])
def test_random_tone_curve_uses_monotone_shared_lut(tmp_path, transform):
    folder = tmp_path / "class"
    folder.mkdir()
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
    pixels = np.stack((ramp, 255 - ramp, ramp), axis=-1)
    Image.fromarray(pixels).save(folder / "ramp.png")

    def augment(p):
        return p.random_tone_curve(0.25)

    image = output(tmp_path, augment, transform=transform, workers=2)[0]
    lut = image[..., 0].flatten().astype(int)
    assert lut[0] == 0
    assert lut[-1] == 255
    assert np.all(np.diff(lut) >= 0)
    assert not np.array_equal(lut, np.arange(256))
    np.testing.assert_array_equal(image[..., 1], lut[pixels[..., 1]])
    np.testing.assert_array_equal(image, output(tmp_path, augment)[0])
    assert not np.array_equal(image, output(tmp_path, augment, epoch=1)[0])
    np.testing.assert_array_equal(
        output(tmp_path, lambda p: p.random_tone_curve(0))[0], pixels
    )


def test_clahe_grayscale_clip_limit_and_small_rgb_tiles(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.repeat(
        np.array([[0, 0], [0, 128]], dtype=np.uint8)[..., None], 3, axis=2
    )
    Image.fromarray(pixels).save(folder / "tiny.png")
    clipped = output(tmp_path, lambda p: p.grayscale().clahe(1.0, grid=(1, 1)))[0]
    unclipped = output(
        tmp_path, lambda p: p.grayscale().clahe(256.0, grid=(1, 1)), transform=True
    )[0]
    np.testing.assert_array_equal(clipped[..., 0], [[128, 128], [128, 255]])
    np.testing.assert_array_equal(unclipped[..., 0], [[191, 191], [191, 255]])
    np.testing.assert_array_equal(
        output(tmp_path, lambda p: p.clahe(grid=(8, 8)))[0], pixels
    )


def test_clahe_blends_neighboring_tile_luminance(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    ramp = np.array([[40, 80, 40, 160]], dtype=np.uint8)
    pixels = np.repeat(ramp[..., None], 3, axis=2)
    Image.fromarray(pixels).save(folder / "tiles.png")
    # Both tiles map 40 to 128; for 80 their LUTs differ by 127.
    # At x=1, the tile-center weights are 3/4 and 1/4.
    actual = output(tmp_path, lambda p: p.clahe(256, grid=(2, 1)))[0]
    expected = np.repeat(np.array([[128, 223, 128, 255]])[..., None], 3, axis=2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("transform", [False, True])
@pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
def test_chromatic_aberration_samples_channels_independently(
    source, transform, interpolation
):
    root, pixels = source
    actual = output(
        root,
        lambda p: p.chromatic_aberration(
            (0.5, 0.5), (-0.5, -0.5), interpolation=interpolation
        ),
        transform=transform,
    )[0]
    h, w = pixels.shape[:2]
    y, x = np.indices((h, w))
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    r2 = ((x - cx) ** 2 + (y - cy) ** 2) / (cx**2 + cy**2)
    expected = pixels.copy()
    for channel, coefficient in ((0, 0.5), (2, -0.5)):
        sx = np.clip(x + (x - cx) * coefficient * r2, 0, w - 1)
        sy = np.clip(y + (y - cy) * coefficient * r2, 0, h - 1)
        if interpolation == "nearest":
            expected[..., channel] = pixels[
                np.floor(sy + 0.5).astype(int), np.floor(sx + 0.5).astype(int), channel
            ]
        else:
            x0, y0 = sx.astype(int), sy.astype(int)
            x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
            plane = pixels[..., channel].astype(float)
            top = plane[y0, x0] + (plane[y0, x1] - plane[y0, x0]) * (sx - x0)
            bottom = plane[y1, x0] + (plane[y1, x1] - plane[y1, x0]) * (sx - x0)
            expected[..., channel] = top + (bottom - top) * (sy - y0)
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual, pixels)


@pytest.mark.parametrize(
    "build",
    [
        lambda p: p.random_tone_curve(-0.1),
        lambda p: p.clahe(grid=(0, 8)),
        lambda p: p.clahe(0),
        lambda p: p.chromatic_aberration(red_range=(0.0, 2.0)),
        lambda p: p.chromatic_aberration(interpolation="bicubic"),
    ],
)
def test_new_color_operators_validate_parameters(build):
    with pytest.raises(ValueError):
        build(rivet.Transform())


@pytest.mark.parametrize("shape", [(5, 7, 3), (1, 5, 3), (1, 1, 3)])
def test_vignetting_radial_falloff(tmp_path, shape):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.full(shape, 200, dtype=np.uint8)
    Image.fromarray(pixels).save(folder / "image.png")

    def fixed(p):
        return p.vignetting((0.5, 0.5), (0.5, 0.5))

    h, w, _ = shape
    actual = output(tmp_path, fixed, transform=True)[0]
    np.testing.assert_array_equal(actual[h // 2, w // 2], 200)
    np.testing.assert_array_equal(actual[0, 0], 200 if h == w == 1 else 100)
    np.testing.assert_array_equal(
        output(tmp_path, lambda p: p.vignetting((0, 0)))[0], pixels
    )
    floating = output(
        tmp_path, lambda p: fixed(p.grayscale().to_dtype("float32", True))
    )[0]
    assert floating.shape == (h, w, 1)
    assert floating.dtype == np.float32
    np.testing.assert_allclose(floating[0, 0, 0], actual[0, 0, 0] / 255)


def test_vignetting_is_seeded_across_workers(source):
    root, _ = source

    def augment(p):
        return p.vignetting()

    serial = output(root, augment)
    parallel = output(root, augment, transform=True, workers=3)
    np.testing.assert_array_equal(serial, parallel)
    assert not np.array_equal(serial, output(root, augment, epoch=1))


def test_planckian_jitter_white_point_and_clipping(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.array([[[128, 128, 128], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
    Image.fromarray(pixels).save(folder / "image.png")

    def tint(kelvin, floating=False):
        def build(p):
            if floating:
                p = p.to_dtype("float32", True)
            return p.planckian_jitter((kelvin, kelvin))

        return output(tmp_path, build, transform=True)[0]

    np.testing.assert_array_equal(tint(6500), pixels)
    warm, cool = tint(3000), tint(15000)
    assert warm[0, 0, 0] > 128 > warm[0, 0, 2]
    assert cool[0, 0, 2] > 128 > cool[0, 0, 0]
    assert warm[0, 1, 0] == cool[0, 1, 2] == 255
    for kelvin, result in [(3000, warm), (15000, cool)]:
        np.testing.assert_array_equal(result[..., 1], pixels[..., 1])
        np.testing.assert_array_equal(result[0, 2], 0)
        floating = tint(kelvin, floating=True)
        assert floating.dtype == np.float32
        np.testing.assert_allclose(floating * 255, result, atol=0.5)


def test_planckian_jitter_seed_and_validation(source):
    root, _ = source

    def augment(p):
        return p.planckian_jitter()

    serial = output(root, augment)
    np.testing.assert_array_equal(serial, output(root, augment, transform=True, workers=3))
    assert not np.array_equal(serial, output(root, augment, epoch=1))
    for interval in [(2999, 6500), (6500, 15001), (6500, 3000), (float("nan"), 6500)]:
        with pytest.raises(ValueError):
            rivet.Transform().planckian_jitter(interval)
    with pytest.raises(ValueError, match="PlanckianJitter"):
        output(root, lambda p: p.grayscale().planckian_jitter())
    with pytest.raises(ValueError, match="PlanckianJitter"):
        output(root, lambda p: p.hwc_to_chw().planckian_jitter())


@pytest.mark.parametrize("shape", [(5, 7, 3), (7, 5, 1), (1, 5, 3), (5, 1, 1), (1, 1, 3)])
def test_median_blur_matches_replicated_border_reference(tmp_path, shape):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.random.default_rng(91).integers(0, 256, shape, dtype=np.uint8)
    grayscale = shape[-1] == 1
    encoded = np.repeat(pixels, 3, axis=-1) if grayscale else pixels
    Image.fromarray(encoded).save(folder / "image.png")

    def augment(p):
        return (p.grayscale() if grayscale else p).median_blur()

    padded = np.pad(pixels, ((1, 1), (1, 1), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(0, 1))
    expected = np.median(windows, axis=(-2, -1)).astype(np.uint8)
    for transform, workers in [(False, 0), (True, 3)]:
        actual = output(tmp_path, augment, transform=transform, workers=workers)[0]
        assert actual.dtype == np.uint8
        np.testing.assert_array_equal(actual, expected)


def test_median_blur_rejects_incompatible_inputs(source):
    root, _ = source
    with pytest.raises(ValueError, match="MedianBlur"):
        output(root, lambda p: p.to_dtype("float32", True).median_blur())
    with pytest.raises(ValueError, match="MedianBlur"):
        output(root, lambda p: p.hwc_to_chw().median_blur())


@pytest.mark.parametrize("shape", [(5, 7, 3), (1, 4, 1), (1, 1, 3)])
@pytest.mark.parametrize("floating", [False, True])
def test_box_blur_reflects_without_repeating_edges(tmp_path, shape, floating):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.random.default_rng(27).integers(0, 256, shape, dtype=np.uint8)
    gray = shape[-1] == 1
    Image.fromarray(np.repeat(pixels, 3, axis=-1) if gray else pixels).save(
        folder / "image.png"
    )

    def augment(p):
        if gray:
            p = p.grayscale()
        if floating:
            p = p.to_dtype("float32", True)
        return p.blur((5, 5))

    padded = np.pad(pixels, ((2, 2), (2, 2), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (5, 5), axis=(0, 1))
    expected = windows.mean(axis=(-2, -1))
    actual = output(tmp_path, augment, transform=True, workers=2)[0]
    if floating:
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected / 255, atol=1e-7)
    else:
        assert actual.dtype == np.uint8
        np.testing.assert_array_equal(actual, expected.round().astype(np.uint8))


def test_box_blur_seed_and_validation(source):
    root, _ = source
    serial = output(root, lambda p: p.blur())
    parallel = output(root, lambda p: p.blur(), transform=True, workers=3)
    np.testing.assert_array_equal(serial, parallel)
    assert not np.array_equal(serial, output(root, lambda p: p.blur(), epoch=1))
    for bounds in [(2, 5), (3, 4), (7, 3), (3, 129)]:
        with pytest.raises(ValueError):
            rivet.Transform().blur(bounds)
    with pytest.raises(ValueError, match="Blur"):
        output(root, lambda p: p.hwc_to_chw().blur())


def test_advanced_blur_rotated_kernel_matches_reference(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.zeros((9, 9, 3), dtype=np.uint8)
    pixels[4, 4] = [255, 128, 0]
    Image.fromarray(pixels).save(folder / "image.png")
    y, x = np.mgrid[-2:3, -2:3]
    k = np.exp(-0.5 * ((y / 2) ** 2 + (x / 0.5) ** 2))
    k /= k.sum()
    for floating in [False, True]:
        def augment(p):
            if floating:
                p = p.to_dtype("float32", True)
            return p.advanced_blur((5, 5), (2, 2), (0.5, 0.5), (90, 90), (1, 1), (1, 1))

        actual = output(tmp_path, augment, transform=True, workers=2)[0]
        expected = np.zeros(pixels.shape, dtype=np.float64)
        expected[2:7, 2:7] = k[..., None] * pixels[4, 4]
        if floating:
            assert actual.dtype == np.float32
            np.testing.assert_allclose(actual, expected / 255, atol=1e-7)
        else:
            assert actual.dtype == np.uint8
            np.testing.assert_array_equal(actual, expected.round().astype(np.uint8))


def test_advanced_blur_seed_and_validation(source):
    root, _ = source
    serial = output(root, lambda p: p.advanced_blur())
    parallel = output(root, lambda p: p.advanced_blur(), transform=True, workers=3)
    np.testing.assert_array_equal(serial, parallel)
    assert not np.array_equal(serial, output(root, lambda p: p.advanced_blur(), epoch=1))
    for params in [
        {"kernel_range": (2, 3)},
        {"sigma_x_range": (0, 1)},
        {"rotate_range": (20, -20)},
        {"beta_range": (1, float("inf"))},
        {"noise_range": (-1, 1)},
    ]:
        with pytest.raises(ValueError):
            rivet.Transform().advanced_blur(**params)
    with pytest.raises(ValueError, match="AdvancedBlur"):
        output(root, lambda p: p.hwc_to_chw().advanced_blur())


@pytest.mark.parametrize("floating", [False, True])
def test_stochastic_convolution_channel_sharing_and_identity(tmp_path, floating):
    folder = tmp_path / "class"
    folder.mkdir()
    gray = np.random.default_rng(83).integers(64, 192, (8, 9), dtype=np.uint8)
    pixels = np.repeat(gray[..., None], 3, axis=-1)
    Image.fromarray(pixels).save(folder / "image.png")

    def apply(strength, per_channel=False):
        def augment(p):
            if floating:
                p = p.to_dtype("float32", True)
            return p.stochastic_convolution((3, 3), (strength, strength), per_channel=per_channel)

        return output(tmp_path, augment, transform=True, workers=2)[0]

    identity = apply(0)
    shared = apply(0.35)
    independent = apply(0.35, per_channel=True)
    if floating:
        assert identity.dtype == shared.dtype == independent.dtype == np.float32
        np.testing.assert_allclose(identity, pixels / 255, atol=1e-7)
        assert 0 <= independent.min() <= independent.max() <= 1
    else:
        assert identity.dtype == shared.dtype == independent.dtype == np.uint8
        np.testing.assert_array_equal(identity, pixels)
    np.testing.assert_array_equal(shared[..., 0], shared[..., 1])
    assert not np.array_equal(identity, shared)
    assert not np.array_equal(independent[..., 0], independent[..., 1])


def test_stochastic_convolution_seed_and_padding(source):
    root, _ = source

    def augment(p, padding="reflect"):
        return p.stochastic_convolution((3, 3), (0.25, 0.25), padding_mode=padding)

    serial = output(root, augment)
    np.testing.assert_array_equal(serial, output(root, augment, transform=True, workers=3))
    assert not np.array_equal(serial, output(root, augment, epoch=1))
    for padding in ["constant", "edge", "symmetric"]:
        alternate = output(root, lambda p: augment(p, padding))
        for a, b in zip(serial, alternate):
            np.testing.assert_array_equal(a[1:-1, 1:-1], b[1:-1, 1:-1])
        assert not np.array_equal(serial, alternate)
    for bounds in [(-0.1, 1), (1, 0), (0, float("inf"))]:
        with pytest.raises(ValueError):
            rivet.Transform().stochastic_convolution(strength_range=bounds)
    with pytest.raises(ValueError, match="StochasticConvolution"):
        output(root, lambda p: augment(p.hwc_to_chw()))


@pytest.mark.parametrize("floating", [False, True])
def test_zoom_blur_averages_centered_samples(tmp_path, floating):
    folder = tmp_path / "class"
    folder.mkdir()
    ramp = np.arange(5, dtype=np.uint8) * 40
    pixels = np.tile(ramp[None, :, None], (1, 1, 3))
    Image.fromarray(pixels).save(folder / "ramp.png")

    def augment(p):
        p = p.grayscale()
        if floating:
            p = p.to_dtype("float32", True)
        return p.zoom_blur((2, 2), (0.5, 0.5))

    actual = output(tmp_path, augment, transform=True, workers=2)[0]
    # For a linear ramp, bilinear interpolation evaluates the ramp at the
    # mean of the original, 1.5x, and 2x source coordinates.
    expected = 80 + (ramp.astype(float) - 80) * (1 + 1 / 1.5 + 1 / 2) / 3
    assert actual.shape == (1, 5, 1)
    if floating:
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual[0, :, 0], expected / 255, atol=1e-7)
    else:
        assert actual.dtype == np.uint8
        np.testing.assert_array_equal(actual[0, :, 0], expected.round())


def test_zoom_blur_identity_seed_and_validation(source):
    root, pixels = source
    identity = output(root, lambda p: p.zoom_blur((1, 1)))
    np.testing.assert_array_equal(identity[0], pixels)
    serial = output(root, lambda p: p.zoom_blur())
    np.testing.assert_array_equal(
        serial, output(root, lambda p: p.zoom_blur(), transform=True, workers=3)
    )
    assert not np.array_equal(serial, output(root, lambda p: p.zoom_blur(), epoch=1))
    for max_factor, step in [((0.9, 1), (0.1, 0.2)), ((1, 2), (0, 0.1)), ((1, 3), (0.001, 0.01))]:
        with pytest.raises(ValueError):
            rivet.Transform().zoom_blur(max_factor, step)
    with pytest.raises(ValueError, match="ZoomBlur"):
        output(root, lambda p: p.hwc_to_chw().zoom_blur())


def test_glass_blur_swaps_whole_pixels_reproducibly(source):
    root, pixels = source

    def augment(p):
        return p.glass_blur(sigma=0, max_delta=2, iterations=2)

    serial = output(root, augment)
    np.testing.assert_array_equal(serial, output(root, augment, transform=True, workers=3))
    assert not np.array_equal(serial, output(root, augment, epoch=1))
    assert not np.array_equal(serial[0], pixels)
    for i, image in enumerate(serial):
        original = np.roll(pixels, i, axis=0)
        assert image.shape == original.shape and image.dtype == np.uint8
        assert sorted(map(tuple, image.reshape(-1, 3))) == sorted(
            map(tuple, original.reshape(-1, 3))
        )
    smoothed = output(root, lambda p: p.glass_blur())[0]
    assert not np.array_equal(smoothed, serial[0])
    assert not np.array_equal(smoothed, pixels)


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 5, 3), (5, 1, 3)])
def test_glass_blur_preserves_uniform_small_float_images(tmp_path, shape):
    folder = tmp_path / "class"
    folder.mkdir()
    Image.fromarray(np.full(shape, 115, dtype=np.uint8)).save(folder / "uniform.png")
    result = output(
        tmp_path, lambda p: p.grayscale().to_dtype("float32", True).glass_blur()
    )[0]
    assert result.dtype == np.float32 and result.shape == (*shape[:2], 1)
    np.testing.assert_allclose(result, 115 / 255, atol=1e-7)


def test_glass_blur_validation(source):
    root, _ = source
    for kwargs in [{"sigma": -0.1}, {"sigma": float("nan")}, {"max_delta": 0}, {"iterations": 0}]:
        with pytest.raises(ValueError):
            rivet.Transform().glass_blur(**kwargs)
    with pytest.raises(ValueError, match="GlassBlur"):
        output(root, lambda p: p.hwc_to_chw().glass_blur())


def test_iso_noise_luma_and_chroma_are_independent(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.broadcast_to(np.array([120, 145, 165], dtype=np.uint8), (8, 11, 3))
    Image.fromarray(pixels).save(folder / "image.png")

    def run(color, intensity):
        return output(
            tmp_path,
            lambda p: p.to_dtype("float32", True).iso_noise(
                (color, color), (intensity, intensity)
            ),
            transform=True,
            workers=3,
        )[0]

    np.testing.assert_allclose(run(1, 0), pixels / 255, atol=1e-7)
    luma_only = run(0, 0.2)
    both = run(0.2, 0.2)
    delta = luma_only - pixels / 255
    assert delta.std() > 0
    np.testing.assert_allclose(delta[..., 0], delta[..., 1], atol=1e-7)
    np.testing.assert_allclose(delta[..., 0], delta[..., 2], atol=1e-7)
    weights = np.array([0.299, 0.587, 0.114])
    np.testing.assert_allclose(both @ weights, luma_only @ weights, atol=1e-7)
    assert not np.array_equal(both, luma_only)


def test_iso_noise_seed_and_validation(source):
    root, pixels = source
    serial = output(root, lambda p: p.iso_noise())
    parallel = output(root, lambda p: p.iso_noise(), transform=True, workers=3)
    np.testing.assert_array_equal(serial, parallel)
    assert serial[0].dtype == np.uint8 and serial[0].shape == pixels.shape
    assert not np.array_equal(serial, output(root, lambda p: p.iso_noise(), epoch=1))
    noisy = output(root, lambda p: p.to_dtype("float32", True).iso_noise((1, 1), (1, 1)))
    assert 0 <= np.min(noisy) <= np.max(noisy) <= 1
    for kwargs in [{"color_shift_range": (-1, 0)}, {"intensity_range": (0, 2)}]:
        with pytest.raises(ValueError):
            rivet.Transform().iso_noise(**kwargs)
    with pytest.raises(ValueError, match="ISONoise"):
        output(root, lambda p: p.grayscale().iso_noise())
    with pytest.raises(ValueError, match="ISONoise"):
        output(root, lambda p: p.hwc_to_chw().iso_noise())


def test_film_grain_scale_and_shadow_strength(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.full((7, 9, 3), 64, dtype=np.uint8)
    pixels[:, 4:] = 192
    Image.fromarray(pixels).save(folder / "grain.png")

    def apply(size, strength=0.05, per_channel=False):
        return output(
            tmp_path,
            lambda p: p.to_dtype("float32", True).film_grain(
                (strength, strength), (size, size), per_channel=per_channel
            ),
            transform=True,
            workers=2,
        )[0]

    base = apply(1, strength=0)
    np.testing.assert_allclose(base, pixels / 255, atol=1e-7)
    coarse = apply(20)
    residual = (coarse - base) / (1 - base)
    np.testing.assert_allclose(residual, residual[0, 0, 0], atol=2e-7)
    assert abs(coarse[0, 0, 0] - base[0, 0, 0]) > abs(coarse[0, -1, 0] - base[0, -1, 0])
    assert np.std(apply(1) - base) > 0
    chromatic = apply(20, per_channel=True)
    assert not np.array_equal(chromatic[..., 0], chromatic[..., 1])
    clipped = apply(1, strength=1, per_channel=True)
    assert clipped.dtype == np.float32 and clipped.shape == pixels.shape
    assert clipped.min() == 0 and clipped.max() == 1


def test_film_grain_seed_and_validation(source):
    root, pixels = source
    serial = output(root, lambda p: p.film_grain())
    np.testing.assert_array_equal(
        serial, output(root, lambda p: p.film_grain(), transform=True, workers=3)
    )
    assert serial[0].dtype == np.uint8 and serial[0].shape == pixels.shape
    assert not np.array_equal(serial, output(root, lambda p: p.film_grain(), epoch=1))
    for options in [{"intensity_range": (0, 2)}, {"grain_size_range": (0, 1)}]:
        with pytest.raises(ValueError):
            rivet.Transform().film_grain(**options)
    with pytest.raises(ValueError, match="FilmGrain"):
        output(root, lambda p: p.hwc_to_chw().film_grain())


@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("elementwise", [False, True])
def test_multiplicative_noise_sharing(tmp_path, per_channel, elementwise):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.full((5, 7, 3), 120, dtype=np.uint8)
    pixels[0, 0] = 0
    Image.fromarray(pixels).save(folder / "image.png")

    def augment(p):
        return p.to_dtype("float32", True).multiplicative_noise(
            (0.5, 1.5), per_channel=per_channel, elementwise=elementwise
        )

    image = output(tmp_path, augment)[0]
    np.testing.assert_array_equal(
        image, output(tmp_path, augment, transform=True, workers=3)[0]
    )
    assert image.dtype == np.float32 and image.shape == pixels.shape
    np.testing.assert_array_equal(image[0, 0], 0)
    assert (not np.array_equal(image[1, 1], image[1, 2])) == elementwise
    assert (image[1, 1, 0] != image[1, 1, 1]) == per_channel
    assert 0.5 * 120 / 255 <= image[1:].min() <= image[1:].max() <= 1.5 * 120 / 255


def test_multiplicative_noise_identity_clipping_and_validation(source):
    root, pixels = source
    identity = output(root, lambda p: p.multiplicative_noise((1, 1)))
    np.testing.assert_array_equal(identity[0], pixels)
    doubled = output(root, lambda p: p.multiplicative_noise((2, 2)))
    np.testing.assert_array_equal(doubled[0], (2 * pixels.astype(int)).clip(0, 255))
    assert doubled[0].dtype == np.uint8
    assert not np.array_equal(
        output(root, lambda p: p.multiplicative_noise()),
        output(root, lambda p: p.multiplicative_noise(), epoch=1),
    )
    for bounds in [(-1, 1), (2, 1), (0, float("inf"))]:
        with pytest.raises(ValueError):
            rivet.Transform().multiplicative_noise(bounds)
    with pytest.raises(ValueError, match="MultiplicativeNoise"):
        output(root, lambda p: p.hwc_to_chw().multiplicative_noise())


@pytest.mark.parametrize("distribution", ["uniform", "gaussian", "laplace", "beta"])
def test_additive_noise_distributions_and_seeding(tmp_path, distribution):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.full((9, 11, 3), 128, dtype=np.uint8)
    for i in range(4):
        Image.fromarray(pixels).save(folder / f"{i}.png")

    def augment(p):
        return p.to_dtype("float32", True).additive_noise(distribution, "shared")

    images = output(tmp_path, augment)
    np.testing.assert_array_equal(
        images, output(tmp_path, augment, transform=True, workers=3)
    )
    assert not np.array_equal(images, output(tmp_path, augment, epoch=1))
    assert images[0].dtype == np.float32 and images[0].shape == pixels.shape
    np.testing.assert_array_equal(images[0][..., 0], images[0][..., 2])
    assert images[0].std() > 0
    assert 0 <= np.min(images) <= np.max(images) <= 1


def test_additive_noise_constant_channel_offsets_and_validation(source):
    root, pixels = source

    def augment(p):
        return p.additive_noise(ranges=[(0.1, 0.1), (0, 0), (-0.1, -0.1)])

    actual = output(root, augment, transform=True)[0]
    expected = np.floor(pixels.astype(float) + np.array([25.5, 0, -25.5]) + 0.5)
    np.testing.assert_allclose(actual, expected.clip(0, 255), atol=1)
    assert actual.dtype == np.uint8
    with pytest.raises(ValueError):
        rivet.Transform().additive_noise("beta", alpha_range=(0, 1))
    with pytest.raises(ValueError):
        rivet.Transform().additive_noise("uniform", "shared", ranges=[(0, 0), (0, 0)])
    with pytest.raises(ValueError, match="AdditiveNoise"):
        output(root, lambda p: p.additive_noise(ranges=[(0, 0), (0, 0)]))
    with pytest.raises(ValueError, match="AdditiveNoise"):
        output(root, lambda p: p.hwc_to_chw().additive_noise())


def test_additive_noise_patch_union_and_seed(source):
    root, pixels = source

    def augment(p):
        return p.additive_noise(
            spatial_mode="patch",
            ranges=[(1, 1)],
            patch_count_range=(1, 1),
            patch_height_range=(0.25, 0.25),
            patch_width_range=(0.25, 0.25),
        )

    serial = output(root, augment)
    np.testing.assert_array_equal(serial, output(root, augment, transform=True, workers=3))
    changed = np.any(serial[0] != pixels, axis=-1)
    ys, xs = np.where(changed)
    assert len(ys) == 3 * 5
    assert np.ptp(ys) == 2 and np.ptp(xs) == 4
    np.testing.assert_array_equal(serial[0][changed], 255)
    with pytest.raises(ValueError):
        rivet.Transform().additive_noise(spatial_mode="patch", patch_height_range=(0, 1))


@pytest.mark.parametrize("mode", ["linear", "corner", "gaussian"])
def test_illumination_seed_and_gain_bounds(source, mode):
    root, pixels = source

    def augment(p):
        return p.illumination(mode, (0.3, 0.3), "darken")

    serial = output(root, augment)
    np.testing.assert_array_equal(serial, output(root, augment, transform=True, workers=3))
    expected = np.stack([np.roll(pixels, i, axis=0) for i in range(len(serial))])
    assert np.all(serial <= expected)
    assert np.all(serial >= np.round(expected * 0.7))
    assert not np.array_equal(serial, expected)
    np.testing.assert_array_equal(
        output(root, lambda p: p.illumination(mode, (0, 0))), expected
    )
    bright = output(
        root, lambda p: p.to_dtype("float32", True).illumination(mode, (1, 1), "brighten")
    )
    assert bright[0].dtype == np.float32
    assert np.all(np.asarray(bright) >= expected / 255 - 1e-7)
    assert np.max(bright) <= 1


@pytest.mark.parametrize("shadow", [False, True])
def test_plasma_light_fields_preserve_layout_and_seed(tmp_path, shadow):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.full((12, 17, 3), 128, dtype=np.uint8)
    for i in range(4):
        Image.fromarray(pixels).save(folder / f"{i}.png")

    def augment(p, amount=0.3):
        if shadow:
            return p.plasma_shadow((amount, amount), plasma_size=8)
        return p.plasma_brightness_contrast(
            (amount, amount), (0.2, 0.2) if amount else (0, 0), plasma_size=8
        )

    serial = output(tmp_path, augment)
    parallel = output(tmp_path, augment, transform=True, workers=3)
    np.testing.assert_array_equal(serial, parallel)
    assert not np.array_equal(serial, output(tmp_path, augment, epoch=1))
    np.testing.assert_array_equal(output(tmp_path, lambda p: augment(p, 0))[0], pixels)
    result = serial[0]
    assert result.dtype == np.uint8 and result.shape == pixels.shape
    assert np.ptp(result) > 0
    np.testing.assert_array_equal(result[..., 0], result[..., 2])
    if shadow:
        assert np.min(result) >= 90 and np.max(result) <= 128
    else:
        assert np.min(result) >= 128
    floats = output(tmp_path, lambda p: augment(p.grayscale().to_dtype("float32", True)))
    assert floats[0].shape == (12, 17, 1) and floats[0].dtype == np.float32
    np.testing.assert_allclose(floats[0][..., 0] * 255, result[..., 0], atol=0.51)


def test_random_shadow_roi_transmission_and_seed(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.full((31, 32, 3), 160, dtype=np.uint8)
    for i in range(4):
        Image.fromarray(pixels).save(folder / f"{i}.png")

    def darken(p, intensity=0.5):
        return p.random_shadow((0, 0.5, 1, 1), (3, 3), 5, (intensity, intensity))

    serial = output(tmp_path, darken)
    np.testing.assert_array_equal(serial, output(tmp_path, darken, transform=True, workers=3))
    assert not np.array_equal(serial, output(tmp_path, darken, epoch=1))
    image = serial[0]
    assert image.dtype == np.uint8 and image.shape == pixels.shape
    np.testing.assert_array_equal(image[:15], pixels[:15])
    np.testing.assert_array_equal(image[..., 0], image[..., 1])
    assert set(np.unique(image)) <= {160, 80, 40, 20}
    assert np.min(image) < 80
    np.testing.assert_array_equal(output(tmp_path, lambda p: darken(p, 0))[0], pixels)
    gray = output(tmp_path, lambda p: darken(p.grayscale().to_dtype("float32", True)))[0]
    assert gray.shape == (31, 32, 1) and gray.dtype == np.float32
    np.testing.assert_allclose(gray[..., 0] * 255, image[..., 0], atol=1e-5)
    with pytest.raises(ValueError):
        rivet.Transform().random_shadow(shadow_roi=(0, 1, 1, 0))


def test_random_sun_flare_source_color_and_radius(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.full((9, 9, 3), 40, dtype=np.uint8)
    Image.fromarray(pixels).save(folder / "image.png")

    def augment(p):
        return p.random_sun_flare((0.5, 0.5, 0.5, 0.5), (0, 0), (0, 0), 3, (255, 96, 48))

    image = output(tmp_path, augment, transform=True, workers=3)[0]
    np.testing.assert_array_equal(image[4, 4], [255, 96, 48])
    np.testing.assert_array_equal(image[0], pixels[0])
    assert image.dtype == np.uint8 and image.shape == pixels.shape


@pytest.mark.parametrize("optical", ["lens", "sun"])
def test_optical_flare_seed_and_float_range(source, optical):
    root, pixels = source

    def augment(p):
        p = p.to_dtype("float32", True)
        if optical == "lens":
            return p.lens_flare()
        return p.random_sun_flare(src_radius=4, physics_based=True)

    serial = output(root, augment)
    np.testing.assert_array_equal(serial, output(root, augment, transform=True, workers=3))
    assert not np.array_equal(serial, output(root, augment, epoch=1))
    assert serial[0].dtype == np.float32 and serial[0].shape == pixels.shape
    assert np.min(serial) >= 0 and np.max(serial) <= 1
    assert np.any(serial[0] > pixels / 255 + 1e-5)
    identity = output(root, lambda p: p.lens_flare(intensity_range=(0, 0)))
    np.testing.assert_array_equal(identity[0], pixels)
    with pytest.raises(ValueError, match="LensFlare"):
        output(root, lambda p: p.grayscale().lens_flare())


def test_random_rain_seed_and_rgb_blend(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.full((40, 48, 3), 100, dtype=np.uint8)
    for i in range(4):
        Image.fromarray(pixels).save(folder / f"{i}.png")

    def augment(p):
        return p.random_rain(
            slant_range=(0, 0), drop_length=8, drop_width=1, drop_color=(200, 100, 0),
            blur_value=1, brightness_coefficient=0.8,
        )

    serial = output(tmp_path, augment)
    np.testing.assert_array_equal(serial, output(tmp_path, augment, transform=True, workers=3))
    assert not np.array_equal(serial, output(tmp_path, augment, epoch=1))
    colors = np.unique(serial[0].reshape(-1, 3), axis=0)
    np.testing.assert_array_equal(colors, [[80, 80, 80], [112, 80, 48]])
    floats = output(tmp_path, lambda p: augment(p.to_dtype("float32", True)))
    assert floats[0].shape == pixels.shape and floats[0].dtype == np.float32
    np.testing.assert_allclose(floats[0] * 255, serial[0], atol=1e-5)
    with pytest.raises(ValueError, match="RandomRain"):
        output(tmp_path, lambda p: p.grayscale().random_rain())
    with pytest.raises(ValueError):
        rivet.Transform().random_rain(blur_value=2)


def test_random_rain_single_pixel(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(folder / "image.png")
    image = output(tmp_path, lambda p: p.random_rain())[0]
    assert image.shape == (1, 1, 3) and image.dtype == np.uint8
    np.testing.assert_array_equal(image, 56)


def test_random_snow_bleach_threshold_and_dtype(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.array([[[32] * 3, [96] * 3, [160] * 3]], dtype=np.uint8)
    Image.fromarray(pixels).save(folder / "image.png")

    def augment(p):
        return p.random_snow((0.25, 0.25), brightness_coeff=2)

    actual = output(tmp_path, augment, transform=True)[0]
    assert actual.dtype == np.uint8 and actual.shape == pixels.shape
    np.testing.assert_array_equal(actual[0, :, 0], [32, 192, 255])
    floats = output(tmp_path, lambda p: augment(p.to_dtype("float32", True)))[0]
    assert floats.dtype == np.float32
    np.testing.assert_allclose(floats * 255, actual, atol=1e-5)


def test_random_snow_texture_depth_seed_and_identity(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.full((23, 17, 3), 70, dtype=np.uint8)
    for i in range(4):
        Image.fromarray(pixels).save(folder / f"{i}.png")

    def augment(p):
        return p.random_snow((0.6, 0.6), method="texture")

    serial = output(tmp_path, augment)
    np.testing.assert_array_equal(serial, output(tmp_path, augment, transform=True, workers=3))
    assert not np.array_equal(serial, output(tmp_path, augment, epoch=1))
    assert serial[0][0].mean() > serial[0][-1].mean()
    assert serial[0][..., 2].mean() > serial[0][..., 0].mean()
    identity = output(tmp_path, lambda p: p.random_snow((0, 0), method="texture"))
    np.testing.assert_array_equal(identity[0], pixels)
    with pytest.raises(ValueError, match="RandomSnow"):
        output(tmp_path, lambda p: p.grayscale().random_snow())


def test_random_gravel_roi_seed_and_dtype(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    pixels = np.full((33, 41, 3), 128, dtype=np.uint8)
    for i in range(4):
        Image.fromarray(pixels).save(folder / f"{i}.png")

    def augment(p):
        return p.random_gravel((0.25, 0.5, 0.75, 0.875), 4)

    serial = output(tmp_path, augment)
    np.testing.assert_array_equal(serial, output(tmp_path, augment, transform=True, workers=3))
    assert not np.array_equal(serial, output(tmp_path, augment, epoch=1))
    result = serial[0]
    assert result.shape == pixels.shape and result.dtype == np.uint8
    roi = np.zeros(pixels.shape[:2], dtype=bool)
    roi[16:29, 10:31] = True
    np.testing.assert_array_equal(result[~roi], pixels[~roi])
    assert np.any(result[roi] != pixels[roi])
    floats = output(tmp_path, lambda p: augment(p.to_dtype("float32", True)))[0]
    assert floats.dtype == np.float32
    np.testing.assert_allclose(floats * 255, result, atol=0.51)
    np.testing.assert_array_equal(
        output(tmp_path, lambda p: p.random_gravel(number_of_patches=0))[0], pixels
    )
    with pytest.raises(ValueError, match="RandomGravel"):
        output(tmp_path, lambda p: p.grayscale().random_gravel())
    with pytest.raises(ValueError):
        rivet.Transform().random_gravel((0, 1, 1, 0))
