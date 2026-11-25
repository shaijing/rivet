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
