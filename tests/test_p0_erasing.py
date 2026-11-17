"""Focused API validation and RNG coverage for blur and random erasing."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import rivet


@pytest.fixture
def image_folder(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    y, x, c = np.indices((16, 18, 3))
    for index in range(4):
        pixels = ((17 * y + 29 * x + 53 * c + 37 * index) % 256).astype(np.uint8)
        Image.fromarray(pixels).save(folder / f"{index:02}.png")
    return tmp_path


def _base(root, *, workers=0, epoch=0):
    return (
        rivet.scan_image_folder(root)
        .decode_image()
        .seed(0xD15EA5E)
        .epoch(epoch)
        .workers(workers)
    )


@pytest.mark.parametrize("entrypoint", ["pipeline", "transform"])
@pytest.mark.parametrize("value", [[], [1.0, 2.0], [1.0, 2.0, 3.0, 4.0]])
def test_random_erasing_rejects_invalid_fill_channel_lengths(
    image_folder, entrypoint, value
):
    with pytest.raises((ValueError, RuntimeError)):
        if entrypoint == "pipeline":
            pipeline = _base(image_folder).random_erasing(
                probability=1.0,
                scale=[0.5, 0.5],
                ratio=[1.0, 1.0],
                value=value,
            )
        else:
            pipeline = _base(image_folder).compose(
                rivet.Transform().random_erasing(
                    probability=1.0,
                    scale=[0.5, 0.5],
                    ratio=[1.0, 1.0],
                    value=value,
                )
            )
        next(pipeline.batch(1).execute())


@pytest.mark.parametrize("value", [[float("nan")], [float("inf")], [float("-inf")]])
def test_random_erasing_rejects_nonfinite_fills(image_folder, value):
    with pytest.raises((ValueError, RuntimeError)):
        pipeline = _base(image_folder).random_erasing(probability=1.0, value=value)
        next(pipeline.batch(1).execute())


def test_random_erasing_rejects_bad_string_and_checks_channels_before_sampling(
    tmp_path,
):
    with pytest.raises((ValueError, RuntimeError)):
        rivet.Transform().random_erasing(value="noise")

    folder = tmp_path / "tiny"
    folder.mkdir()
    Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(folder / "tiny.png")
    # No rectangle can fit, but torchvision validates fill channels after the
    # probability check and before trying rectangle dimensions.
    pipeline = (
        _base(tmp_path)
        .random_erasing(
            probability=1.0,
            scale=[1.0, 1.0],
            ratio=[1.0, 1.0],
            value=[1.0, 2.0],
        )
        .batch(1)
    )
    with pytest.raises((ValueError, RuntimeError)):
        next(pipeline.execute())


def test_random_erasing_probability_zero_skips_channel_count_validation(image_folder):
    original = np.concatenate(
        [batch["images"] for batch in _base(image_folder).batch(2).execute()]
    )
    pipeline = _base(image_folder).random_erasing(probability=0.0, value=[1.0, 2.0])
    erased = np.concatenate([batch["images"] for batch in pipeline.batch(2).execute()])
    np.testing.assert_array_equal(erased, original)


@pytest.mark.parametrize(
    "build",
    [
        lambda p: p.gaussian_blur(1.0, kernel_size=0),
        lambda p: p.gaussian_blur(1.0, kernel_size=[3, 0]),
        lambda p: p.gaussian_blur(1.0, kernel_size=[2, 3]),
        lambda p: p.gaussian_blur(-0.1),
        lambda p: p.gaussian_blur([-0.1, 0.5], kernel_size=[3, 3]),
        lambda p: p.gaussian_blur(float("nan")),
        lambda p: p.random_gaussian_blur(4),
        lambda p: p.random_gaussian_blur([3, 5], sigma=[2.0, 0.1]),
        lambda p: p.random_gaussian_blur([3, 5], sigma=[0.1, 0.0]),
    ],
)
def test_blur_rejects_invalid_kernel_and_sigma(image_folder, build):
    with pytest.raises((ValueError, RuntimeError)):
        pipeline = build(_base(image_folder))
        next(pipeline.batch(1).execute())


def test_gaussian_blur_preserves_zero_sigma_identity(image_folder):
    original = np.concatenate(
        [batch["images"] for batch in _base(image_folder).batch(2).execute()]
    )
    scalar_zero = _base(image_folder).gaussian_blur(0.0)
    scalar_output = np.concatenate(
        [batch["images"] for batch in scalar_zero.batch(2).execute()]
    )
    np.testing.assert_array_equal(scalar_output, original)

    axis_zero = _base(image_folder).gaussian_blur([0.0, 0.8], kernel_size=[1, 3])
    axis_output = np.concatenate(
        [batch["images"] for batch in axis_zero.batch(2).execute()]
    )
    assert axis_output.shape == original.shape


@pytest.mark.parametrize("dtype", ["uint8", "float32"])
@pytest.mark.parametrize("layout", ["hwc", "chw"])
@pytest.mark.parametrize("entrypoint", ["pipeline", "transform"])
def test_random_erasing_random_fill_repeats_across_workers_and_epochs(
    image_folder, dtype, layout, entrypoint
):
    def collect(workers, epoch):
        pipeline = _base(image_folder, workers=workers, epoch=epoch)
        if dtype == "float32":
            pipeline = (
                pipeline.to_tensor()
                if layout == "chw"
                else pipeline.convert_image_dtype("float32")
            )
        elif layout == "chw":
            pipeline = pipeline.pil_to_tensor()
        options = {
            "probability": 1.0,
            "scale": [0.3, 0.3],
            "ratio": [1.0, 1.0],
            "value": "random",
        }
        operation = (
            pipeline.random_erasing(**options)
            if entrypoint == "pipeline"
            else pipeline.compose(rivet.Transform().random_erasing(**options))
        )
        return np.concatenate(
            [batch["images"] for batch in operation.batch(2).execute()]
        )

    baseline = collect(0, 2)
    np.testing.assert_array_equal(baseline, collect(0, 2))
    np.testing.assert_array_equal(baseline, collect(2, 2))
    assert not np.array_equal(baseline, collect(0, 3))
