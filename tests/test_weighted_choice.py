"""Weighted composition behavior on both public Python entry points."""

import numpy as np
import pytest
from PIL import Image

from rivet import vision as rv


@pytest.fixture
def image_folder(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    for index in range(48):
        Image.fromarray(np.full((5, 7, 3), index, dtype=np.uint8)).save(
            folder / f"{index:03}.png"
        )
    return tmp_path


def collect(root, weights, *, nested=False, workers=0, epoch=0):
    pipeline = (
        rv.scan_image_folder(root).decode_image().seed(41).epoch(epoch).workers(workers)
    )
    choices = [rv.Transform(), rv.Transform().invert()]
    if nested:
        pipeline = pipeline.compose(rv.Transform().random_choice(choices, p=weights))
    else:
        pipeline = pipeline.random_choice(choices, p=weights)
    return np.concatenate([batch["images"] for batch in pipeline.batch(8).execute()])


@pytest.mark.parametrize("nested", [False, True])
def test_zero_weights_never_select_a_branch(image_folder, nested):
    original = collect(image_folder, [1, 0], nested=nested)
    inverted = collect(image_folder, [0, 4], nested=nested)
    np.testing.assert_array_equal(inverted, 255 - original)
    np.testing.assert_array_equal(original[:, 0, 0, 0], np.arange(48))


@pytest.mark.parametrize("nested", [False, True])
def test_relative_weights_and_worker_reproducibility(image_folder, nested):
    expected = collect(image_folder, [1, 3], nested=nested)
    np.testing.assert_array_equal(
        expected, collect(image_folder, [2, 6], nested=nested)
    )
    np.testing.assert_array_equal(
        expected, collect(image_folder, [1, 3], nested=nested, workers=4)
    )
    changed = collect(image_folder, [1, 3], nested=nested, epoch=1)
    assert not np.array_equal(expected, changed)
    np.testing.assert_array_equal(
        changed, collect(image_folder, [1, 3], nested=nested, epoch=1, workers=4)
    )
    pixels = expected[:, 0, 0, 0]
    assert np.any(pixels < 128)
    assert np.any(pixels > 128)


@pytest.mark.parametrize("nested", [False, True])
def test_omitted_weights_preserve_uniform_sampler(image_folder, nested):
    source = rv.scan_image_folder(image_folder).decode_image().seed(41)
    choices = [rv.Transform(), rv.Transform().invert()]
    transform = rv.Transform().random_choice(choices)
    default = source.compose(transform) if nested else source.random_choice(choices)
    actual = np.concatenate([b["images"] for b in default.batch(8).execute()])
    np.testing.assert_array_equal(actual, collect(image_folder, None, nested=nested))


@pytest.mark.parametrize(
    "weights",
    [
        [],
        [1],
        [1, 2, 3],
        [-1, 2],
        [0, 0],
        [float("nan"), 1],
        [float("inf"), 1],
        [1e308, 1e308],
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_invalid_weights_are_rejected(image_folder, weights, nested):
    owner = rv.Transform() if nested else rv.scan_image_folder(image_folder)
    with pytest.raises(ValueError, match="weights"):
        owner.random_choice([rv.Transform(), rv.Transform().invert()], p=weights)


@pytest.mark.parametrize("owner", [rv.Transform(), None])
def test_empty_choices_are_rejected(image_folder, owner):
    owner = owner if owner is not None else rv.scan_image_folder(image_folder)
    with pytest.raises(ValueError, match="at least one"):
        owner.random_choice([])
