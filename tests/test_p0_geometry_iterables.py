"""Iterable fill values are materialized before crossing the PyO3 boundary."""

import numpy as np
import pytest
from PIL import Image

from rivet import vision as rv

F = pytest.importorskip("torchvision.transforms.functional")


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    image = Image.fromarray(
        np.random.default_rng(31).integers(0, 256, (9, 13, 3), dtype=np.uint8)
    )
    image.save(folder / "sample.png")
    return tmp_path, image


def _run(source, nested, name, fill):
    pipeline = rv.scan_image_folder(source[0]).decode_image()
    build = (
        (lambda target: target.pad(2, fill=fill))
        if name == "pad"
        else (lambda target: target.arbitrary_rotate(23, center=[1, 5], fill=fill))
    )
    if nested:
        pipeline = pipeline.compose(build(rv.Transform()))
    else:
        pipeline = build(pipeline)
    return next(pipeline.batch(1).execute())["images"][0]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("name", ["pad", "arbitrary_rotate"])
def test_generator_channel_fill_is_supported(source, nested, name):
    fill = [17, 91, 203]
    if name == "pad":
        actual = _run(source, nested, name, (value for value in fill))
        expected = np.asarray(F.pad(source[1], [2], fill=fill))
    else:
        actual = _run(source, nested, name, (value for value in fill))
        expected = np.asarray(F.rotate(source[1], 23, center=[1, 5], fill=fill))
    np.testing.assert_array_equal(actual, expected)
