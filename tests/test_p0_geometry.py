"""Differential coverage for extended geometry arguments on both APIs."""

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
        np.random.default_rng(13).integers(0, 256, (17, 23, 3), dtype=np.uint8)
    )
    image.save(folder / "sample.png")
    return tmp_path, image


def output(source, build, nested):
    pipeline = rv.scan_image_folder(source[0]).decode_image()
    pipeline = pipeline.compose(build(rv.Transform())) if nested else build(pipeline)
    return next(pipeline.batch(1).execute())["images"][0]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("angle", [27, 90])
@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
def test_rotate_center_and_rgb_fill(source, nested, expand, angle, mode):
    options = {"center": [2.0, 8.0], "expand": expand, "fill": [17, 91, 203]}
    expected = np.asarray(
        F.rotate(
            source[1],
            angle,
            interpolation=getattr(F.InterpolationMode, mode.upper()),
            **options,
        )
    )
    actual = output(
        source,
        lambda p: p.arbitrary_rotate(angle, interpolation=mode, **options),
        nested,
    )
    np.testing.assert_array_equal(actual, expected)
    sampled = output(
        source,
        lambda p: p.random_rotation([angle, angle], interpolation=mode, **options),
        nested,
    )
    np.testing.assert_array_equal(sampled, expected)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
def test_random_affine_fixed_intervals_match_functional(source, nested, mode):
    actual = output(
        source,
        lambda p: p.random_affine(
            [17, 17],
            shear_ranges=[5, 5, -3, -3],
            interpolation=mode,
            fill=[17, 91, 203],
        ),
        nested,
    )
    expected = F.affine(
        source[1],
        17,
        [0, 0],
        1,
        [5, -3],
        interpolation=getattr(F.InterpolationMode, mode.upper()),
        fill=[17, 91, 203],
    )
    np.testing.assert_array_equal(actual, np.asarray(expected))


@pytest.mark.parametrize("nested", [False, True])
def test_perspective_rgb_fill(source, nested):
    start = [[0, 0], [22, 0], [22, 16], [0, 16]]
    end = [[2, 1], [20, 2], [19, 14], [1, 15]]
    actual = output(
        source, lambda p: p.perspective(start, end, fill=[17, 91, 203]), nested
    )
    expected = F.perspective(source[1], start, end, fill=[17, 91, 203])
    np.testing.assert_array_equal(actual, np.asarray(expected))


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "build",
    [
        lambda p: p.arbitrary_rotate(20, center=[0, 0, 0]),
        lambda p: p.arbitrary_rotate(20, center=[float("nan"), 1]),
        lambda p: p.random_rotation(20, center=[0, float("inf")]),
        lambda p: p.random_affine([2, -2]),
        lambda p: p.random_affine([0, float("inf")]),
        lambda p: p.random_affine(20, shear_ranges=[2, -2]),
        lambda p: p.random_affine(20, shear_ranges=[1, 2, 3]),
        lambda p: p.random_affine(20, shear=[1, 2], shear_ranges=[-2, 2]),
        lambda p: p.pad(1, fill=[1, 2]),
        lambda p: p.arbitrary_rotate(20, fill=[1, 2]),
    ],
)
def test_invalid_geometry_parameters(source, nested, build):
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        output(source, build, nested)
