import numpy as np
import pytest

from rivet import VolumeSample


def test_resize3d_trilinear_half_pixel_grid_and_xyz_keypoints():
    z, y, x = np.indices((2, 3, 4), dtype=np.float32)
    image = np.stack([100*z + 10*y + x, x-z], axis=-1)
    mask = np.arange(24, dtype=np.int64).reshape(2, 3, 4) + 2**53
    sample = VolumeSample(image, mask=mask, keypoints=np.array([[1, 1, 0]]), label=7)
    output = sample.resize3d((3, 4, 5)).to_dict()
    coords = [(np.arange(n) + 0.5) * d / n for n, d in zip((3, 4, 5), (2, 3, 4))]
    z, y, x = np.meshgrid(*(np.clip(p - 0.5, 0, d-1) for p, d in zip(coords, (2, 3, 4))), indexing="ij")
    np.testing.assert_allclose(output["volume"], np.stack([100*z + 10*y + x, x-z], axis=-1), atol=2e-5)
    np.testing.assert_array_equal(output["mask"], mask[np.ix_(*(p.astype(int) for p in coords))])
    np.testing.assert_allclose(output["keypoints"], [[1.375, 1.5, 0.25]])
    assert output["volume"].dtype == np.float32 and output["mask"].dtype == np.int64
    assert output["label"] == 7


def test_resize3d_nearest_grayscale_and_singleton():
    gray = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
    out = VolumeSample(gray).resize3d((2, 1, 1), "nearest").to_dict()
    np.testing.assert_array_equal(out["volume"], np.full((2, 1, 1), 40, dtype=np.uint8))
    with pytest.raises(ValueError):
        VolumeSample(gray, mask=np.zeros((2, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        VolumeSample(gray).resize3d((0, 2, 2))
    with pytest.raises(ValueError):
        VolumeSample(gray).resize3d((1, 2, 2), "cubic")


def test_pad3d_asymmetric_channels_masks_and_xyz():
    image = np.arange(24, dtype=np.uint8).reshape(2, 2, 3, 2)
    mask = np.full((2, 2, 3), 2**53 + 1, np.int64)
    sample = VolumeSample(image, mask=mask, keypoints=[[1, 0, 1]], label=9)
    result = sample.pad3d((1, 2, 3, 0, 2, 1), fill=[111, 222], fill_mask=-7).to_dict()
    expected = np.empty((5, 5, 6, 2), np.uint8)
    expected[:] = [111, 222]
    expected[1:3, 3:5, 2:5] = image
    np.testing.assert_array_equal(result["volume"], expected)
    np.testing.assert_array_equal(result["mask"], np.pad(mask, ((1, 2), (3, 0), (2, 1)), constant_values=-7))
    np.testing.assert_array_equal(result["keypoints"], [[3, 3, 2]])
    assert result["label"] == 9


def test_pad_if_needed3d_minimum_divisor_and_random_seed():
    image = np.ones((2, 3, 5), np.float32)
    sample = VolumeSample(image, mask=np.ones((2, 3, 5), np.uint8), keypoints=[[0, 0, 0]])
    centered = sample.pad_if_needed3d((3, 4, 7), (2, 3, 4), fill=0.25).to_dict()
    assert centered["volume"].shape == (4, 6, 8)
    np.testing.assert_array_equal(centered["volume"][1:3, 1:4, 1:6], image)
    np.testing.assert_array_equal(centered["keypoints"], [[1, 1, 1]])
    args = dict(min_zyx=(5, 6, 9), position="random", seed=73)
    a = sample.pad_if_needed3d(**args).to_dict()
    b = sample.pad_if_needed3d(**args).to_dict()
    np.testing.assert_array_equal(a["volume"], b["volume"])
    np.testing.assert_array_equal(a["mask"], b["mask"])
    x,y,z = a["keypoints"][0].astype(int)
    np.testing.assert_array_equal(a["volume"][z:z+2, y:y+3, x:x+5], image)
    with pytest.raises(ValueError):
        sample.pad_if_needed3d()
    with pytest.raises(ValueError):
        sample.pad_if_needed3d(pad_divisor_zyx=(0, 1, 2))
    with pytest.raises(ValueError):
        sample.pad3d(1, fill_mask=-1)
