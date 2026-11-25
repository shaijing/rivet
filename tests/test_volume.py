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
