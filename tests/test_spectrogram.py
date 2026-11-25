from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from rivet import SpectrogramSample


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [False, True])
def test_time_reverse(dtype, channels):
    data = np.arange(35, dtype=dtype).reshape(5, 7)
    if channels:
        data = np.stack([data, data + 1])
    sample = SpectrogramSample(data, label=19)
    out = sample.time_reverse().to_dict()
    np.testing.assert_array_equal(out["spectrogram"], data[..., ::-1])
    assert out["spectrogram"].dtype == dtype and out["label"] == 19
    np.testing.assert_array_equal(sample.time_reverse().time_reverse().to_dict()["spectrogram"], data)


@pytest.mark.parametrize("method,axis,reduce_axes", [
    ("time_masking", 2, (0, 1)),
    ("frequency_masking", 1, (0, 2)),
])
def test_seeded_mask_is_one_shared_band(method, axis, reduce_axes):
    data = np.arange(1, 71, dtype=np.float32).reshape(2, 5, 7)
    sample = SpectrogramSample(data, label=23)

    def run(seed):
        return getattr(sample, method)(30, seed=seed).to_dict()

    widths = set()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for seed, result in enumerate(pool.map(run, range(24))):
            out = result["spectrogram"]
            bands = np.flatnonzero(np.all(out == 0, axis=reduce_axes))
            widths.add(len(bands))
            if len(bands):
                np.testing.assert_array_equal(bands, np.arange(bands[0], bands[-1] + 1))
            expected = data.copy()
            index = [slice(None)] * 3
            index[axis] = bands
            expected[tuple(index)] = 0
            np.testing.assert_array_equal(out, expected)
            np.testing.assert_array_equal(out, run(seed)["spectrogram"])
            assert out.dtype == data.dtype and result["label"] == 23
    assert len(widths) > 1


def test_spectrogram_validation():
    for shape in ((7,), (2, 0), (1, 2, 3, 4)):
        with pytest.raises(ValueError):
            SpectrogramSample(np.ones(shape, np.float32))
    with pytest.raises(ValueError):
        SpectrogramSample(np.ones((3, 4), np.int64))
    with pytest.raises(ValueError):
        SpectrogramSample(np.ones((3, 4), np.uint8)).time_masking(0)
