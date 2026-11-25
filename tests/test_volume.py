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


def test_center_crop3d_mixed_padding_and_crop():
    volume = np.arange(2 * 5 * 7 * 2, dtype=np.uint8).reshape(2, 5, 7, 2)
    mask = volume[..., 0].astype(np.int64) + 2**53
    sample = VolumeSample(volume, mask=mask, keypoints=[[2, 1, 0], [0, 0, 0]], label=17)
    out = sample.center_crop3d((4, 3, 3), pad_if_needed=True, fill=[1, 2], fill_mask=-3).to_dict()
    expected = np.empty((4, 3, 3, 2), dtype=np.uint8)
    expected[:] = [1, 2]
    expected[1:3] = volume[:, 1:4, 2:5]
    np.testing.assert_array_equal(out["volume"], expected)
    np.testing.assert_array_equal(out["mask"], np.pad(mask[:, 1:4, 2:5], ((1, 1), (0, 0), (0, 0)), constant_values=-3))
    np.testing.assert_array_equal(out["keypoints"], [[0, 0, 1], [-2, -1, 1]])
    assert out["label"] == 17
    with pytest.raises(ValueError):
        sample.center_crop3d((4, 3, 3))
    with pytest.raises(ValueError):
        sample.center_crop3d((0, 3, 3))


def test_random_crop3d_seeded_window_and_mask():
    from concurrent.futures import ThreadPoolExecutor

    volume = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    mask = volume.astype(np.uint8)
    sample = VolumeSample(volume, mask=mask, keypoints=[[0, 0, 0]], label=23)

    def crop(seed):
        return sample.random_crop3d((2, 3, 4), seed=seed).to_dict()

    offsets = set()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for seed, out in enumerate(pool.map(crop, range(12))):
            x, y, z = -out["keypoints"][0].astype(int)
            offsets.add((x, y, z))
            np.testing.assert_array_equal(out["volume"], volume[z:z+2, y:y+3, x:x+4])
            np.testing.assert_array_equal(out["mask"], mask[z:z+2, y:y+3, x:x+4])
            np.testing.assert_array_equal(out["volume"], crop(seed)["volume"])
            assert out["label"] == 23
    assert len(offsets) > 1
    padded = sample.random_crop3d((6, 7, 8), pad_if_needed=True, seed=3).to_dict()
    x, y, z = padded["keypoints"][0].astype(int)
    np.testing.assert_array_equal(padded["volume"][z:z+4, y:y+5, x:x+6], volume)


def test_flip3d_and_rotate90_match_numpy():
    volume = np.arange(60, dtype=np.uint8).reshape(2, 3, 5, 2)
    mask = volume[..., 0].astype(np.int64) + 2**53
    sample = VolumeSample(volume, mask=mask, keypoints=[[3, 2, 1]], label=29)
    flipped = sample.flip3d(flip_axes=(0, 2)).to_dict()
    np.testing.assert_array_equal(flipped["volume"], np.flip(volume, (0, 2)))
    np.testing.assert_array_equal(flipped["mask"], np.flip(mask, (0, 2)))
    np.testing.assert_array_equal(flipped["keypoints"], [[1, 2, 0]])
    assert flipped["label"] == 29
    for pair in ((0, 1), (0, 2), (1, 2)):
        for turns, group in enumerate(("e", "r90", "r180", "r270")):
            out = sample.random_rotate90_3d(axis_pair=pair, group_element=group).to_dict()
            np.testing.assert_array_equal(out["volume"], np.rot90(volume, turns, pair))
            np.testing.assert_array_equal(out["mask"], np.rot90(mask, turns, pair))
            x, y, z = out["keypoints"][0].astype(int)
            assert out["mask"][z, y, x] == mask[1, 2, 3]
            assert out["label"] == 29
    with pytest.raises(ValueError):
        sample.flip3d((0,), flip_axes=(2,))
    with pytest.raises(ValueError):
        sample.random_rotate90_3d(axis_pair=(2, 0))


def test_all_cubic_symmetries_and_seeded_reorientation():
    from concurrent.futures import ThreadPoolExecutor

    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    sample = VolumeSample(volume, mask=volume.astype(np.uint8), keypoints=[[3, 2, 1]])
    orientations = set()
    for element in range(48):
        out = sample.cubic_symmetry(element=element).to_dict()
        orientations.add((out["volume"].shape, out["volume"].tobytes()))
        np.testing.assert_array_equal(out["volume"], out["mask"])
        x, y, z = out["keypoints"][0].astype(int)
        assert out["volume"][z, y, x] == 23
    assert len(orientations) == 48
    with pytest.raises(ValueError):
        sample.cubic_symmetry(element=48)

    def apply(seed):
        return tuple(
            op(seed=seed).to_dict()["volume"]
            for op in (sample.flip3d, sample.random_rotate90_3d, sample.cubic_symmetry)
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        for seed, results in enumerate(pool.map(apply, range(12))):
            for parallel, serial in zip(results, apply(seed), strict=True):
                np.testing.assert_array_equal(parallel, serial)


def test_grid_shuffle3d_uneven_cells_and_seed():
    from concurrent.futures import ThreadPoolExecutor

    z, y, x = np.indices((5, 6, 7))
    mask = (z * 42 + y * 7 + x).astype(np.int64) + 2**53
    volume = np.stack([mask - 2**53, 255 - (mask - 2**53)], axis=-1).astype(np.uint8)
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)
    points += 0.25
    points = np.concatenate([points, [[-1, 3, 2]]])
    sample = VolumeSample(volume, mask=mask, keypoints=points, label=37)

    def shuffle(seed):
        return sample.grid_shuffle3d((2, 3, 3), seed=seed).to_dict()

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(shuffle, range(4)))
    changed = False
    for seed, out in enumerate(outputs):
        xv, yv, zv = out["keypoints"][:-1].astype(int).T
        np.testing.assert_array_equal(out["mask"][zv, yv, xv], mask.ravel())
        np.testing.assert_array_equal(out["volume"][zv, yv, xv], volume.reshape(-1, 2))
        np.testing.assert_array_equal(out["keypoints"][:-1] % 1, points[:-1] % 1)
        np.testing.assert_array_equal(out["keypoints"][-1], points[-1])
        np.testing.assert_array_equal(out["volume"], shuffle(seed)["volume"])
        assert out["label"] == 37
        changed |= not np.array_equal(out["volume"], volume)
        # Every X-run in a cell is transplanted intact, including the short cells.
        for start, stop in ((0, 3), (3, 5), (5, 7)):
            np.testing.assert_array_equal(np.diff(out["mask"][..., start:stop], axis=2), 1)
    assert changed
    gray = VolumeSample(volume[..., 0].astype(np.float32))
    np.testing.assert_array_equal(
        gray.grid_shuffle3d((1, 1, 1)).to_dict()["volume"], volume[..., 0]
    )
    for grid in ((0, 2, 2), (6, 2, 2)):
        with pytest.raises(ValueError):
            sample.grid_shuffle3d(grid)


def test_coarse_dropout3d_synchronizes_overlapping_cuboids():
    from concurrent.futures import ThreadPoolExecutor

    mask = np.full((3, 5, 7), 2**53 + 1, dtype=np.int64)
    image = np.full((*mask.shape, 2), 127, dtype=np.uint8)
    z, y, x = np.indices(mask.shape)
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    sample = VolumeSample(image, mask=mask, keypoints=points, label=41)
    args = dict(
        num_holes_range=(3, 4), hole_depth_range=(0.6, 0.8),
        hole_height_range=(0.6, 0.8), hole_width_range=(0.6, 0.8), fill=(1, 2),
    )

    def run(seed):
        return sample.coarse_dropout3d(**args, fill_mask=-3, seed=seed).to_dict()

    with ThreadPoolExecutor(max_workers=4) as pool:
        for seed, out in enumerate(pool.map(run, range(8))):
            holes = out["mask"] == -3
            assert holes.any()
            np.testing.assert_array_equal(out["mask"][~holes], 2**53 + 1)
            np.testing.assert_array_equal(out["volume"][..., 0], np.where(holes, 1, 127))
            np.testing.assert_array_equal(out["volume"][..., 1], np.where(holes, 2, 127))
            np.testing.assert_array_equal(out["keypoints"], points[~holes.ravel()])
            np.testing.assert_array_equal(out["mask"], run(seed)["mask"])
            assert out["label"] == 41
    kept = sample.coarse_dropout3d(**args, remove_invisible=False).to_dict()
    np.testing.assert_array_equal(kept["mask"], mask)
    np.testing.assert_array_equal(kept["keypoints"], points)


def test_coarse_dropout3d_full_empty_and_invalid():
    sample = VolumeSample(np.ones((1, 2, 3), np.float32), keypoints=[[2, 1, 0]])
    opts = dict(hole_depth_range=(1, 1), hole_height_range=(1, 1), hole_width_range=(1, 1))
    out = sample.coarse_dropout3d(**opts, fill=-0.25).to_dict()
    np.testing.assert_array_equal(out["volume"], -0.25)
    assert out["keypoints"].shape == (0, 3)
    np.testing.assert_array_equal(sample.coarse_dropout3d((0, 0), **opts).to_dict()["volume"], 1)
    with pytest.raises(ValueError):
        sample.coarse_dropout3d(hole_depth_range=(0.5, 0.1))
    with pytest.raises(ValueError):
        sample.coarse_dropout3d(fill=(1, 2))


def test_anisotropy3d_filters_only_selected_axis_and_keeps_annotations():
    signal = np.array([0, 0, 9, 0, 0, 0], dtype=np.float32)
    image = np.broadcast_to(signal[:, None, None, None], (6, 2, 3, 2)).copy()
    image[..., 1] += 12
    mask = np.arange(36, dtype=np.int64).reshape(6, 2, 3) + 2**53
    sample = VolumeSample(image, mask=mask, keypoints=[[1, 1, 4]], label=47)
    args = dict(axes=(0,), downscale_factor_range=(3, 3))
    out = sample.anisotropy3d(**args).to_dict()
    expected = np.array([3, 3, 2, 1, 0, 0], dtype=np.float32)[:, None, None, None]
    np.testing.assert_allclose(out["volume"], np.broadcast_to(expected + [0, 12], image.shape))
    np.testing.assert_array_equal(out["mask"], mask)
    np.testing.assert_array_equal(out["keypoints"], [[1, 1, 4]])
    assert out["volume"].dtype == np.float32 and out["label"] == 47
    no_filter = sample.anisotropy3d(**args, antialias=False).to_dict()
    np.testing.assert_array_equal(no_filter["volume"][..., 0], 0)


def test_anisotropy3d_seeded_axes_on_uint8():
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(22)
    volume = rng.integers(0, 256, (5, 7, 9), dtype=np.uint8)
    sample = VolumeSample(volume)

    def run(seed):
        return sample.anisotropy3d(num_axes_range=(1, 3), seed=seed).to_dict()["volume"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(run, range(8)))
    assert len({out.tobytes() for out in outputs}) > 1
    for seed, out in enumerate(outputs):
        assert out.dtype == volume.dtype and out.shape == volume.shape
        np.testing.assert_array_equal(out, run(seed))
    for kwargs in ({"axes": (0, 0)}, {"axes": (1,), "num_axes_range": (1, 2)}, {"downscale_factor_range": (1, 2)}):
        with pytest.raises(ValueError):
            sample.anisotropy3d(**kwargs)


def test_affine3d_fractional_translation_and_border():
    sample = VolumeSample(
        np.array([[[0, 4, 8, 12]]], np.float32),
        mask=np.array([[[2**53 + 1, 2**53 + 2, 2**53 + 3, 2**53 + 4]]], np.int64),
        keypoints=[[2, 0, 0]], label=51,
    )
    translation = {"x": (0.125, 0.125), "y": (0, 0), "z": (0, 0)}
    out = sample.affine3d(translate_percent_range=translation, fill=10).to_dict()
    np.testing.assert_array_equal(out["volume"], [[[5, 2, 6, 10]]])
    np.testing.assert_array_equal(out["mask"], sample.to_dict()["mask"])
    np.testing.assert_array_equal(out["keypoints"], [[2.5, 0, 0]])
    assert out["label"] == 51
    replicated = sample.affine3d(translate_percent_range=translation, border_mode="replicate")
    np.testing.assert_array_equal(replicated.to_dict()["volume"], [[[0, 2, 6, 10]]])
    translation["x"] = (0.25, 0.25)
    shifted = sample.affine3d(translate_percent_range=translation, fill_mask=-9).to_dict()
    np.testing.assert_array_equal(shifted["mask"], [[[-9, 2**53 + 1, 2**53 + 2, 2**53 + 3]]])


def test_affine3d_rotation_and_random_reproducibility():
    from concurrent.futures import ThreadPoolExecutor

    volume = np.arange(54, dtype=np.uint8).reshape(3, 3, 3, 2)
    sample = VolumeSample(volume, keypoints=[[2, 1, 0]])
    rotate = {"x": (90, 90), "y": (0, 0), "z": (0, 0)}
    out = sample.affine3d(rotate_range=rotate, interpolation="nearest").to_dict()
    np.testing.assert_array_equal(out["volume"], np.rot90(volume, -1, (0, 1)))
    np.testing.assert_allclose(out["keypoints"], [[2, 2, 1]])
    np.testing.assert_array_equal(sample.affine3d().to_dict()["volume"], volume)

    def run(seed):
        return sample.affine3d(
            rotate_range={"x": (-20, 20), "y": (-30, 30), "z": (-15, 15)},
            scale_range={"x": (0.7, 1.4), "y": (0.8, 1.2), "z": (0.9, 1.0)},
            seed=seed,
        ).to_dict()

    with ThreadPoolExecutor(max_workers=4) as pool:
        for seed, result in enumerate(pool.map(run, range(8))):
            serial = run(seed)
            np.testing.assert_array_equal(result["volume"], serial["volume"])
            np.testing.assert_array_equal(result["keypoints"], serial["keypoints"])
    with pytest.raises(ValueError):
        sample.affine3d(scale_range={"x": (0, 1), "y": (1, 1), "z": (1, 1)})


def test_elastic_transform3d_shared_smooth_geometry():
    from concurrent.futures import ThreadPoolExecutor
    from itertools import product

    z, y, x = np.indices((17, 18, 19))
    grid = np.stack([x, y, z], axis=-1).astype(np.float32)
    mask = ((z * 18 + y) * 19 + x).astype(np.int64) + 2**53
    source = np.array([8, 7, 9], np.float32)
    sample = VolumeSample(grid, mask=mask, keypoints=[source], label=59)

    def run(seed):
        return sample.elastic_transform3d(
            displacement_range=(0.1, 0.1), control_grid_shape=(5, 5),
            border_mode="replicate", seed=seed,
        ).to_dict()

    with ThreadPoolExecutor(max_workers=4) as pool:
        for seed, out in enumerate(pool.map(run, range(4))):
            xyz = np.floor(out["volume"] + 0.5).astype(int)
            np.testing.assert_array_equal(out["mask"], mask[xyz[..., 2], xyz[..., 1], xyz[..., 0]])
            destination = out["keypoints"][0]
            assert not np.array_equal(destination, source)
            base = np.floor(destination).astype(int)
            fraction = destination - base
            back = np.zeros(3)
            for dx, dy, dz in product(range(2), repeat=3):
                corner = np.array([dx, dy, dz])
                xx, yy, zz = base + corner
                weight = np.prod(np.where(corner, fraction, 1 - fraction))
                back += weight * out["volume"][zz, yy, xx]
            np.testing.assert_allclose(back, source, atol=0.01)
            np.testing.assert_array_equal(out["volume"], run(seed)["volume"])
            assert out["label"] == 59


def test_elastic_transform3d_identity_singleton_and_validation():
    image = np.full((1, 3, 5), 9, dtype=np.uint8)
    sample = VolumeSample(image, keypoints=[[2, 1, 0]])
    np.testing.assert_array_equal(
        sample.elastic_transform3d(displacement_range=(0, 0)).to_dict()["volume"], image
    )
    singleton = VolumeSample(np.full((1, 1, 1), 99, dtype=np.uint8))
    np.testing.assert_array_equal(singleton.elastic_transform3d().to_dict()["volume"], 99)
    flat = sample.elastic_transform3d(border_mode="replicate").to_dict()
    np.testing.assert_array_equal(flat["volume"], image)
    assert flat["keypoints"][0, 2] == 0
    with pytest.raises(ValueError):
        sample.elastic_transform3d(control_grid_shape=(3, 7))
    with pytest.raises(ValueError):
        sample.elastic_transform3d(displacement_range=(0.1, 1))


def test_rician_volume_noise_floor_channel_modes_and_annotations():
    from concurrent.futures import ThreadPoolExecutor

    image = np.zeros((8, 24, 32, 2), dtype=np.float32)
    mask = np.full(image.shape[:3], 2**53 + 1, dtype=np.int64)
    sample = VolumeSample(image, mask=mask, keypoints=[[1, 2, 3]], label=61)

    def run(seed):
        return sample.rician_noise((0.1, 0.1), seed=seed).to_dict()

    with ThreadPoolExecutor(max_workers=3) as pool:
        for seed, out in enumerate(pool.map(run, range(3))):
            field = out["volume"]
            assert 0.12 < field.mean() < 0.13
            assert np.all(field >= 0)
            np.testing.assert_array_equal(field[..., 0], field[..., 1])
            np.testing.assert_array_equal(field, run(seed)["volume"])
            np.testing.assert_array_equal(out["mask"], mask)
            np.testing.assert_array_equal(out["keypoints"], [[1, 2, 3]])
            assert out["label"] == 61
    independent = sample.rician_noise((0.1, 0.1), per_channel=True).to_dict()["volume"]
    assert not np.array_equal(independent[..., 0], independent[..., 1])
    gray = VolumeSample(np.full((2, 3, 4), 5, np.uint8))
    baseline = gray.to_dict()["volume"]
    np.testing.assert_array_equal(gray.rician_noise((0, 0)).to_dict()["volume"], baseline)
    noisy = gray.rician_noise((0.5, 0.5)).to_dict()["volume"]
    assert noisy.dtype == np.uint8 and not np.array_equal(noisy, baseline)
    with pytest.raises(ValueError):
        sample.rician_noise((0.5, 0.1))


def test_bias_field_volume_is_smooth_and_preserves_annotations():
    from concurrent.futures import ThreadPoolExecutor

    image = np.full((11, 13, 15, 2), 0.125, dtype=np.float32)
    image[..., 1] *= 2
    mask = np.full(image.shape[:3], 2**53 + 3, dtype=np.int64)
    sample = VolumeSample(image, mask=mask, keypoints=[[2, 4, 6]], label=67)

    def run(seed):
        return sample.bias_field(std_range=(0.3, 0.3), seed=seed).to_dict()

    with ThreadPoolExecutor(max_workers=3) as pool:
        for seed, out in enumerate(pool.map(run, range(4))):
            augmented = out["volume"]
            assert augmented.dtype == image.dtype and augmented.shape == image.shape
            assert np.ptp(augmented) > 0.02 and np.all(augmented > 0)
            np.testing.assert_allclose(augmented[..., 1], 2 * augmented[..., 0])
            assert np.any(augmented[0] != augmented[-1])
            log_gain = np.log(augmented[3:8, ..., 0] / 0.125)
            np.testing.assert_allclose(np.diff(log_gain, n=2, axis=0), 0, atol=5e-7)
            np.testing.assert_array_equal(augmented, run(seed)["volume"])
            np.testing.assert_array_equal(out["mask"], mask)
            np.testing.assert_array_equal(out["keypoints"], [[2, 4, 6]])
            assert out["label"] == 67
    separate = sample.bias_field(std_range=(0.3, 0.3), per_channel=True).to_dict()["volume"]
    assert not np.allclose(separate[..., 1], 2 * separate[..., 0])
    np.testing.assert_array_equal(sample.bias_field(std_range=(0, 0)).to_dict()["volume"], image)
    with pytest.raises(ValueError):
        sample.bias_field(scale_range=(0, 0.1))


def test_gibbs_ringing_matches_rectangular_numpy_fft():
    rng = np.random.default_rng(29)
    image = rng.uniform(0.1, 0.9, (5, 6, 7, 2)).astype(np.float32)
    mask = rng.integers(0, 4, image.shape[:3], dtype=np.int64) + 2**53
    sample = VolumeSample(image, mask=mask, keypoints=[[2, 3, 4]], label=71)
    axes = (0, 1, 2)
    for fraction in (0, 0.62, 1):
        out = sample.gibbs_ringing((fraction, fraction)).to_dict()
        spectrum = np.fft.fftn(image, axes=axes)
        for axis, n in enumerate(image.shape[:3]):
            keep = np.abs(np.rint(np.fft.fftfreq(n) * n)) <= np.floor(fraction * n / 2)
            shape = [1, 1, 1, 1]
            shape[axis] = n
            spectrum *= keep.reshape(shape)
        expected = np.clip(np.fft.ifftn(spectrum, axes=axes).real, 0, 1)
        np.testing.assert_allclose(out["volume"], expected, atol=2e-7)
        np.testing.assert_array_equal(out["mask"], mask)
        np.testing.assert_array_equal(out["keypoints"], [[2, 3, 4]])
        assert out["label"] == 71 and out["volume"].dtype == np.float32
    gray = VolumeSample(np.array([[[0, 2, 4]]], dtype=np.uint8))
    np.testing.assert_array_equal(gray.gibbs_ringing((0, 0)).to_dict()["volume"], 2)
    with pytest.raises(ValueError):
        sample.gibbs_ringing((-1, 1))
