"""Differential coverage for the native torchvision operator migration."""

import numpy as np
import pytest

import rivet

torch = pytest.importorskip("torch")
F = pytest.importorskip("torchvision.transforms.functional")
T = pytest.importorskip("torchvision.transforms")
Image = pytest.importorskip("PIL.Image")


@pytest.fixture
def source(tmp_path):
    pixels = np.random.default_rng(17).integers(0, 256, (37, 41, 3), dtype=np.uint8)
    folder = tmp_path / "class"
    folder.mkdir()
    image = Image.fromarray(pixels)
    image.save(folder / "sample.png")
    return tmp_path, image, pixels


def output(root, build):
    return next(build(rivet.scan_image_folder(root).decode_image()).batch(1).execute())[
        "images"
    ][0]


@pytest.mark.parametrize("factor", [0.0, 0.3, 1.0, 1.7])
def test_saturation(source, factor):
    root, image, _ = source
    np.testing.assert_array_equal(
        output(root, lambda p: p.saturation(factor)),
        np.asarray(F.adjust_saturation(image, factor)),
    )


@pytest.mark.parametrize(
    "gamma,gain",
    [(0.0, 1.0), (0.3, 0.5), (1.0, 1.0), (1.7, 1.4), (2.2, 1.0), (0.7, -0.5)],
)
def test_gamma(source, gamma, gain):
    root, image, _ = source
    np.testing.assert_array_equal(
        output(root, lambda p: p.gamma(gamma, gain)),
        np.asarray(F.adjust_gamma(image, gamma, gain)),
    )


@pytest.mark.parametrize(
    "mode", ["nearest", "bilinear", "bicubic", "lanczos3", "box", "hamming"]
)
def test_resized_crop_and_new_filters(source, mode):
    root, image, _ = source
    interpolation = (
        F.InterpolationMode.LANCZOS
        if mode == "lanczos3"
        else getattr(F.InterpolationMode, mode.upper())
    )
    actual = output(root, lambda p: p.resized_crop(2, 1, 17, 19, 23, 11, mode))
    expected = F.resized_crop(image, 1, 2, 19, 17, [11, 23], interpolation)
    np.testing.assert_array_equal(actual, np.asarray(expected))


@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
@pytest.mark.parametrize("center", [None, [0.0, 0.0]])
def test_deterministic_affine(source, mode, center):
    root, image, _ = source
    actual = output(
        root,
        lambda p: p.affine(
            13.0,
            translate=[2, -1],
            scale=0.85,
            shear=[7, -3],
            interpolation=mode,
            fill=37,
            center=center,
        ),
    )
    expected = F.affine(
        image,
        13.0,
        [2, -1],
        0.85,
        [7, -3],
        interpolation=getattr(F.InterpolationMode, mode.upper()),
        fill=37,
        center=center,
    )
    np.testing.assert_allclose(
        actual.astype(np.int16),
        np.asarray(expected).astype(np.int16),
        atol=0 if mode == "nearest" else 1,
        rtol=0,
    )


@pytest.mark.parametrize("value", [23.0, -1.5, 300.5, [0.0, 127.0, 255.0]])
def test_erase_uint8(source, value):
    root, _, pixels = source
    actual = output(root, lambda p: p.erase(2, 1, 9, 7, value))
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    fill = torch.tensor(
        [value] if isinstance(value, float) else value, dtype=torch.float32
    ).reshape(-1, 1, 1)
    expected = F.erase(tensor, 1, 2, 7, 9, fill).permute(1, 2, 0)
    np.testing.assert_array_equal(actual, expected.numpy())


def test_erase_float(source):
    root, _, pixels = source
    actual = output(
        root,
        lambda p: p.convert_image_dtype("float32").erase(2, 1, 9, 7, [0.1, 0.2, 0.3]),
    )
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
    expected = F.erase(
        tensor, 1, 2, 7, 9, torch.tensor([0.1, 0.2, 0.3]).reshape(3, 1, 1)
    ).permute(1, 2, 0)
    np.testing.assert_array_equal(actual, expected.numpy())


@pytest.mark.parametrize("expand", [False, True])
def test_random_rotation_fixed_range(source, expand):
    root, image, _ = source
    actual = output(
        root, lambda p: p.random_rotation([17.0, 17.0], expand=expand, fill=37)
    )
    expected = F.rotate(image, 17.0, expand=expand, fill=37)
    np.testing.assert_array_equal(actual, np.asarray(expected))


def test_random_gaussian_fixed_range(source):
    root, image, _ = source
    actual = output(root, lambda p: p.random_gaussian_blur(5, [0.7, 0.7]))
    expected = F.gaussian_blur(image, 5, [0.7, 0.7])
    np.testing.assert_allclose(
        actual.astype(np.int16), np.asarray(expected).astype(np.int16), atol=1, rtol=0
    )


def test_rectangular_gaussian_blur_axis_sigmas(source):
    root, image, _ = source
    kernel_size = (value for value in [3, 5])  # torchvision order: width, height
    sigma = (value for value in [0.45, 1.2])  # horizontal, vertical
    actual = output(
        root,
        lambda p: p.gaussian_blur(sigma, kernel_size=kernel_size),
    )
    expected = F.gaussian_blur(image, [3, 5], [0.45, 1.2])
    np.testing.assert_allclose(
        actual.astype(np.int16), np.asarray(expected).astype(np.int16), atol=1, rtol=0
    )


def test_random_erasing_channel_values_and_random_mode(source):
    root, image, _ = source
    values = [12.0, 100.0, 225.0]
    actual = output(
        root,
        lambda p: p.random_erasing(
            probability=1.0,
            scale=[0.25, 0.25],
            ratio=[1.0, 1.0],
            value=values,
        ),
    )
    # The native RNG chooses another rectangle, so verify the channel values
    # wherever the erase occurred instead of comparing coordinates.
    erased = np.all(actual == np.asarray(values), axis=-1)
    assert erased.any()

    float_values = [0.125, 0.5, 0.875]
    float_actual = output(
        root,
        lambda p: p.convert_image_dtype("float32").random_erasing_with_options(
            probability=1.0,
            scale=[0.25, 0.25],
            ratio=[1.0, 1.0],
            value=float_values,
        ),
    )
    assert np.any(np.all(float_actual == float_values, axis=-1))

    random_actual = output(
        root,
        lambda p: p.convert_image_dtype("float32").random_erasing_with_options(
            probability=1.0,
            scale=[0.25, 0.25],
            ratio=[1.0, 1.0],
            value="random",
        ),
    )
    assert random_actual.shape == np.asarray(image).shape
    assert np.isfinite(random_actual).all()


@pytest.mark.parametrize(
    "name,args,reference",
    [
        ("random_vertical_flip", [], F.vflip),
        ("random_invert", [], F.invert),
        ("random_posterize", [4], lambda im: F.posterize(im, 4)),
        ("random_solarize", [128], lambda im: F.solarize(im, 128)),
        ("random_adjust_sharpness", [1.7], lambda im: F.adjust_sharpness(im, 1.7)),
        ("random_autocontrast", [], F.autocontrast),
        ("random_equalize", [], F.equalize),
    ],
)
@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_random_wrappers(source, name, args, reference, probability):
    root, image, pixels = source
    actual = output(root, lambda p: getattr(p, name)(*args, probability=probability))
    np.testing.assert_array_equal(
        actual, np.asarray(reference(image)) if probability else pixels
    )


@pytest.mark.parametrize(
    "builder",
    [
        lambda p: p.auto_augment("imagenet"),
        lambda p: p.auto_augment("cifar10"),
        lambda p: p.auto_augment("svhn"),
        lambda p: p.rand_augment(),
        lambda p: p.trivial_augment_wide(),
        lambda p: p.augmix(),
        lambda p: p.color_jitter(saturation=0.4),
        lambda p: p.random_rotation(20),
        lambda p: p.random_gaussian_blur(5),
    ],
)
def test_native_random_policies_reproducible_across_workers(source, builder):
    root, _, pixels = source

    def run(workers):
        return next(
            builder(rivet.scan_image_folder(root).decode_image())
            .seed(37)
            .workers(workers)
            .batch(1)
            .execute()
        )["images"]

    a = run(0)
    np.testing.assert_array_equal(a, run(0))
    np.testing.assert_array_equal(a, run(2))
    assert a.dtype == np.uint8
    assert a.shape == (1, *pixels.shape)


@pytest.mark.parametrize(
    "builder",
    [
        lambda p: p.saturation(-1),
        lambda p: p.gamma(-1),
        lambda p: p.affine(0, scale=0),
        lambda p: p.erase(1, 1, 3, 3, []),
        lambda p: p.random_rotation([2, -2]),
        lambda p: p.random_gaussian_blur(4),
        lambda p: p.auto_augment("invalid"),
        lambda p: p.rand_augment(magnitude=31),
        lambda p: p.trivial_augment_wide(num_magnitude_bins=1),
        lambda p: p.augmix(alpha=0),
        lambda p: p.color_jitter(saturation=-1),
    ],
)
def test_invalid_new_parameters_rejected(source, builder):
    root, _, _ = source
    with pytest.raises((ValueError, RuntimeError)):
        output(root, builder)


@pytest.mark.parametrize("scaled", [False, True])
def test_native_tensor_conversion_equivalents(source, scaled):
    root, image, _ = source
    actual = output(root, lambda p: p.to_tensor() if scaled else p.pil_to_tensor())
    expected = F.to_tensor(image) if scaled else F.pil_to_tensor(image)
    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-7, rtol=0)


@pytest.mark.parametrize("chw", [False, True])
def test_linear_transformation_float32(source, chw):
    root, image, _ = source
    matrix = torch.randn(18, 18, generator=torch.Generator().manual_seed(7)) * 0.1
    mean = torch.linspace(0.1, 0.8, 18)

    def build(p):
        p = p.resize(3, 2, "nearest")
        p = p.to_tensor() if chw else p.convert_image_dtype("float32")
        return p.linear_transform(matrix.tolist(), mean.tolist())

    actual = output(root, build)
    tensor = F.to_tensor(F.resize(image, [2, 3], F.InterpolationMode.NEAREST))
    expected = T.LinearTransformation(matrix, mean)(tensor).numpy()
    if not chw:
        expected = expected.transpose(1, 2, 0)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
