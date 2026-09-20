"""Numerical compatibility checks against torchvision image transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rivet

torch = pytest.importorskip("torch")
transforms = pytest.importorskip("torchvision.transforms")
functional = pytest.importorskip("torchvision.transforms.functional")
InterpolationMode = functional.InterpolationMode
Image = pytest.importorskip("PIL.Image")


def _write_test_image(root: Path) -> tuple[Path, np.ndarray]:
    """Write an RGB image with enough high-frequency detail to expose drift."""
    pixels = np.fromfunction(
        lambda y, x, channel: (17 * y + 29 * x + 53 * channel + x * y) % 256,
        (11, 13, 3),
        dtype=int,
    ).astype(np.uint8)
    image_path = root / "sample.png"
    Image.fromarray(pixels, mode="RGB").save(image_path)
    return image_path, pixels


def _rivet_image(root: Path, pipeline: rivet.Pipeline) -> np.ndarray:
    batch = next(pipeline.batch(1).execute())
    return batch["images"][0]


def test_geometry_and_normalize_match_torchvision(tmp_path: Path) -> None:
    """Crop/flip and normalization have the same values as torchvision."""
    class_dir = tmp_path / "class"
    class_dir.mkdir()
    image_path, _ = _write_test_image(class_dir)
    mean = [0.25, 0.5, 0.75]
    std = [0.5, 0.25, 0.125]

    actual = _rivet_image(
        class_dir.parent,
        rivet.scan_image_folder(class_dir.parent)
        .decode_image()
        .center_crop(9, 7)
        .horizontal_flip()
        .normalize(mean, std),
    )
    expected = transforms.Compose(
        [
            transforms.CenterCrop((7, 9)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )(Image.open(image_path).convert("RGB"))

    np.testing.assert_allclose(
        actual,
        expected.numpy().transpose(1, 2, 0),
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("rivet_mode", "torch_mode", "max_error"),
    [
        ("nearest", InterpolationMode.NEAREST, 0),
        ("bilinear", InterpolationMode.BILINEAR, 1),
        # image::CatmullRom and Pillow's bicubic kernel have small edge drift.
        ("bicubic", InterpolationMode.BICUBIC, 8),
        # The two Lanczos implementations have wider ringing differences.
        ("lanczos3", InterpolationMode.LANCZOS, 16),
    ],
)
def test_resize_stays_within_bounded_u8_error_of_torchvision(
    tmp_path: Path,
    rivet_mode: str,
    torch_mode: InterpolationMode,
    max_error: int,
) -> None:
    """The resize backend may round differently, but drift must stay bounded."""
    class_dir = tmp_path / "class"
    class_dir.mkdir()
    image_path, _ = _write_test_image(class_dir)

    actual = _rivet_image(
        class_dir.parent,
        rivet.scan_image_folder(class_dir.parent)
        .decode_image()
        .resize(17, 15, rivet_mode),
    )
    expected = transforms.Resize(
        (15, 17), interpolation=torch_mode
    )(Image.open(image_path).convert("RGB"))

    np.testing.assert_allclose(
        actual.astype(np.int16),
        np.asarray(expected, dtype=np.int16),
        rtol=0.0,
        atol=max_error,
    )


def test_crop_flip_and_pad_match_torchvision_exactly(tmp_path: Path) -> None:
    """Indexing-only operations must not introduce any pixel error."""
    class_dir = tmp_path / "class"
    class_dir.mkdir()
    image_path, _ = _write_test_image(class_dir)

    actual = _rivet_image(
        class_dir.parent,
        rivet.scan_image_folder(class_dir.parent)
        .decode_image()
        .crop(2, 1, 8, 7)
        .vertical_flip()
        .pad(2, fill=37),
    )
    expected = functional.pad(
        functional.vflip(
            functional.crop(Image.open(image_path).convert("RGB"), 1, 2, 7, 8)
        ),
        2,
        fill=37,
    )

    np.testing.assert_array_equal(actual, np.asarray(expected))


@pytest.mark.parametrize("channels", [1, 3])
def test_grayscale_stays_within_known_luma_difference_of_torchvision(
    tmp_path: Path, channels: int
) -> None:
    """Bound the Rec.709 (Rivet) versus torchvision luma-coefficient drift."""
    class_dir = tmp_path / "class"
    class_dir.mkdir()
    image_path, _ = _write_test_image(class_dir)

    actual = _rivet_image(
        class_dir.parent,
        rivet.scan_image_folder(class_dir.parent)
        .decode_image()
        .grayscale(channels),
    )
    expected = transforms.Grayscale(num_output_channels=channels)(
        Image.open(image_path).convert("RGB")
    )
    expected_array = np.asarray(expected)
    if channels == 1:
        expected_array = expected_array[:, :, np.newaxis]

    np.testing.assert_allclose(
        actual.astype(np.int16),
        expected_array.astype(np.int16),
        rtol=0.0,
        # Rivet uses Rec.709 coefficients; torchvision/Pillow uses the
        # traditional RGB luma coefficients. Their theoretical u8 gap is < 34.
        atol=34,
    )


@pytest.mark.parametrize(
    ("name", "argument", "torch_operation"),
    [
        ("invert", None, functional.invert),
        ("posterize", 4, functional.posterize),
        # Avoid an exact-threshold pixel: Pillow and image differ on whether
        # that value itself is inverted.
        ("solarize", 122, functional.solarize),
        ("autocontrast", None, functional.autocontrast),
        ("equalize", None, functional.equalize),
    ],
)
def test_u8_color_operations_match_torchvision(
    tmp_path: Path, name: str, argument: int | None, torch_operation: object
) -> None:
    """Pixel-wise color operations use the same uint8 contract as torchvision."""
    class_dir = tmp_path / "class"
    class_dir.mkdir()
    image_path, _ = _write_test_image(class_dir)
    pipeline = rivet.scan_image_folder(class_dir.parent).decode_image()
    actual = _rivet_image(
        class_dir.parent,
        getattr(pipeline, name)() if argument is None else getattr(pipeline, name)(argument),
    )
    image = Image.open(image_path).convert("RGB")
    expected = (
        torch_operation(image)  # type: ignore[operator]
        if argument is None
        else torch_operation(image, argument)  # type: ignore[operator]
    )

    np.testing.assert_allclose(
        actual.astype(np.int16),
        np.asarray(expected, dtype=np.int16),
        rtol=0.0,
        # Equalize histogram bins round differently in the two backends.
        atol=9 if name == "equalize" else 0,
    )


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_fixed_rotate_matches_torchvision(tmp_path: Path, angle: int) -> None:
    class_dir = tmp_path / "class"
    class_dir.mkdir()
    image_path, _ = _write_test_image(class_dir)

    actual = _rivet_image(
        class_dir.parent,
        rivet.scan_image_folder(class_dir.parent).decode_image().rotate(angle),
    )
    expected = functional.rotate(
        Image.open(image_path).convert("RGB"),
        # image::rotate90 is clockwise, whereas torchvision's positive
        # angles are counter-clockwise.
        -angle,
        interpolation=InterpolationMode.NEAREST,
        expand=True,
    )

    np.testing.assert_array_equal(actual, np.asarray(expected))


def test_dtype_layout_and_asymmetric_padding_match_torchvision(tmp_path: Path) -> None:
    """Verify exported batch dtype/layout operations and four-side padding."""
    class_dir = tmp_path / "class"
    class_dir.mkdir()
    image_path, pixels = _write_test_image(class_dir)

    padded = _rivet_image(
        class_dir.parent,
        rivet.scan_image_folder(class_dir.parent)
        .decode_image()
        .pad_with_sides(1, 2, 3, 4, fill=37),
    )
    expected_padded = functional.pad(
        Image.open(image_path).convert("RGB"), [1, 2, 3, 4], fill=37
    )
    np.testing.assert_array_equal(padded, np.asarray(expected_padded))

    converted = _rivet_image(
        class_dir.parent,
        rivet.scan_image_folder(class_dir.parent)
        .decode_image()
        .convert_image_dtype("float32")
        .hwc_to_chw(),
    )
    expected_converted = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
    np.testing.assert_allclose(converted, expected_converted.numpy(), rtol=0.0, atol=1e-7)


def test_configurable_random_ops_have_torchvision_comparable_degenerate_cases(
    tmp_path: Path,
) -> None:
    """Exported stochastic configuration can be constrained to deterministic output."""
    class_dir = tmp_path / "class"
    class_dir.mkdir()
    image_path, pixels = _write_test_image(class_dir)
    ratio = 13.0 / 11.0

    resized = _rivet_image(
        class_dir.parent,
        rivet.scan_image_folder(class_dir.parent)
        .decode_image()
        .random_resized_crop_with_options(
            17, 15, scale=[1.0, 1.0], ratio=[ratio, ratio]
        ),
    )
    expected_resized = functional.resize(
        Image.open(image_path).convert("RGB"),
        [15, 17],
        interpolation=InterpolationMode.BILINEAR,
    )
    np.testing.assert_allclose(
        resized.astype(np.int16),
        np.asarray(expected_resized, dtype=np.int16),
        rtol=0.0,
        atol=1,
    )

    erased = _rivet_image(
        class_dir.parent,
        rivet.scan_image_folder(class_dir.parent)
        .decode_image()
        .random_erasing_with_options(
            probability=1.0,
            scale=[1.0, 1.0],
            ratio=[ratio, ratio],
            value=23,
        ),
    )
    expected_erased = np.full_like(pixels, 23)
    np.testing.assert_array_equal(erased, expected_erased)


@pytest.mark.parametrize(
    "build",
    [
        lambda pipeline: pipeline.random_crop(13, 11),
        lambda pipeline: pipeline.random_horizontal_flip(probability=0.0),
        lambda pipeline: pipeline.random_grayscale(probability=0.0),
        lambda pipeline: pipeline.random_erasing(probability=0.0),
        lambda pipeline: pipeline.sharpness(0.0),
        lambda pipeline: pipeline.arbitrary_rotate(0.0, expand=False),
        lambda pipeline: pipeline.random_affine(0.0),
        lambda pipeline: pipeline.perspective(
            [(0.0, 0.0), (12.0, 0.0), (12.0, 10.0), (0.0, 10.0)],
            [(0.0, 0.0), (12.0, 0.0), (12.0, 10.0), (0.0, 10.0)],
        ),
        lambda pipeline: pipeline.random_perspective(probability=0.0),
        lambda pipeline: pipeline.elastic_transform(0.0, 1.0),
    ],
)
def test_identity_parameterizations_match_torchvision_identity(
    tmp_path: Path, build: object
) -> None:
    """Stochastic transforms expose PyTorch-equivalent deterministic limits."""
    class_dir = tmp_path / "class"
    class_dir.mkdir()
    _, pixels = _write_test_image(class_dir)
    pipeline = rivet.scan_image_folder(class_dir.parent).decode_image()
    actual = _rivet_image(class_dir.parent, build(pipeline))  # type: ignore[operator]

    np.testing.assert_array_equal(actual, pixels)
