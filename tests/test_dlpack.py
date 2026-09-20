from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest

import rivet

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")


def _loader(root: Path, *, channels_first: bool = False) -> tuple[object, np.ndarray]:
    class_dir = root / "class"
    class_dir.mkdir(parents=True)
    pixels = np.fromfunction(
        lambda y, x, channel: (11 * y + 17 * x + 31 * channel) % 256,
        (4, 5, 3),
        dtype=int,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(class_dir / "sample.png")

    pipeline = rivet.scan_image_folder(root).decode_image()
    if channels_first:
        pipeline = pipeline.hwc_to_chw()
    return pipeline.batch(1).execute(), pixels


def test_dlpack_matches_torch_layout_dtype_and_values(tmp_path: Path) -> None:
    loader, pixels = _loader(tmp_path)
    batch = loader.next_dlpack()

    images = torch.utils.dlpack.from_dlpack(batch["images"])
    labels = torch.utils.dlpack.from_dlpack(batch["labels"])

    expected_images = torch.from_numpy(pixels).unsqueeze(0)
    assert images.dtype == torch.uint8
    assert tuple(images.shape) == (1, 4, 5, 3)
    assert tuple(images.stride()) == tuple(expected_images.stride())
    torch.testing.assert_close(images, expected_images)
    assert labels.dtype == torch.int64
    assert tuple(labels.shape) == (1,)
    assert tuple(labels.stride()) == (1,)
    assert labels.tolist() == [0]


def test_dlpack_keeps_tensor_alive_after_producer_is_dropped(tmp_path: Path) -> None:
    loader, pixels = _loader(tmp_path)
    batch = loader.next_dlpack()
    images = torch.utils.dlpack.from_dlpack(batch["images"])
    expected = torch.from_numpy(pixels).unsqueeze(0)

    del batch
    del loader
    gc.collect()

    torch.testing.assert_close(images, expected)


def test_dlpack_capsule_is_consumed_once(tmp_path: Path) -> None:
    loader, _ = _loader(tmp_path)
    producer = loader.next_dlpack()["images"]
    capsule = producer.__dlpack__()

    first = torch.utils.dlpack.from_dlpack(capsule)
    assert first.numel() == 4 * 5 * 3
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        torch.utils.dlpack.from_dlpack(capsule)


def test_into_dlpack_transfers_ownership_and_consumes_producer(tmp_path: Path) -> None:
    loader, _ = _loader(tmp_path)
    producer = loader.next_dlpack()["images"]
    capsule = producer.into_dlpack()
    images = torch.utils.dlpack.from_dlpack(capsule)

    images.add_(1)
    with pytest.raises(RuntimeError, match="transferred ownership"):
        producer.into_dlpack()
    with pytest.raises(RuntimeError, match="transferred ownership"):
        producer.__dlpack_device__()


def test_dlpack_version_and_cpu_argument_validation(tmp_path: Path) -> None:
    loader, _ = _loader(tmp_path)
    producer = loader.next_dlpack()["images"]

    with pytest.raises(ValueError, match="stream"):
        producer.__dlpack__(stream=1)
    with pytest.raises(BufferError, match="CPU device 0"):
        producer.__dlpack__(dl_device=(2, 0))
    with pytest.raises(ValueError, match="too old"):
        producer.__dlpack__(max_version=(0, 9))
    with pytest.raises(BufferError, match="zero-copy"):
        producer.__dlpack__(copy=True)

    accepted_loader, _ = _loader(tmp_path / "accepted")
    accepted = accepted_loader.next_dlpack()["images"]
    assert accepted.__dlpack__(stream=0, dl_device=(1, 0), copy=False)

    versioned = producer.__dlpack__(max_version=(1, 0))
    tensor = torch.utils.dlpack.from_dlpack(versioned)
    assert tensor.dtype == torch.uint8


def test_dlpack_preserves_permuted_strides(tmp_path: Path) -> None:
    loader, pixels = _loader(tmp_path, channels_first=True)
    batch = loader.next_dlpack()
    images = torch.utils.dlpack.from_dlpack(batch["images"])

    expected = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    assert tuple(images.shape) == (1, 3, 4, 5)
    assert tuple(images.stride()) == (pixels.size, *expected.stride()[1:])
    torch.testing.assert_close(images, expected)
