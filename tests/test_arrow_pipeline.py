from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import rivet


def scan(arrow_file: Path) -> rivet.Pipeline:
    return rivet.scan_arrow([arrow_file])


def labels(batch: dict[str, Any]) -> list[int]:
    values = batch["labels"]
    return values.tolist() if hasattr(values, "tolist") else list(values)


def test_decode_batch(arrow_file: Path) -> None:
    batch = next(scan(arrow_file).decode_image().batch(8).execute())

    assert isinstance(batch["images"], np.ndarray)
    assert isinstance(batch["labels"], np.ndarray)
    assert batch["images"].shape == (8, 32, 32, 3)
    assert batch["images"].dtype == np.uint8
    assert batch["labels"].dtype == np.int64
    assert batch["dtype"] == "uint8"
    assert batch["layout"] == "NHWC"


def test_resize(arrow_file: Path) -> None:
    batch = next(scan(arrow_file).decode_image().resize(24, 20).batch(2).execute())

    assert batch["images"].shape == (2, 20, 24, 3)
    assert batch["dtype"] == "uint8"


def test_center_crop(arrow_file: Path) -> None:
    batch = next(scan(arrow_file).decode_image().center_crop(20, 18).batch(2).execute())

    assert batch["images"].shape == (2, 18, 20, 3)


def test_layout_hwc_to_chw(arrow_file: Path) -> None:
    batch = next(scan(arrow_file).decode_image().hwc_to_chw().batch(2).execute())

    assert batch["images"].shape == (2, 3, 32, 32)
    assert batch["layout"] == "NCHW"


def test_normalize_dtype(arrow_file: Path) -> None:
    batch = next(
        scan(arrow_file)
        .decode_image()
        .normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        .batch(2)
        .execute()
    )

    assert batch["images"].dtype == np.float32
    assert batch["dtype"] == "float32"


def test_normalize_then_hwc_to_chw(arrow_file: Path) -> None:
    batch = next(
        scan(arrow_file)
        .decode_image()
        .normalize([0.5], [0.5])
        .hwc_to_chw()
        .batch(2)
        .execute()
    )

    assert batch["images"].shape == (2, 3, 32, 32)
    assert batch["images"].dtype == np.float32
    assert batch["layout"] == "NCHW"


def test_skip_take(arrow_file: Path) -> None:
    base = next(scan(arrow_file).decode_image().batch(8).execute())
    subset = next(scan(arrow_file).skip(2).take(3).decode_image().batch(4).execute())

    assert subset["images"].shape == (3, 32, 32, 3)
    assert labels(subset) == labels(base)[2:5]


def test_drop_last(arrow_file: Path) -> None:
    loader = scan(arrow_file).take(3).decode_image().batch(4, drop_last=True).execute()

    with pytest.raises(StopIteration):
        next(loader)


def test_take_larger_than_dataset(arrow_file: Path) -> None:
    dataset = rivet.ArrowDataset([arrow_file])
    batch = next(scan(arrow_file).take(len(dataset) + 10).decode_image().batch(4).execute())

    assert batch["images"].shape == (4, 32, 32, 3)


def test_skip_larger_than_dataset(arrow_file: Path) -> None:
    dataset = rivet.ArrowDataset([arrow_file])
    loader = scan(arrow_file).skip(len(dataset) + 10).decode_image().batch(4).execute()

    with pytest.raises(StopIteration):
        next(loader)


def test_batch_zero_rejected(arrow_file: Path) -> None:
    # Builders are infallible; validation surfaces at execute/compile.
    with pytest.raises(ValueError, match="batch size"):
        scan(arrow_file).decode_image().batch(0).execute()


def test_resize_before_decode_rejected(arrow_file: Path) -> None:
    with pytest.raises(ValueError, match="decoded"):
        scan(arrow_file).resize(16, 16).batch(1).execute()


def test_decode_twice_rejected(arrow_file: Path) -> None:
    with pytest.raises(ValueError, match="encoded"):
        scan(arrow_file).decode_image().decode_image().batch(1).execute()


def test_resize_after_normalize_rejected(arrow_file: Path) -> None:
    with pytest.raises(ValueError, match="uint8 HWC"):
        (
            scan(arrow_file)
            .decode_image()
            .normalize([0.5], [0.5])
            .resize(16, 16)
            .batch(1)
            .execute()
        )


def test_normalize_channel_mismatch(arrow_file: Path) -> None:
    loader = scan(arrow_file).decode_image().normalize([0.5, 0.5], [0.5, 0.5]).batch(1).execute()

    with pytest.raises(ValueError, match="channel count"):
        next(loader)
