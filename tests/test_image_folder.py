from __future__ import annotations

import binascii
import gc
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

import rivet

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n"
    + struct.pack(">I", 13)
    + b"IHDR"
    + struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
)
PNG_1X1 += struct.pack(">I", binascii.crc32(PNG_1X1[12:]) & 0xFFFFFFFF)
_IDAT = b"IDAT" + zlib.compress(b"\x00\xff\x00\x00")
PNG_1X1 += struct.pack(">I", len(_IDAT) - 4) + _IDAT
PNG_1X1 += struct.pack(">I", binascii.crc32(_IDAT) & 0xFFFFFFFF)
PNG_1X1 += b"\x00\x00\x00\x00IEND\xaeB`\x82"


def labels(batch: dict[str, object]) -> list[int]:
    values = batch["labels"]
    return values.tolist() if hasattr(values, "tolist") else list(values)  # type: ignore[union-attr]


def test_image_folder_metadata_is_stable(tmp_path: Path) -> None:
    (tmp_path / "zebra").mkdir()
    (tmp_path / "ant" / "nested").mkdir(parents=True)
    (tmp_path / "zebra" / "b.PNG").write_bytes(PNG_1X1)
    (tmp_path / "ant" / "nested" / "a.jpg").write_bytes(PNG_1X1)
    (tmp_path / "ant" / "ignore.txt").write_text("not an image")

    dataset = rivet.ImageFolder(tmp_path)

    assert dataset.classes == ["ant", "zebra"]
    assert dataset.class_to_idx == {"ant": 0, "zebra": 1}
    assert len(dataset) == 2
    assert [sample["label"] for sample in dataset.samples] == [0, 1]
    assert [Path(sample["path"]).name for sample in dataset.samples] == ["a.jpg", "b.PNG"]


def test_image_folder_pipeline(tmp_path: Path) -> None:
    (tmp_path / "cat").mkdir()
    (tmp_path / "cat" / "one.png").write_bytes(PNG_1X1)

    batch = next(rivet.scan_image_folder(tmp_path).decode_image().batch(1).execute())

    assert batch["images"].shape == (1, 1, 1, 3)
    assert labels(batch) == [0]


def test_image_folder_pipeline_returns_readonly_zero_copy_numpy_views(
    tmp_path: Path,
) -> None:
    (tmp_path / "cat").mkdir()
    (tmp_path / "cat" / "one.png").write_bytes(PNG_1X1)

    batch = next(rivet.scan_image_folder(tmp_path).decode_image().batch(1).execute())
    images = batch["images"]
    batch_labels = batch["labels"]

    assert isinstance(images, np.ndarray)
    assert isinstance(batch_labels, np.ndarray)
    assert not images.flags.owndata
    assert not batch_labels.flags.owndata
    assert not images.flags.writeable
    assert not batch_labels.flags.writeable
    assert images.base is not None
    assert batch_labels.base is not None
    with pytest.raises(ValueError, match="read-only"):
        images[0, 0, 0, 0] = 0

    del batch
    gc.collect()
    np.testing.assert_array_equal(images, np.array([[[[255, 0, 0]]]], dtype=np.uint8))
    np.testing.assert_array_equal(batch_labels, np.array([0], dtype=np.int64))

    normalized = next(
        rivet.scan_image_folder(tmp_path)
        .decode_image()
        .normalize([0.5], [0.5])
        .batch(1)
        .execute()
    )["images"]
    assert normalized.dtype == np.float32
    assert not normalized.flags.owndata
    assert not normalized.flags.writeable
