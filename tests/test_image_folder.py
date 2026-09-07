from __future__ import annotations

import binascii
import struct
import zlib
from pathlib import Path

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
