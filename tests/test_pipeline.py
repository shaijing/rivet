from __future__ import annotations

import binascii
import os
import struct
import zlib
from pathlib import Path
from typing import Any

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


def _candidate_arrow_files() -> list[Path]:
    explicit = os.environ.get("RIVET_TEST_ARROW_FILE")
    if explicit:
        return [Path(explicit)]

    roots = []
    if datasets_cache := os.environ.get("HF_DATASETS_CACHE"):
        roots.append(Path(datasets_cache))
    if hf_home := os.environ.get("HF_HOME"):
        roots.append(Path(hf_home) / "datasets")
    roots.append(Path.home() / ".cache" / "huggingface" / "datasets")

    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)
        files.extend(root.glob("uoft-cs___cifar10/**/cifar10-train.arrow"))
    return sorted(files)


@pytest.fixture(scope="module")
def arrow_file() -> Path:
    for path in _candidate_arrow_files():
        if path.exists():
            return path
    pytest.skip("local CIFAR-10 Arrow cache not found; set RIVET_TEST_ARROW_FILE")


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
    with pytest.raises(ValueError, match="batch size"):
        scan(arrow_file).decode_image().batch(0)


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


def test_hf_indices_rejected() -> None:
    datasets = pytest.importorskip("datasets")

    try:
        ds = datasets.load_dataset("uoft-cs/cifar10", split="train[:8]")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not load cached CIFAR-10 dataset: {exc}")

    shuffled = ds.shuffle(seed=42)
    with pytest.raises(ValueError, match="flatten_indices"):
        rivet.scan_hf(shuffled)


def test_rust_numpy_export_does_not_clone_image_batch() -> None:
    loader_rs = Path(__file__).parents[1] / "crates" / "rivet-python" / "src" / "loader.rs"

    assert "values.clone()" not in loader_rs.read_text()


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
