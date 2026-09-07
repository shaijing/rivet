"""Multi-arrow end-to-end tests against a real multi-shard HF Arrow cache.

The fixture targets the Hugging Face `clane9/imagenet-100` cache (17 shards
-> 17 Arrow files), but any multi-file cache can be supplied explicitly via
the comma-separated `RIVET_TEST_ARROW_FILES` environment variable.

```bash
python - <<'EOF'
from datasets import load_dataset
load_dataset("clane9/imagenet-100", split="train")
EOF
pytest tests/test_multi_arrow.py -q
```
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow.ipc as pa_ipc
import pytest

import rivet

IMAGE_COLUMN = "image"  # clane9/imagenet-100 layout: struct<bytes>, int64 label


def _candidate_arrow_files() -> list[Path]:
    explicit = os.environ.get("RIVET_TEST_ARROW_FILES")
    if explicit:
        files = [Path(p) for p in explicit.split(",") if p.strip()]
        if files:
            return files

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
        files.extend(root.glob("clane9___imagenet-100/**/imagenet-100-train-*.arrow"))
    return sorted(files)


@pytest.fixture(scope="module")
def arrow_files() -> list[Path]:
    files = _candidate_arrow_files()
    if len(files) < 2:
        pytest.skip(
            "multi-file Arrow cache not found; load clane9/imagenet-100 or "
            "set RIVET_TEST_ARROW_FILES=/path/a.arrow,/path/b.arrow"
        )
    return files


@pytest.fixture(scope="module")
def labels(arrow_files: list[Path]) -> np.ndarray:
    """Concatenated upstream labels across files, in file order."""
    parts = []
    for path in arrow_files:
        with pa_ipc.open_stream(path) as reader:
            parts.extend(batch.column("label").to_pylist() for batch in reader)
    return np.asarray([label for part in parts for label in part], dtype=np.int64)


@pytest.fixture(scope="module")
def dataset(arrow_files: list[Path]) -> rivet.ArrowDataset:
    return rivet.ArrowDataset(arrow_files, image_column=IMAGE_COLUMN)


def test_multiple_arrow_files(arrow_files: list[Path]) -> None:
    assert len(arrow_files) >= 2
    assert all(path.exists() for path in arrow_files)


def test_row_indexing_matches_upstream_labels(
    dataset: rivet.ArrowDataset,
    labels: np.ndarray,
) -> None:
    assert len(dataset) == len(labels)

    # Stride over the whole dataset so shard and intra-file batch boundaries
    # are crossed; also cover both ends explicitly.
    indices = [0, len(labels) - 1] + list(range(0, len(labels), 997))
    for index in indices:
        sample = dataset.get_encoded(index)
        assert sample["label"] == labels[index], f"row {index}"


def test_decoded_batch_spans_files(
    arrow_files: list[Path],
    dataset: rivet.ArrowDataset,
    labels: np.ndarray,
) -> None:
    loader = (
        rivet.scan_arrow(arrow_files, image_column=IMAGE_COLUMN)
        .decode_image()
        .resize(224, 224)
        .batch(32)
        .execute()
    )

    batch = next(loader)
    assert batch["images"].shape == (32, 224, 224, 3)
    assert batch["dtype"] == "uint8"
    assert batch["labels"].tolist() == labels[:32].tolist()

    batch = next(loader)
    assert batch["images"].shape == (32, 224, 224, 3)
    assert batch["labels"].tolist() == labels[32:64].tolist()

    first = dataset.get_encoded(0)["image"]
    last = dataset.get_encoded(len(labels) - 1)["image"]
    assert len(first) > 0
    assert len(last) > 0
