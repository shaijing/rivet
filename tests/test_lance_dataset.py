from __future__ import annotations

import json
from pathlib import Path

import lance
import pyarrow as pa
import pytest

import rivet


def _write_split(path: Path, labels: list[int]) -> None:
    schema = pa.schema(
        [
            pa.field("image", pa.binary(), nullable=False),
            pa.field("label", pa.int32(), nullable=False),
        ]
    )
    rows = [{"image": b"encoded", "label": label} for label in labels]
    lance.write_dataset(
        pa.RecordBatch.from_pylist(rows, schema=schema),
        str(path),
        schema=schema,
        mode="overwrite",
    )


def test_load_dataset_discovers_and_selects_splits(tmp_path: Path) -> None:
    _write_split(tmp_path / "train.lance", [0, 1])
    _write_split(tmp_path / "test.lance", [2])

    dataset = rivet.load_dataset(tmp_path)

    assert isinstance(dataset, rivet.DatasetDict)
    assert dataset.keys() == ["test", "train"]
    assert len(dataset) == 2
    assert "train" in dataset
    assert len(dataset["train"]) == 2
    assert dataset["train"].get_encoded(1)["label"] == 1
    cached_train = dataset["train"].cache("encoded", chunk_size=1)
    assert [cached_train.get_encoded(index)["label"] for index in [1, 0]] == [1, 0]
    assert isinstance(rivet.load_dataset(tmp_path, split="test"), rivet.LanceDataset)

    with pytest.raises(KeyError, match="available splits: test, train"):
        dataset["dev"]


def test_manifest_is_authoritative_and_physical_paths_stay_single(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path / "train.lance", [0, 1])
    _write_split(tmp_path / "ignored.lance", [2, 3])
    (tmp_path / "dataset.rivet.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "name": "fixture",
                "splits": {
                    "train": {"path": "train.lance", "num_rows": 2},
                },
            }
        )
    )

    dataset = rivet.load_dataset(tmp_path)
    assert dataset.keys() == ["train"]
    assert len(rivet.load_dataset(tmp_path / "train.lance")) == 2

    with pytest.raises(ValueError, match="single .lance"):
        rivet.load_dataset(tmp_path / "train.lance", split="train")


def test_encoded_cache_is_explicit_and_keeps_original_lazy_dataset(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path / "train.lance", [0, 1, 2])

    lazy = rivet.load_dataset(tmp_path / "train.lance")
    cached = lazy.cache("encoded", chunk_size=1)

    assert cached is not lazy
    assert len(cached) == len(lazy) == 3
    assert [cached.get_encoded(index)["label"] for index in [2, 0, 2]] == [2, 0, 2]
    assert [lazy.get_encoded(index)["label"] for index in [2, 0, 2]] == [2, 0, 2]

    with pytest.raises(ValueError, match="only the 'encoded' cache"):
        lazy.cache("decoded")
    with pytest.raises(ValueError, match="chunk_size"):
        lazy.cache(chunk_size=0)
