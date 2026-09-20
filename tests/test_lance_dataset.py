from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from PIL import Image

import rivet

PNG_1X1 = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90\x77\x53\xde\x00\x00\x00\x0c\x49\x44\x41\x54\x78\x9c\x63\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82"


def _png_2x1() -> bytes:
    output = BytesIO()
    with Image.new("RGB", (2, 1)) as image:
        image.putdata([(255, 0, 0), (0, 0, 255)])
        image.save(output, format="PNG")
    return output.getvalue()


def _write_split(path: Path, labels: list[int], image: bytes = b"encoded") -> None:
    schema = pa.schema(
        [
            pa.field("image", pa.binary(), nullable=False),
            pa.field("label", pa.int32(), nullable=False),
        ]
    )
    rows = [{"image": image, "label": label} for label in labels]
    lance.write_dataset(
        pa.RecordBatch.from_pylist(rows, schema=schema),
        str(path),
        schema=schema,
        mode="overwrite",
    )


def _write_custom_split(path: Path, labels: list[int]) -> None:
    schema = pa.schema(
        [
            pa.field("jpeg_bytes", pa.binary(), nullable=False),
            pa.field("target", pa.int64(), nullable=False),
        ]
    )
    lance.write_dataset(
        pa.RecordBatch.from_pylist(
            [{"jpeg_bytes": PNG_1X1, "target": label} for label in labels], schema=schema
        ),
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


def test_v2_manifest_binds_semantic_features_to_lance_columns(tmp_path: Path) -> None:
    _write_custom_split(tmp_path / "train.lance", [4, 7])
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "dataset": {
                    "name": "custom-images",
                    "version": "1.0",
                    "modality": "image",
                },
                "features": {
                    "input": {
                        "type": "image",
                        "column": "jpeg_bytes",
                        "representation": "encoded",
                        "encoding": "png",
                    },
                    "target": {
                        "type": "class_label",
                        "column": "target",
                        "dtype": "int64",
                        "num_classes": 10,
                    },
                },
                "splits": {"train": {"uri": "train.lance", "num_rows": 2}},
                "created_by": {"tool": "rivet", "version": "0.1.0"},
                "extensions": {"example.vendor": {"enabled": True}},
            }
        )
    )

    dataset = rivet.load_dataset(tmp_path, image_column="ignored", label_column="ignored")
    assert dataset.keys() == ["train"]
    assert dataset["train"].get_encoded(1)["label"] == 7


def test_v2_manifest_rejects_missing_semantic_binding(tmp_path: Path) -> None:
    _write_split(tmp_path / "train.lance", [0])
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "dataset": {"name": "invalid", "modality": "image"},
                "features": {
                    "input": {
                        "type": "image",
                        "column": "image",
                        "representation": "encoded",
                    }
                },
                "splits": {"train": {"uri": "train.lance", "num_rows": 1}},
            }
        )
    )

    with pytest.raises(ValueError, match="class_label"):
        rivet.load_dataset(tmp_path)


def test_v2_manifest_validates_class_label_dtype(tmp_path: Path) -> None:
    _write_split(tmp_path / "train.lance", [0])
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "dataset": {"name": "invalid", "modality": "image"},
                "features": {
                    "input": {
                        "type": "image",
                        "column": "image",
                        "representation": "encoded",
                    },
                    "target": {
                        "type": "class_label",
                        "column": "label",
                        "dtype": "int64",
                    },
                },
                "splits": {"train": {"uri": "train.lance", "num_rows": 1}},
            }
        )
    )

    with pytest.raises(ValueError, match="expects int64"):
        rivet.load_dataset(tmp_path)


def test_encoded_cache_is_explicit_and_keeps_original_lazy_dataset(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path / "train.lance", [0, 1, 2], image=PNG_1X1)

    lazy = rivet.load_dataset(tmp_path / "train.lance")
    cached = lazy.cache("encoded", chunk_size=1)

    assert cached is not lazy
    assert len(cached) == len(lazy) == 3
    assert [cached.get_encoded(index)["label"] for index in [2, 0, 2]] == [2, 0, 2]
    assert [lazy.get_encoded(index)["label"] for index in [2, 0, 2]] == [2, 0, 2]

    decoded = lazy.cache("decoded", chunk_size=1)
    assert [decoded.get_decoded(index)["labels"].tolist() for index in [2, 0]] == [
        [2],
        [0],
    ]
    batch = next(decoded.pipeline().batch(2).execute())
    assert batch["images"].shape == (2, 1, 1, 3)
    assert batch["labels"].tolist() == [0, 1]
    lazy_batch = next(lazy.pipeline().decode_image().batch(2).execute())
    assert batch["images"].tolist() == lazy_batch["images"].tolist()

    with pytest.raises(ValueError, match="Decode requires an encoded image"):
        decoded.pipeline().decode_image().batch(1).execute()
    with pytest.raises(ValueError, match="max_bytes"):
        lazy.cache("decoded", max_bytes=2)
    with pytest.raises(ValueError, match="only supported for decoded"):
        lazy.cache("encoded", max_bytes=1)
    with pytest.raises(ValueError, match="chunk_size"):
        lazy.cache(chunk_size=0)


def test_decoded_cache_skips_decode_and_preserves_random_pipeline(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path / "train.lance", [0], image=_png_2x1())

    lazy = rivet.load_dataset(tmp_path / "train.lance")
    decoded = lazy.cache("decoded")

    def run(dataset: rivet.LanceDataset, seed: int, decode: bool) -> bytes:
        pipeline = dataset.pipeline()
        if decode:
            pipeline = pipeline.decode_image()
        batch = next(
            pipeline
            .random_horizontal_flip(0.5)
            .normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            .shuffle(seed)
            .batch(1)
            .execute()
        )
        assert batch["images"].shape == (1, 1, 2, 3)
        assert batch["images"].dtype.name == "float32"
        return batch["images"].tobytes()

    lazy_outputs = [run(lazy, seed, decode=True) for seed in range(32)]
    decoded_outputs = [run(decoded, seed, decode=False) for seed in range(32)]

    assert decoded_outputs == lazy_outputs
    assert len(set(decoded_outputs)) > 1
