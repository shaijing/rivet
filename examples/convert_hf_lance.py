"""Convert Hugging Face-style Lance image datasets to Rivet's native schema.

The input image column is expected to look like:

    img: struct<bytes: binary, path: string>

The output schema is:

    image: binary
    label: int32

Examples:

    python examples/convert_hf_lance.py \
        /data/datasets/rivet/cifar10 \
        /data/datasets/rivet/cifar10_rivet

    python examples/convert_hf_lance.py \
        /data/datasets/rivet/cifar100 \
        /data/datasets/rivet/cifar100_rivet \
        --label-column fine_label \
        --keep-path
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pyarrow.compute as pc

MANIFEST_NAME = "dataset.rivet.json"


def _is_lance_path(path: Path) -> bool:
    return path.name.endswith(".lance")


def _split_inputs(input_path: Path) -> list[tuple[str | None, Path]]:
    if _is_lance_path(input_path):
        return [(None, input_path)]

    if not input_path.is_dir():
        raise ValueError(f"input must be a .lance dataset or directory: {input_path}")

    splits = sorted(
        (
            entry.stem,
            entry,
        )
        for entry in input_path.iterdir()
        if entry.is_dir() and entry.suffix == ".lance"
    )
    if not splits:
        raise ValueError(
            f"no immediate *.lance split directories found in {input_path}"
        )
    return splits


def _image_bytes_array(
    batch: pa.RecordBatch,
    image_column: str,
) -> pa.Array:
    image = batch.column(image_column)
    if not pa.types.is_struct(image.type):
        raise ValueError(
            f"{image_column!r} must be a struct column with a bytes field; "
            f"got {image.type}"
        )

    try:
        image_bytes = image.field("bytes")
    except KeyError as exc:
        raise ValueError(f"{image_column!r} has no bytes field") from exc

    if not pa.types.is_binary(image_bytes.type) and not pa.types.is_large_binary(
        image_bytes.type
    ):
        raise ValueError(
            f"{image_column}.bytes must be binary or large_binary; "
            f"got {image_bytes.type}"
        )
    if image_bytes.null_count:
        raise ValueError(f"{image_column}.bytes contains null values")

    if pa.types.is_large_binary(image_bytes.type):
        return pc.cast(image_bytes, pa.binary(), safe=True)
    return image_bytes


def _label_array(batch: pa.RecordBatch, label_column: str) -> pa.Array:
    labels = batch.column(label_column)
    if not pa.types.is_integer(labels.type):
        raise ValueError(
            f"{label_column!r} must be an integer column; got {labels.type}"
        )
    if labels.null_count:
        raise ValueError(f"{label_column!r} contains null values")
    try:
        return pc.cast(labels, pa.int32(), safe=True)
    except pa.ArrowInvalid as exc:
        raise ValueError(
            f"{label_column!r} contains values outside the int32 range"
        ) from exc


def _path_array(batch: pa.RecordBatch, image_column: str) -> pa.Array:
    image = batch.column(image_column)
    try:
        paths = image.field("path")
    except KeyError as exc:
        raise ValueError(
            f"{image_column!r} has no path field; remove --keep-path or use an "
            "input containing img.path"
        ) from exc
    if not pa.types.is_string(paths.type) and not pa.types.is_large_string(paths.type):
        raise ValueError(f"{image_column}.path must be string; got {paths.type}")
    if pa.types.is_large_string(paths.type):
        return pc.cast(paths, pa.string(), safe=True)
    return paths


def _output_schema(keep_path: bool) -> pa.Schema:
    fields = [
        pa.field("image", pa.binary(), nullable=False),
        pa.field("label", pa.int32(), nullable=False),
    ]
    if keep_path:
        fields.append(pa.field("path", pa.string(), nullable=True))
    return pa.schema(fields)


def _source_columns(
    dataset: lance.LanceDataset,
    image_column: str,
    label_column: str,
) -> None:
    schema = dataset.schema
    if image_column not in schema.names:
        raise ValueError(f"image column {image_column!r} is missing from {schema}")
    if label_column not in schema.names:
        raise ValueError(f"label column {label_column!r} is missing from {schema}")

    image_type = schema.field(image_column).type
    if not pa.types.is_struct(image_type):
        raise ValueError(
            f"{image_column!r} must be a struct column with a bytes field; "
            f"got {image_type}"
        )
    image_field_names = {field.name for field in image_type}
    if "bytes" not in image_field_names:
        raise ValueError(f"{image_column!r} has no bytes field")


def _converted_batches(
    dataset: lance.LanceDataset,
    *,
    image_column: str,
    label_column: str,
    keep_path: bool,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    columns = [image_column, label_column]
    schema = _output_schema(keep_path)
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        arrays = [
            _image_bytes_array(batch, image_column),
            _label_array(batch, label_column),
        ]

        if keep_path:
            arrays.append(_path_array(batch, image_column))

        yield pa.RecordBatch.from_arrays(arrays, schema=schema)


def _convert_split(
    input_path: Path,
    output_path: Path,
    *,
    image_column: str,
    label_column: str,
    keep_path: bool,
    batch_size: int,
) -> int:
    source = lance.dataset(str(input_path))
    _source_columns(source, image_column, label_column)
    schema = _output_schema(keep_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted = lance.write_dataset(
        _converted_batches(
            source,
            image_column=image_column,
            label_column=label_column,
            keep_path=keep_path,
            batch_size=batch_size,
        ),
        str(output_path),
        schema=schema,
        mode="overwrite",
    )
    rows = converted.count_rows()
    print(f"{input_path} -> {output_path}: {rows} rows")
    return rows


def _manifest(
    input_path: Path,
    outputs: list[tuple[str, Path, int]],
) -> dict[str, Any]:
    name = input_path.name
    return {
        "format_version": 1,
        "name": name,
        "splits": {
            split: {
                "path": output.name,
                "num_rows": rows,
            }
            for split, output, rows in outputs
        },
    }


def convert(
    input_path: Path,
    output_path: Path,
    *,
    image_column: str,
    label_column: str,
    keep_path: bool,
    batch_size: int,
    overwrite: bool,
) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path or output_path.is_relative_to(input_path):
        raise ValueError("output must not be the input path or a child of the input")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"output already exists: {output_path}; pass --overwrite to replace it"
            )
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    split_inputs = _split_inputs(input_path)
    if len(split_inputs) == 1 and split_inputs[0][0] is None:
        _convert_split(
            split_inputs[0][1],
            output_path,
            image_column=image_column,
            label_column=label_column,
            keep_path=keep_path,
            batch_size=batch_size,
        )
        return

    output_path.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[str, Path, int]] = []
    for split, split_input in split_inputs:
        assert split is not None
        split_output = output_path / f"{split}.lance"
        rows = _convert_split(
            split_input,
            split_output,
            image_column=image_column,
            label_column=label_column,
            keep_path=keep_path,
            batch_size=batch_size,
        )
        outputs.append((split, split_output, rows))

    (output_path / MANIFEST_NAME).write_text(
        json.dumps(_manifest(input_path, outputs), indent=2) + "\n"
    )
    print(f"wrote manifest: {output_path / MANIFEST_NAME}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face Lance image columns to Rivet's native schema."
    )
    parser.add_argument("input", type=Path, help="one .lance dataset or a split root")
    parser.add_argument("output", type=Path, help="output .lance dataset or split root")
    parser.add_argument("--image-column", default="img")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--keep-path",
        action="store_true",
        help="copy img.path to an optional native path column",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output path if it already exists",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    convert(
        args.input,
        args.output,
        image_column=args.image_column,
        label_column=args.label_column,
        keep_path=args.keep_path,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
