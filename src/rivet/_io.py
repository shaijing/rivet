"""Arrow batch readers and Hugging Face cache helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ._rivet import read_image_batch as _read_image_batch
from ._utils import _maybe_numpy_batch, _normalize_arrow_files, _validate_hf_dataset


def hf_arrow_files(dataset: Any) -> list[str]:
    """Return local Hugging Face Arrow cache files for a loaded dataset split."""
    _validate_hf_dataset(dataset)
    files: list[str] = []

    for cache_file in getattr(dataset, "cache_files", []):
        filename = cache_file.get("filename") if isinstance(cache_file, dict) else None
        if filename:
            files.append(str(Path(filename)))

    if not files:
        raise ValueError(
            "dataset has no local Arrow cache files; pass a loaded Hugging Face dataset split"
        )

    return files


def load_hf_arrow_files(
    path: str,
    *,
    split: str,
    name: str | None = None,
    **load_dataset_kwargs: Any,
) -> list[str]:
    """Load a Hugging Face dataset split and return its local Arrow cache files."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "load_hf_arrow_files requires the `datasets` package; install it with "
            "`pip install datasets`"
        ) from exc

    dataset = load_dataset(path, name=name, split=split, **load_dataset_kwargs)
    return hf_arrow_files(dataset)


def read_image_batch(
    arrow_files: Iterable[str | Path],
    batch_size: int,
    *,
    start: int = 0,
    image_column: str = "img",
    label_column: str = "label",
    as_numpy: bool = True,
) -> dict[str, Any]:
    """Read decoded RGB images from Arrow files through the Rust backend.

    The returned image axis order is NHWC and dtype is uint8. NumPy values
    are read-only zero-copy views over Rust-owned CPU storage.
    """
    result = _read_image_batch(
        _normalize_arrow_files(arrow_files),
        batch_size,
        start,
        image_column,
        label_column,
    )
    return _maybe_numpy_batch(result, as_numpy)


def read_hf_image_batch(
    dataset: Any,
    batch_size: int,
    *,
    start: int = 0,
    image_column: str = "img",
    label_column: str = "label",
    as_numpy: bool = True,
) -> dict[str, Any]:
    """Read a decoded RGB batch from a loaded Hugging Face dataset split."""
    return read_image_batch(
        hf_arrow_files(dataset),
        batch_size,
        start=start,
        image_column=image_column,
        label_column=label_column,
        as_numpy=as_numpy,
    )


def load_hf_image_batch(
    path: str,
    *,
    split: str,
    batch_size: int,
    start: int = 0,
    name: str | None = None,
    image_column: str = "img",
    label_column: str = "label",
    as_numpy: bool = True,
    **load_dataset_kwargs: Any,
) -> dict[str, Any]:
    """Load a Hugging Face split, pass its Arrow files to Rust, and return a batch."""
    return read_image_batch(
        load_hf_arrow_files(path, split=split, name=name, **load_dataset_kwargs),
        batch_size,
        start=start,
        image_column=image_column,
        label_column=label_column,
        as_numpy=as_numpy,
    )
