"""Shared conversions and validation for the Python image API."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from pathlib import Path
from typing import Any


def _normalize_arrow_files(arrow_files: Iterable[str | Path]) -> list[str]:
    return [str(Path(path)) for path in arrow_files]


def _normalize_fill(fill: float | Iterable[float]) -> float | list[float]:
    return fill if isinstance(fill, (int, float)) else list(fill)


def _normalize_jpeg_quality(quality: int | Iterable[int]) -> list[int]:
    values = [quality, quality] if isinstance(quality, Integral) else list(quality)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise TypeError("JPEG quality must be an integer or a pair of integers")
    return [int(value) for value in values]


def _validate_hf_dataset(dataset: Any) -> None:
    if getattr(dataset, "_indices", None) is not None:
        raise ValueError(
            "Rivet currently requires a contiguous Hugging Face Dataset; "
            "call dataset.flatten_indices() first."
        )


def _maybe_numpy_batch(batch: dict[str, Any], as_numpy: bool) -> dict[str, Any]:
    """Keep the native read-only NumPy views or serialize them explicitly."""
    if as_numpy:
        return batch

    images = batch["images"]
    labels = batch["labels"]

    if hasattr(images, "tobytes"):
        batch["images"] = images.tobytes()
    if hasattr(labels, "tolist"):
        batch["labels"] = labels.tolist()

    return batch
