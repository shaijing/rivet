"""Shared dataset loading and small measurement helpers for the benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import rivet

try:
    import resource
except ImportError:  # pragma: no cover - resource is Unix-only
    resource = None

CIFAR10_SOURCE = "torch:CIFAR10"
DEFAULT_DATASET = os.environ.get("RIVET_CIFAR10_ROOT", CIFAR10_SOURCE)


def add_dataset_args(parser: argparse.ArgumentParser, legacy_flag: str) -> None:
    parser.add_argument(
        "--dataset",
        legacy_flag,
        default=DEFAULT_DATASET,
        help="Rivet dataset source or local Lance path (default: %(default)s)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use cached dataset data without downloading",
    )


def load_train_dataset(
    source: str | Path, *, offline: bool = False
) -> rivet.LanceDataset:
    loaded = rivet.load_dataset(source, offline=offline)
    return loaded["train"] if isinstance(loaded, rivet.DatasetDict) else loaded


def detect_encoding(dataset: Any) -> str:
    raw = dataset.get_encoded(0)["image"]
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if raw.startswith(b"\xff\xd8"):
        return "JPEG"
    return f"unknown ({raw[:8].hex()})"


def _peak_rss_mb() -> float:
    """Return the process high-water RSS; Linux reports this in KiB."""
    if resource is None:
        return float("nan")
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.name == "nt":
        return value / (1024 * 1024)
    return value / 1024
