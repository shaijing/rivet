"""Shared paths and small measurement helpers for the benchmark."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - resource is Unix-only
    resource = None

DEFAULT_LANCE_ROOT = Path(
    os.environ.get("RIVET_CIFAR10_ROOT", "/data/datasets/rivet/cifar10")
)


def _train_path(lance_root: Path) -> Path:
    if lance_root.name.endswith(".lance"):
        return lance_root
    train_path = lance_root / "train.lance"
    if not train_path.is_dir():
        raise SystemExit(f"train Lance split not found: {train_path}")
    return train_path


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
