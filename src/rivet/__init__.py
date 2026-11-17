"""Rivet's backwards-compatible root namespace.

The canonical image API is available from :mod:`rivet.vision`. The root
exports remain during the namespace migration so existing applications can
upgrade without changing every import at once.
"""

from __future__ import annotations

from typing import Any

from ._datasets import (
    ArrowDataset,
    DatasetDict,
    ImageFolder,
    LanceDataset,
    dataset,
    load_dataset,
    scan_arrow,
    scan_hf,
    scan_image_folder,
    scan_lance,
)
from ._io import (
    hf_arrow_files,
    load_hf_arrow_files,
    load_hf_image_batch,
    read_hf_image_batch,
    read_image_batch,
)
from ._loader import DataLoader
from ._pipeline import Pipeline
from ._transforms import Compose, Transform

__all__ = [
    "ArrowDataset",
    "Compose",
    "DataLoader",
    "DatasetDict",
    "ImageFolder",
    "LanceDataset",
    "Pipeline",
    "Transform",
    "dataset",
    "hf_arrow_files",
    "load_dataset",
    "load_hf_arrow_files",
    "load_hf_image_batch",
    "read_hf_image_batch",
    "read_image_batch",
    "scan_arrow",
    "scan_hf",
    "scan_image_folder",
    "scan_lance",
    "vision",
]


def __getattr__(name: str) -> Any:
    if name == "vision":
        import importlib

        module = importlib.import_module(".vision", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
