"""Canonical image namespace for Rivet.

Only image APIs live here for now. Future modalities should get their own
namespace only when their public implementation exists; this module does not
reserve placeholder namespaces.
"""

from . import (
    ArrowDataset,
    Compose,
    DataLoader,
    DatasetDict,
    ImageFolder,
    LanceDataset,
    Pipeline,
    Transform,
    dataset,
    hf_arrow_files,
    load_dataset,
    load_hf_arrow_files,
    load_hf_image_batch,
    read_hf_image_batch,
    read_image_batch,
    scan_arrow,
    scan_hf,
    scan_image_folder,
    scan_lance,
)

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
]
