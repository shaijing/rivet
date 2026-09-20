"""Canonical image namespace for Rivet.

Only image APIs live here for now. Future modalities should get their own
namespace only when their public implementation exists; this module does not
reserve placeholder namespaces.
"""

from . import (
    ArrowDataset,
    DataLoader,
    DatasetDict,
    ImageFolder,
    LanceDataset,
    Pipeline,
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
    "DataLoader",
    "DatasetDict",
    "ImageFolder",
    "LanceDataset",
    "Pipeline",
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
