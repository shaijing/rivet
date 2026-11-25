"""Canonical image namespace for Rivet.

Includes image pipelines, structured detection/segmentation samples and
decoded video temporal sampling. Pixel processing runs in the native runtime;
Python owns configuration and framework handoff.
"""

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
from ._download import download_dataset
from ._io import (
    hf_arrow_files,
    load_hf_arrow_files,
    load_hf_image_batch,
    read_hf_image_batch,
    read_image_batch,
)
from ._loader import DataLoader
from ._pipeline import Pipeline
from ._structured import (
    AnnotatedSample,
    CutMix,
    MixUp,
    StructuredDataLoader,
    StructuredPipeline,
    UniformTemporalSubsample,
    VideoSample,
    uniform_temporal_subsample,
)
from ._transforms import Compose, Transform

__all__ = [
    "AnnotatedSample",
    "ArrowDataset",
    "Compose",
    "CutMix",
    "DataLoader",
    "DatasetDict",
    "ImageFolder",
    "LanceDataset",
    "MixUp",
    "Pipeline",
    "StructuredDataLoader",
    "StructuredPipeline",
    "Transform",
    "UniformTemporalSubsample",
    "VideoSample",
    "dataset",
    "download_dataset",
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
    "uniform_temporal_subsample",
]
