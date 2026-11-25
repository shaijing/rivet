"""Reusable components for the CIFAR-10 Lance benchmark."""

from .cli import main, sweep_rivet
from .common import DEFAULT_DATASET, _peak_rss_mb, detect_encoding
from .rivet_backend import (
    _rivet_checks,
    _rivet_pipeline,
    bench_rivet,
    bench_rivet_cache,
    bench_rivet_encoded_cache,
)
from .torch_backend import TorchLanceDataset, bench_torch

__all__ = [
    "DEFAULT_DATASET",
    "TorchLanceDataset",
    "_peak_rss_mb",
    "_rivet_checks",
    "_rivet_pipeline",
    "bench_rivet",
    "bench_rivet_cache",
    "bench_rivet_encoded_cache",
    "bench_torch",
    "detect_encoding",
    "main",
    "sweep_rivet",
]
