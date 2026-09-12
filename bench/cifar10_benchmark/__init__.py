"""Reusable components for the CIFAR-10 Lance benchmark."""

from .cli import main, sweep_rivet
from .common import DEFAULT_LANCE_ROOT, _peak_rss_mb, _train_path, detect_encoding
from .rivet_backend import (
    _rivet_checks,
    _rivet_pipeline,
    bench_rivet,
    bench_rivet_cache,
    bench_rivet_encoded_cache,
)
from .torch_backend import TorchLanceDataset, bench_torch

__all__ = [
    "DEFAULT_LANCE_ROOT",
    "TorchLanceDataset",
    "_peak_rss_mb",
    "_rivet_checks",
    "_rivet_pipeline",
    "_train_path",
    "bench_rivet",
    "bench_rivet_cache",
    "bench_rivet_encoded_cache",
    "bench_torch",
    "detect_encoding",
    "main",
    "sweep_rivet",
]
