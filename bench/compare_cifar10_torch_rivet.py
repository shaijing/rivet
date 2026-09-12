"""Benchmark PyTorch and Rivet on a shared CIFAR-10 Lance split.

The implementation is organized under :mod:`cifar10_benchmark`; this file is
kept as the stable executable entry point.

Usage, after building the release extension:

    maturin develop --release
    python bench/compare_cifar10_torch_rivet.py --modes A,B,C
        --batch 64 --batches 0 --resize 64
        --rivet-workers 4 --torch-workers 4 --epochs 2

Add ``--cache-compare`` to report Lance lazy versus encoded in-memory Rivet
performance, including cache construction and per-epoch timings.

Use ``--lance-root`` to select another converted CIFAR-10 root. The default is
``/data/datasets/rivet/cifar10``. Add ``--sweep`` for a Rivet worker scaling
table.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .cifar10_benchmark import (
        DEFAULT_LANCE_ROOT,
        TorchLanceDataset,
        _peak_rss_mb,
        _rivet_checks,
        _rivet_pipeline,
        _train_path,
        bench_rivet,
        bench_rivet_encoded_cache,
        bench_torch,
        detect_encoding,
        main,
        sweep_rivet,
    )
else:
    # Direct execution puts ``bench`` on sys.path, so make the sibling
    # benchmark package importable without requiring an installation step.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cifar10_benchmark import (
        DEFAULT_LANCE_ROOT,
        TorchLanceDataset,
        _peak_rss_mb,
        _rivet_checks,
        _rivet_pipeline,
        _train_path,
        bench_rivet,
        bench_rivet_encoded_cache,
        bench_torch,
        detect_encoding,
        main,
        sweep_rivet,
    )

__all__ = [
    "DEFAULT_LANCE_ROOT",
    "TorchLanceDataset",
    "_peak_rss_mb",
    "_rivet_checks",
    "_rivet_pipeline",
    "_train_path",
    "bench_rivet",
    "bench_rivet_encoded_cache",
    "bench_torch",
    "detect_encoding",
    "main",
    "sweep_rivet",
]


if __name__ == "__main__":
    main()
