"""Compare torchvision's native CIFAR-10 loader with Rivet decoded cache.

The two backends use the same standard CIFAR-10 training transform:

    RandomCrop(32, padding=4)
    RandomHorizontalFlip()
    ToTensor()
    Normalize(mean, std)

Torch uses ``torchvision.datasets.CIFAR10`` and its regular ``DataLoader``.
Rivet materializes the Lance split with ``cache("decoded")`` once, then runs
the random transforms and normalization without a decode operation.

Example:

    .venv/bin/python bench/compare_cifar10_torchvision_rivet_decoded.py \
        --torch-root /data/datasets/pytorch \
        --rivet-root /data/datasets/rivet/cifar10 \
        --batch 128 --workers 4 --epochs 3
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from cifar10_benchmark.common import DEFAULT_LANCE_ROOT, _train_path, detect_encoding

import rivet

DEFAULT_TORCHVISION_ROOT = Path(
    os.environ.get("TORCHVISION_CIFAR10_ROOT", "/data/datasets/pytorch")
)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _build_torch_transform(crop_size: int, padding: int) -> Any:
    try:
        from torchvision import transforms
    except ImportError as error:
        raise SystemExit(
            "Torch benchmark requires torch and torchvision to be installed"
        ) from error

    return transforms.Compose(
        [
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _check_torch_batch(batch: Any, crop_size: int) -> int:
    import torch

    assert isinstance(batch, (list, tuple)) and len(batch) == 2
    images, labels = batch
    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert images.shape[1:] == (3, crop_size, crop_size)
    assert images.dtype == torch.float32
    return int(images.shape[0])


def _check_rivet_batch(batch: dict[str, Any], crop_size: int) -> int:
    import numpy as np

    images = batch["images"]
    assert isinstance(images, np.ndarray)
    assert images.shape[1:] == (3, crop_size, crop_size)
    assert images.dtype == np.float32
    return len(images)


def _summarize(epoch_seconds: list[float], epoch_rows: list[int]) -> dict[str, float]:
    later_seconds = epoch_seconds[1:]
    later_rows = epoch_rows[1:]
    return {
        "first_epoch_seconds": epoch_seconds[0] if epoch_seconds else 0.0,
        "first_epoch_images_per_second": (
            epoch_rows[0] / epoch_seconds[0]
            if epoch_seconds and epoch_seconds[0]
            else 0.0
        ),
        "steady_epoch_seconds": (
            sum(later_seconds) / len(later_seconds) if later_seconds else 0.0
        ),
        "steady_images_per_second": (
            sum(later_rows) / sum(later_seconds)
            if later_seconds and sum(later_seconds)
            else 0.0
        ),
    }


def bench_torchvision(
    root: Path,
    batch_size: int,
    workers: int,
    epochs: int,
    crop_size: int,
    padding: int,
    batches: int,
    download: bool,
) -> dict[str, Any]:
    try:
        from torch.utils.data import DataLoader
        from torchvision.datasets import CIFAR10
    except ImportError as error:
        raise SystemExit(
            "Torch benchmark requires torch and torchvision to be installed"
        ) from error

    transform = _build_torch_transform(crop_size, padding)
    prepare_start = time.perf_counter()
    dataset = CIFAR10(
        root=str(root),
        train=True,
        transform=transform,
        download=download,
    )
    prepare_seconds = time.perf_counter() - prepare_start

    epoch_seconds: list[float] = []
    epoch_rows: list[int] = []
    for _ in range(epochs):
        loader_options: dict[str, Any] = {
            "batch_size": batch_size,
            "num_workers": workers,
            "shuffle": False,
            "drop_last": False,
        }
        if workers:
            loader_options["prefetch_factor"] = 2
        loader = DataLoader(dataset, **loader_options)

        rows = 0
        start = time.perf_counter()
        for index, output in enumerate(loader):
            if batches and index >= batches:
                break
            rows += _check_torch_batch(output, crop_size)
        epoch_seconds.append(time.perf_counter() - start)
        epoch_rows.append(rows)

    result = _summarize(epoch_seconds, epoch_rows)
    result["load_seconds"] = prepare_seconds
    result["rows"] = float(epoch_rows[0] if epoch_rows else 0)
    return result


def bench_rivet_decoded(
    lance_root: Path,
    batch_size: int,
    workers: int,
    epochs: int,
    crop_size: int,
    padding: int,
    batches: int,
    cache_chunk: int,
    max_bytes: int | None,
) -> dict[str, Any]:
    lance_path = _train_path(lance_root)
    load_start = time.perf_counter()
    dataset = rivet.load_dataset(lance_path)
    encoding = detect_encoding(dataset)
    load_seconds = time.perf_counter() - load_start
    cache_start = time.perf_counter()
    cached = dataset.cache(
        "decoded",
        chunk_size=cache_chunk,
        max_bytes=max_bytes,
    )
    cache_seconds = time.perf_counter() - cache_start

    epoch_seconds: list[float] = []
    epoch_rows: list[int] = []
    for _ in range(epochs):
        pipeline = (
            cached.pipeline()
            .random_crop(crop_size, crop_size, padding)
            .random_horizontal_flip(0.5)
            .normalize(list(CIFAR10_MEAN), list(CIFAR10_STD))
            .hwc_to_chw()
            .workers(workers)
            .prefetch_batches(2)
            .batch(batch_size)
        )

        rows = 0
        start = time.perf_counter()
        for index, output in enumerate(pipeline.execute()):
            if batches and index >= batches:
                break
            rows += _check_rivet_batch(output, crop_size)
        epoch_seconds.append(time.perf_counter() - start)
        epoch_rows.append(rows)

    result = _summarize(epoch_seconds, epoch_rows)
    result["load_seconds"] = load_seconds
    result["cache_seconds"] = cache_seconds
    result["rows"] = float(epoch_rows[0] if epoch_rows else 0)
    result["encoding"] = encoding
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare torchvision CIFAR-10 loading with Rivet decoded cache"
        )
    )
    parser.add_argument(
        "--torch-root",
        type=Path,
        default=DEFAULT_TORCHVISION_ROOT,
        help="torchvision dataset root (default: %(default)s)",
    )
    parser.add_argument(
        "--rivet-root",
        type=Path,
        default=DEFAULT_LANCE_ROOT,
        help="Rivet CIFAR-10 root or train.lance path (default: %(default)s)",
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batches", type=int, default=0, help="0 = drain all")
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--cache-chunk", type=int, default=4096)
    parser.add_argument("--max-bytes", type=int, default=None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="allow torchvision to download CIFAR-10 if absent",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.batch <= 0 or args.workers < 0 or args.epochs <= 0:
        raise SystemExit("batch and epochs must be positive; workers must be non-negative")
    if args.batches < 0 or args.crop_size <= 0 or args.padding < 0:
        raise SystemExit("batches/crop-size must be non-negative/positive and padding non-negative")
    if args.cache_chunk <= 0 or (args.max_bytes is not None and args.max_bytes < 0):
        raise SystemExit("cache-chunk must be positive and max-bytes must be non-negative")

    torch_result = bench_torchvision(
        args.torch_root,
        args.batch,
        args.workers,
        args.epochs,
        args.crop_size,
        args.padding,
        args.batches,
        args.download,
    )
    rivet_result = bench_rivet_decoded(
        args.rivet_root,
        args.batch,
        args.workers,
        args.epochs,
        args.crop_size,
        args.padding,
        args.batches,
        args.cache_chunk,
        args.max_bytes,
    )

    print(f"torchvision root        : {args.torch_root}")
    print(f"rivet root              : {_train_path(args.rivet_root)}")
    print(f"rivet encoding          : {rivet_result['encoding']}")
    print(
        f"batch={args.batch} workers={args.workers} epochs={args.epochs} "
        f"crop={args.crop_size} padding={args.padding} "
        f"batches={args.batches or 'all'}"
    )
    print()
    print(
        f"{'backend':<14} {'load s':>9} {'cache s':>10} {'first s':>11} "
        f"{'steady s':>11} {'steady img/s':>15} {'speedup':>9}"
    )
    torch_rate = torch_result["steady_images_per_second"]
    rivet_rate = rivet_result["steady_images_per_second"]
    print(
        f"{'torchvision':<14} {torch_result['load_seconds']:>9.3f} "
        f"{'-':>10} "
        f"{torch_result['first_epoch_seconds']:>11.3f} "
        f"{torch_result['steady_epoch_seconds']:>11.3f} "
        f"{torch_rate:>15.0f} {'1.00x':>9}"
    )
    print(
        f"{'rivet decoded':<14} {rivet_result['load_seconds']:>9.3f} "
        f"{rivet_result['cache_seconds']:>10.3f} "
        f"{rivet_result['first_epoch_seconds']:>11.3f} "
        f"{rivet_result['steady_epoch_seconds']:>11.3f} "
        f"{rivet_rate:>15.0f} "
        f"{rivet_rate / torch_rate if torch_rate else float('nan'):>8.2f}x"
    )
    print(
        "load s = dataset/source initialization; cache s = one-time Rivet "
        "decoded-cache materialization"
    )


if __name__ == "__main__":
    main()
