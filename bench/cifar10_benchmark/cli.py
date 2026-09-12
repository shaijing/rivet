"""Command-line orchestration for the CIFAR-10 Lance benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import rivet

from .common import DEFAULT_LANCE_ROOT, _train_path, detect_encoding
from .rivet_backend import bench_rivet, bench_rivet_encoded_cache
from .torch_backend import bench_torch


def sweep_rivet(
    dataset: rivet.LanceDataset,
    batch: int,
    epochs: int,
    resize: int,
) -> None:
    workloads = ["decode", "decode+resize", "decode+resize+floatCHW"]
    worker_configs = [
        (0, 0),
        (1, 0),
        (2, 0),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 2),
    ]
    print()
    print("Rivet scaling sweep (images/s, full drain per row):")
    header = f"{'workload':<30} " + " ".join(
        f"w{workers}/p{prefetch:<5}" for workers, prefetch in worker_configs
    )
    print(header)
    for workload in workloads:
        row = [f"{workload:<30}"]
        mode = "A" if workload == "decode" else "B"
        if workload == "decode+resize+floatCHW":
            mode = "C"
        for workers, prefetch in worker_configs:
            rate = bench_rivet(
                dataset,
                mode,
                resize,
                batch,
                0,
                workers,
                epochs,
            )
            row.append(f"{rate:>10.0f}")
        print(" ".join(row))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch and Rivet on a shared CIFAR-10 Lance split"
    )
    parser.add_argument("--lance-root", type=Path, default=DEFAULT_LANCE_ROOT)
    parser.add_argument("--modes", default="A,B,C")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--batches", type=int, default=0, help="batches per pass; 0 = drain all"
    )
    parser.add_argument("--resize", type=int, default=64)
    parser.add_argument("--rivet-workers", type=int, default=4)
    parser.add_argument("--torch-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--cache-compare",
        action="store_true",
        help="compare lazy Lance access with an encoded in-memory cache",
    )
    parser.add_argument(
        "--cache-chunk",
        type=int,
        default=4096,
        help="rows fetched per Lance request while constructing the encoded cache",
    )
    parser.add_argument(
        "--sweep", action="store_true", help="print a Rivet worker scaling table"
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    train_dataset = rivet.load_dataset(args.lance_root, split="train")
    train_path = _train_path(args.lance_root)
    encoding = detect_encoding(train_dataset)
    print(f"shared Lance dataset     : {train_path}")
    print(f"encoded rows detected    : {encoding}")
    print(f"rows                     : {len(train_dataset)}")
    print(
        f"batches={args.batches or 'all'} batch={args.batch} resize={args.resize} "
        f"epochs={args.epochs} (one warm-up batch before each timed pass)"
    )

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    print()
    print(f"{'mode':<8} {'rivet img/s':>12} {'torch img/s':>12} {'ratio':>7}")
    for mode in modes:
        rivet_rate = bench_rivet(
            train_dataset,
            mode,
            args.resize,
            args.batch,
            args.batches,
            args.rivet_workers,
            args.epochs,
        )
        torch_rate = bench_torch(
            train_path,
            mode,
            args.resize,
            args.batch,
            args.batches,
            args.torch_workers,
            args.epochs,
        )
        ratio = rivet_rate / torch_rate if torch_rate else float("nan")
        print(f"{mode:<8} {rivet_rate:>12.0f} {torch_rate:>12.0f} {ratio:>6.2f}x")

    if args.sweep:
        sweep_rivet(train_dataset, args.batch, args.epochs, args.resize)

    if args.cache_compare:
        print()
        print("Rivet cache comparison (same pipeline and sampler):")
        print(
            f"{'mode':<8} {'cache s':>10} {'first s':>10} {'later s':>10} "
            f"{'later img/s':>14} {'peak RSS MB':>13}"
        )
        for mode in modes:
            result = bench_rivet_encoded_cache(
                train_dataset,
                mode,
                args.resize,
                args.batch,
                args.batches,
                args.rivet_workers,
                args.epochs,
                args.cache_chunk,
            )
            print(
                f"{mode:<8} {result['cache_seconds']:>10.3f} "
                f"{result['first_epoch_seconds']:>10.3f} "
                f"{result['later_epoch_seconds']:>10.3f} "
                f"{result['later_epoch_images_per_second']:>14.0f} "
                f"{result['peak_rss_mb']:>13.1f}"
            )
