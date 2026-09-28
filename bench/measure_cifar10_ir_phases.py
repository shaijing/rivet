"""Measure pipeline startup separately from decoded CIFAR iteration.

Use PYTHONPATH to select isolated release packages for alternating A/B runs.
The standard comparison benchmark includes startup in its epoch timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import rivet
from rivet import _rivet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rivet-root", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--startup-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.epochs < 1 or args.warmup < 0 or args.startup_repeats < 1:
        parser.error("epochs/startup-repeats must be positive; warmup must be nonnegative")
    path = args.rivet_root
    if not path.name.endswith(".lance"):
        path /= "train.lance"
    cached = rivet.load_dataset(path).cache("decoded", chunk_size=4096)
    expected_rows = len(cached)

    def pipeline():
        return (
            cached.pipeline()
            .random_crop(32, 32, 4)
            .random_horizontal_flip(0.5)
            .normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
            .hwc_to_chw()
            .seed(args.seed)
            .workers(args.workers)
            .prefetch_batches(2)
            .batch(args.batch)
        )

    phases: dict[str, list[float]] = {
        name: [] for name in ("startup", "first_batch", "remaining", "cleanup", "total")
    }
    digest = None
    for epoch in range(-args.warmup, args.epochs):
        builder = pipeline()
        start = time.perf_counter_ns()
        loader = builder.execute()
        compiled = time.perf_counter_ns()
        first = next(loader)
        first_ready = time.perf_counter_ns()
        rows = len(first["labels"])
        for output in loader:
            rows += len(output["labels"])
        finished = time.perf_counter_ns()
        del loader
        cleaned = time.perf_counter_ns()
        if rows != expected_rows:
            raise RuntimeError(f"expected {expected_rows} rows, received {rows}")
        if digest is None:
            # Outside timing; the same seed must produce the same first batch.
            digest = hashlib.sha256(
                first["images"].tobytes() + first["labels"].tobytes()
            ).hexdigest()
        if epoch >= 0:
            for name, elapsed in zip(
                phases,
                (
                    compiled - start,
                    first_ready - compiled,
                    finished - first_ready,
                    cleaned - finished,
                    cleaned - start,
                ),
                strict=True,
            ):
                phases[name].append(elapsed / 1e6)

    startup_only = []
    builder = pipeline()
    for _ in range(args.startup_repeats):
        start = time.perf_counter_ns()
        loader = builder.execute()
        startup_only.append((time.perf_counter_ns() - start) / 1e6)
        del loader
    print(
        json.dumps(
            {
                "extension": _rivet.__file__,
                "rows": expected_rows,
                "seed": args.seed,
                "batch": args.batch,
                "workers": args.workers,
                "epochs": args.epochs,
                "warmup": args.warmup,
                "startup_repeats": args.startup_repeats,
                "first_batch_sha256": digest,
                "median_ms": {
                    name: statistics.median(values) for name, values in phases.items()
                },
                "startup_only_median_ms": statistics.median(startup_only),
                "mean_throughput": expected_rows / (statistics.mean(phases["total"]) / 1000),
                "median_throughput": expected_rows / (statistics.median(phases["total"]) / 1000),
                "epoch_ms": phases,
            }
        )
    )


if __name__ == "__main__":
    main()
