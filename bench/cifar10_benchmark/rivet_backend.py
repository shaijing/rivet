"""Rivet-side benchmark implementations."""

from __future__ import annotations

import time

import numpy as np

import rivet

from .common import _peak_rss_mb


def _rivet_pipeline(
    dataset: rivet.LanceDataset,
    mode: str,
    resize: int,
    batch: int,
    workers: int,
    prefetch: int,
):
    pipeline = dataset.pipeline().decode_image()
    pipeline = pipeline.resize(resize, resize)
    if mode == "C":
        pipeline = pipeline.normalize(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        )
    if mode in ("B", "C"):
        pipeline = pipeline.hwc_to_chw()
    return pipeline.workers(workers).prefetch_batches(prefetch).batch(batch)


def _rivet_checks(batch: dict[str, object], mode: str) -> None:
    images = batch["images"]
    assert isinstance(images, np.ndarray)
    if mode == "A":
        assert images.ndim == 4 and images.shape[3] == 3
        assert images.dtype == np.uint8
    else:
        assert images.shape[1] == 3, f"expected NCHW, got {images.shape}"
        expected = np.float32 if mode == "C" else np.uint8
        assert images.dtype == expected, f"mode {mode}: got {images.dtype}"


def bench_rivet(
    dataset: rivet.LanceDataset,
    mode: str,
    resize: int,
    batch: int,
    batches: int,
    workers: int,
    epochs: int,
) -> float:
    total = 0
    elapsed = 0.0
    for _ in range(epochs):
        loader = _rivet_pipeline(dataset, mode, resize, batch, workers, 2).execute()
        iterator = iter(loader)
        next(iterator)  # warm-up: thread startup and compile happen before timing
        start = time.perf_counter()
        for index, output in enumerate(iterator):
            if batches and index >= batches:
                break
            _rivet_checks(output, mode)
            total += len(output["images"])
        elapsed += time.perf_counter() - start
    return total / elapsed if elapsed else 0.0


def bench_rivet_encoded_cache(
    dataset: rivet.LanceDataset,
    mode: str,
    resize: int,
    batch: int,
    batches: int,
    workers: int,
    epochs: int,
    chunk_size: int,
) -> dict[str, float]:
    """Measure encoded cache construction and epoch timings separately."""
    cache_start = time.perf_counter()
    cached = dataset.cache("encoded", chunk_size=chunk_size)
    cache_seconds = time.perf_counter() - cache_start

    epoch_seconds: list[float] = []
    epoch_rows: list[int] = []
    for _ in range(epochs):
        rows = 0
        start = time.perf_counter()
        loader = _rivet_pipeline(
            cached, mode, resize, batch, workers, 2
        ).execute()
        for index, output in enumerate(loader):
            if batches and index >= batches:
                break
            _rivet_checks(output, mode)
            rows += len(output["images"])
        epoch_seconds.append(time.perf_counter() - start)
        epoch_rows.append(rows)

    later = epoch_seconds[1:]
    later_seconds = sum(later) / len(later) if later else 0.0
    later_images_per_second = (
        sum(epoch_rows[1:]) / sum(epoch_seconds[1:]) if later else 0.0
    )
    return {
        "cache_seconds": cache_seconds,
        "first_epoch_seconds": epoch_seconds[0] if epoch_seconds else 0.0,
        "later_epoch_seconds": later_seconds,
        "first_epoch_images_per_second": (
            epoch_rows[0] / epoch_seconds[0]
            if epoch_seconds and epoch_seconds[0]
            else 0.0
        ),
        "later_epoch_images_per_second": later_images_per_second,
        "peak_rss_mb": _peak_rss_mb(),
    }
