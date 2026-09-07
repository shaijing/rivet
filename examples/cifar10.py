"""Simple CIFAR-10 example: load the dataset and iterate numpy batches.

A tiny self-test in the ``__main__`` block: every batch from the ``for``
loop must be plain numpy (``dtype=uint8``, ``NHWC`` images plus labels),
the loader must reach a sticky ``StopIteration``, and worker mode must
agree with the inline path.

Prerequisites (one of):

* ``pip install datasets`` then::

    from datasets import load_dataset
    load_dataset("uoft-cs/cifar10", split="train")   # fills the HF cache

* or point ``RIVET_TEST_ARROW_FILE`` at any CIFAR-10 Arrow IPC file.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

import rivet

BATCH_SIZE = 64
N_BATCHES = 4  # keep the demo quick; set to None to drain everything


def _arrow_file() -> Path:
    explicit = os.environ.get("RIVET_TEST_ARROW_FILE")
    if explicit:
        return Path(explicit)

    roots = [
        Path(os.environ.get("HF_DATASETS_CACHE", "")),
        Path(os.environ.get("HF_HOME", "")) / "datasets",
        Path.home() / ".cache" / "huggingface" / "datasets",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        matches = sorted(root.glob("uoft-cs___cifar10/**/cifar10-train.arrow"))
        if matches:
            return matches[0]

    raise SystemExit(
        "no CIFAR-10 Arrow file found: load 'uoft-cs/cifar10' with the "
        "datasets library or set RIVET_TEST_ARROW_FILE"
    )


def run(loader: object, max_batches: int | None) -> list[dict[str, object]]:
    """Exercise the documented pattern: `for batch in loader` -> numpy."""
    batches: list[dict[str, object]] = []
    for index, batch in enumerate(loader):  # type: ignore[assignment]
        if max_batches is not None and index >= max_batches:
            break
        assert isinstance(batch["images"], np.ndarray), "images must be numpy"
        x: np.ndarray = batch["images"]
        y: np.ndarray = batch["labels"]
        assert x.dtype == np.uint8, f"dtype {x.dtype}"
        assert len(x) == len(y)
        assert x.ndim == 4 and x.shape[3] == 3  # NHWC
        print(
            f"batch {index}: x={x.shape} dtype={x.dtype} "
            f"y={y.tolist()} labels {int(y.min())}..{int(y.max())}"
        )
        batches.append(batch)

    # Sticky EOF only applies after natural exhaustion (no batch cap).
    if max_batches is None:
        try:
            next(loader)  # type: ignore[arg-type]
        except StopIteration:
            pass
        else:
            raise AssertionError("loader must be exhausted after the for loop")
    return batches


def main() -> int:
    arrow_file = _arrow_file()
    print(f"arrow file: {arrow_file}")

    pipeline = (
        rivet.scan_arrow([arrow_file])
        .take(512)
        .decode_image()
        .batch(BATCH_SIZE)
    )

    inline = run(
        pipeline.execute(),
        N_BATCHES,
    )
    workers = run(
        pipeline.workers(4).prefetch_batches(2).execute(),
        N_BATCHES,
    )

    assert len(inline) == len(workers)
    for a, b in zip(inline, workers):
        assert np.array_equal(a["images"], b["images"])
        assert a["labels"].tolist() == b["labels"].tolist()
    print(
        f"ok: {len(inline)} batches x {BATCH_SIZE} rows, "
        "numpy results identical for inline and workers(4)+prefetch(2)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
