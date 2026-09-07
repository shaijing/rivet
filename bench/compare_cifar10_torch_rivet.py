"""Benchmark: PyTorch (datasets -> PIL -> DataLoader) vs Rivet, same source.

Both loaders read the *same* Hugging Face CIFAR-10 Arrow cache, so rows
are identical compressed image bytes (PNG/JPEG - detected at runtime, not
hard-coded). Three output representations are compared:

mode A  user-facing paths  rivet: uint8 NHWC numpy
                           torch: float32 CHW tensor (Resize + ToTensor)
mode B  strict equivalent  both:  uint8 CHW  (rivet hwc_to_chw,
                           torch Resize + PILToTensor)
mode C  training input     both:  float32 CHW in [0, 1] (rivet /255 via
                           normalize(0,1) + hwc_to_chw, torch ToTensor)

Per-image work is decode + resize (same encoded bytes, different codec
and resize implementations). Wall-clock throughput in images/second over
`--epochs` passes, with one warm-up batch before every timed pass so
worker/process startup is excluded.

Usage (release build required for meaningful rust timings):

    maturin develop --release
    python bench/compare_cifar10_torch_rivet.py [--modes A,B,C]
        [--batch 64] [--batches 0] [--resize 64]
        [--rivet-workers 4] [--torch-workers 4] [--epochs 2]

`--batches 0` drains the whole dataset. Add `--sweep` to print a rivet
workers x prefetch scaling table (decode / +resize / +float CHW).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

import rivet


def rivet_arrow_file() -> Path:
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
        "no CIFAR-10 Arrow file: load 'uoft-cs/cifar10' with the datasets "
        "library or set RIVET_TEST_ARROW_FILE"
    )


def detect_encoding(arrow_file: Path) -> str:
    raw = rivet.ArrowDataset([arrow_file]).get_encoded(0)["image"]
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if raw.startswith(b"\xff\xd8"):
        return "JPEG"
    return f"unknown ({raw[:8].hex()})"


def _rivet_pipeline(
    arrow_file: Path,
    mode: str,
    resize: int,
    batch: int,
    workers: int,
    prefetch: int,
):
    p = rivet.scan_arrow([arrow_file]).decode_image()
    if mode in ("A", "B", "C"):
        p = p.resize(resize, resize)
    if mode == "C":
        p = p.normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])  # uint8 -> f32 /255
    if mode in ("B", "C"):
        p = p.hwc_to_chw()
    return p.workers(workers).prefetch_batches(prefetch).batch(batch)


def _rivet_checks(batch: dict[str, object], mode: str) -> None:
    x = batch["images"]
    assert isinstance(x, np.ndarray)
    if mode in ("B", "C"):
        assert x.shape[1] == 3, f"expected NCHW, got {x.shape}"
        expected = np.float32 if mode == "C" else np.uint8
        assert x.dtype == expected, f"mode {mode}: got {x.dtype}"


def bench_rivet(
    arrow_file: Path,
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
        loader = _rivet_pipeline(arrow_file, mode, resize, batch, workers, 2).execute()
        it = iter(loader)
        next(it)  # warm-up: thread startup and compile happen before timing
        start = time.perf_counter()
        for index, out in enumerate(it):
            if batches and index >= batches:
                break
            _rivet_checks(out, mode)
            total += len(out["images"])
        elapsed += time.perf_counter() - start
    return total / elapsed if elapsed else 0.0


def bench_torch(
    arrow_file: Path,
    mode: str,
    resize: int,
    batch: int,
    batches: int,
    workers: int,
    epochs: int,
) -> float:
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from torchvision import transforms

    ds = load_dataset("uoft-cs/cifar10", split="train")
    transform = transforms.Compose(
        [
            transforms.Resize((resize, resize)),
            transforms.ToTensor() if mode in ("A", "C") else transforms.PILToTensor(),
        ]
    )

    def decode(ex: dict[str, object]) -> dict[str, object]:
        images = ex["img"]  # list of PIL images (batch fetch)
        ex["img"] = (
            [transform(image) for image in images]
            if isinstance(images, list)
            else transform(images)
        )
        return ex

    ds.set_transform(decode)

    total = 0
    elapsed = 0.0
    for _ in range(epochs):
        loader = DataLoader(
            ds,
            batch_size=batch,
            num_workers=workers,
            prefetch_factor=2,
            shuffle=False,
            drop_last=False,
        )
        it = iter(loader)
        next(it)  # warm-up: worker processes start before timing
        start = time.perf_counter()
        for index, out in enumerate(it):
            if batches and index >= batches:
                break
            x = out["img"]
            assert x.shape[1] == 3
            if mode == "B":
                assert x.dtype == torch.uint8
            total += x.shape[0]
        elapsed += time.perf_counter() - start
    return total / elapsed if elapsed else 0.0


def sweep_rivet(arrow_file: Path, batch: int, epochs: int, resize: int) -> None:
    workloads = ["decode", "decode+resize", "decode+resize+floatCHW"]
    print()
    print("rivet scaling sweep (images/s, full drain per row):")
    header = f"{'workload':<22} " + " ".join(
        f"w{workers}/p{pf:<5}" for workers, pf in [(0, 0), (1, 0), (2, 0), (4, 0), (4, 1), (4, 2), (4, 4), (8, 2)]
    )
    print(header)
    for mode in workloads:
        row = [f"{mode:<22}"]
        for workers, prefetch in [(0, 0), (1, 0), (2, 0), (4, 0), (4, 1), (4, 2), (4, 4), (8, 2)]:
            mode_id = "A" if mode == "decode" else ("B" if mode == "decode+resize" else "C")
            rate = bench_rivet(arrow_file, mode_id, resize, batch, 0, workers, epochs)
            row.append(f"{rate:>10.0f}")
        print(" ".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modes", default="A,B,C")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--batches", type=int, default=0, help="batches per pass; 0 = drain all")
    parser.add_argument("--resize", type=int, default=64)
    parser.add_argument("--rivet-workers", type=int, default=4)
    parser.add_argument("--torch-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--sweep", action="store_true", help="print rivet scaling table")
    args = parser.parse_args()

    arrow_file = rivet_arrow_file()
    encoding = detect_encoding(arrow_file)
    print(f"shared HF Arrow cache : {arrow_file}")
    print(f"encoded rows detected : {encoding}")
    print(
        f"batches={args.batches or 'all'} batch={args.batch} resize={args.resize} "
        f"epochs={args.epochs} (one warm-up batch before each timed pass)"
    )

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print()
    print(f"{'mode':<8} {'rivet img/s':>12} {'torch img/s':>12} {'ratio':>7}")
    for mode in modes:
        rivet_rate = bench_rivet(
            arrow_file, mode, args.resize, args.batch, args.batches, args.rivet_workers, args.epochs
        )
        torch_rate = bench_torch(
            arrow_file, mode, args.resize, args.batch, args.batches, args.torch_workers, args.epochs
        )
        ratio = rivet_rate / torch_rate if torch_rate else float("nan")
        print(f"{mode:<8} {rivet_rate:>12.0f} {torch_rate:>12.0f} {ratio:>6.2f}x")

    if args.sweep:
        sweep_rivet(arrow_file, args.batch, args.epochs, args.resize)


if __name__ == "__main__":
    main()
