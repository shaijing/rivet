"""Benchmark PyTorch and Rivet using the same native CIFAR-10 Lance split.

Both loaders read the same Rivet-native CIFAR-10 Lance split:
image: binary and label: int32. The Torch dataset implements __getitems__,
so one DataLoader batch performs one Lance take call, matching Rivet's
batch-first source access.

Three output representations are compared:

mode A  Rivet uint8 NHWC numpy; Torch float32 CHW tensor
mode B  both uint8 CHW (Rivet hwc_to_chw, Torch PILToTensor)
mode C  both float32 CHW in [0, 1]

Usage, after building the release extension:

    maturin develop --release
    python bench/compare_cifar10_torch_rivet.py --modes A,B,C
        --batch 64 --batches 0 --resize 64
        --rivet-workers 4 --torch-workers 4 --epochs 2

Use --lance-root to select another converted CIFAR-10 root. The default is
/data/datasets/rivet/cifar10. Add --sweep for a Rivet worker scaling table.
"""

from __future__ import annotations

import argparse
import io
import os
import time
from pathlib import Path

import numpy as np

import rivet

DEFAULT_LANCE_ROOT = Path(
    os.environ.get("RIVET_CIFAR10_ROOT", "/data/datasets/rivet/cifar10")
)


def _train_path(lance_root: Path) -> Path:
    if lance_root.name.endswith(".lance"):
        return lance_root
    train_path = lance_root / "train.lance"
    if not train_path.is_dir():
        raise SystemExit(f"train Lance split not found: {train_path}")
    return train_path


def detect_encoding(dataset: rivet.LanceDataset) -> str:
    raw = dataset.get_encoded(0)["image"]
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if raw.startswith(b"\xff\xd8"):
        return "JPEG"
    return f"unknown ({raw[:8].hex()})"


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


class TorchLanceDataset:
    """Map-style Torch dataset that fetches one Lance batch per DataLoader batch."""

    def __init__(self, path: Path, transform: object) -> None:
        import lance

        self.path = path
        self.transform = transform
        source = lance.dataset(str(path))
        self.length = source.count_rows()
        self._source = None

    def _open(self):
        if self._source is None:
            import lance

            self._source = lance.dataset(str(self.path))
        return self._source

    def __len__(self) -> int:
        return self.length

    def __getitems__(self, indices: list[int]) -> list[dict[str, object]]:
        from PIL import Image

        table = self._open().take(indices, columns=["image", "label"])
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        return [
            {
                "image": self.transform(Image.open(io.BytesIO(image)).convert("RGB")),
                "label": int(label),
            }
            for image, label in zip(images, labels, strict=True)
        ]

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.__getitems__([index])[0]


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


def bench_torch(
    lance_path: Path,
    mode: str,
    resize: int,
    batch: int,
    batches: int,
    workers: int,
    epochs: int,
) -> float:
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except ImportError as error:
        raise SystemExit(
            "Torch benchmark requires torch and torchvision to be installed"
        ) from error

    transform = transforms.Compose(
        [
            transforms.Resize((resize, resize)),
            transforms.ToTensor() if mode in ("A", "C") else transforms.PILToTensor(),
        ]
    )

    dataset = TorchLanceDataset(lance_path, transform)
    total = 0
    elapsed = 0.0
    for _ in range(epochs):
        loader_options = {
            "batch_size": batch,
            "num_workers": workers,
            "shuffle": False,
            "drop_last": False,
        }
        if workers:
            loader_options["prefetch_factor"] = 2
        loader = DataLoader(dataset, **loader_options)
        iterator = iter(loader)
        next(iterator)  # warm-up: worker processes start before timing
        start = time.perf_counter()
        for index, output in enumerate(iterator):
            if batches and index >= batches:
                break
            images = output["image"]
            assert images.shape[1] == 3
            if mode == "B":
                assert images.dtype == torch.uint8
            else:
                assert images.dtype == torch.float32
            total += images.shape[0]
        elapsed += time.perf_counter() - start
    return total / elapsed if elapsed else 0.0


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
        "--sweep", action="store_true", help="print Rivet scaling table"
    )
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
