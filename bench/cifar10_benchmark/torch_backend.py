"""Torch-side Lance dataset and benchmark implementation."""

from __future__ import annotations

import io
import time
from pathlib import Path


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
