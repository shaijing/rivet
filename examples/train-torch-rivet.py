"""Train a PyTorch ResNet18 on CIFAR-10 with Rivet data loading.

The model, optimizer, training loop, and epoch timing mirror
``jax/train-torch.py``.  Only dataset loading and preprocessing are changed
to use Rivet's native Lance pipeline.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import rivet


@dataclass(frozen=True)
class Config:
    batch_size: int = 128

    # Rivet pipeline workers
    num_workers: int = 8
    prefetch_factor: int = 2
    shuffle: bool = True
    drop_last: bool = True

    # Training
    num_epochs: int = 200
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    seed: int = 42


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
DEFAULT_CIFAR10_ROOT = Path.home() / ".cache" / "rivet" / "datasets" / "cifar10"


def cifar10_root() -> Path:
    """Return the Rivet-native Lance CIFAR-10 root."""
    return Path(os.environ.get("RIVET_CIFAR10_ROOT", DEFAULT_CIFAR10_ROOT))


def load_cifar10_split(split: str):
    """Load a decoded Rivet CIFAR-10 split."""
    if split not in ("train", "test"):
        raise ValueError(f"unknown split {split!r}")

    return rivet.load_dataset(cifar10_root(), split=split).cache(
        level="decoded",
    )


def build_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    seed: int | None = None,
    shuffle: bool = False,
    drop_last: bool = True,
    augment: bool = False,
):
    """Build a Rivet iterator yielding normalized NCHW NumPy batches."""
    if augment and seed is None:
        raise ValueError("seed is required when augment=True")

    pipeline = dataset.pipeline()
    if augment:
        pipeline = (
            pipeline.random_crop(32, 32, padding=4)
            .random_horizontal_flip(probability=0.5)
        )

    pipeline = pipeline.normalize(
        list(CIFAR10_MEAN),
        list(CIFAR10_STD),
    ).hwc_to_chw()

    if shuffle:
        if seed is None:
            raise ValueError("seed is required when shuffle=True")
        pipeline = pipeline.shuffle(seed)

    return (
        {key: batch[key] for key in ("images", "labels")}
        for batch in (
            pipeline.workers(num_workers)
            .prefetch_batches(prefetch_factor)
            .batch(batch_size, drop_last=drop_last)
            .execute()
        )
    )


def to_torch_batch(batch, *, torch, device, pin_memory: bool):
    """Convert a Rivet NCHW batch to a PyTorch batch through DLPack."""
    images = torch.from_dlpack(batch["images"])
    labels = torch.from_dlpack(batch["labels"]).long()

    if pin_memory:
        images = images.pin_memory()
        labels = labels.pin_memory()

    return (
        images.to(device, non_blocking=pin_memory),
        labels.to(device, non_blocking=pin_memory),
    )


def main() -> None:
    import torch
    from torch import nn

    from train_support import resnet18_c, setup_logger

    config = Config()
    log_path = Path(
        os.environ.get(
            "TRAIN_LOG",
            "tmp/train-torch-rivet.log",
        )
    )
    logger = setup_logger(log_path)
    logger.info(f"Training log: {log_path}")

    torch.manual_seed(config.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_dataset = load_cifar10_split("train")
    test_dataset = load_cifar10_split("test")
    train_rows = len(train_dataset)
    test_rows = len(test_dataset)
    steps_per_epoch = train_rows // config.batch_size

    model = resnet18_c(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    total_steps = config.num_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=0.0,
    )

    logger.info(f"Train size: {train_rows}")
    logger.info(f"Test size: {test_rows}")
    logger.info(f"Torch device: {device}")
    logger.info(f"Rivet workers: {config.num_workers}")

    for epoch in range(config.num_epochs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()

        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0
        train_loader = build_loader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            seed=config.seed + epoch,
            shuffle=config.shuffle,
            drop_last=config.drop_last,
            augment=True,
        )

        for batch in train_loader:
            images, labels = to_torch_batch(
                batch,
                torch=torch,
                device=device,
                pin_memory=pin_memory,
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_samples += batch_size

        train_loss = train_loss_sum / train_samples
        train_acc = train_correct / train_samples

        model.eval()
        test_loss_sum = 0.0
        test_correct = 0
        test_samples = 0
        test_loader = build_loader(
            test_dataset,
            batch_size=config.batch_size,
            num_workers=0,
            prefetch_factor=1,
            shuffle=False,
            drop_last=False,
        )

        with torch.no_grad():
            for batch in test_loader:
                images, labels = to_torch_batch(
                    batch,
                    torch=torch,
                    device=device,
                    pin_memory=pin_memory,
                )

                logits = model(images)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                test_loss_sum += loss.item() * batch_size
                test_correct += (logits.argmax(dim=1) == labels).sum().item()
                test_samples += batch_size

        test_loss = test_loss_sum / test_samples
        test_acc = test_correct / test_samples

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start

        logger.info(
            f"Epoch {epoch + 1:03d}/{config.num_epochs:03d} "
            f"| train loss {train_loss:.4f} "
            f"| train acc {train_acc * 100:.2f}% "
            f"| test loss {test_loss:.4f} "
            f"| test acc {test_acc * 100:.2f}% "
            f"| epoch time {epoch_seconds:.2f}s"
        )


if __name__ == "__main__":
    main()
