"""Train a PyTorch ResNet18 on CIFAR-10 for throughput comparison.

The training configuration and epoch-level logging intentionally mirror
``jax/train-jax.py``.  This script uses torchvision's CIFAR-10 dataset and
PyTorch's DataLoader so the per-epoch timings can be compared directly.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    batch_size: int = 128

    # DataLoader
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
DEFAULT_CIFAR10_ROOT = Path.home() / ".data" / "pytorch"


def cifar10_root() -> Path:
    """Return the local directory used by torchvision for CIFAR-10."""
    return Path(os.environ.get("CIFAR10_ROOT", DEFAULT_CIFAR10_ROOT))


def main() -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from train_support import resnet18_c, setup_logger

    config = Config()
    log_path = Path(
        os.environ.get(
            "TRAIN_LOG",
            "tmp/train-torch.log",
        )
    )
    logger = setup_logger(log_path)
    logger.info(f"Training log: {log_path}")

    torch.manual_seed(config.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=cifar10_root(),
        train=True,
        transform=train_transform,
        download=True,
    )
    test_dataset = datasets.CIFAR10(
        root=cifar10_root(),
        train=False,
        transform=test_transform,
        download=True,
    )

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = resnet18_c(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    total_steps = config.num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=0.0,
    )

    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Test size: {len(test_dataset)}")
    logger.info(f"Torch device: {device}")
    logger.info(f"DataLoader workers: {config.num_workers}")

    for epoch in range(config.num_epochs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()

        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

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

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)

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
