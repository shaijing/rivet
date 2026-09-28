"""Local helpers for the standalone training examples.

These helpers intentionally live beside the examples instead of in Rivet's
runtime package. The examples can therefore be run directly from a checkout
without depending on the separate research project.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(log_path: str | Path) -> logging.Logger:
    """Create a console and file logger for a training example."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rivet-training")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def resnet18_c(num_classes: int = 10):
    """Build a CIFAR-sized ResNet-18 using torchvision's implementation."""
    from torch import nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
