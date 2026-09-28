from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import rivet

# ============================================================
# Config
# ============================================================


@dataclass(frozen=True)
class Config:
    batch_size: int = 128

    # DataLoader
    num_workers: int = 4
    prefetch_factor: int = 2
    shuffle: bool = True
    drop_last: bool = True

    # Training
    num_epochs: int = 200
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    seed: int = 42


# ============================================================
# Plain Python constants only.
#
# IMPORTANT:
# Do not create jnp.array / jax.Array at module import time.
# Spawned worker processes import this module again.
# ============================================================


CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)


# ============================================================
# Rivet-backed loading and augmentation
#
# Everything below runs as rivet pipeline ops on rust workers that are
# CPU-only and never touch JAX. The python side only pulls numpy batches
# from a thin iterator:
#
#     {"images": float32 NHWC normalized numpy, "labels": int64 numpy}
#
# Deterministic per epoch: stochastic ops (random crop / flip) draw from
# the same seed as the shuffle, so a (seed, epoch) rebuild reproduces the
# sample order and every augmentation exactly, regardless of worker count.
# ============================================================


DEFAULT_CIFAR10_ROOT = Path.home() / ".cache" / "rivet" / "datasets" / "cifar10"


def cifar10_root() -> Path:
    """Return the Rivet-native Lance CIFAR-10 root."""
    return Path(os.environ.get("RIVET_CIFAR10_ROOT", DEFAULT_CIFAR10_ROOT))


def load_cifar10_split(split: str):
    """Load and materialize one Rivet-native CIFAR-10 Lance split.

    Decoded caching is done once before the training loop. The returned
    dataset's pipeline therefore starts with uint8 HWC images and must not
    add a second ``decode_image`` operation.
    """
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
    """Create a Rivet DataLoader over a decoded CIFAR-10 Lance split.

    The decoded Lance cache supplies uint8 HWC images. Augmentation,
    normalization and the optional deterministic shuffle are Rivet pipeline
    ops running on its Rust worker pool. With ``augment``, each image is
    random-cropped (4 px zero padding) and random-horizontally-flipped
    (p = 0.5) before per-channel normalization; both stochastic ops draw
    from the shuffle seed.
    Returns a python iterator of numpy batches.
    """
    if augment and seed is None:
        raise ValueError("seed is required when augment=True")

    # ``dataset`` is already decoded, so adding decode_image() here would be
    # invalid and would defeat the decoded cache.
    pipeline = dataset.pipeline()

    if augment:
        pipeline = pipeline.random_crop(32, 32, padding=4).random_horizontal_flip(
            probability=0.5
        )

    pipeline = pipeline.normalize(
        list(CIFAR10_MEAN),
        list(CIFAR10_STD),
    )

    if shuffle:
        if seed is None:
            raise ValueError("seed is required when shuffle=True")
        pipeline = pipeline.shuffle(seed)

    loader = (
        pipeline.workers(num_workers)
        .prefetch_batches(prefetch_factor)
        .batch(batch_size, drop_last=drop_last)
        .execute()
    )

    # JAX-only tensors: strip rivet's metadata keys (dtype/layout/shape).
    return ({key: batch[key] for key in ("images", "labels")} for batch in loader)


# ============================================================
# Main
#
# Actual JAX / CUDA initialization happens only here.
# Spawn workers import this module but do not execute main().
# ============================================================


def main():
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from train_jax_support import flax_resnet18_c
    from train_support import setup_logger

    log_path = Path(
        os.environ.get(
            "TRAIN_LOG",
            "tmp/train-jax.log",
        )
    )
    logger = setup_logger(log_path)
    logger.info(f"Training log: {log_path}")

    config = Config(
        batch_size=128,
        num_workers=8,
        prefetch_factor=2,
        num_epochs=200,
        learning_rate=0.1,
        weight_decay=5e-4,
        momentum=0.9,
        seed=42,
    )

    # --------------------------------------------------------
    # Dataset
    #
    # Rivet reads the native CIFAR-10 Lance splits and materializes decoded
    # uint8 images once. Augmentation (train), normalization and shuffle are
    # Rivet pipeline ops on its Rust workers. Loaders are rebuilt per epoch so
    # the epoch seed is deterministic without repeating cache construction.
    # --------------------------------------------------------

    train_dataset = load_cifar10_split("train")
    test_dataset = load_cifar10_split("test")

    train_rows = len(train_dataset)
    test_rows = len(test_dataset)

    logger.info(f"Train size: {train_rows}")
    logger.info(f"Test size: {test_rows}")

    # --------------------------------------------------------
    # JAX device
    # --------------------------------------------------------

    devices = jax.devices()

    logger.info(f"JAX devices: {devices}")

    device = devices[0]

    logger.info(f"Default backend: {jax.default_backend()}")

    # ========================================================
    # Metrics
    # ========================================================

    def accuracy(
        logits,
        labels,
    ):
        predictions = jnp.argmax(
            logits,
            axis=-1,
        )

        return jnp.mean(predictions == labels)

    # ========================================================
    # Train step
    #
    # Batches arrive already augmented + normalized from the rivet
    # pipeline; forward + backward + optimizer are one JIT region.
    # ========================================================

    @nnx.jit
    def train_step(
        model,
        optimizer,
        batch,
    ):
        images = batch["images"]

        labels = batch["labels"]

        def loss_fn(model):
            logits = model(images, training=True)

            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits,
                labels,
            ).mean()

            acc = accuracy(
                logits,
                labels,
            )

            return loss, acc

        (
            (
                loss,
                acc,
            ),
            grads,
        ) = nnx.value_and_grad(
            loss_fn,
            has_aux=True,
        )(model)

        optimizer.update(
            model,
            grads,
        )

        return (
            loss,
            acc,
        )

    # ========================================================
    # Eval step
    # ========================================================

    @nnx.jit
    def eval_step(
        model,
        batch,
    ):
        images = batch["images"]

        labels = batch["labels"]

        logits = model(images, training=False)

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits,
            labels,
        ).mean()

        acc = accuracy(
            logits,
            labels,
        )

        return (
            loss,
            acc,
        )

    # ========================================================
    # Model + optimizer
    # ========================================================

    steps_per_epoch = train_rows // config.batch_size
    total_steps = config.num_epochs * steps_per_epoch
    lr_schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=total_steps,
        alpha=0.0,
    )
    tx = optax.sgd(
        learning_rate=lr_schedule,
        momentum=config.momentum,
    )

    model = flax_resnet18_c(
        num_classes=10,
        rngs=nnx.Rngs(
            config.seed,
        ),
    )
    optimizer = nnx.Optimizer(
        model,
        tx,
        wrt=nnx.Param,
    )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(config.num_epochs):
        epoch_start = time.perf_counter()

        # Deterministic epoch: rivet rebuilds each epoch with seed =
        # config.seed + epoch, reproducing the same shuffle order and the
        # same per-sample augmentations at any worker count.
        train_loader = build_loader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            seed=config.seed + epoch,
            shuffle=True,
            drop_last=True,
            augment=True,
        )

        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            # ============================================
            # CPU / GPU boundary
            #
            # Worker side (Rivet Rust pool):
            #   decoded Lance cache -> random crop/flip ->
            #   normalize -> float32 numpy batch
            #
            # Main process from here:
            #   NumPy -> JAX/CUDA
            # ============================================

            batch = jax.device_put(
                batch,
                device=device,
            )

            (
                loss,
                acc,
            ) = train_step(
                model,
                optimizer,
                batch,
            )

            train_loss_sum += float(loss)

            train_acc_sum += float(acc)

            train_batches += 1

        train_loss = train_loss_sum / train_batches

        train_acc = train_acc_sum / train_batches

        # ================================================
        # Evaluation (rivet pipeline, no augmentation,
        # inline workers, tiny)
        # ================================================

        test_loader = build_loader(
            test_dataset,
            batch_size=config.batch_size,
            num_workers=0,
            prefetch_factor=1,
            shuffle=False,
            drop_last=False,
        )

        test_loss_sum = 0.0
        test_acc_sum = 0.0
        test_batches = 0

        for batch in test_loader:
            batch = jax.device_put(
                batch,
                device=device,
            )

            (
                test_loss,
                test_acc,
            ) = eval_step(
                model,
                batch,
            )

            test_loss_sum += float(test_loss)

            test_acc_sum += float(test_acc)

            test_batches += 1

        test_loss = test_loss_sum / test_batches

        test_acc = test_acc_sum / test_batches
        epoch_seconds = time.perf_counter() - epoch_start

        logger.info(
            f"Epoch {epoch + 1:03d}/{config.num_epochs:03d} "
            f"| train loss {train_loss:.4f} "
            f"| train acc {train_acc * 100:.2f}% "
            f"| test loss {test_loss:.4f} "
            f"| test acc {test_acc * 100:.2f}% "
            f"| epoch time {epoch_seconds:.2f}s"
        )


# ============================================================
# Entry point
# ============================================================


if __name__ == "__main__":
    main()
