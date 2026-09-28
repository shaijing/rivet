"""Local JAX/Flax model helper for the standalone training example."""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx


class _BasicBlock(nnx.Module):
    expansion = 1

    def __init__(
        self,
        in_features: int,
        features: int,
        stride: int = 1,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=features,
            kernel_size=(3, 3),
            strides=stride,
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(features, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(features, rngs=rngs)

        self.use_projection = stride != 1 or in_features != features
        if self.use_projection:
            self.shortcut_conv = nnx.Conv(
                in_features=in_features,
                out_features=features,
                kernel_size=(1, 1),
                strides=stride,
                padding="SAME",
                use_bias=False,
                rngs=rngs,
            )
            self.shortcut_bn = nnx.BatchNorm(features, rngs=rngs)

    def __call__(self, x, *, training: bool = True):
        identity = x
        if self.use_projection:
            identity = self.shortcut_conv(identity)
            identity = self.shortcut_bn(
                identity,
                use_running_average=not training,
            )

        out = self.conv1(x)
        out = self.bn1(out, use_running_average=not training)
        out = nnx.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, use_running_average=not training)
        return nnx.relu(out + identity)


class _ResNet(nnx.Module):
    def __init__(
        self,
        block_counts: tuple[int, int, int, int],
        *,
        num_classes: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.stem_conv = nnx.Conv(
            in_features=3,
            out_features=64,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.stem_bn = nnx.BatchNorm(64, rngs=rngs)
        self.stages = nnx.List()

        in_features = 64
        for features, count, stage_stride in zip(
            (64, 128, 256, 512),
            block_counts,
            (1, 2, 2, 2),
            strict=True,
        ):
            blocks = []
            for block_index in range(count):
                stride = stage_stride if block_index == 0 else 1
                blocks.append(
                    _BasicBlock(
                        in_features,
                        features,
                        stride,
                        rngs=rngs,
                    )
                )
                in_features = features
            self.stages.append(nnx.Sequential(*blocks))

        self.fc = nnx.Linear(
            in_features=in_features,
            out_features=num_classes,
            rngs=rngs,
        )

    def __call__(self, x, *, training: bool = True, return_feat: bool = False):
        x = self.stem_conv(x)
        x = self.stem_bn(x, use_running_average=not training)
        x = nnx.relu(x)
        for stage in self.stages:
            x = stage(x, training=training)
        features = jnp.mean(x, axis=(1, 2))
        logits = self.fc(features)
        if return_feat:
            return logits, features
        return logits


def flax_resnet18_c(
    num_classes: int = 10,
    *,
    rngs: nnx.Rngs | None = None,
) -> _ResNet:
    """Build a CIFAR-sized Flax NNX ResNet-18."""
    return _ResNet(
        (2, 2, 2, 2),
        num_classes=num_classes,
        rngs=nnx.Rngs(0) if rngs is None else rngs,
    )
