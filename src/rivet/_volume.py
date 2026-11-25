"""Decoded volumetric samples and geometry."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral, Real
from typing import Any

import numpy as np

from ._rivet import _VolumeSample


class VolumeSample:
    """A DHW or DHWC volume with optional DHW mask and XYZ keypoints."""

    def __init__(
        self,
        volume: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        keypoints: np.ndarray | None = None,
        label: int = 0,
    ) -> None:
        self._inner = _VolumeSample(
            np.asarray(volume),
            mask=None if mask is None else np.asarray(mask),
            keypoints=None if keypoints is None else np.asarray(keypoints, dtype=np.float32),
            label=label,
        )

    def resize3d(
        self, size: tuple[int, int, int], interpolation: str = "linear"
    ) -> VolumeSample:
        """Resize in depth, height, width order using half-pixel coordinates.

        Masks always use nearest-neighbor sampling to preserve class IDs.
        """
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.resize3d(size, interpolation)
        return result

    def pad3d(
        self,
        padding: int | Iterable[int],
        *,
        fill: float | Iterable[float] = 0,
        fill_mask: int = 0,
    ) -> VolumeSample:
        """Pad with a scalar, symmetric DHW amounts, or six explicit sides.

        Explicit sides are front, back, top, bottom, left, right.
        """
        amounts = [padding] * 6 if isinstance(padding, Integral) else list(padding)
        if len(amounts) == 3:
            amounts = [side for amount in amounts for side in (amount, amount)]
        values = [fill] if isinstance(fill, Real) else list(fill)
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.pad3d(amounts, values, fill_mask)
        return result

    def pad_if_needed3d(
        self,
        min_zyx: tuple[int, int, int] | None = None,
        pad_divisor_zyx: tuple[int, int, int] | None = None,
        *,
        position: str = "center",
        fill: float | Iterable[float] = 0,
        fill_mask: int = 0,
        seed: int = 0,
    ) -> VolumeSample:
        """Meet minimum dimensions, then round each axis up to its divisor.

        Random placement is repeatable with the same seed.
        """
        values = [fill] if isinstance(fill, Real) else list(fill)
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.pad_if_needed3d(
            min_zyx, pad_divisor_zyx, position, values, fill_mask, seed
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        return self._inner.to_dict()
