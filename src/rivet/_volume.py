"""Decoded volumetric samples and geometry."""

from __future__ import annotations

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

    def to_dict(self) -> dict[str, Any]:
        return self._inner.to_dict()
