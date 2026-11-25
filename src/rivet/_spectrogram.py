"""Augment precomputed audio spectrograms."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._rivet import _SpectrogramSample


class SpectrogramSample:
    """A real spectrogram with shape (frequency, time) or (channel, frequency, time)."""

    def __init__(self, spectrogram: np.ndarray, *, label: int = 0) -> None:
        self._inner = _SpectrogramSample(np.asarray(spectrogram), label)

    def time_masking(self, time_mask_param: int = 40, *, seed: int = 0) -> SpectrogramSample:
        """Zero a random interval on the last axis, shared across frequencies and channels."""
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.mask(False, time_mask_param, seed)
        return result

    def frequency_masking(self, freq_mask_param: int = 30, *, seed: int = 0) -> SpectrogramSample:
        """Zero a random interval on the frequency axis, shared across time and channels."""
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.mask(True, freq_mask_param, seed)
        return result

    def time_reverse(self) -> SpectrogramSample:
        """Reverse time along the last axis."""
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.time_reverse()
        return result

    def to_dict(self) -> dict[str, Any]:
        return self._inner.to_dict()
