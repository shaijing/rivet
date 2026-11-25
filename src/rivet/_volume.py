"""Decoded volumetric samples and geometry."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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

    def center_crop3d(
        self,
        size: tuple[int, int, int],
        *,
        pad_if_needed: bool = False,
        fill: float | Iterable[float] = 0,
        fill_mask: int = 0,
    ) -> VolumeSample:
        """Take a centered DHW crop and translate all XYZ points."""
        return self._crop3d(size, pad_if_needed, fill, fill_mask, None)

    def random_crop3d(
        self,
        size: tuple[int, int, int],
        *,
        pad_if_needed: bool = False,
        fill: float | Iterable[float] = 0,
        fill_mask: int = 0,
        seed: int = 0,
    ) -> VolumeSample:
        """Take a seeded DHW crop and translate all XYZ points."""
        return self._crop3d(size, pad_if_needed, fill, fill_mask, seed)

    def _crop3d(
        self,
        size: tuple[int, int, int],
        pad_if_needed: bool,
        fill: float | Iterable[float],
        fill_mask: int,
        seed: int | None,
    ) -> VolumeSample:
        values = [fill] if isinstance(fill, Real) else list(fill)
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.crop3d(size, pad_if_needed, values, fill_mask, seed)
        return result

    def flip3d(
        self,
        axes: tuple[int, ...] = (0, 1, 2),
        *,
        flip_axes: tuple[int, ...] | None = None,
        seed: int = 0,
    ) -> VolumeSample:
        """Reflect explicit axes or independently sample the allowed DHW axes."""
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.flip3d(axes, flip_axes, seed)
        return result

    def random_rotate90_3d(
        self,
        axis_pairs: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2)),
        *,
        axis_pair: tuple[int, int] | None = None,
        group_element: str | None = None,
        seed: int = 0,
    ) -> VolumeSample:
        """Rotate counterclockwise across an ascending DHW axis pair."""
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.random_rotate90_3d(axis_pairs, axis_pair, group_element, seed)
        return result

    def cubic_symmetry(self, *, element: int | None = None, seed: int = 0) -> VolumeSample:
        """Apply a signed DHW axis permutation selected from 48 possibilities."""
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.cubic_symmetry(element, seed)
        return result

    def grid_shuffle3d(
        self, grid_zyx: tuple[int, int, int] = (2, 2, 2), *, seed: int = 0
    ) -> VolumeSample:
        """Shuffle equal-sized DHW cells and move their XYZ keypoints."""
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.grid_shuffle3d(grid_zyx, seed)
        return result

    def coarse_dropout3d(
        self,
        num_holes_range: tuple[int, int] = (1, 1),
        *,
        hole_depth_range: tuple[float, float] = (0.1, 0.2),
        hole_height_range: tuple[float, float] = (0.1, 0.2),
        hole_width_range: tuple[float, float] = (0.1, 0.2),
        fill: float | Iterable[float] = 0,
        fill_mask: int | None = None,
        remove_invisible: bool = True,
        seed: int = 0,
    ) -> VolumeSample:
        """Fill seeded cuboids and optionally fill masks or keep occluded points."""
        values = [fill] if isinstance(fill, Real) else list(fill)
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.coarse_dropout3d(
            num_holes_range, (hole_depth_range, hole_height_range, hole_width_range),
            values, fill_mask, remove_invisible, seed,
        )
        return result

    def anisotropy3d(
        self,
        axes: tuple[int, ...] = (0, 1, 2),
        *,
        num_axes_range: tuple[int, int] = (1, 1),
        downscale_factor_range: tuple[float, float] = (1.5, 4.0),
        antialias: bool = True,
        seed: int = 0,
    ) -> VolumeSample:
        """Degrade selected DHW axes and restore the grid with trilinear interpolation.

        Area averaging antialiases the reduction. Annotations retain their coordinates.
        """
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.anisotropy3d(
            axes, num_axes_range, downscale_factor_range, antialias, seed,
        )
        return result

    def affine3d(
        self,
        *,
        rotate_range: Mapping[str, tuple[float, float]] | None = None,
        scale_range: Mapping[str, tuple[float, float]] | None = None,
        translate_percent_range: Mapping[str, tuple[float, float]] | None = None,
        interpolation: str = "linear",
        border_mode: str = "constant",
        fill: float | Iterable[float] = 0,
        fill_mask: int = 0,
        seed: int = 0,
    ) -> VolumeSample:
        """Scale, rotate in XYZ order about the voxel center, and translate on the same grid.

        Rotation ranges use degrees, and translation ranges use fractions of axis lengths.
        Mask sampling is always nearest-neighbor to preserve integer IDs.
        """
        def xyz(ranges: Mapping[str, tuple[float, float]] | None, default: float):
            if ranges is None:
                return [(default, default)] * 3
            if set(ranges) != {"x", "y", "z"}:
                raise ValueError("Affine3D axis ranges must contain x, y, and z")
            return [ranges[axis] for axis in "xyz"]

        values = [fill] if isinstance(fill, Real) else list(fill)
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.affine3d(
            xyz(rotate_range, 0), xyz(scale_range, 1), xyz(translate_percent_range, 0),
            interpolation, border_mode, values, fill_mask, seed,
        )
        return result

    def elastic_transform3d(
        self,
        *,
        displacement_range: tuple[float, float] = (0.02, 0.05),
        control_grid_shape: tuple[int, int] = (7, 7),
        interpolation: str = "linear",
        border_mode: str = "constant",
        fill: float | Iterable[float] = 0,
        fill_mask: int = 0,
        seed: int = 0,
    ) -> VolumeSample:
        """Apply a seeded cubic displacement field on the original voxel grid.

        Mask IDs use nearest-neighbor sampling. XYZ keypoints follow the inverse
        of the lookup used to sample voxels.
        """
        values = [fill] if isinstance(fill, Real) else list(fill)
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.elastic_transform3d(
            displacement_range, control_grid_shape, interpolation, border_mode,
            values, fill_mask, seed,
        )
        return result

    def rician_noise(
        self, std_range: tuple[float, float] = (0.05, 0.15), *, per_channel: bool = False, seed: int = 0
    ) -> VolumeSample:
        """Apply a seeded MRI magnitude noise field on the entire DHW volume."""
        result = self.__class__.__new__(self.__class__)
        result._inner = self._inner.rician_noise(std_range, per_channel, seed)
        return result

    def to_dict(self) -> dict[str, Any]:
        return self._inner.to_dict()
