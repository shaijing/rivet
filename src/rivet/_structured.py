"""Native batch augmentation, instance annotations and decoded video samples."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from typing import Any

import numpy as np

from ._rivet import (
    _AnnotatedSample,
    _StructuredPipeline,
    _VideoSample,
)
from ._rivet import (
    mix_batch as _mix_batch,
)
from ._rivet import (
    temporal_subsample as _temporal_subsample,
)
from ._transforms import Transform


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _labels(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        if array.size == 0:
            return np.asarray(array, dtype=np.int64)
        if array.dtype.kind not in "iu":
            raise ValueError("index labels must contain integers")
        if (
            array.dtype.kind == "u"
            and array.size
            and array.max() > np.iinfo(np.int64).max
        ):
            raise ValueError("labels exceed int64 range")
        return np.asarray(array, dtype=np.int64)
    if array.ndim == 2:
        return np.asarray(array, dtype=np.float32)
    raise ValueError("labels must have shape [B] or [B, num_classes]")


class MixUp:
    """Native MixUp over NumPy batches, returning an image batch dictionary.

    Pairing rolls by one along B. One Beta(alpha, alpha) draw is shared by
    images and labels. Images must be float32. Labels are int64 [B] (requiring
    num_classes) or float32 [B, C]. Random draws depend on seed, epoch and
    batch_index, never on Python/Torch RNG state.
    """

    def __init__(self, alpha: float = 1.0, num_classes: int | None = None) -> None:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha = float(alpha)
        self.num_classes = (
            None
            if num_classes is None
            else _positive_integer(num_classes, "num_classes")
        )

    def __call__(
        self,
        images: np.ndarray,
        labels: Any,
        *,
        layout: str = "NHWC",
        seed: int = 0,
        epoch: int = 0,
        batch_index: int = 0,
        dlpack: bool = False,
    ) -> dict[str, Any]:
        return _mix_batch(
            np.asarray(images),
            _labels(labels),
            self.num_classes,
            self.alpha,
            isinstance(self, CutMix),
            layout,
            seed,
            epoch,
            batch_index,
            dlpack,
        )


class CutMix(MixUp):
    """Native CutMix, accepting uint8/float32 images and using actual patch area.

    The coefficient's Beta sample determines a rectangle centered on a random
    pixel and clipped to the canvas. Labels use the clipped rectangle area.
    See MixUp for pairing, soft-label and random-context conventions.
    """


class AnnotatedSample:
    """Decoded image with aligned instance fields, owned by the native runtime.

    Boxes are float32 [N,4] in xyxy/xywh/cxcywh edge coordinates, labels are
    int64 [N], masks are uint8/int64 [N,H,W] (instances) or [H,W]
    (semantic segmentation), keypoints are float32 [N,K,2]. Instance fields
    share N; semantic maps are not filtered by sanitize. N=0 is supported. Input NumPy views
    (including negative strides) are copied once at construction. Exported
    arrays are read-only and retain their native owner.
    """

    def __init__(
        self,
        image: np.ndarray,
        *,
        boxes: Any = None,
        labels: Any = None,
        masks: Any = None,
        keypoints: Any = None,
        box_format: str = "xyxy",
        layout: str = "HWC",
    ) -> None:
        label_array = None if labels is None else _labels(labels)
        if label_array is not None and label_array.ndim != 1:
            raise ValueError("instance labels must have shape [N]")
        box_array = None if boxes is None else np.asarray(boxes, dtype=np.float32)
        if box_array is not None:
            if box_array.ndim == 1 and box_array.size == 0:
                box_array = box_array.reshape(0, 4)
            if box_array.ndim != 2 or box_array.shape[1] != 4:
                raise ValueError("boxes must have shape [N,4]")
        self._inner = _AnnotatedSample(
            np.asarray(image),
            box_array,
            label_array,
            None if masks is None else np.asarray(masks),
            None if keypoints is None else np.asarray(keypoints, dtype=np.float32),
            box_format,
            layout,
        )

    @classmethod
    def _from_inner(cls, inner: _AnnotatedSample) -> AnnotatedSample:
        out = cls.__new__(cls)
        out._inner = inner
        return out

    def to_dict(self, *, dlpack: bool = False) -> dict[str, Any]:
        """Export image/targets and canvas, format, shape and dtype metadata."""
        return self._inner.to_dict(dlpack)

    def apply(
        self,
        transform: Transform,
        *,
        seed: int = 0,
        epoch: int = 0,
        sample_index: int = 0,
    ) -> AnnotatedSample:
        """Apply one shared geometric plan to image and every target field."""
        return self._from_inner(
            self._inner.apply(transform._inner, seed, epoch, sample_index)
        )


class StructuredPipeline:
    """Lazy native pipeline over decoded AnnotatedSample objects.

    Geometry uses one sampled plan per source identity. Masks always use
    nearest-neighbor interpolation and zero constant fill. Boxes are clamped
    after crop/warp; invalid instances remain until explicit sanitize.
    Batches contain stacked images and a ragged list of target dictionaries.
    Worker count and batch size do not change per-source random decisions.
    """

    def __init__(self, samples: Iterable[AnnotatedSample]) -> None:
        self._inner = _StructuredPipeline([sample._inner for sample in samples])

    @classmethod
    def _from_inner(cls, inner: _StructuredPipeline) -> StructuredPipeline:
        out = cls.__new__(cls)
        out._inner = inner
        return out

    def compose(self, transform: Transform) -> StructuredPipeline:
        return self._from_inner(self._inner.compose(transform._inner))

    def convert_bounding_box_format(self, format: str) -> StructuredPipeline:
        return self._from_inner(self._inner.convert_bounding_box_format(format))

    def clamp_bounding_boxes(self) -> StructuredPipeline:
        return self._from_inner(self._inner.clamp_bounding_boxes())

    def clamp_keypoints(self) -> StructuredPipeline:
        """Clamp keypoints to the first and last pixel centers on each axis."""
        return self._from_inner(self._inner.clamp_keypoints())

    def sanitize_keypoints(self) -> StructuredPipeline:
        """Drop instances with any out-of-canvas keypoint and their aligned fields."""
        return self._from_inner(self._inner.sanitize_keypoints())

    def sanitize_bounding_boxes(
        self,
        min_size: float = 1.0,
        min_area: float = 1.0,
    ) -> StructuredPipeline:
        """Remove invalid boxes and corresponding labels, masks and keypoints."""
        return self._from_inner(self._inner.sanitize_bounding_boxes(min_size, min_area))

    def random_iou_crop(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Iterable[float] | None = None,
        trials: int = 40,
        *,
        max_attempts: int = 100,
    ) -> StructuredPipeline:
        """Sample an SSD crop; mark excluded boxes zero for later sanitize.

        After bounded unsuccessful option rounds, retain the original sample.
        This avoids the unbounded retry of impossible upstream configurations.
        """
        return self._from_inner(
            self._inner.random_iou_crop(
                min_scale,
                max_scale,
                min_aspect_ratio,
                max_aspect_ratio,
                None if sampler_options is None else list(sampler_options),
                _positive_integer(trials, "trials"),
                _positive_integer(max_attempts, "max_attempts"),
            )
        )

    def seed(self, seed: int) -> StructuredPipeline:
        return self._from_inner(self._inner.seed(seed))

    def epoch(self, epoch: int) -> StructuredPipeline:
        return self._from_inner(self._inner.epoch(epoch))

    def workers(self, count: int) -> StructuredPipeline:
        return self._from_inner(self._inner.workers(count))

    def shuffle(self) -> StructuredPipeline:
        return self._from_inner(self._inner.shuffle())

    def batch(self, size: int, *, drop_last: bool = False) -> StructuredPipeline:
        return self._from_inner(
            self._inner.batch(_positive_integer(size, "size"), drop_last)
        )

    def execute(self, *, as_numpy: bool = True) -> StructuredDataLoader:
        return StructuredDataLoader(self._inner.execute(), as_numpy)


class StructuredDataLoader:
    def __init__(self, inner: Any, as_numpy: bool) -> None:
        self._inner = inner
        self.as_numpy = as_numpy

    def __iter__(self) -> StructuredDataLoader:
        return self

    def __next__(self) -> dict[str, Any]:
        batch = next(self._inner)
        if not self.as_numpy:
            batch["images"] = batch["images"].tobytes()
            for target in batch["targets"]:
                for name in target["shapes"]:
                    target[name] = target[name].tobytes()
        return batch

    def next_dlpack(self) -> dict[str, Any]:
        batch = self._inner.next_dlpack()
        if batch is None:
            raise StopIteration
        return batch

    def next_torch(self) -> dict[str, Any]:
        import torch

        batch = self.next_dlpack()
        batch["images"] = torch.utils.dlpack.from_dlpack(batch["images"])
        for target in batch["targets"]:
            for name in target["shapes"]:
                target[name] = torch.utils.dlpack.from_dlpack(target[name])
        return batch


class VideoSample:
    """Decoded THWC/TCHW frames with optional nondecreasing float64 timestamps.

    Time is axis 0; labels describe the entire sequence. Uniform subsampling
    selects both endpoint frames, truncates intermediate linspace indices,
    repeats frames when oversampling and preserves timestamps and the label.
    One requested sample selects the first frame. Empty sequences are rejected.
    """

    def __init__(
        self,
        frames: np.ndarray,
        *,
        timestamps: Any = None,
        label: int = 0,
        layout: str = "THWC",
    ) -> None:
        self._inner = _VideoSample(
            np.asarray(frames),
            None if timestamps is None else np.asarray(timestamps, dtype=np.float64),
            label,
            layout,
        )

    def uniform_temporal_subsample(self, num_samples: int) -> VideoSample:
        out = self.__class__.__new__(self.__class__)
        out._inner = self._inner.uniform_temporal_subsample(
            _positive_integer(num_samples, "num_samples")
        )
        return out

    def to_dict(self, *, dlpack: bool = False) -> dict[str, Any]:
        return self._inner.to_dict(dlpack)


def uniform_temporal_subsample(
    video: np.ndarray,
    num_samples: int,
    *,
    temporal_dim: int = -4,
    dlpack: bool = False,
) -> Any:
    """Subsample a NumPy video (including batched prefixes) in native Rust.

    Default axis -4 follows [...,T,C,H,W]. For THWC use temporal_dim=0;
    for NTHWC use temporal_dim=1. Output may be a read-only NumPy view or a
    public DLPack producer. No video file decoder is implied.
    """
    return _temporal_subsample(
        np.asarray(video),
        _positive_integer(num_samples, "num_samples"),
        temporal_dim,
        dlpack,
    )


class UniformTemporalSubsample:
    def __init__(self, num_samples: int, *, temporal_dim: int = -4) -> None:
        self.num_samples = _positive_integer(num_samples, "num_samples")
        self.temporal_dim = temporal_dim

    def __call__(self, video: VideoSample | np.ndarray) -> VideoSample | np.ndarray:
        if isinstance(video, VideoSample):
            if self.temporal_dim not in (0, -4):
                raise ValueError("VideoSample time is axis 0")
            return video.uniform_temporal_subsample(self.num_samples)
        return uniform_temporal_subsample(
            video, self.num_samples, temporal_dim=self.temporal_dim
        )
