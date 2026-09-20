"""Rivet's backwards-compatible root namespace.

The canonical image API is available from :mod:`rivet.vision`. The root
exports remain during the namespace migration so existing applications can
upgrade without changing every import at once.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ._rivet import (
    _ArrowDataset,
    _ImageFolderDataset,
    _ImagePipeline,
    _Transform,
)
from ._rivet import read_image_batch as _read_image_batch

try:
    from ._rivet import (
        _LanceDataset,
        _LanceDatasetDict,
    )
    from ._rivet import (
        load_lance_split as _load_lance_split,
    )
except ImportError:
    _LanceDataset = None
    _LanceDatasetDict = None
    _load_lance_split = None

__all__ = [
    "ArrowDataset",
    "Compose",
    "DataLoader",
    "DatasetDict",
    "ImageFolder",
    "LanceDataset",
    "Pipeline",
    "Transform",
    "dataset",
    "hf_arrow_files",
    "load_dataset",
    "load_hf_arrow_files",
    "load_hf_image_batch",
    "read_hf_image_batch",
    "read_image_batch",
    "scan_arrow",
    "scan_hf",
    "scan_image_folder",
    "scan_lance",
    "vision",
]


def __getattr__(name: str) -> Any:
    if name == "vision":
        import importlib

        module = importlib.import_module(".vision", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _normalize_arrow_files(arrow_files: Iterable[str | Path]) -> list[str]:
    return [str(Path(path)) for path in arrow_files]


def _validate_hf_dataset(dataset: Any) -> None:
    if getattr(dataset, "_indices", None) is not None:
        raise ValueError(
            "Rivet currently requires a contiguous Hugging Face Dataset; "
            "call dataset.flatten_indices() first."
        )


def _maybe_numpy_batch(batch: dict[str, Any], as_numpy: bool) -> dict[str, Any]:
    if as_numpy:
        return batch

    images = batch["images"]
    labels = batch["labels"]

    if hasattr(images, "tobytes"):
        batch["images"] = images.tobytes()
    if hasattr(labels, "tolist"):
        batch["labels"] = labels.tolist()

    return batch


def _require_lance() -> tuple[Any, Any, Any]:
    if _LanceDataset is None or _LanceDatasetDict is None or _load_lance_split is None:
        raise RuntimeError(
            "Lance support is disabled; rebuild rivet-python with the 'lance' feature"
        )
    return _LanceDataset, _LanceDatasetDict, _load_lance_split


class ArrowDataset:
    """Arrow-backed encoded image dataset."""

    def __init__(
        self,
        files: Iterable[str | Path],
        *,
        image_column: str = "img",
        label_column: str = "label",
    ) -> None:
        self._inner = _ArrowDataset(
            _normalize_arrow_files(files),
            image_column,
            label_column,
        )

    @classmethod
    def from_huggingface(
        cls,
        dataset: Any,
        *,
        image_column: str = "img",
        label_column: str = "label",
    ) -> ArrowDataset:
        """Build an ArrowDataset from a loaded Hugging Face dataset split."""
        _validate_hf_dataset(dataset)
        return cls(
            hf_arrow_files(dataset),
            image_column=image_column,
            label_column=label_column,
        )

    def __len__(self) -> int:
        return len(self._inner)

    def get_encoded(self, index: int) -> dict[str, Any]:
        return self._inner.get_encoded(index)

    def get_decoded(self, index: int, *, as_numpy: bool = True) -> dict[str, Any]:
        return _maybe_numpy_batch(self._inner.get_decoded(index), as_numpy)

    def pipeline(self) -> Pipeline:
        return Pipeline(self._inner.pipeline())


class ImageFolder:
    """Directory-backed encoded image dataset."""

    def __init__(self, root: str | Path) -> None:
        self._inner = _ImageFolderDataset(str(Path(root)))

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def classes(self) -> list[str]:
        return self._inner.classes

    @property
    def class_to_idx(self) -> dict[str, int]:
        return self._inner.class_to_idx

    @property
    def samples(self) -> list[dict[str, Any]]:
        return self._inner.samples

    def get_encoded(self, index: int) -> dict[str, Any]:
        return self._inner.get_encoded(index)

    def get_decoded(self, index: int, *, as_numpy: bool = True) -> dict[str, Any]:
        return _maybe_numpy_batch(self._inner.get_decoded(index), as_numpy)

    def pipeline(self) -> Pipeline:
        return Pipeline(self._inner.pipeline())


class DatasetDict:
    """Lightweight collection of named Lance dataset splits."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, split: str) -> LanceDataset:
        return LanceDataset._from_inner(self._inner[split])

    def __contains__(self, split: object) -> bool:
        return isinstance(split, str) and split in self._inner

    def keys(self) -> list[str]:
        return self._inner.keys()

    def values(self) -> list[LanceDataset]:
        return [self[split] for split in self.keys()]

    def items(self) -> list[tuple[str, LanceDataset]]:
        return [(split, self[split]) for split in self.keys()]

    def __iter__(self):
        return iter(self.keys())

    def __repr__(self) -> str:
        entries = ", ".join(
            f"{split}: Dataset(num_rows={len(dataset)})"
            for split, dataset in self.items()
        )
        return f"DatasetDict({{{entries}}})"


class LanceDataset:
    """Lance-backed encoded image dataset.

    Dataset access is lazy by default. Use :meth:`cache` to explicitly
    materialize encoded or decoded image payloads in memory.

    The native Rivet schema is ``image: binary`` and ``label: int32`` or
    ``int64``. A configured Hugging Face-compatible ``img.bytes`` struct is
    also accepted for compatibility. Only the configured image and label
    columns are projected from Lance during training.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        image_column: str = "image",
        label_column: str = "label",
    ) -> None:
        lance_dataset, _, _ = _require_lance()
        self._inner = lance_dataset(
            str(path),
            image_column,
            label_column,
        )

    @classmethod
    def _from_inner(cls, inner: Any) -> LanceDataset:
        dataset = cls.__new__(cls)
        dataset._inner = inner
        return dataset

    def __len__(self) -> int:
        return len(self._inner)

    def get_encoded(self, index: int) -> dict[str, Any]:
        return self._inner.get_encoded(index)

    def get_decoded(self, index: int, *, as_numpy: bool = True) -> dict[str, Any]:
        return _maybe_numpy_batch(self._inner.get_decoded(index), as_numpy)

    def cache(
        self,
        level: str = "encoded",
        *,
        chunk_size: int = 4096,
        max_bytes: int | None = None,
    ) -> LanceDataset:
        """Return a new dataset materialized at the requested cache level.

        ``encoded`` stores compressed image bytes and still decodes in the
        pipeline. ``decoded`` stores shared uint8 HWC pixels, so the pipeline
        starts after decoding and should omit ``decode_image()``. The original
        dataset is unchanged. ``max_bytes`` applies to decoded caching.
        """
        if level not in {"encoded", "decoded"}:
            raise ValueError(
                "cache level must be 'encoded' or 'decoded'"
            )
        if chunk_size <= 0:
            raise ValueError("cache chunk_size must be greater than 0")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("cache max_bytes must not be negative")
        return LanceDataset._from_inner(
            self._inner.cache(level, chunk_size, max_bytes)
        )

    def pipeline(self) -> Pipeline:
        return Pipeline(self._inner.pipeline())


class Transform:
    """Reusable ordered image transforms for pipeline composition."""

    def __init__(self, inner: _Transform | None = None) -> None:
        self._inner = _Transform() if inner is None else inner

    def compose(self, other: Transform) -> Transform:
        return Transform(self._inner.compose(other._inner))

    def decode_image(self) -> Transform:
        return Transform(self._inner.decode_image())

    def resize(
        self, width: int, height: int, interpolation: str = "bilinear"
    ) -> Transform:
        return Transform(
            self._inner.resize(width, height, interpolation)
        )

    def crop(self, x: int, y: int, width: int, height: int) -> Transform:
        return Transform(self._inner.crop(x, y, width, height))

    def center_crop(self, width: int, height: int) -> Transform:
        return Transform(self._inner.center_crop(width, height))

    def pad(self, padding: int, fill: float = 0.0) -> Transform:
        return Transform(self._inner.pad(padding, fill))

    def pad_with_sides(
        self, left: int, top: int, right: int, bottom: int, fill: float = 0.0
    ) -> Transform:
        return Transform(self._inner.pad_with_sides(left, top, right, bottom, fill))

    def horizontal_flip(self) -> Transform:
        return Transform(self._inner.horizontal_flip())

    def vertical_flip(self) -> Transform:
        return Transform(self._inner.vertical_flip())

    def random_crop(
        self, width: int, height: int, padding: int = 0
    ) -> Transform:
        return Transform(self._inner.random_crop(width, height, padding))

    def random_resized_crop(self, width: int, height: int) -> Transform:
        return Transform(self._inner.random_resized_crop(width, height))

    def random_resized_crop_with_options(
        self,
        width: int,
        height: int,
        scale: Iterable[float] | None = None,
        ratio: Iterable[float] | None = None,
        interpolation: str = "bilinear",
    ) -> Transform:
        return Transform(
            self._inner.random_resized_crop_with_options(
                width,
                height,
                None if scale is None else list(scale),
                None if ratio is None else list(ratio),
                interpolation,
            )
        )

    def random_horizontal_flip(self, probability: float = 0.5) -> Transform:
        return Transform(self._inner.random_horizontal_flip(probability))

    def brightness(self, value: int) -> Transform:
        return Transform(self._inner.brightness(value))

    def contrast(self, value: float) -> Transform:
        return Transform(self._inner.contrast(value))

    def color_jitter(
        self, brightness: int = 0, contrast: float = 0.0, hue: int = 0
    ) -> Transform:
        return Transform(self._inner.color_jitter(brightness, contrast, hue))

    def invert(self) -> Transform:
        return Transform(self._inner.invert())

    def posterize(self, bits: int) -> Transform:
        return Transform(self._inner.posterize(bits))

    def solarize(self, threshold: int) -> Transform:
        return Transform(self._inner.solarize(threshold))

    def autocontrast(self) -> Transform:
        return Transform(self._inner.autocontrast())

    def equalize(self) -> Transform:
        return Transform(self._inner.equalize())

    def sharpness(self, amount: float) -> Transform:
        return Transform(self._inner.sharpness(amount))

    def gaussian_blur(self, sigma: float) -> Transform:
        return Transform(self._inner.gaussian_blur(sigma))

    def hue(self, degrees: int) -> Transform:
        return Transform(self._inner.hue(degrees))

    def grayscale(self, num_output_channels: int = 1) -> Transform:
        return Transform(self._inner.grayscale(num_output_channels))

    def random_grayscale(
        self, probability: float = 0.1, num_output_channels: int = 1
    ) -> Transform:
        return Transform(
            self._inner.random_grayscale(probability, num_output_channels)
        )

    def random_erasing(self, probability: float = 0.5) -> Transform:
        return Transform(self._inner.random_erasing(probability))

    def random_erasing_with_options(
        self,
        probability: float = 0.5,
        scale: Iterable[float] | None = None,
        ratio: Iterable[float] | None = None,
        value: float = 0.0,
    ) -> Transform:
        return Transform(
            self._inner.random_erasing_with_options(
                probability,
                None if scale is None else list(scale),
                None if ratio is None else list(ratio),
                value,
            )
        )

    def convert_image_dtype(self, dtype: str) -> Transform:
        return Transform(self._inner.convert_image_dtype(dtype))

    def rotate(self, angle: int) -> Transform:
        return Transform(self._inner.rotate(angle))

    def arbitrary_rotate(
        self,
        angle: float,
        interpolation: str = "bilinear",
        expand: bool = True,
        fill: int = 0,
    ) -> Transform:
        return Transform(
            self._inner.arbitrary_rotate(angle, interpolation, expand, fill)
        )

    def random_affine(
        self,
        degrees: float,
        translate: Iterable[float] | None = None,
        scale: Iterable[float] | None = None,
        shear: Iterable[float] | None = None,
        interpolation: str = "bilinear",
        fill: int = 0,
    ) -> Transform:
        return Transform(
            self._inner.random_affine(
                degrees,
                None if translate is None else list(translate),
                None if scale is None else list(scale),
                None if shear is None else list(shear),
                interpolation,
                fill,
            )
        )

    def perspective(
        self,
        start_points: Iterable[Iterable[float]],
        end_points: Iterable[Iterable[float]],
        interpolation: str = "bilinear",
        fill: int = 0,
    ) -> Transform:
        return Transform(
            self._inner.perspective(
                [list(point) for point in start_points],
                [list(point) for point in end_points],
                interpolation,
                fill,
            )
        )

    def random_perspective(
        self,
        distortion_scale: float = 0.5,
        probability: float = 0.5,
        interpolation: str = "bilinear",
        fill: int = 0,
    ) -> Transform:
        return Transform(
            self._inner.random_perspective(
                distortion_scale, probability, interpolation, fill
            )
        )

    def elastic_transform(
        self,
        alpha: float,
        sigma: float,
        interpolation: str = "bilinear",
        fill: int = 0,
    ) -> Transform:
        return Transform(
            self._inner.elastic_transform(alpha, sigma, interpolation, fill)
        )

    def normalize(self, mean: list[float], std: list[float]) -> Transform:
        return Transform(self._inner.normalize(mean, std))

    def hwc_to_chw(self) -> Transform:
        return Transform(self._inner.hwc_to_chw())

    def chw_to_hwc(self) -> Transform:
        return Transform(self._inner.chw_to_hwc())

    def random_apply(self, probability: float, nested: Transform) -> Transform:
        return Transform(self._inner.random_apply(probability, nested._inner))

    def random_choice(self, choices: Iterable[Transform]) -> Transform:
        return Transform(
            self._inner.random_choice([choice._inner for choice in choices])
        )

    def random_order(self, nested: Transform) -> Transform:
        return Transform(self._inner.random_order(nested._inner))


Compose = Transform


class Pipeline:
    """Lazy image input pipeline."""

    def __init__(self, inner: _ImagePipeline, *, as_numpy: bool = True) -> None:
        self._inner = inner
        self.as_numpy = as_numpy

    def decode_image(self) -> Pipeline:
        return Pipeline(self._inner.decode_image(), as_numpy=self.as_numpy)

    def resize(
        self, width: int, height: int, interpolation: str = "bilinear"
    ) -> Pipeline:
        return Pipeline(
            self._inner.resize(width, height, interpolation), as_numpy=self.as_numpy
        )

    def crop(self, x: int, y: int, width: int, height: int) -> Pipeline:
        return Pipeline(self._inner.crop(x, y, width, height), as_numpy=self.as_numpy)

    def center_crop(self, width: int, height: int) -> Pipeline:
        return Pipeline(self._inner.center_crop(width, height), as_numpy=self.as_numpy)

    def pad(self, padding: int, fill: float = 0.0) -> Pipeline:
        return Pipeline(self._inner.pad(padding, fill), as_numpy=self.as_numpy)

    def pad_with_sides(
        self, left: int, top: int, right: int, bottom: int, fill: float = 0.0
    ) -> Pipeline:
        return Pipeline(
            self._inner.pad_with_sides(left, top, right, bottom, fill),
            as_numpy=self.as_numpy,
        )

    def horizontal_flip(self) -> Pipeline:
        return Pipeline(self._inner.horizontal_flip(), as_numpy=self.as_numpy)

    def vertical_flip(self) -> Pipeline:
        return Pipeline(self._inner.vertical_flip(), as_numpy=self.as_numpy)

    def random_crop(
        self,
        width: int,
        height: int,
        padding: int = 0,
    ) -> Pipeline:
        """Zero-pad each image by `padding` pixels on every side, then crop
        a `width`x`height` window at a per-sample random offset.

        Stochastic ops are seeded by the pipeline seed: when the pipeline
        shuffles (``.shuffle(seed)``), the same seed reproduces the same
        sample order *and* the same augmentations at any worker count; use
        ``.epoch(epoch)`` for a fresh augmentation stream per epoch. Without
        a shuffle the augmentations are still deterministic.
        Requires decoded uint8 HWC input.
        """
        return Pipeline(
            self._inner.random_crop(width, height, padding),
            as_numpy=self.as_numpy,
        )

    def random_resized_crop(self, width: int, height: int) -> Pipeline:
        return Pipeline(
            self._inner.random_resized_crop(width, height), as_numpy=self.as_numpy
        )

    def random_resized_crop_with_options(
        self,
        width: int,
        height: int,
        scale: Iterable[float] | None = None,
        ratio: Iterable[float] | None = None,
        interpolation: str = "bilinear",
    ) -> Pipeline:
        return Pipeline(
            self._inner.random_resized_crop_with_options(
                width,
                height,
                None if scale is None else list(scale),
                None if ratio is None else list(ratio),
                interpolation,
            ),
            as_numpy=self.as_numpy,
        )

    def random_horizontal_flip(self, probability: float = 0.5) -> Pipeline:
        """Randomly flip each image horizontally with `probability`
        (one draw per sample from the pipeline seed, see `random_crop`).
        Requires decoded uint8 HWC input.
        """
        return Pipeline(
            self._inner.random_horizontal_flip(probability),
            as_numpy=self.as_numpy,
        )

    def brightness(self, value: int) -> Pipeline:
        return Pipeline(self._inner.brightness(value), as_numpy=self.as_numpy)

    def contrast(self, value: float) -> Pipeline:
        return Pipeline(self._inner.contrast(value), as_numpy=self.as_numpy)

    def color_jitter(
        self, brightness: int = 0, contrast: float = 0.0, hue: int = 0
    ) -> Pipeline:
        return Pipeline(
            self._inner.color_jitter(brightness, contrast, hue),
            as_numpy=self.as_numpy,
        )

    def compose(self, sequence: Transform) -> Pipeline:
        return Pipeline(self._inner.compose(sequence._inner), as_numpy=self.as_numpy)

    def random_apply(self, probability: float, sequence: Transform) -> Pipeline:
        return Pipeline(
            self._inner.random_apply(probability, sequence._inner),
            as_numpy=self.as_numpy,
        )

    def random_choice(self, choices: Iterable[Transform]) -> Pipeline:
        return Pipeline(
            self._inner.random_choice([choice._inner for choice in choices]),
            as_numpy=self.as_numpy,
        )

    def random_order(self, sequence: Transform) -> Pipeline:
        return Pipeline(
            self._inner.random_order(sequence._inner),
            as_numpy=self.as_numpy,
        )

    def invert(self) -> Pipeline:
        return Pipeline(self._inner.invert(), as_numpy=self.as_numpy)

    def posterize(self, bits: int) -> Pipeline:
        return Pipeline(self._inner.posterize(bits), as_numpy=self.as_numpy)

    def solarize(self, threshold: int) -> Pipeline:
        return Pipeline(self._inner.solarize(threshold), as_numpy=self.as_numpy)

    def autocontrast(self) -> Pipeline:
        return Pipeline(self._inner.autocontrast(), as_numpy=self.as_numpy)

    def equalize(self) -> Pipeline:
        return Pipeline(self._inner.equalize(), as_numpy=self.as_numpy)

    def sharpness(self, amount: float) -> Pipeline:
        return Pipeline(self._inner.sharpness(amount), as_numpy=self.as_numpy)

    def gaussian_blur(self, sigma: float) -> Pipeline:
        return Pipeline(self._inner.gaussian_blur(sigma), as_numpy=self.as_numpy)

    def hue(self, degrees: int) -> Pipeline:
        return Pipeline(self._inner.hue(degrees), as_numpy=self.as_numpy)

    def grayscale(self, num_output_channels: int = 1) -> Pipeline:
        return Pipeline(
            self._inner.grayscale(num_output_channels), as_numpy=self.as_numpy
        )

    def random_grayscale(
        self, probability: float = 0.1, num_output_channels: int = 1
    ) -> Pipeline:
        return Pipeline(
            self._inner.random_grayscale(probability, num_output_channels),
            as_numpy=self.as_numpy,
        )

    def random_erasing(self, probability: float = 0.5) -> Pipeline:
        return Pipeline(
            self._inner.random_erasing(probability), as_numpy=self.as_numpy
        )

    def random_erasing_with_options(
        self,
        probability: float = 0.5,
        scale: Iterable[float] | None = None,
        ratio: Iterable[float] | None = None,
        value: float = 0.0,
    ) -> Pipeline:
        return Pipeline(
            self._inner.random_erasing_with_options(
                probability,
                None if scale is None else list(scale),
                None if ratio is None else list(ratio),
                value,
            ),
            as_numpy=self.as_numpy,
        )

    def convert_image_dtype(self, dtype: str) -> Pipeline:
        return Pipeline(
            self._inner.convert_image_dtype(dtype), as_numpy=self.as_numpy
        )

    def rotate(self, angle: int) -> Pipeline:
        return Pipeline(self._inner.rotate(angle), as_numpy=self.as_numpy)

    def arbitrary_rotate(
        self,
        angle: float,
        interpolation: str = "bilinear",
        expand: bool = True,
        fill: int = 0,
    ) -> Pipeline:
        return Pipeline(
            self._inner.arbitrary_rotate(angle, interpolation, expand, fill),
            as_numpy=self.as_numpy,
        )

    def random_affine(
        self,
        degrees: float,
        translate: Iterable[float] | None = None,
        scale: Iterable[float] | None = None,
        shear: Iterable[float] | None = None,
        interpolation: str = "bilinear",
        fill: int = 0,
    ) -> Pipeline:
        return Pipeline(
            self._inner.random_affine(
                degrees,
                None if translate is None else list(translate),
                None if scale is None else list(scale),
                None if shear is None else list(shear),
                interpolation,
                fill,
            ),
            as_numpy=self.as_numpy,
        )

    def perspective(
        self,
        start_points: Iterable[Iterable[float]],
        end_points: Iterable[Iterable[float]],
        interpolation: str = "bilinear",
        fill: int = 0,
    ) -> Pipeline:
        return Pipeline(
            self._inner.perspective(
                [list(point) for point in start_points],
                [list(point) for point in end_points],
                interpolation,
                fill,
            ),
            as_numpy=self.as_numpy,
        )

    def random_perspective(
        self,
        distortion_scale: float = 0.5,
        probability: float = 0.5,
        interpolation: str = "bilinear",
        fill: int = 0,
    ) -> Pipeline:
        return Pipeline(
            self._inner.random_perspective(
                distortion_scale, probability, interpolation, fill
            ),
            as_numpy=self.as_numpy,
        )

    def elastic_transform(
        self,
        alpha: float,
        sigma: float,
        interpolation: str = "bilinear",
        fill: int = 0,
    ) -> Pipeline:
        return Pipeline(
            self._inner.elastic_transform(alpha, sigma, interpolation, fill),
            as_numpy=self.as_numpy,
        )

    def normalize(self, mean: list[float], std: list[float]) -> Pipeline:
        return Pipeline(self._inner.normalize(mean, std), as_numpy=self.as_numpy)

    def hwc_to_chw(self) -> Pipeline:
        return Pipeline(self._inner.hwc_to_chw(), as_numpy=self.as_numpy)

    def chw_to_hwc(self) -> Pipeline:
        return Pipeline(self._inner.chw_to_hwc(), as_numpy=self.as_numpy)

    def skip(self, count: int) -> Pipeline:
        return Pipeline(self._inner.skip(count), as_numpy=self.as_numpy)

    def take(self, count: int) -> Pipeline:
        return Pipeline(self._inner.take(count), as_numpy=self.as_numpy)

    def batch(self, size: int, *, drop_last: bool = False) -> Pipeline:
        return Pipeline(
            self._inner.batch(size, drop_last),
            as_numpy=self.as_numpy,
        )

    def seed(self, seed: int) -> Pipeline:
        """Set the pipeline-owned seed for sampling and transforms."""
        return Pipeline(self._inner.seed(seed), as_numpy=self.as_numpy)

    def shuffle(self, seed: int | None = None) -> Pipeline:
        """Deterministically shuffle the sampled window.

        The same seed reproduces the same order at any worker count.
        If ``seed`` is omitted, use the pipeline-owned seed configured by
        :meth:`seed`; ``.epoch(epoch)`` changes both sampler and transform
        namespaces.
        """
        return Pipeline(
            self._inner.shuffle(seed),
            as_numpy=self.as_numpy,
        )

    def epoch(self, epoch: int) -> Pipeline:
        """Set the deterministic augmentation epoch without reordering data."""
        return Pipeline(self._inner.epoch(epoch), as_numpy=self.as_numpy)

    def workers(self, num_workers: int) -> Pipeline:
        """Load samples on a persistent pool of `num_workers` threads.

        Zero (the default) keeps synchronous loading; ordering and batch
        contents are identical for any worker count.
        """
        return Pipeline(
            self._inner.workers(num_workers),
            as_numpy=self.as_numpy,
        )

    def prefetch_batches(self, prefetch: int) -> Pipeline:
        """Prepare `prefetch` future batches ahead (worker pools only).

        The current batch plus `prefetch` more are in flight; ``0`` keeps
        only the current batch running.
        """
        return Pipeline(
            self._inner.prefetch_batches(prefetch),
            as_numpy=self.as_numpy,
        )

    def with_numpy(self, enabled: bool = True) -> Pipeline:
        return Pipeline(self._inner, as_numpy=enabled)

    def execute(self, *, as_numpy: bool | None = None) -> DataLoader:
        return DataLoader(
            self,
            as_numpy=self.as_numpy if as_numpy is None else as_numpy,
        )


def scan_arrow(
    files: Iterable[str | Path],
    *,
    image_column: str = "img",
    label_column: str = "label",
) -> Pipeline:
    dataset = ArrowDataset(
        files,
        image_column=image_column,
        label_column=label_column,
    )
    return dataset.pipeline()


def scan_hf(
    dataset: Any,
    *,
    image_column: str = "img",
    label_column: str = "label",
) -> Pipeline:
    _validate_hf_dataset(dataset)
    return ArrowDataset.from_huggingface(
        dataset,
        image_column=image_column,
        label_column=label_column,
    ).pipeline()


def scan_image_folder(root: str | Path) -> Pipeline:
    return ImageFolder(root).pipeline()


def scan_lance(
    path: str | Path,
    *,
    image_column: str = "image",
    label_column: str = "label",
) -> Pipeline:
    return LanceDataset(
        path,
        image_column=image_column,
        label_column=label_column,
    ).pipeline()


def load_dataset(
    path: str | Path,
    *,
    split: str | None = None,
    image_column: str = "image",
    label_column: str = "label",
) -> LanceDataset | DatasetDict:
    """Load one physical Lance dataset or a logical multi-split dataset.

    A path ending in .lance is one physical dataset. A directory is resolved
    as a split collection using dataset.json when present (with legacy
    dataset.rivet.json support), or immediate *.lance child directories
    otherwise.
    """
    path = Path(path)
    _, lance_dataset_dict, load_lance_split = _require_lance()
    is_physical = path.name.endswith(".lance")

    if split is None:
        if is_physical:
            return LanceDataset(
                path,
                image_column=image_column,
                label_column=label_column,
            )
        return DatasetDict(
            lance_dataset_dict(
                str(path),
                image_column,
                label_column,
            )
        )

    if is_physical:
        raise ValueError(
            "split cannot be specified when loading a single .lance dataset path"
        )

    return LanceDataset._from_inner(
        load_lance_split(
            str(path),
            image_column,
            label_column,
            split,
        )
    )


def dataset(
    path: str | Path,
    *,
    image_column: str = "image",
    label_column: str = "label",
) -> LanceDataset:
    """Open a native Lance image dataset."""
    return LanceDataset(
        path,
        image_column=image_column,
        label_column=label_column,
    )


class DataLoader:
    """Sequential decoded image batch loader, usable as a python iterator.

    The standard protocol applies:

    .. code-block:: python

        for batch in loader:
            x = batch["images"]      # numpy array (or bytes when
            y = batch["labels"]      # as_numpy=False)

    ``iter(loader)`` returns the loader itself and exhaustion is sticky
    (StopIteration). Loading always happens inside Rust on the pipeline's
    persistent worker pool (``pipeline.workers(n)``) with the GIL released
    while a batch is prepared, so this iterator stays a thin control-flow
    wrapper and python threads are not blocked by data preparation.
    """

    def __init__(
        self,
        source: ArrowDataset | LanceDataset | ImageFolder | Pipeline,
        *,
        batch_size: int | None = None,
        as_numpy: bool = True,
    ) -> None:
        if isinstance(source, Pipeline):
            if batch_size is not None:
                raise ValueError("batch_size belongs in pipeline.batch(size)")
            self._inner = source._inner.execute()
        else:
            if batch_size is None:
                raise ValueError("batch_size is required when source is a dataset")
            self._inner = source.pipeline().decode_image().batch(batch_size)._inner.execute()

        self.as_numpy = as_numpy

    def __iter__(self) -> DataLoader:
        return self

    def __next__(self) -> dict[str, Any]:
        return _maybe_numpy_batch(next(self._inner), self.as_numpy)


def hf_arrow_files(dataset: Any) -> list[str]:
    """Return local Hugging Face Arrow cache files for a loaded dataset split."""
    _validate_hf_dataset(dataset)
    files: list[str] = []

    for cache_file in getattr(dataset, "cache_files", []):
        filename = cache_file.get("filename") if isinstance(cache_file, dict) else None
        if filename:
            files.append(str(Path(filename)))

    if not files:
        raise ValueError(
            "dataset has no local Arrow cache files; pass a loaded Hugging Face dataset split"
        )

    return files


def load_hf_arrow_files(
    path: str,
    *,
    split: str,
    name: str | None = None,
    **load_dataset_kwargs: Any,
) -> list[str]:
    """Load a Hugging Face dataset split and return its local Arrow cache files."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "load_hf_arrow_files requires the `datasets` package; install it with "
            "`pip install datasets`"
        ) from exc

    dataset = load_dataset(path, name=name, split=split, **load_dataset_kwargs)
    return hf_arrow_files(dataset)


def read_image_batch(
    arrow_files: Iterable[str | Path],
    batch_size: int,
    *,
    start: int = 0,
    image_column: str = "img",
    label_column: str = "label",
    as_numpy: bool = True,
) -> dict[str, Any]:
    """Read decoded RGB images from Arrow files through the Rust backend.

    The returned image axis order is NHWC and dtype is uint8.
    """
    result = _read_image_batch(
        _normalize_arrow_files(arrow_files),
        batch_size,
        start,
        image_column,
        label_column,
    )
    return _maybe_numpy_batch(result, as_numpy)


def read_hf_image_batch(
    dataset: Any,
    batch_size: int,
    *,
    start: int = 0,
    image_column: str = "img",
    label_column: str = "label",
    as_numpy: bool = True,
) -> dict[str, Any]:
    """Read a decoded RGB batch from a loaded Hugging Face dataset split."""
    return read_image_batch(
        hf_arrow_files(dataset),
        batch_size,
        start=start,
        image_column=image_column,
        label_column=label_column,
        as_numpy=as_numpy,
    )


def load_hf_image_batch(
    path: str,
    *,
    split: str,
    batch_size: int,
    start: int = 0,
    name: str | None = None,
    image_column: str = "img",
    label_column: str = "label",
    as_numpy: bool = True,
    **load_dataset_kwargs: Any,
) -> dict[str, Any]:
    """Load a Hugging Face split, pass its Arrow files to Rust, and return a batch."""
    return read_image_batch(
        load_hf_arrow_files(path, split=split, name=name, **load_dataset_kwargs),
        batch_size,
        start=start,
        image_column=image_column,
        label_column=label_column,
        as_numpy=as_numpy,
    )
