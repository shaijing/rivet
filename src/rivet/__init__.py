from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ._rivet import _ArrowDataset, _ImageFolderDataset, _ImagePipeline
from ._rivet import read_image_batch as _read_image_batch

__all__ = [
    "ArrowDataset",
    "DataLoader",
    "ImageFolder",
    "Pipeline",
    "hf_arrow_files",
    "load_hf_arrow_files",
    "load_hf_image_batch",
    "read_hf_image_batch",
    "read_image_batch",
    "scan_arrow",
    "scan_hf",
    "scan_image_folder",
]


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


class Pipeline:
    """Lazy image input pipeline."""

    def __init__(self, inner: _ImagePipeline, *, as_numpy: bool = True) -> None:
        self._inner = inner
        self.as_numpy = as_numpy

    def decode_image(self) -> Pipeline:
        return Pipeline(self._inner.decode_image(), as_numpy=self.as_numpy)

    def resize(self, width: int, height: int) -> Pipeline:
        return Pipeline(self._inner.resize(width, height), as_numpy=self.as_numpy)

    def crop(self, x: int, y: int, width: int, height: int) -> Pipeline:
        return Pipeline(self._inner.crop(x, y, width, height), as_numpy=self.as_numpy)

    def center_crop(self, width: int, height: int) -> Pipeline:
        return Pipeline(self._inner.center_crop(width, height), as_numpy=self.as_numpy)

    def horizontal_flip(self) -> Pipeline:
        return Pipeline(self._inner.horizontal_flip(), as_numpy=self.as_numpy)

    def vertical_flip(self) -> Pipeline:
        return Pipeline(self._inner.vertical_flip(), as_numpy=self.as_numpy)

    def brightness(self, value: int) -> Pipeline:
        return Pipeline(self._inner.brightness(value), as_numpy=self.as_numpy)

    def contrast(self, value: float) -> Pipeline:
        return Pipeline(self._inner.contrast(value), as_numpy=self.as_numpy)

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
        source: ArrowDataset | ImageFolder | Pipeline,
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

    The returned image layout is NHWC and dtype is uint8.
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
