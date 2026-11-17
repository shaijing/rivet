"""Dataset wrappers and scan entry points."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ._io import hf_arrow_files
from ._pipeline import Pipeline
from ._rivet import _ArrowDataset, _ImageFolderDataset
from ._utils import _maybe_numpy_batch, _normalize_arrow_files, _validate_hf_dataset

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
            raise ValueError("cache level must be 'encoded' or 'decoded'")
        if chunk_size <= 0:
            raise ValueError("cache chunk_size must be greater than 0")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("cache max_bytes must not be negative")
        return LanceDataset._from_inner(self._inner.cache(level, chunk_size, max_bytes))

    def pipeline(self) -> Pipeline:
        return Pipeline(self._inner.pipeline())


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
