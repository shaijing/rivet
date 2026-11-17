"""Python iterator and framework handoff for image batches."""

from __future__ import annotations

from typing import Any

from ._datasets import ArrowDataset, ImageFolder, LanceDataset
from ._pipeline import Pipeline
from ._utils import _maybe_numpy_batch


class DataLoader:
    """Sequential decoded image batch loader, usable as a python iterator.

    The standard protocol applies:

    .. code-block:: python

        for batch in loader:
            x = batch["images"]      # numpy array (or bytes when
            y = batch["labels"]      # as_numpy=False)

    NumPy batches are read-only zero-copy views over Rust-owned CPU storage;
    keep a reference to the batch while using the views. Use
    :meth:`next_dlpack` for a zero-copy framework handoff or
    :meth:`next_torch` for a Torch tensor batch.

    Multi-view pipelines flatten each source's views consecutively and repeat
    its label. Their batches include ``views_per_sample`` and
    ``source_batch_size``; the first image dimension is their product.

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
            self._inner = (
                source.pipeline().decode_image().batch(batch_size)._inner.execute()
            )

        self.as_numpy = as_numpy

    def __iter__(self) -> DataLoader:
        return self

    def __next__(self) -> dict[str, Any]:
        return _maybe_numpy_batch(next(self._inner), self.as_numpy)

    def next_dlpack(self) -> dict[str, Any]:
        """Return the next batch through public zero-copy DLPack handles.

        The ``images`` and ``labels`` values are repeatable shared DLPack
        producers; each capsule returned by ``__dlpack__`` is one-shot.
        ``into_dlpack()`` is the explicit ownership-transfer form. After
        that call, Rivet no longer retains that Tensor alias and the handle
        cannot be exported again.
        """
        batch = self._inner.next_dlpack()
        if batch is None:
            raise StopIteration
        return batch

    def next_torch(self) -> dict[str, Any]:
        """Return the next batch as Torch tensors through DLPack."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "DataLoader.next_torch requires the `torch` package"
            ) from exc

        batch = self.next_dlpack()
        batch["images"] = torch.utils.dlpack.from_dlpack(batch["images"])
        batch["labels"] = torch.utils.dlpack.from_dlpack(batch["labels"])
        return batch
