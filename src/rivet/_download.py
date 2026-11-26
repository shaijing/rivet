"""Dataset download bindings and cache discovery."""

from pathlib import Path

from ._rivet import _dataset_cache, _list_datasets, download_dataset

__all__ = ["download_dataset", "list_datasets"]


def list_datasets(cache_dir: str | Path | None = None) -> list[dict]:
    """List downloaded, derived Lance, and local Lance datasets in the cache."""
    root = _dataset_cache() if cache_dir is None else Path(cache_dir).expanduser()
    return _list_datasets(root)
