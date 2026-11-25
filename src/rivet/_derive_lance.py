"""Materialize tabular download artifacts as Lance split datasets."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

_SUFFIXES = {".parquet", ".arrow", ".csv", ".tsv", ".jsonl", ".ndjson", ".json"}
_SPLIT_NAMES = {"train", "test", "dev", "val", "validation"}


def _valid_split(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"invalid Lance split name: {name!r}")
    return name


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.anchor or ".." in path.parts or not path.parts:
        raise ValueError(f"data_files must be relative to the download: {value!r}")
    return path


def _select_files(
    payload: Path, data_files: str | Mapping[str, tuple[str, ...]] | None
) -> dict[str, list[Path]]:
    base = payload if payload.is_dir() else payload.parent
    if isinstance(data_files, str):
        path = base / _relative_path(data_files)
        if not path.is_file() or (payload.is_file() and path != payload):
            raise ValueError(f"downloaded file not found: {data_files!r}")
        return {"train": [path]}
    if data_files is not None:
        groups = {}
        for split, patterns in data_files.items():
            paths = {}
            for pattern in patterns:
                _relative_path(pattern)
                for path in sorted(base.glob(pattern)):
                    if path.is_file() and (payload.is_dir() or path == payload):
                        paths[path] = None
            if not paths:
                raise ValueError(f"no downloaded files matched split {split!r}")
            groups[split] = list(paths)
        return groups

    files = [payload] if payload.is_file() else sorted(payload.rglob("*"))
    files = [
        path
        for path in files
        if path.is_file()
        and path.suffix.lower() in _SUFFIXES
        and not any(
            part.startswith(".") or part == "__MACOSX"
            for part in path.relative_to(base).parts
        )
        and (
            payload.is_file() or path.name not in {"dataset.json", "dataset.rivet.json"}
        )
        and not path.name.endswith(".metadata.json")
    ]
    if not files:
        raise ValueError("no supported data files found in the download")
    if len({path.suffix.lower() for path in files}) > 1:
        raise ValueError("multiple input formats found; select splits with data_files")

    groups: dict[str, list[Path]] = {}
    for path in files:
        parts = path.relative_to(base).with_suffix("").parts
        tokens = re.split(r"[-_.]", parts[-1].lower())
        split = next((token for token in tokens if token in _SPLIT_NAMES), None)
        if split is None:
            split = next(
                (part for part in reversed(parts[:-1]) if part in _SPLIT_NAMES), "train"
            )
        groups.setdefault(split, []).append(path)
    return groups


def _derive_lance(
    cache: Path,
    name: str,
    source_key: str,
    data_files: str | Mapping[str, str | Sequence[str]] | None,
    json_columns: Sequence[str] | None,
    download: Callable[[], Path],
) -> Path:
    files = data_files
    if isinstance(data_files, Mapping):
        if not data_files:
            raise ValueError("data_files must declare at least one split")
        files = {
            _valid_split(split): (patterns,)
            if isinstance(patterns, str)
            else tuple(patterns)
            for split, patterns in sorted(data_files.items())
        }
    columns = tuple(json_columns) if json_columns is not None else None
    if json_columns is not None and (
        isinstance(json_columns, str)
        or not columns
        or any(not isinstance(name, str) or not name for name in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("json_columns must contain unique, nonempty column names")
    name = re.sub(r"[^A-Za-z0-9_-]", "-", name).strip("-")[:80] or "dataset"
    key = hashlib.sha256(
        json.dumps([1, source_key, files, columns]).encode()
    ).hexdigest()
    parent = cache / "derived" / "lance" / name
    output = parent / key
    if (output / "dataset.json").is_file():
        return output
    try:
        from ._rivet import _convert_lance_files
    except ImportError as exc:
        raise RuntimeError(
            "Lance conversion requires Rivet to be built with the 'lance' feature"
        ) from exc

    payload = download()
    groups = _select_files(payload, files)
    root = (payload if payload.is_dir() else payload.parent).resolve()
    for paths in groups.values():
        for path in paths:
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"data file leaves the download directory: {path}")

    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".convert-", dir=parent) as tmp:
        staging = Path(tmp) / "dataset"
        staging.mkdir(mode=0o700)
        _convert_lance_files(
            staging, name, groups, columns, not isinstance(files, Mapping)
        )
        try:
            staging.rename(output)
        except OSError:
            if not (output / "dataset.json").is_file():
                raise
    return output
