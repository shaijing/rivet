"""Download dataset sources into the Rivet cache."""

from __future__ import annotations

import hashlib
import json
import ntpath
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import unquote, urlsplit


def _dataset_cache() -> Path:
    xdg = Path(os.environ.get("XDG_CACHE_HOME", ""))
    base = xdg if xdg.is_absolute() else Path.home() / ".cache"
    return base / "rivet" / "datasets"


def list_datasets(cache_dir: str | Path | None = None) -> list[dict]:
    """List downloaded, derived Lance, and local Lance datasets in the cache."""
    from ._rivet import _list_datasets

    root = _dataset_cache() if cache_dir is None else Path(cache_dir).expanduser()
    return _list_datasets(root)


def _is_remote(source: str | Path) -> bool:
    return isinstance(source, str) and source.startswith(
        ("hf:", "git:", "github:", "torch:", "http://", "https://")
    )


def _git_url(source: str) -> str:
    if "://" not in source:
        parts = source.split("/")
        if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
            raise ValueError("Git source must be owner/repo or a Git URL")
        source = f"https://github.com/{source}"
    url = urlsplit(source)
    if url.scheme == "file" and not url.netloc and url.path.startswith("/"):
        return source
    if url.scheme not in {"https", "http", "ssh", "git"} or not url.hostname:
        raise ValueError("unsupported Git URL")
    return source


def _download_git(url: str, destination: Path, revision: str | None) -> None:
    env = {
        **os.environ,
        "GIT_ALLOW_PROTOCOL": "file"
        if url.startswith("file:")
        else "http:https:ssh:git",
    }

    def git(*args: str) -> None:
        subprocess.run(["git", *args], check=True, env=env)

    if revision is None:
        git("clone", "--depth=1", "--no-checkout", "--", url, str(destination))
        git("-C", str(destination), "checkout", "--detach", "HEAD")
    else:
        git("init", str(destination))
        git("-C", str(destination), "remote", "add", "origin", url)
        git("-C", str(destination), "fetch", "--depth=1", "--", "origin", revision)
        git("-C", str(destination), "checkout", "--detach", "FETCH_HEAD")
    shutil.rmtree(destination / ".git")
    repository = destination.resolve()
    for path in destination.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(repository):
            raise ValueError(
                f"Git symbolic link leaves the repository: {path.relative_to(destination)}"
            )


def _zip_path(destination: Path, name: str) -> Path:
    posix = PurePosixPath(name)
    if (
        not posix.parts
        or posix.is_absolute()
        or ".." in posix.parts
        or any(
            "\\" in part or ":" in part or ntpath.isreserved(part)
            for part in posix.parts
        )
    ):
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return destination.joinpath(*posix.parts)


def _unpack_archive(archive: Path, destination: Path) -> bool:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                target = _zip_path(destination, entry.filename)
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError(
                        f"ZIP member is a symbolic link: {entry.filename!r}"
                    )
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(entry) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
        destination.mkdir(exist_ok=True)
        return True
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            for entry in tar:
                if not (entry.isfile() or entry.isdir()):
                    raise ValueError(f"unsupported TAR member: {entry.name!r}")
                tar.extract(entry, destination, filter="data")
        destination.mkdir(exist_ok=True)
        return True
    return False


def _download_http(source: str, destination: Path, extract: bool) -> Path:
    import httpx

    destination.mkdir()
    name = Path(unquote(urlsplit(source).path.rsplit("/", 1)[-1])).name
    if (
        name in {"", ".", ".."}
        or "\\" in name
        or ":" in name
        or ntpath.isreserved(name)
    ):
        name = "download"
    artifact = destination / name
    with httpx.stream("GET", source, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with artifact.open("wb") as output:
            for chunk in response.iter_bytes():
                output.write(chunk)

    if extract:
        unpacked = destination.with_name("unpacked")
        if _unpack_archive(artifact, unpacked):
            shutil.rmtree(destination)
            unpacked.rename(destination)
            return destination
    return artifact


def download_dataset(
    source: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    extract: bool = True,
    format: Literal["raw", "lance"] = "raw",
    data_files: str | Mapping[str, str | Sequence[str]] | None = None,
    json_columns: Sequence[str] | None = None,
) -> Path:
    """Return a cached repository, extracted archive, or downloaded file.

    Git and Hugging Face accept a separate revision. Cache hits reuse the
    first successful download for the same source, revision, and options.
    Set format="lance" to convert the source into cached Lance splits.
    data_files selects a relative file or maps split names to relative globs.
    json_columns names fields in positional JSON records.
    A torch: source returns a root for the original torchvision dataset class.
    """
    if format not in {"raw", "lance"}:
        raise ValueError("format must be 'raw' or 'lance'")
    if format == "raw" and (data_files is not None or json_columns is not None):
        raise ValueError("data_files and json_columns require format='lance'")
    if not _is_remote(source):
        raise ValueError(
            "dataset source must use hf:, git:, github:, torch:, http://, or https://"
        )
    provider, location = source.split(":", 1)
    if revision == "" or (
        revision is not None and provider in {"http", "https", "torch"}
    ):
        raise ValueError("revision must be a nonempty Git or Hugging Face reference")
    if provider in {"git", "github"}:
        provider, location = "git", _git_url(location)
    elif provider in {"http", "https"}:
        provider, location = "http", source
    cache = (
        _dataset_cache()
        if cache_dir is None
        else Path(cache_dir).expanduser().absolute()
    )
    if provider == "torch":
        from ._torch_dataset import _derive_torch, _torch_source

        name, options = _torch_source(location)
        key = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()

        def download_torch() -> Path:
            return _download_raw(
                provider, location, None, False, cache / provider / name / key, offline
            )

        if format == "lance":
            if data_files is not None or json_columns is not None:
                raise ValueError(
                    "torch: selects data with torchvision constructor options"
                )
            return _derive_torch(cache, name, key, location, download_torch)
        return download_torch()
    extract = provider == "http" and extract
    key = hashlib.sha256(json.dumps([location, revision, extract]).encode()).hexdigest()
    if format == "lance":
        from ._derive_lance import _derive_lance

        name = PurePosixPath(urlsplit(location).path).name.removesuffix(".git")
        return _derive_lance(
            cache,
            name,
            f"{provider}/{key}",
            data_files,
            json_columns,
            lambda: _download_raw(
                provider, location, revision, extract, cache / provider / key, offline
            ),
        )
    return _download_raw(
        provider, location, revision, extract, cache / provider / key, offline
    )


def _download_raw(
    provider: str,
    location: str,
    revision: str | None,
    extract: bool,
    entry: Path,
    offline: bool,
) -> Path:
    root = entry.parent
    if entry.is_dir():
        return entry / (entry / "artifact").read_text(encoding="utf-8")
    if offline:
        raise FileNotFoundError("dataset source is absent from the Rivet cache")

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=root) as temp:
        staging = Path(temp) / "entry"
        staging.mkdir(mode=0o700)
        payload = staging / "data"
        artifact = payload
        if provider == "hf":
            from huggingface_hub import snapshot_download

            snapshot_download(
                location, repo_type="dataset", revision=revision, local_dir=payload
            )
        elif provider == "git":
            _download_git(location, payload, revision)
        elif provider == "torch":
            from ._torch_dataset import _download_torch

            payload.mkdir()
            _download_torch(location, payload)
        else:
            artifact = _download_http(location, payload, extract)
        if provider == "hf":
            (staging / "dataset-name").write_text(location, encoding="utf-8")
        elif provider in {"git", "http"}:
            name = PurePosixPath(urlsplit(location).path).name.removesuffix(".git")
            if name:
                (staging / "dataset-name").write_text(name, encoding="utf-8")
        (staging / "artifact").write_text(
            str(artifact.relative_to(staging)), encoding="utf-8"
        )
        try:
            staging.rename(entry)
        except OSError:
            # Another process may have published a complete entry first.
            if not entry.is_dir():
                raise
    return entry / (entry / "artifact").read_text(encoding="utf-8")
