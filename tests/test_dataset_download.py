from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import httpx
import pytest

import rivet
from rivet.vision import download_dataset


def serve(monkeypatch, content: bytes, status: int = 200) -> list[str]:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(status, content=content)

    stream = httpx.Client(transport=httpx.MockTransport(respond)).stream
    monkeypatch.setattr(httpx, "stream", stream)
    return requests


def test_http_cache_and_offline(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    requests = serve(monkeypatch, b"data")
    source = "https://example.test/train.parquet?download=1"
    with pytest.raises(FileNotFoundError):
        download_dataset(source, offline=True)
    path = download_dataset(source)
    assert path.is_relative_to(tmp_path / "rivet" / "datasets")
    assert path.name == "train.parquet"
    assert path.read_bytes() == b"data"
    assert download_dataset(source, offline=True) == path
    assert download_dataset(source) == path
    assert requests == [source]


def test_http_failure_and_unsafe_archive_are_not_cached(
    monkeypatch, tmp_path: Path
) -> None:
    source = "https://example.test/data.zip"
    serve(monkeypatch, b"missing", 404)
    with pytest.raises(httpx.HTTPStatusError):
        download_dataset(source, cache_dir=tmp_path)

    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("../outside", "unsafe")
    serve(monkeypatch, data.getvalue())
    with pytest.raises(ValueError, match="unsafe ZIP"):
        download_dataset(source, cache_dir=tmp_path)
    assert list((tmp_path / "http").iterdir()) == []

    raw = download_dataset(source, cache_dir=tmp_path, extract=False)
    assert raw.read_bytes() == data.getvalue()


def test_huggingface_snapshot(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub

    def snapshot(repo_id, *, repo_type, revision, local_dir):
        assert (repo_id, repo_type, revision) == ("owner/data", "dataset", "v1")
        local_dir.mkdir()
        (local_dir / "README.md").write_text("dataset")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    path = download_dataset("hf:owner/data", revision="v1", cache_dir=tmp_path)
    assert (path / "README.md").read_text() == "dataset"


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is unavailable")
def test_git_download_at_revision(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True)

    git("init")
    git("config", "user.email", "test@example.test")
    git("config", "user.name", "Test")
    (repo / "data").write_text("first")
    git("add", "data")
    git("commit", "-m", "first")
    git("-c", "tag.gpgSign=false", "tag", "-m", "v1", "v1")
    (repo / "data").write_text("second")
    git("commit", "-am", "second")

    path = download_dataset(
        f"git:{repo.as_uri()}", revision="v1", cache_dir=tmp_path / "cache"
    )
    assert (path / "data").read_text() == "first"
    assert not (path / ".git").exists()


@pytest.mark.parametrize("extension", ["zip", "tar.gz"])
def test_remote_lance_archive_keeps_legacy_manifest(
    monkeypatch, tmp_path: Path, extension: str
) -> None:
    import pyarrow as pa

    lance = pytest.importorskip("lance")
    bundle = tmp_path / ("bundle.lance" if extension == "zip" else "bundle")
    train = bundle / "train.lance"
    lance.write_dataset(
        pa.table({"image": [b"encoded"], "label": pa.array([7], type=pa.int32())}),
        str(train),
    )
    (bundle / "dataset.rivet.json").write_text(
        json.dumps({"format_version": 1, "splits": {"train": {"path": "train.lance"}}})
    )
    data = io.BytesIO()
    if extension == "zip":
        with zipfile.ZipFile(data, "w") as archive:
            for path in bundle.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(tmp_path))
    else:
        with tarfile.open(fileobj=data, mode="w:gz") as archive:
            archive.add(bundle, arcname="bundle")
    serve(monkeypatch, data.getvalue())

    source = f"https://example.test/lance.{extension}"
    cache = tmp_path / "cache"
    dataset = rivet.load_dataset(source, split="train", cache_dir=cache)
    assert dataset.get_encoded(0)["label"] == 7
    assert rivet.load_dataset(source, cache_dir=cache, offline=True).keys() == ["train"]


def test_download_parquet_as_lance_and_reuse(monkeypatch, tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    lance = pytest.importorskip("lance")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    buffer = io.BytesIO()
    table = pa.table({"image": [b"encoded"], "label": pa.array([7], type=pa.int32())})
    pq.write_table(table, buffer)
    source = "https://example.test/train.parquet"
    requests = serve(monkeypatch, buffer.getvalue())

    with monkeypatch.context() as no_pylance:
        no_pylance.setitem(sys.modules, "lance", None)
        derived = download_dataset(source, format="lance")
    assert derived.is_relative_to(tmp_path / "rivet" / "datasets" / "derived" / "lance")
    assert lance.dataset(derived / "train.lance").to_table().equals(table)
    assert rivet.load_dataset(derived, split="train").get_encoded(0)["label"] == 7
    assert download_dataset(source, offline=True).name == "train.parquet"
    assert download_dataset(source, format="lance", offline=True) == derived
    assert requests == [source]


def test_download_grouped_json_preserves_splits(monkeypatch, tmp_path: Path) -> None:
    lance = pytest.importorskip("lance")
    source = "https://example.test/records.json"
    requests = serve(monkeypatch, b'{"train":[["hello","greeting"]],"test":[]}')
    options = {"format": "lance", "cache_dir": tmp_path, "data_files": "records.json"}
    with pytest.raises(ValueError, match="match json_columns"):
        download_dataset(source, json_columns=("text",), **options)
    assert not list((tmp_path / "derived").rglob("dataset.json"))

    derived = download_dataset(
        source, json_columns=("text", "label"), offline=True, **options
    )
    train = lance.dataset(derived / "train.lance").to_table()
    test = lance.dataset(derived / "test.lance").to_table()
    assert train.to_pylist() == [{"text": "hello", "label": "greeting"}]
    assert test.num_rows == 0
    assert test.schema == train.schema
    assert requests == [source]


def test_download_lance_merges_selected_shards(monkeypatch, tmp_path: Path) -> None:
    lance = pytest.importorskip("lance")
    source = "https://example.test/shards.zip"
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("first.jsonl", '{"value":1,"text":"first"}\n')
        archive.writestr("second.jsonl", '{"value":2.5}\n')
    serve(monkeypatch, data.getvalue())

    derived = download_dataset(
        source, format="lance", cache_dir=tmp_path, data_files={"train": "*.jsonl"}
    )
    assert lance.dataset(derived / "train.lance").to_table().to_pylist() == [
        {"value": 1.0, "text": "first"},
        {"value": 2.5, "text": None},
    ]


def test_torchvision_download_preserves_native_root(
    monkeypatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    calls = []

    class VisionDataset:
        pass

    class CIFAR10(VisionDataset):
        def __init__(self, root, train=True, download=False):
            calls.append((train, download))
            raw = Path(root) / "Digits" / "raw"
            source = raw / ("train" if train else "test")
            if download:
                raw.mkdir(parents=True, exist_ok=True)
                source.write_bytes(b"native bytes")
            assert source.is_file()
            import numpy as np

            self.data = np.full((1, 2, 3, 3), 42 if train else 75, dtype=np.uint8)
            self.targets = [0 if train else 1]
            self.classes = ["first", "second"]

    class ExtendedDigits(CIFAR10):
        def __init__(self, root, variant, **kwargs):
            assert variant == "letters"
            super().__init__(root, **kwargs)

    datasets = SimpleNamespace(
        __all__=["CIFAR10", "ExtendedDigits"],
        VisionDataset=VisionDataset,
        CIFAR10=CIFAR10,
        ExtendedDigits=ExtendedDigits,
    )
    monkeypatch.setitem(sys.modules, "torchvision", SimpleNamespace(datasets=datasets))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    root = download_dataset("torch:CIFAR10")
    assert root.is_relative_to(tmp_path / "rivet" / "datasets" / "torch" / "cifar10")
    assert (root / "Digits" / "raw" / "train").read_bytes() == b"native bytes"
    assert (root / "Digits" / "raw" / "test").read_bytes() == b"native bytes"
    assert download_dataset("torch:cifar10", offline=True) == root

    test = download_dataset("torch:CIFAR10?split=test")
    assert (test / "Digits" / "raw" / "test").is_file()
    assert not (test / "Digits" / "raw" / "train").exists()

    with pytest.raises(ValueError, match="invalid or repeated"):
        download_dataset("torch:CIFAR10?download=false")

    bundle = rivet.load_dataset("torch:CIFAR10", offline=True)
    assert bundle.keys() == ["test", "train"]
    assert bundle["train"].get_encoded(0)["label"] == 0
    assert (
        bundle["test"].get_decoded(0)["images"][0].tolist() == [[[75, 75, 75]] * 3] * 2
    )
    derived = download_dataset("torch:CIFAR10", format="lance", offline=True)
    assert derived.is_relative_to(tmp_path / "rivet" / "datasets" / "derived" / "lance")
    assert (
        rivet.load_dataset(
            "torch:CIFAR10?split=test", split="test", offline=True
        ).get_encoded(0)["label"]
        == 1
    )
    with pytest.raises(ValueError, match="variant"):
        download_dataset("torch:ExtendedDigits")
    download_dataset("torch:ExtendedDigits?train=false&variant=letters")
    assert calls == [
        (True, True),
        (False, True),
        (False, True),
        (True, False),
        (False, False),
        (False, False),
        (False, True),
    ]
