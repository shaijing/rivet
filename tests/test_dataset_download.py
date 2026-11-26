from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

import rivet
from rivet.vision import download_dataset


@pytest.fixture
def server():
    requests = []
    routes = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            status, headers, content = routes.get(self.path, (404, {}, b"missing"))
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(content)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as http:
        thread = Thread(target=http.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{http.server_port}", routes, requests
        finally:
            http.shutdown()
            thread.join()


def test_http_cache_and_offline(monkeypatch, tmp_path: Path, server) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    endpoint, routes, requests = server
    routes["/train.parquet?download=1"] = (200, {}, b"data")
    source = endpoint + "/train.parquet?download=1"
    with pytest.raises(FileNotFoundError):
        download_dataset(source, offline=True)
    path = download_dataset(source)
    key = hashlib.sha256(json.dumps([source, None, True]).encode()).hexdigest()
    assert (
        path
        == tmp_path / "rivet" / "datasets" / "http" / key / "data" / "train.parquet"
    )
    assert path.read_bytes() == b"data"
    assert download_dataset(source, offline=True) == path
    assert download_dataset(source) == path
    assert requests == ["/train.parquet?download=1"]


def test_http_failure_and_unsafe_archive_are_not_cached(tmp_path: Path, server) -> None:
    endpoint, routes, _ = server
    source = endpoint + "/data.zip"
    with pytest.raises(OSError, match="404"):
        download_dataset(source, cache_dir=tmp_path)

    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("../outside", "unsafe")
    routes["/data.zip"] = (200, {}, data.getvalue())
    with pytest.raises(ValueError, match="unsafe ZIP"):
        download_dataset(source, cache_dir=tmp_path)
    assert list((tmp_path / "http").iterdir()) == []

    raw = download_dataset(source, cache_dir=tmp_path, extract=False)
    assert raw.read_bytes() == data.getvalue()


def test_huggingface_snapshot(monkeypatch, tmp_path: Path, server) -> None:
    endpoint, routes, requests = server
    monkeypatch.setenv("HF_ENDPOINT", endpoint)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    commit = "a" * 40
    api = "/api/datasets/owner/data"
    tree = f"{api}/tree/{commit}"
    resolve = f"/datasets/owner/data/resolve/{commit}"
    routes[f"{api}/revision/refs%2Fv1"] = (
        200,
        {},
        json.dumps({"sha": commit}).encode(),
    )
    routes[f"{tree}?recursive=true&expand=false"] = (
        200,
        {"Link": f'<{tree}?cursor=next>; rel="next"'},
        b'[{"type":"directory","path":"data"},{"type":"file","path":"README.md"}]',
    )
    routes[f"{tree}?cursor=next"] = (
        200,
        {},
        b'[{"type":"file","path":"data/test rows.jsonl"}]',
    )
    routes[f"{resolve}/README.md"] = (200, {}, b"dataset")
    routes[f"{resolve}/data/test%20rows.jsonl"] = (200, {}, b'{"label":1}\n')

    options = {"revision": "refs/v1", "cache_dir": tmp_path}
    path = download_dataset("hf:owner/data", **options)
    assert (path / "README.md").read_text() == "dataset"
    assert (path / "data/test rows.jsonl").read_text() == '{"label":1}\n'
    assert download_dataset("hf:owner/data", offline=True, **options) == path
    assert requests == [
        f"{api}/revision/refs%2Fv1",
        f"{tree}?recursive=true&expand=false",
        f"{resolve}/README.md",
        f"{tree}?cursor=next",
        f"{resolve}/data/test%20rows.jsonl",
    ]


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


@pytest.mark.parametrize("extension", ["zip", "tar", "tar.gz", "tar.bz2", "tar.xz"])
def test_remote_lance_archive_keeps_legacy_manifest(
    tmp_path: Path, extension: str, server
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
        compression = extension.removeprefix("tar").removeprefix(".")
        with tarfile.open(fileobj=data, mode=f"w:{compression}") as archive:
            archive.add(bundle, arcname="bundle")
    endpoint, routes, _ = server
    routes[f"/lance.{extension}"] = (200, {}, data.getvalue())
    source = f"{endpoint}/lance.{extension}"
    cache = tmp_path / "cache"
    dataset = rivet.load_dataset(source, split="train", cache_dir=cache)
    assert dataset.get_encoded(0)["label"] == 7
    assert rivet.load_dataset(source, cache_dir=cache, offline=True).keys() == ["train"]


def test_remote_lance_manifest_cannot_open_split_outside_cache(tmp_path: Path, server):
    endpoint, routes, _ = server
    external_split = tmp_path / "outside.lance"
    external_split.mkdir()
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr(
            "dataset.rivet.json",
            json.dumps(
                {
                    "format_version": 1,
                    "splits": {"train": {"path": str(external_split)}},
                }
            ),
        )
    routes["/bundle.zip"] = (200, {}, content.getvalue())

    with pytest.raises(ValueError, match="leaves the cache"):
        rivet.load_dataset(f"{endpoint}/bundle.zip", cache_dir=tmp_path / "cache")


def test_download_parquet_as_lance_and_reuse(
    monkeypatch, tmp_path: Path, server
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    lance = pytest.importorskip("lance")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    buffer = io.BytesIO()
    table = pa.table({"image": [b"encoded"], "label": pa.array([7], type=pa.int32())})
    pq.write_table(table, buffer)
    endpoint, routes, requests = server
    routes["/train.parquet"] = (200, {}, buffer.getvalue())
    source = endpoint + "/train.parquet"

    with monkeypatch.context() as no_pylance:
        no_pylance.setitem(sys.modules, "lance", None)
        derived = download_dataset(source, format="lance")
    raw_key = hashlib.sha256(json.dumps([source, None, True]).encode()).hexdigest()
    derived_key = hashlib.sha256(
        json.dumps([1, f"http/{raw_key}", None, None]).encode()
    ).hexdigest()
    assert derived == (
        tmp_path
        / "rivet"
        / "datasets"
        / "derived"
        / "lance"
        / "train-parquet"
        / derived_key
    )
    assert lance.dataset(derived / "train.lance").to_table().equals(table)
    assert rivet.load_dataset(derived, split="train").get_encoded(0)["label"] == 7
    assert download_dataset(source, offline=True).name == "train.parquet"
    assert download_dataset(source, format="lance", offline=True) == derived
    assert requests == ["/train.parquet"]


def test_download_grouped_json_preserves_splits(tmp_path: Path, server) -> None:
    lance = pytest.importorskip("lance")
    endpoint, routes, requests = server
    routes["/records.json"] = (200, {}, b'{"train":[["hello","greeting"]],"test":[]}')
    source = endpoint + "/records.json"
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
    assert requests == ["/records.json"]


def test_download_lance_merges_selected_shards(tmp_path: Path, server) -> None:
    lance = pytest.importorskip("lance")
    endpoint, routes, _ = server
    source = endpoint + "/shards.zip"
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("first.jsonl", '{"value":1,"text":"first"}\n')
        archive.writestr("second.jsonl", '{"value":2.5}\n')
    routes["/shards.zip"] = (200, {}, data.getvalue())

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


def test_list_datasets_includes_completed_downloads_and_lance(
    monkeypatch, tmp_path: Path, server
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cache = tmp_path / "rivet" / "datasets"
    assert rivet.list_datasets() == []

    endpoint, routes, requests = server
    routes["/data.parquet"] = (200, {}, b"file")
    source = endpoint + "/data.parquet"
    raw = download_dataset(source)
    index = cache / "manifests"
    index.mkdir()
    (index / "import.json").write_text(
        json.dumps({"datasets": {"named": {"path": str(raw), "source": source}}})
    )
    derived = cache / "derived" / "lance" / "converted" / "snapshot"
    (derived / "train.lance").mkdir(parents=True)
    (derived / "dataset.json").write_text(
        json.dumps({"format_version": 1, "splits": {"train": {"path": "train.lance"}}})
    )
    local = cache / "local" / "lance" / "single.lance"
    (local / "_versions").mkdir(parents=True)
    pending = cache / "hf" / ".download-in-progress"
    (pending / "data").mkdir(parents=True)
    (pending / "artifact").write_text("data")
    incomplete = cache / "git" / "missing"
    incomplete.mkdir(parents=True)
    (incomplete / "artifact").write_text("data")

    entries = rivet.vision.list_datasets(cache)
    assert [(entry["name"], entry["kind"], entry["path"]) for entry in entries] == [
        ("converted", "derived", derived),
        ("named", "download", raw),
        ("single.lance", "local", local),
    ]
    assert entries[0]["splits"] == ["train"]
    assert entries[1]["provider"] == "http"
    assert entries[1]["source"] == source
    assert requests == ["/data.parquet"]
