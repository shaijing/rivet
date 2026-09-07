from __future__ import annotations

from pathlib import Path

import pytest

import rivet


def test_hf_indices_rejected() -> None:
    datasets = pytest.importorskip("datasets")

    try:
        ds = datasets.load_dataset("uoft-cs/cifar10", split="train[:8]")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not load cached CIFAR-10 dataset: {exc}")

    shuffled = ds.shuffle(seed=42)
    with pytest.raises(ValueError, match="flatten_indices"):
        rivet.scan_hf(shuffled)


def test_rust_numpy_export_does_not_clone_image_batch() -> None:
    loader_rs = Path(__file__).parents[1] / "crates" / "rivet-python" / "src" / "loader.rs"

    assert "values.clone()" not in loader_rs.read_text()
