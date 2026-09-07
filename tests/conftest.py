from __future__ import annotations

import os
from pathlib import Path

import pytest


def _candidate_arrow_files() -> list[Path]:
    explicit = os.environ.get("RIVET_TEST_ARROW_FILE")
    if explicit:
        return [Path(explicit)]

    roots = []
    if datasets_cache := os.environ.get("HF_DATASETS_CACHE"):
        roots.append(Path(datasets_cache))
    if hf_home := os.environ.get("HF_HOME"):
        roots.append(Path(hf_home) / "datasets")
    roots.append(Path.home() / ".cache" / "huggingface" / "datasets")

    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)
        files.extend(root.glob("uoft-cs___cifar10/**/cifar10-train.arrow"))
    return sorted(files)


@pytest.fixture(scope="module")
def arrow_file() -> Path:
    for path in _candidate_arrow_files():
        if path.exists():
            return path
    pytest.skip("local CIFAR-10 Arrow cache not found; set RIVET_TEST_ARROW_FILE")
