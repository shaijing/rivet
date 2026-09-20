from __future__ import annotations

import rivet
from rivet import vision


def test_vision_is_the_canonical_image_namespace() -> None:
    assert vision.ArrowDataset is rivet.ArrowDataset
    assert vision.Pipeline is rivet.Pipeline
    assert vision.scan_image_folder is rivet.scan_image_folder
    assert "vision" in rivet.__all__
