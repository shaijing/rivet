"""P0 RandomCrop padding order compatibility tests."""

import numpy as np
import pytest
from PIL import Image

import rivet

F = pytest.importorskip("torchvision.transforms.functional")


def test_random_crop_pad_if_needed_applies_symmetric_padding_sequentially(tmp_path):
    class_dir = tmp_path / "class"
    class_dir.mkdir()
    pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    image = Image.fromarray(pixels)
    image.save(class_dir / "sample.png")

    # torchvision pads the explicit request first, then pads the width and
    # height deficits in separate calls. Symmetric padding is not associative
    # here, so combining these borders into one Pad changes the possible crops.
    padded = F.pad(image, (1, 0, 0, 0), padding_mode="symmetric")
    padded = F.pad(padded, [3, 0], padding_mode="symmetric")
    padded = F.pad(padded, [0, 4], padding_mode="symmetric")
    expected_crops = {
        np.asarray(F.crop(padded, top, left, 6, 6)).tobytes()
        for top in range(padded.height - 5)
        for left in range(padded.width - 5)
    }

    results = []
    for seed in range(8):
        pipeline = (
            rivet.scan_image_folder(tmp_path)
            .decode_image()
            .random_crop(
                6,
                6,
                padding=(1, 0, 0, 0),
                pad_if_needed=True,
                padding_mode="symmetric",
            )
            .seed(seed)
        )
        actual = next(pipeline.batch(1).execute())["images"][0]
        results.append(actual.tobytes())

    assert all(result in expected_crops for result in results)
