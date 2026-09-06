from pathlib import Path

import numpy as np

import rivet


def main() -> None:
    arrow_file = Path(
        "/home/ling/.cache/huggingface/datasets/uoft-cs___cifar10/plain_text/"
        "0.0.0/0b2714987fa478483af9968de7c934580d0bb9a2/cifar10-train.arrow"
    )

    try:
        from datasets import load_dataset

        hf_dataset = load_dataset("uoft-cs/cifar10", split="train[:16]")
        pipeline = rivet.scan_hf(hf_dataset).decode_image().batch(8)
    except ImportError:
        # Local fallback for this workspace when `datasets` is not installed.
        pipeline = rivet.scan_arrow([arrow_file]).decode_image().batch(8)

    loader = pipeline.execute(as_numpy=False)
    batch = next(loader)

    print(batch["shape"])
    print(batch["dtype"], batch["layout"])
    print(batch["labels"])
    print(len(batch["images"]))

    image_ops_batch = next(
        rivet.scan_arrow([arrow_file])
        .decode_image()
        .resize(24, 24)
        .center_crop(20, 20)
        .horizontal_flip()
        .brightness(4)
        .contrast(1.0)
        .normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        .hwc_to_chw()
        .batch(2)
        .execute(as_numpy=False)
    )

    assert image_ops_batch["shape"] == (2, 3, 20, 20)
    assert image_ops_batch["dtype"] == "float32"
    assert image_ops_batch["layout"] == "NCHW"
    assert len(image_ops_batch["images"]) == 2 * 3 * 20 * 20 * 4

    numpy_batch = next(rivet.scan_arrow([arrow_file]).decode_image().batch(2).execute())
    assert isinstance(numpy_batch["images"], np.ndarray)
    assert isinstance(numpy_batch["labels"], np.ndarray)
    assert numpy_batch["images"].shape == (2, 32, 32, 3)
    assert numpy_batch["images"].dtype == np.uint8
    assert numpy_batch["labels"].dtype == np.int64


if __name__ == "__main__":
    main()
