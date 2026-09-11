from collections.abc import Iterator
from pathlib import Path

import lance
import pyarrow as pa

ROOT = Path("/data/datasets/custom/nuimages_classification")
OUTPUT = Path("/tmp/my_dataset")

BATCH_SIZE = 1024

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


schema = pa.schema(
    [
        pa.field("image", pa.binary(), nullable=False),
        pa.field("label", pa.int32(), nullable=False),
        pa.field("label_name", pa.string(), nullable=False),
        pa.field("path", pa.string(), nullable=False),
    ]
)


def build_class_mapping(split_dir: Path):
    classes = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])

    return {class_name: index for index, class_name in enumerate(classes)}


def iter_samples(
    split_dir: Path,
    class_to_idx: dict[str, int],
) -> Iterator[dict]:

    for class_name, label in class_to_idx.items():
        class_dir = split_dir / class_name

        for path in class_dir.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            with path.open("rb") as f:
                image_bytes = f.read()

            yield {
                "image": image_bytes,
                "label": label,
                "label_name": class_name,
                "path": str(path.relative_to(split_dir)),
            }


def make_batches(
    split_dir: Path,
    class_to_idx: dict[str, int],
    batch_size: int,
) -> Iterator[pa.RecordBatch]:

    rows = []

    for row in iter_samples(split_dir, class_to_idx):
        rows.append(row)

        if len(rows) >= batch_size:
            yield pa.RecordBatch.from_pylist(
                rows,
                schema=schema,
            )
            rows.clear()

    if rows:
        yield pa.RecordBatch.from_pylist(
            rows,
            schema=schema,
        )


def convert_split(split: str):
    split_dir = ROOT / split
    output_path = OUTPUT / f"{split}.lance"

    class_to_idx = build_class_mapping(split_dir)

    print(f"{split}:")
    print(class_to_idx)

    dataset = lance.write_dataset(
        make_batches(
            split_dir,
            class_to_idx,
            BATCH_SIZE,
        ),
        str(output_path),
        schema=schema,
        mode="overwrite",
    )

    print(f"wrote {dataset.count_rows()} rows -> {output_path}")


OUTPUT.mkdir(parents=True, exist_ok=True)

convert_split("train")
convert_split("val")
