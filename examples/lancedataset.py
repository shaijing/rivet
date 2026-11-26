"""Inspect an image sample in a local or remote Lance bundle."""

import argparse
import io

from PIL import Image

from rivet import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local bundle path or remote dataset source")
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    dataset = load_dataset(args.source, split=args.split)
    print("Rows:", len(dataset))
    sample = dataset.get_encoded(args.index)
    encoded = sample["image"]
    print("Label:", sample["label"])
    print("Encoded bytes:", len(encoded))

    with Image.open(io.BytesIO(encoded)) as image:
        print("Format:", image.format)
        print("Size:", image.size)
        print("Mode:", image.mode)
