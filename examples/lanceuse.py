"""Download tabular data and inspect the derived Lance split."""

import argparse

import lance

from rivet import download_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default="hf:SetFit/sst5")
    parser.add_argument("--split", default="train")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    root = download_dataset(args.source, format="lance", offline=args.offline)
    dataset = lance.dataset(str(root / f"{args.split}.lance"))
    print(dataset.count_rows())
    print(dataset.schema)
