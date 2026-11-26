"""Inspect the schema and contents of a physical Lance table."""

import argparse

import lance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="physical .lance dataset path")
    args = parser.parse_args()
    dataset = lance.dataset(args.path)

    print("=== Schema ===")
    print(dataset.schema)
    print("\n=== Rows ===")
    print(dataset.count_rows())
    print("\n=== Version ===")
    print(dataset.version)
    print(dataset.to_table())
