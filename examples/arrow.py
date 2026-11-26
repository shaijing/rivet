"""Convert an Arrow IPC stream to a Lance dataset."""

import argparse
from pathlib import Path

import lance
import pyarrow as pa
from pyarrow import ipc

from rivet import download_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="local Arrow IPC stream file or HTTP(S) URL")
    parser.add_argument("output", type=Path, help="output .lance path")
    args = parser.parse_args()

    path = (
        download_dataset(args.input, extract=False)
        if args.input.startswith(("http://", "https://"))
        else Path(args.input).expanduser()
    )
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_stream(source)
        dataset = lance.write_dataset(reader, str(args.output), schema=reader.schema)

    print(f"written: {args.output}")
    print("rows:", dataset.count_rows())
    print("schema:", dataset.schema)


if __name__ == "__main__":
    main()
