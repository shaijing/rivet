"""Command-line access to Rivet's dataset cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from subprocess import SubprocessError
from tarfile import TarError
from zipfile import BadZipFile

from httpx import HTTPError

from ._download import download_dataset, list_datasets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rivet", description="Manage Rivet datasets.")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('rivet')}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="list cached datasets")
    listing.add_argument(
        "--cache-dir", type=Path, help="override the XDG dataset cache"
    )
    listing.add_argument("--json", action="store_true", help="emit JSON")
    listing.add_argument("--all", action="store_true", help="show every cached variant")

    download = commands.add_parser("download", help="download or convert a dataset")
    download.add_argument("source", help="hf:, git:, torch:, or HTTP(S) source")
    download.add_argument(
        "--cache-dir", type=Path, help="override the XDG dataset cache"
    )
    download.add_argument("--revision", help="Git or Hugging Face revision")
    download.add_argument(
        "--offline", action="store_true", help="use cached files only"
    )
    download.add_argument("--format", choices=("raw", "lance"), default="raw")
    download.add_argument(
        "--no-extract", action="store_true", help="retain the HTTP archive"
    )
    files = download.add_mutually_exclusive_group()
    files.add_argument("--data-file", help="select a file relative to the download")
    files.add_argument(
        "--split-file",
        action="append",
        metavar="SPLIT=GLOB",
        help="select files for a Lance split (repeatable)",
    )
    download.add_argument(
        "--json-column",
        action="append",
        help="name a positional JSON column (repeatable)",
    )
    return parser


def _preferred_datasets(entries: list[dict]) -> list[dict]:
    # The torchvision download cache hashes the constructor options.
    default_torch_key = hashlib.sha256(b"{}").hexdigest()
    kind_order = {"derived": 0, "local": 1, "download": 2}

    def priority(entry: dict) -> tuple:
        path = entry["path"]
        return (
            kind_order[entry["kind"]],
            -len(entry["splits"]),
            not (
                entry["provider"] == "torch" and path.parent.name == default_torch_key
            ),
            not path.is_dir(),
            path,
        )

    preferred = {}
    for entry in sorted(entries, key=priority):
        preferred.setdefault(entry["name"], entry)
    return [preferred[name] for name in sorted(preferred)]


def _print_datasets(entries: list[dict]) -> None:
    if not entries:
        print("No cached datasets found.")
        return

    headers = ("NAME", "KIND", "PROVIDER", "SPLITS", "PATH")
    rows = [
        (
            entry["name"],
            entry["kind"],
            entry["provider"] or "-",
            ", ".join(entry["splits"]) or "-",
            str(entry["path"]),
        )
        for entry in entries
    ]
    widths = [max(map(len, column)) for column in zip(headers, *rows, strict=True)]
    for row in (headers, tuple("-" * width for width in widths), *rows):
        print(
            "  ".join(
                value.ljust(width)
                for value, width in zip(row[:-1], widths[:-1], strict=True)
            ),
            row[-1],
            sep="  ",
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            entries = list_datasets(args.cache_dir)
            if not args.all:
                entries = _preferred_datasets(entries)
            if args.json:
                print(json.dumps(entries, default=str, ensure_ascii=False))
            else:
                _print_datasets(entries)
        else:
            data_files = args.data_file
            if args.split_file:
                data_files = {}
                for value in args.split_file:
                    split, _, pattern = value.partition("=")
                    if not split or not pattern:
                        parser.error("--split-file requires SPLIT=GLOB")
                    data_files.setdefault(split, []).append(pattern)
            path = download_dataset(
                args.source,
                revision=args.revision,
                cache_dir=args.cache_dir,
                offline=args.offline,
                extract=not args.no_extract,
                format=args.format,
                data_files=data_files,
                json_columns=args.json_column,
            )
            print(path)
    except KeyboardInterrupt:
        return 130
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        ImportError,
        HTTPError,
        SubprocessError,
        TarError,
        BadZipFile,
    ) as exc:
        print(f"rivet: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
