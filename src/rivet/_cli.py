"""Command-line access to Rivet's dataset cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from subprocess import SubprocessError
from tarfile import TarError
from zipfile import BadZipFile

from httpx import HTTPError

from ._download import _dataset_cache, download_dataset, list_datasets


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


def _display_path(path: Path, cache_root: Path) -> str:
    if not path.is_relative_to(cache_root):
        return str(path)
    relative = path.relative_to(cache_root)
    parts = (
        re.sub(r"(^|[-_])([0-9a-f]{8})[0-9a-f]{56}$", r"\1\2…", part)
        for part in relative.parts
    )
    return str(Path(*parts))


def _print_datasets(entries: list[dict], cache_dir: Path) -> None:
    if not entries:
        print("No cached datasets found.")
        return

    cache_root = cache_dir.resolve()
    headers = ("NAME", "KIND", "PROVIDER", "SPLITS", "PATH (CACHE RELATIVE)")
    rows = [
        (
            entry["name"],
            entry["kind"],
            entry["provider"] or "-",
            ", ".join(entry["splits"]) or "-",
            _display_path(entry["path"], cache_root),
        )
        for entry in entries
    ]
    widths = [max(map(len, column)) for column in zip(headers, *rows, strict=True)]
    use_color = (
        sys.stdout.isatty()
        and not os.environ.get("NO_COLOR")
        and os.environ.get("TERM") != "dumb"
    )

    def color(value: str, style: str) -> str:
        return f"\033[{style}m{value}\033[0m" if use_color else value

    def cells(row: tuple[str, ...]) -> list[str]:
        return [
            value.ljust(width) if index < len(row) - 1 else value
            for index, (value, width) in enumerate(zip(row, widths, strict=True))
        ]

    print(color("  ".join(cells(headers)), "1"))
    print(color("  ".join("-" * width for width in widths), "2"))
    kind_colors = {"derived": "32", "local": "36", "download": "33"}
    for row in rows:
        columns = cells(row)
        columns[1] = color(columns[1], kind_colors[row[1]])
        print("  ".join(columns))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            cache_dir = args.cache_dir.expanduser() if args.cache_dir else _dataset_cache()
            entries = list_datasets(cache_dir)
            if not args.all:
                entries = _preferred_datasets(entries)
            if args.json:
                print(json.dumps(entries, default=str, ensure_ascii=False))
            else:
                _print_datasets(entries, cache_dir)
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
