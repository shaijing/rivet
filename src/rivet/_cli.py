"""Command-line access to Rivet's dataset cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

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
    download.add_argument("source", help="hf:, git:, github:, torch:, or HTTP(S) source")
    download.add_argument(
        "--cache-dir", type=Path, help="override the XDG dataset cache"
    )
    download.add_argument("--revision", help="Git or Hugging Face revision")
    download.add_argument(
        "--offline", action="store_true", help="use cached files only"
    )
    download.add_argument("--format", choices=("raw", "arrow", "arrow-ipc-file", "arrow-ipc-stream", "lance"), default="raw")
    download.add_argument(
        "--no-extract", action="store_true", help="skip HTTP archive extraction during download"
    )
    files = download.add_mutually_exclusive_group()
    files.add_argument("--data-file", help="select a file relative to the download")
    files.add_argument(
        "--split-file",
        action="append",
        metavar="SPLIT=GLOB",
        help="select files for a prepared split (repeatable)",
    )
    download.add_argument(
        "--json-column",
        action="append",
        help="name a positional JSON column (repeatable)",
    )
    return parser


def _dataset_rows(entries: list[dict], show_all: bool) -> list[dict]:
    originals = {entry["path"]: entry for entry in entries if entry["kind"] == "download"}
    rows = []
    used = set()
    for entry in entries:
        if entry["kind"] == "download":
            continue
        original = originals.get(entry["raw_path"])
        if original:
            used.add(original["path"])
        rows.append((original, entry))
    rows.extend((entry, None) for path, entry in originals.items() if path not in used)

    def priority(row: tuple[dict | None, dict | None]) -> tuple:
        original, prepared = row
        entry = original or prepared
        return (
            entry["name"],
            not (prepared and "arrow-ipc-file" in prepared["formats"]),
            prepared is None,
            original is None,
            -len(prepared["splits"]) if prepared else 0,
            entry["path"],
        )

    ordered = sorted(rows, key=priority)
    if not show_all:
        default_torch_key = hashlib.sha256(b"{}").hexdigest()

        def original_priority(entry: dict) -> tuple:
            path = entry["path"]
            return (
                entry["provider"] == "torch" and path.parent.name != default_torch_key,
                not path.is_dir(),
                path,
            )

        preferred_originals = {}
        for entry in sorted(originals.values(), key=original_priority):
            preferred_originals.setdefault(entry["name"], entry)

        merged = {}
        for original, prepared in ordered:
            name = (original or prepared)["name"]
            merged.setdefault(
                name,
                (original if prepared and original else preferred_originals.get(name), prepared),
            )
        ordered = list(merged.values())

    return [
        {
            "name": (original or prepared)["name"],
            "provider": (original or prepared)["provider"],
            "original": original,
            "prepared": prepared,
        }
        for original, prepared in ordered
    ]


def _print_datasets(entries: list[dict]) -> None:
    if not entries:
        print("No cached datasets found.")
        return

    headers = (
        "NAME",
        "PROVIDER",
        "ORIGINAL FORMAT",
        "PREPARED FORMAT",
        "SPLITS",
    )

    def formats(entry: dict | None) -> str:
        return ", ".join(entry["formats"]) if entry else "-"

    rows = [
        (
            entry["name"],
            entry["provider"] or "-",
            formats(entry["original"]),
            formats(entry["prepared"]),
            ", ".join((entry["prepared"] or {}).get("splits", ())) or "-",
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
    for row in rows:
        columns = cells(row)
        columns[2] = color(columns[2], "33")
        columns[3] = color(columns[3], "32")
        print("  ".join(columns))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            cache_dir = args.cache_dir.expanduser() if args.cache_dir else _dataset_cache()
            entries = _dataset_rows(list_datasets(cache_dir), args.all)
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
    ) as exc:
        print(f"rivet: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
