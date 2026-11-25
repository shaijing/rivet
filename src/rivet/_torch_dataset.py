"""Download through torchvision while retaining its original root layout."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qsl


def _torch_source(location: str) -> tuple[str, dict[str, str]]:
    name, _, query = location.removeprefix("//").partition("?")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError("torch: requires a torchvision dataset class name")
    options = {}
    for key, value in parse_qsl(query, keep_blank_values=True, strict_parsing=True):
        if (
            not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", key)
            or key
            in {
                "root",
                "download",
                "transform",
                "target_transform",
                "transforms",
                "loader",
            }
            or key in options
        ):
            raise ValueError(f"invalid or repeated torchvision option: {key!r}")
        options[key] = value
    return name.lower(), options


def _torch_calls(location: str):
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError(
            "torch: downloads require torchvision in the active environment"
        ) from exc

    name, options = _torch_source(location)
    exported = next((item for item in datasets.__all__ if item.lower() == name), None)
    dataset = getattr(datasets, exported) if exported is not None else None
    if not inspect.isclass(dataset) or not issubclass(dataset, datasets.VisionDataset):
        raise ValueError(f"unknown torchvision dataset: {name!r}")

    signature = inspect.signature(dataset)
    parameters = dict(signature.parameters)
    if "download" not in parameters and any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        # EMNIST and QMNIST expose download via their MNIST base constructor.
        for base in dataset.__mro__[1:]:
            inherited = inspect.signature(base).parameters
            if "download" in inherited:
                parameters = {**inherited, **parameters}
                break
    keyword_kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    if any(
        key not in parameters or parameters[key].kind not in keyword_kinds
        for key in ("root", "download")
    ):
        raise ValueError(
            f"torchvision dataset {dataset.__name__} requires manual acquisition"
        )

    if "split" in options and "split" not in parameters and "train" in parameters:
        split = options.pop("split")
        if split not in {"train", "test"} or "train" in options:
            raise ValueError("use either train=true/false or split=train/test")
        options["train"] = "true" if split == "train" else "false"

    kwargs = {}
    for key, value in options.items():
        parameter = parameters.get(key)
        if parameter is None or parameter.kind not in keyword_kinds:
            raise ValueError(f"{dataset.__name__} does not accept option {key!r}")
        if isinstance(parameter.default, str):
            kwargs[key] = value
            continue
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = value
        if isinstance(parameter.default, bool) and not isinstance(decoded, bool):
            raise TypeError(f"torchvision option {key!r} requires true or false")
        kwargs[key] = decoded

    calls = [kwargs]
    if (
        "train" not in kwargs
        and "train" in parameters
        and isinstance(parameters["train"].default, bool)
    ):
        calls = [{**kwargs, "train": True}, {**kwargs, "train": False}]
    for options in calls:
        try:
            signature.bind(root="", download=True, **options)
        except TypeError as exc:
            raise ValueError(
                f"invalid options for torchvision dataset {dataset.__name__}: {exc}"
            ) from exc
    return dataset, calls


def _download_torch(location: str, root: Path) -> None:
    dataset, calls = _torch_calls(location)
    for options in calls:
        dataset(root=str(root), download=True, **options)


def _derive_torch(
    cache: Path, name: str, key: str, location: str, download: Callable[[], Path]
) -> Path:
    """Stage torchvision's in-memory pixel arrays for the Rust image writer."""
    import tempfile

    import numpy as np

    supported = {
        "cifar10",
        "cifar100",
        "mnist",
        "fashionmnist",
        "kmnist",
        "emnist",
        "usps",
        "svhn",
        "stl10",
    }
    if name not in supported:
        raise ValueError(
            f"Lance conversion is unavailable for torchvision dataset {name!r}"
        )

    output = cache / "derived" / "lance" / name / f"torch-v1-{key}"
    if (output / "dataset.json").is_file():
        return output
    try:
        from ._rivet import _convert_torch_images
    except ImportError as exc:
        raise RuntimeError(
            "Lance conversion requires Rivet's 'lance' build feature"
        ) from exc
    root = download()
    dataset_class, calls = _torch_calls(location)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".convert-torch-", dir=output.parent
    ) as temp:
        work = Path(temp)
        splits = {}
        classes = None
        for options in calls:
            split = options.get("split") if name in {"svhn", "stl10"} else None
            if split is None:
                split = "train" if options.get("train", True) else "test"
            dataset = dataset_class(root=str(root), download=False, **options)
            split = (
                getattr(dataset, "split", split) if name in {"svhn", "stl10"} else split
            )
            split_classes = getattr(dataset, "classes", None)
            if split_classes is not None:
                classes = list(split_classes)
            pixels = np.asarray(dataset.data)
            if name in {"svhn", "stl10"}:
                pixels = pixels.transpose(0, 2, 3, 1)
            if pixels.ndim == 3:
                pixels = pixels[..., None]
            if (
                pixels.ndim != 4
                or pixels.dtype != np.uint8
                or pixels.shape[3] not in {1, 3}
            ):
                raise ValueError(
                    f"unsupported torchvision pixel layout: {pixels.shape}, {pixels.dtype}"
                )
            targets = dataset.targets if hasattr(dataset, "targets") else dataset.labels
            if targets is None:
                raise ValueError(f"torchvision split {split!r} has no class labels")
            targets = np.asarray(targets)
            if targets.shape != (len(pixels),) or targets.dtype.kind not in "iu":
                raise ValueError(
                    f"torchvision split {split!r} requires one integer label per image"
                )
            if targets.size and (
                targets.min() < 0 or targets.max() > np.iinfo(np.int32).max
            ):
                raise ValueError(
                    f"torchvision split {split!r} has invalid class labels"
                )
            index = len(splits)
            images, labels = work / f"images-{index}", work / f"labels-{index}"
            np.ascontiguousarray(pixels).tofile(images)
            targets.astype("<i4", copy=False).tofile(labels)
            rows, height, width, channels = pixels.shape
            splits[split] = (images, labels, rows, width, height, channels)
        staging = work / name
        _convert_torch_images(staging, splits, classes)
        try:
            staging.rename(output)
        except OSError:
            if not (output / "dataset.json").is_file():
                raise
    return output
