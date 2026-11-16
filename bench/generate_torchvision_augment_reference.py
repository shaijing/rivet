"""Regenerate independent CPU/PIL reference data for native augmentation tests."""

import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import autoaugment as A
from torchvision.transforms import functional as F

torch.set_num_threads(1)
pixels = np.fromfunction(
    lambda y, x, c: (17 * y + 29 * x + 53 * c + x * y) % 256, (7, 9, 3), dtype=int
).astype(np.uint8)
image = Image.fromarray(pixels)
tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
ops = [
    ("Identity", 0),
    ("ShearX", 0.17),
    ("ShearY", -0.23),
    ("TranslateX", 2),
    ("TranslateY", -1),
    ("Rotate", 17),
    ("Brightness", 0.3),
    ("Color", 0.3),
    ("Contrast", -0.4),
    ("Sharpness", 0.7),
    ("Posterize", 4),
    ("Solarize", 127.25),
    ("AutoContrast", 0),
    ("Equalize", 0),
    ("Invert", 0),
]
cases = []
for backend, source in [("pil", image), ("tensor", tensor)]:
    for name, m in ops:
        m = float(np.float32(m))
        for mode in (
            ["nearest", "bilinear"]
            if name in {"ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"}
            else ["nearest"]
        ):
            result = A._apply_op(
                source,
                name,
                m,
                getattr(F.InterpolationMode, mode.upper()),
                [37, 37, 37],
            )
            values = (
                np.asarray(result)
                if backend == "pil"
                else result.permute(1, 2, 0).numpy()
            )
            cases.append(
                {
                    "backend": backend,
                    "op": name,
                    "magnitude": m,
                    "interpolation": mode,
                    "expected": values.reshape(-1).tolist(),
                }
            )
spaces = []
for kind, transform, bins in [
    ("auto", A.AutoAugment(), 10),
    ("rand", A.RandAugment(), 31),
    ("wide", A.TrivialAugmentWide(), 31),
    ("mix", A.AugMix(), 10),
]:
    meta = (
        transform._augmentation_space(bins)
        if kind == "wide"
        else transform._augmentation_space(bins, (7, 9))
    )
    for op, (magnitudes, signed) in meta.items():
        spaces.append(
            {
                "kind": kind,
                "op": op,
                "signed": signed,
                "bins": bins,
                "magnitudes": magnitudes.float().reshape(-1).tolist(),
            }
        )
# A fixed mixed plan exercises Tensor branches plus mixing/truncation order.
chains = [
    [("ShearX", 0.17), ("Color", 0.3)],
    [("Rotate", 17), ("Contrast", -0.4), ("Equalize", 0)],
]
weights = [0.25, 0.5]
original = 0.25
mix = tensor.float() * original
for chain, weight in zip(chains, weights):
    aug = tensor
    for op, m in chain:
        aug = A._apply_op(
            aug, op, float(np.float32(m)), F.InterpolationMode.BILINEAR, [37] * 3
        )
    mix.add_(weight * aug)
data = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "height": 7,
    "width": 9,
    "pixels": pixels.reshape(-1).tolist(),
    "cases": cases,
    "spaces": spaces,
    "mix": {
        "original_weight": original,
        "weights": weights,
        "chains": chains,
        "expected": mix.to(torch.uint8).permute(1, 2, 0).reshape(-1).tolist(),
    },
}
data["policies"] = {
    policy.value: A.AutoAugment(policy).policies for policy in A.AutoAugmentPolicy
}
path = (
    Path(__file__).resolve().parents[1]
    / "crates/rivet-vision/tests/fixtures/torchvision_augment.json"
)
path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
print(path)
