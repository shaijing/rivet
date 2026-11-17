"""Reusable image transform composition."""

from __future__ import annotations

from collections.abc import Iterable

from ._rivet import _Transform


class Transform:
    """Reusable ordered image transforms for pipeline composition."""

    def __init__(self, inner: _Transform | None = None) -> None:
        self._inner = _Transform() if inner is None else inner

    def compose(self, other: Transform) -> Transform:
        return Transform(self._inner.compose(other._inner))

    def decode_image(self) -> Transform:
        return Transform(self._inner.decode_image())

    def resize(
        self, width: int, height: int, interpolation: str = "bilinear"
    ) -> Transform:
        return Transform(self._inner.resize(width, height, interpolation))

    def crop(self, x: int, y: int, width: int, height: int) -> Transform:
        return Transform(self._inner.crop(x, y, width, height))

    def center_crop(self, width: int, height: int) -> Transform:
        return Transform(self._inner.center_crop(width, height))

    def pad(
        self, padding: int, fill: float = 0.0, padding_mode: str = "constant"
    ) -> Transform:
        return Transform(self._inner.pad(padding, fill, padding_mode))

    def pad_with_sides(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
        fill: float = 0.0,
        padding_mode: str = "constant",
    ) -> Transform:
        return Transform(
            self._inner.pad_with_sides(left, top, right, bottom, fill, padding_mode)
        )

    def auto_augment(
        self, policy: str = "imagenet", interpolation: str = "nearest", fill: int = 0
    ) -> Transform:
        return Transform(self._inner.auto_augment(policy, interpolation, fill))

    def rand_augment(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: str = "nearest",
        fill: int = 0,
    ) -> Transform:
        return Transform(
            self._inner.rand_augment(
                num_ops, magnitude, num_magnitude_bins, interpolation, fill
            )
        )

    def trivial_augment_wide(
        self,
        num_magnitude_bins: int = 31,
        interpolation: str = "nearest",
        fill: int = 0,
    ) -> Transform:
        return Transform(
            self._inner.trivial_augment_wide(num_magnitude_bins, interpolation, fill)
        )

    def augmix(
        self,
        severity: int = 3,
        mixture_width: int = 3,
        chain_depth: int = -1,
        alpha: float = 1.0,
        all_ops: bool = True,
        interpolation: str = "bilinear",
        fill: int = 0,
    ) -> Transform:
        return Transform(
            self._inner.augmix(
                severity,
                mixture_width,
                chain_depth,
                alpha,
                all_ops,
                interpolation,
                fill,
            )
        )

    def to_tensor(self) -> Transform:
        """Convert decoded HWC pixels to float32 CHW, scaling uint8 by 255."""
        return Transform(self._inner.to_tensor())

    def pil_to_tensor(self) -> Transform:
        """Convert decoded HWC pixels to CHW without changing dtype/range."""
        return Transform(self._inner.pil_to_tensor())

    def linear_transform(
        self, matrix: Iterable[Iterable[float]], mean: Iterable[float]
    ) -> Transform:
        return Transform(
            self._inner.linear_transform([list(row) for row in matrix], list(mean))
        )

    def saturation(self, factor: float) -> Transform:
        return Transform(self._inner.saturation(factor))

    def gamma(self, gamma: float, gain: float = 1.0) -> Transform:
        return Transform(self._inner.gamma(gamma, gain))

    def affine(
        self,
        angle: float,
        translate: Iterable[float] | None = None,
        scale: float = 1.0,
        shear: Iterable[float] | None = None,
        interpolation: str = "nearest",
        fill: int = 0,
        center: Iterable[float] | None = None,
    ) -> Transform:
        return Transform(
            self._inner.affine(
                angle,
                None if translate is None else list(translate),
                scale,
                None if shear is None else list(shear),
                interpolation,
                fill,
                None if center is None else list(center),
            )
        )

    def erase(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        value: float | Iterable[float] = 0.0,
    ) -> Transform:
        values = [float(value)] if isinstance(value, (int, float)) else list(value)
        return Transform(self._inner.erase(x, y, width, height, values))

    def resized_crop(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        out_width: int,
        out_height: int,
        interpolation: str = "bilinear",
    ) -> Transform:
        return Transform(
            self._inner.resized_crop(
                x, y, width, height, out_width, out_height, interpolation
            )
        )

    def random_rotation(
        self,
        degrees: float | Iterable[float],
        interpolation: str = "nearest",
        expand: bool = False,
        fill: int = 0,
    ) -> Transform:
        if isinstance(degrees, (int, float)):
            if degrees < 0:
                raise ValueError("degrees must be non-negative")
            degrees = [-float(degrees), float(degrees)]
        return Transform(
            self._inner.random_rotation(list(degrees), interpolation, expand, fill)
        )

    def random_gaussian_blur(
        self, kernel_size: int, sigma: Iterable[float] | None = None
    ) -> Transform:
        return Transform(
            self._inner.random_gaussian_blur(
                kernel_size, None if sigma is None else list(sigma)
            )
        )

    def random_vertical_flip(self, probability: float = 0.5) -> Transform:
        return Transform(self._inner.random_vertical_flip(probability))

    def random_invert(self, probability: float = 0.5) -> Transform:
        return Transform(self._inner.random_invert(probability))

    def random_posterize(self, bits: int, probability: float = 0.5) -> Transform:
        return Transform(self._inner.random_posterize(bits, probability))

    def random_solarize(self, threshold: int, probability: float = 0.5) -> Transform:
        return Transform(self._inner.random_solarize(threshold, probability))

    def random_adjust_sharpness(
        self, factor: float, probability: float = 0.5
    ) -> Transform:
        return Transform(self._inner.random_adjust_sharpness(factor, probability))

    def random_autocontrast(self, probability: float = 0.5) -> Transform:
        return Transform(self._inner.random_autocontrast(probability))

    def random_equalize(self, probability: float = 0.5) -> Transform:
        return Transform(self._inner.random_equalize(probability))

    def horizontal_flip(self) -> Transform:
        return Transform(self._inner.horizontal_flip())

    def vertical_flip(self) -> Transform:
        return Transform(self._inner.vertical_flip())

    def random_crop(self, width: int, height: int, padding: int = 0) -> Transform:
        return Transform(self._inner.random_crop(width, height, padding))

    def random_resized_crop(self, width: int, height: int) -> Transform:
        return Transform(self._inner.random_resized_crop(width, height))

    def random_resized_crop_with_options(
        self,
        width: int,
        height: int,
        scale: Iterable[float] | None = None,
        ratio: Iterable[float] | None = None,
        interpolation: str = "bilinear",
    ) -> Transform:
        return Transform(
            self._inner.random_resized_crop_with_options(
                width,
                height,
                None if scale is None else list(scale),
                None if ratio is None else list(ratio),
                interpolation,
            )
        )

    def random_horizontal_flip(self, probability: float = 0.5) -> Transform:
        return Transform(self._inner.random_horizontal_flip(probability))

    def brightness(self, value: float) -> Transform:
        """Adjust brightness by a nonnegative factor (1 is identity)."""
        return Transform(self._inner.brightness(value))

    def contrast(self, value: float) -> Transform:
        return Transform(self._inner.contrast(value))

    def color_jitter(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        hue: float = 0.0,
        saturation: float = 0.0,
    ) -> Transform:
        return Transform(
            self._inner.color_jitter(brightness, contrast, hue, saturation)
        )

    def invert(self) -> Transform:
        return Transform(self._inner.invert())

    def posterize(self, bits: int) -> Transform:
        return Transform(self._inner.posterize(bits))

    def solarize(self, threshold: int) -> Transform:
        return Transform(self._inner.solarize(threshold))

    def autocontrast(self) -> Transform:
        return Transform(self._inner.autocontrast())

    def equalize(self) -> Transform:
        return Transform(self._inner.equalize())

    def sharpness(self, amount: float) -> Transform:
        return Transform(self._inner.sharpness(amount))

    def gaussian_blur(self, sigma: float, kernel_size: int | None = None) -> Transform:
        return Transform(self._inner.gaussian_blur(sigma, kernel_size))

    def hue(self, factor: float) -> Transform:
        """Shift HSV hue by a factor in [-0.5, 0.5]."""
        return Transform(self._inner.hue(factor))

    def grayscale(self, num_output_channels: int = 1) -> Transform:
        return Transform(self._inner.grayscale(num_output_channels))

    def random_grayscale(
        self, probability: float = 0.1, num_output_channels: int = 3
    ) -> Transform:
        return Transform(self._inner.random_grayscale(probability, num_output_channels))

    def random_erasing(self, probability: float = 0.5) -> Transform:
        return Transform(self._inner.random_erasing(probability))

    def random_erasing_with_options(
        self,
        probability: float = 0.5,
        scale: Iterable[float] | None = None,
        ratio: Iterable[float] | None = None,
        value: float = 0.0,
    ) -> Transform:
        return Transform(
            self._inner.random_erasing_with_options(
                probability,
                None if scale is None else list(scale),
                None if ratio is None else list(ratio),
                value,
            )
        )

    def convert_image_dtype(self, dtype: str) -> Transform:
        return Transform(self._inner.convert_image_dtype(dtype))

    def rotate(self, angle: int) -> Transform:
        return Transform(self._inner.rotate(angle))

    def arbitrary_rotate(
        self,
        angle: float,
        interpolation: str = "nearest",
        expand: bool = False,
        fill: int = 0,
    ) -> Transform:
        return Transform(
            self._inner.arbitrary_rotate(angle, interpolation, expand, fill)
        )

    def random_affine(
        self,
        degrees: float,
        translate: Iterable[float] | None = None,
        scale: Iterable[float] | None = None,
        shear: Iterable[float] | None = None,
        interpolation: str = "nearest",
        fill: int = 0,
    ) -> Transform:
        return Transform(
            self._inner.random_affine(
                degrees,
                None if translate is None else list(translate),
                None if scale is None else list(scale),
                None if shear is None else list(shear),
                interpolation,
                fill,
            )
        )

    def perspective(
        self,
        start_points: Iterable[Iterable[float]],
        end_points: Iterable[Iterable[float]],
        interpolation: str = "bilinear",
        fill: int = 0,
    ) -> Transform:
        return Transform(
            self._inner.perspective(
                [list(point) for point in start_points],
                [list(point) for point in end_points],
                interpolation,
                fill,
            )
        )

    def random_perspective(
        self,
        distortion_scale: float = 0.5,
        probability: float = 0.5,
        interpolation: str = "bilinear",
        fill: int = 0,
    ) -> Transform:
        return Transform(
            self._inner.random_perspective(
                distortion_scale, probability, interpolation, fill
            )
        )

    def elastic_transform(
        self,
        alpha: float,
        sigma: float,
        interpolation: str = "bilinear",
        fill: int = 0,
    ) -> Transform:
        return Transform(
            self._inner.elastic_transform(alpha, sigma, interpolation, fill)
        )

    def normalize(self, mean: list[float], std: list[float]) -> Transform:
        return Transform(self._inner.normalize(mean, std))

    def hwc_to_chw(self) -> Transform:
        return Transform(self._inner.hwc_to_chw())

    def chw_to_hwc(self) -> Transform:
        return Transform(self._inner.chw_to_hwc())

    def random_apply(self, probability: float, nested: Transform) -> Transform:
        return Transform(self._inner.random_apply(probability, nested._inner))

    def random_choice(self, choices: Iterable[Transform]) -> Transform:
        return Transform(
            self._inner.random_choice([choice._inner for choice in choices])
        )

    def random_order(self, nested: Transform) -> Transform:
        return Transform(self._inner.random_order(nested._inner))


Compose = Transform
