"""Reusable image transform composition."""

from __future__ import annotations

import math

from collections.abc import Iterable, Sequence

from ._rivet import _Transform
from ._utils import _normalize_fill, _normalize_jpeg_quality


class Transform:
    """Reusable ordered image transforms for pipeline composition."""

    def __init__(self, inner: _Transform | None = None) -> None:
        self._inner = _Transform() if inner is None else inner

    def compose(self, other: Transform) -> Transform:
        return Transform(self._inner.compose(other._inner))

    def decode_image(self) -> Transform:
        return Transform(self._inner.decode_image())

    def resize(
        self,
        width: int,
        height: int | None = None,
        interpolation: str = "bilinear",
        *,
        max_size: int | None = None,
    ) -> Transform:
        """Resize to explicit width/height, or set the short edge to ``width``.

        With no height, preserve aspect ratio; ``max_size`` optionally caps
        the long edge and must exceed the requested short edge."""
        return Transform(self._inner.resize(width, height, interpolation, max_size))

    def crop(self, x: int, y: int, width: int, height: int) -> Transform:
        return Transform(self._inner.crop(x, y, width, height))

    def five_crop(self, width: int, height: int) -> Transform:
        """Expand each image into top-left, top-right, bottom-left, bottom-right,
        and center crops. ``batch(B)`` yields ``B * 5`` views and repeated labels.
        """
        return Transform(self._inner.five_crop(width, height))

    def ten_crop(
        self, width: int, height: int, *, vertical_flip: bool = False
    ) -> Transform:
        """Emit five original crops followed by five crops of the flipped image.

        The flip is horizontal by default. ``batch(B)`` yields ``B * 10`` views.
        """
        return Transform(self._inner.ten_crop(width, height, vertical_flip))

    def longest_max_size(
        self,
        max_size: int | Iterable[int] | None = None,
        *,
        max_size_hw: tuple[int | None, int | None] | None = None,
        interpolation: str = "bilinear",
    ) -> Transform:
        """Resize uint8 HWC RGB proportionally to the given bounds.

        Supply a longest side (or seeded choices), or (height, width) limits.
        None leaves one axis unconstrained. The limiting axis is reached when
        upscaling or downscaling. Structured masks use nearest sampling.
        """
        sizes = None if max_size is None else ([max_size] if isinstance(max_size, int) else list(max_size))
        inner = self._inner.longest_max_size(sizes, max_size_hw=max_size_hw, interpolation=interpolation)
        return Transform(inner)

    def center_crop(self, width: int, height: int) -> Transform:
        return Transform(self._inner.center_crop(width, height))

    def pad(
        self,
        padding: int,
        fill: float | Iterable[float] = 0.0,
        padding_mode: str = "constant",
    ) -> Transform:
        return Transform(self._inner.pad(padding, _normalize_fill(fill), padding_mode))

    def pad_with_sides(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
        fill: float | Iterable[float] = 0.0,
        padding_mode: str = "constant",
    ) -> Transform:
        return Transform(
            self._inner.pad_with_sides(
                left, top, right, bottom, _normalize_fill(fill), padding_mode
            )
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
        shear: float | Iterable[float] | None = None,
        interpolation: str = "nearest",
        fill: int | Iterable[int] = 0,
        center: Iterable[float] | None = None,
    ) -> Transform:
        return Transform(
            self._inner.affine(
                angle,
                None if translate is None else list(translate),
                scale,
                None
                if shear is None
                else (
                    [float(shear), 0.0]
                    if isinstance(shear, (int, float))
                    else list(shear)
                ),
                interpolation,
                _normalize_fill(fill),
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
        fill: int | Iterable[int] = 0,
        center: Iterable[float] | None = None,
    ) -> Transform:
        if isinstance(degrees, (int, float)):
            if degrees < 0:
                raise ValueError("degrees must be non-negative")
            degrees = [-float(degrees), float(degrees)]
        return Transform(
            self._inner.random_rotation(
                list(degrees),
                interpolation,
                expand,
                _normalize_fill(fill),
                None if center is None else list(center),
            )
        )

    def random_gaussian_blur(
        self,
        kernel_size: int | Iterable[int],
        sigma: float | Iterable[float] | None = None,
    ) -> Transform:
        """Blur with a scalar or (width, height) kernel.

        Sigma is a fixed scalar or [min, max] interval. One sampled sigma
        is shared by both axes; the pair does not specify axis sigmas."""
        return Transform(
            self._inner.random_gaussian_blur(
                kernel_size if isinstance(kernel_size, int) else list(kernel_size),
                sigma
                if sigma is None or isinstance(sigma, (int, float))
                else list(sigma),
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

    def random_crop(
        self,
        width: int,
        height: int,
        padding: int | Sequence[int] = 0,
        *,
        pad_if_needed: bool = False,
        fill: float | Sequence[float] = 0,
        padding_mode: str = "constant",
    ) -> Transform:
        """Apply explicit padding, optional minimum-size padding, then crop.

        Padding is scalar, horizontal/vertical, or left/top/right/bottom.
        Fill is scalar or RGB; modes are constant, edge, reflect, symmetric.
        """
        return Transform(
            self._inner.random_crop(
                width, height, padding, pad_if_needed, fill, padding_mode
            )
        )

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

    def gaussian_blur(
        self,
        sigma: float | Iterable[float],
        kernel_size: int | Iterable[int] | None = None,
    ) -> Transform:
        """Blur with scalar or (x, y) sigma and (width, height) kernel size.

        Omitting kernel size retains Rivet's sigma-derived kernel convenience;
        sigma zero acts as identity on that axis."""
        return Transform(
            self._inner.gaussian_blur(
                sigma if isinstance(sigma, (int, float)) else list(sigma),
                kernel_size
                if kernel_size is None or isinstance(kernel_size, int)
                else list(kernel_size),
            )
        )

    def hue(self, factor: float) -> Transform:
        """Shift HSV hue by a factor in [-0.5, 0.5]."""
        return Transform(self._inner.hue(factor))

    def grayscale(self, num_output_channels: int = 1) -> Transform:
        return Transform(self._inner.grayscale(num_output_channels))

    def random_grayscale(
        self, probability: float = 0.1, num_output_channels: int = 3
    ) -> Transform:
        return Transform(self._inner.random_grayscale(probability, num_output_channels))

    def random_erasing(
        self,
        probability: float = 0.5,
        *,
        scale: Iterable[float] | None = None,
        ratio: Iterable[float] | None = None,
        value: float | Iterable[float] | str = 0.0,
    ) -> Transform:
        """Randomly erase a rectangle with scalar, channel, or ``"random"`` fill.

        Random fill uses independent standard-normal values. Float32 preserves
        fill values; uint8 conversion truncates and wraps modulo 256."""
        return self.random_erasing_with_options(probability, scale, ratio, value)

    def random_erasing_with_options(
        self,
        probability: float = 0.5,
        scale: Iterable[float] | None = None,
        ratio: Iterable[float] | None = None,
        value: float | Iterable[float] | str = 0.0,
    ) -> Transform:
        return Transform(
            self._inner.random_erasing_with_options(
                probability,
                None if scale is None else list(scale),
                None if ratio is None else list(ratio),
                value if isinstance(value, (int, float, str)) else list(value),
            )
        )

    def channel_swap(self, channel_order: Iterable[int] = (2, 1, 0)) -> Transform:
        """Apply a fixed channel order. The default swaps red and blue.

        The order must contain each input channel index exactly once.
        """
        return self.permute_channels(channel_order)

    def permute_channels(self, permutation: Iterable[int]) -> Transform:
        """Reorder channels with a permutation containing each channel once."""
        return Transform(self._inner.permute_channels(list(permutation)))

    def random_channel_permutation(self) -> Transform:
        """Uniformly shuffle image channels using the pipeline random stream."""
        return Transform(self._inner.random_channel_permutation())

    def random_photometric_distort(
        self,
        brightness: Iterable[float] = (0.875, 1.125),
        contrast: Iterable[float] = (0.5, 1.5),
        saturation: Iterable[float] = (0.5, 1.5),
        hue: Iterable[float] = (-0.05, 0.05),
        probability: float = 0.5,
    ) -> Transform:
        """Sample brightness, contrast placement, saturation, hue and channels.

        Each color adjustment and the final channel shuffle uses its own
        probability draw. Contrast runs either before or after saturation/hue.
        """
        return Transform(
            self._inner.random_photometric_distort(
                list(brightness),
                list(contrast),
                list(saturation),
                list(hue),
                probability,
            )
        )

    def random_resize(
        self, min_size: int, max_size: int, interpolation: str = "bilinear"
    ) -> Transform:
        """Resize the short edge to an integer in [min_size, max_size)."""
        return Transform(self._inner.random_resize(min_size, max_size, interpolation))

    def random_shortest_size(
        self,
        min_size: int | Iterable[int],
        max_size: int | None = None,
        interpolation: str = "bilinear",
    ) -> Transform:
        """Choose a short edge uniformly, optionally capping the long edge."""
        sizes = [min_size] if isinstance(min_size, int) else list(min_size)
        return Transform(
            self._inner.random_shortest_size(sizes, max_size, interpolation)
        )

    def scale_jitter(
        self,
        width: int,
        height: int,
        scale_range: Iterable[float] = (0.1, 2.0),
        interpolation: str = "bilinear",
    ) -> Transform:
        """Fit within target width/height, then sample a multiplicative scale."""
        return Transform(
            self._inner.scale_jitter(width, height, list(scale_range), interpolation)
        )

    def random_zoom_out(
        self,
        fill: float | Iterable[float] = 0.0,
        side_range: Iterable[float] = (1.0, 4.0),
        probability: float = 0.5,
    ) -> Transform:
        """Place the image randomly on an expanded canvas with native-range fill."""
        return Transform(
            self._inner.random_zoom_out(
                _normalize_fill(fill), list(side_range), probability
            )
        )

    def gaussian_noise(
        self, mean: float = 0.0, sigma: float = 0.1, clip: bool = True
    ) -> Transform:
        """Add Gaussian noise using mean/sigma in the [0, 1] convention.

        Float32 output clips to [0, 1]; uint8 noise is scaled by 255 and clips
        to [0, 255]. With ``clip=False``, uint8 arithmetic wraps on conversion.
        """
        return Transform(self._inner.gaussian_noise(mean, sigma, clip))

    def letterbox(
        self,
        width: int,
        height: int,
        *,
        interpolation: str = "bilinear",
        fill: float = 0.0,
        mask_fill: int = 0,
    ) -> Transform:
        """Fit uint8 RGB inside a fixed canvas and center it.

        When applied to annotated samples, masks use nearest-neighbor resizing
        and their own padding value. Odd padding goes on the right or bottom.
        """
        return Transform(
            self._inner.letterbox(
                width,
                height,
                interpolation=interpolation,
                fill=fill,
                mask_fill=mask_fill,
            )
        )

    def additive_noise(
        self, noise_type: str = "uniform", spatial_mode: str = "constant", *,
        ranges: list[tuple[float, float]] | None = None,
        mean_range: tuple[float, float] = (0.0, 0.0),
        scale_range: tuple[float, float] = (0.1, 0.1),
        alpha_range: tuple[float, float] = (0.5, 1.5),
        beta_range: tuple[float, float] = (0.5, 1.5),
        patch_count_range: tuple[int, int] = (1, 1),
        patch_height_range: tuple[float, float] = (0.1, 1.0),
        patch_width_range: tuple[float, float] = (0.1, 1.0),
        per_channel: bool = False,
    ) -> Transform:
        """Add seeded offsets to uint8 or normalized float32 HWC and clip.

        Constant samples per channel, per_pixel samples per element, and
        shared samples one offset per pixel for all channels. Uniform accepts
        one range or one per channel, with one required for shared mode.
        Scale is Gaussian standard deviation, Laplace scale, or beta amplitude
        after mapping beta variates to [-1, 1]. Beta shapes are in [1e-4, 1e6].
        Patch mode samples noise once at each pixel in the union of sampled
        rectangles. Patch dimensions are fractions of the image dimensions.
        Set per_channel in patch mode to sample each channel independently.
        """
        inner = self._inner.additive_noise(
            noise_type, spatial_mode, ranges=ranges, mean_range=mean_range,
            scale_range=scale_range, alpha_range=alpha_range, beta_range=beta_range,
            patch_count_range=patch_count_range, patch_height_range=patch_height_range,
            patch_width_range=patch_width_range, per_channel=per_channel,
        )
        return Transform(inner)

    def multiplicative_noise(
        self,
        multiplier: tuple[float, float] = (0.9, 1.1),
        *,
        per_channel: bool = False,
        elementwise: bool = False,
    ) -> Transform:
        """Multiply uint8 or normalized float32 HWC by seeded uniform gains.

        Gain bounds are finite and nonnegative. Set per_channel for separate
        gains by channel and elementwise for separate gains by pixel. The
        default samples one gain per image. Output is clipped.
        """
        inner = self._inner.multiplicative_noise(
            multiplier, per_channel=per_channel, elementwise=elementwise
        )
        return Transform(inner)

    def random_sun_flare(
        self,
        flare_roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.5),
        angle_range: tuple[float, float] = (0.0, 1.0),
        num_flare_circles_range: tuple[int, int] = (6, 10),
        src_radius: int = 400,
        src_color: tuple[int, int, int] = (255, 255, 255),
        physics_based: bool = False,
    ) -> Transform:
        """Add a seeded optical flare to uint8 or normalized float32 HWC RGB.

        Angles are fractions of a turn, radius is in pixels, and color is RGB.
        Set physics_based for diffraction rays and screen blending.
        """
        inner = self._inner.random_sun_flare(flare_roi, angle_range, num_flare_circles_range, src_radius, src_color, physics_based)
        return Transform(inner)

    def lens_flare(
        self,
        flare_roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.5),
        num_ghosts_range: tuple[int, int] = (3, 7),
        intensity_range: tuple[float, float] = (0.3, 0.7),
        num_rays_range: tuple[int, int] = (4, 8),
        bloom_range: tuple[float, float] = (0.01, 0.05),
    ) -> Transform:
        """Add a seeded optical flare to uint8 or normalized float32 HWC RGB.

        Source-position coordinates and bloom radius are normalized.
        Ghosts and starburst rays use screen blending.
        """
        inner = self._inner.lens_flare(flare_roi, num_ghosts_range, intensity_range, num_rays_range, bloom_range)
        return Transform(inner)

    def random_brightness_contrast(
        self,
        brightness_range: tuple[float, float] = (-0.2, 0.2),
        contrast_range: tuple[float, float] = (-0.2, 0.2),
        *,
        brightness_by_max: bool = False,
        ensure_safe_output: bool = False,
    ) -> Transform:
        """Sample brightness and contrast once per uint8 or float32 HWC image.

        By default apply mean-centered contrast followed by brightness
        multiplication, each using 1 + adjustment. With brightness_by_max,
        apply contrast about zero then add brightness in normalized units.
        Safe output projects the combined gain and offset into the range
        that maps [0, 1] into [0, 1].
        """
        inner = self._inner.random_brightness_contrast(
            brightness_range, contrast_range, brightness_by_max=brightness_by_max,
            ensure_safe_output=ensure_safe_output,
        )
        return Transform(inner)

    def hue_saturation_value(
        self,
        hue_shift_range: tuple[float, float] = (-20, 20),
        sat_shift_range: tuple[float, float] = (-30, 30),
        val_shift_range: tuple[float, float] = (-20, 20),
    ) -> Transform:
        """Sample HSV offsets per uint8 or normalized float32 HWC image.

        Hue wraps in degrees. Saturation and value use 8-bit units and
        are clipped after adjustment. Grayscale uses the value offset alone.
        """
        inner = self._inner.hue_saturation_value(
            hue_shift_range, sat_shift_range, val_shift_range
        )
        return Transform(inner)

    def halftone(
        self,
        dot_size_range: tuple[int, int] = (4, 10),
        blend_range: tuple[float, float] = (0.0, 0.5),
    ) -> Transform:
        """Render colored dots on black for uint8 or normalized float32 HWC.

        Sample the cell size in pixels and the original-image blend per
        image. Mean cell luma controls radius. Edge cells use their actual
        dimensions. A blend of one keeps the original.
        """
        return Transform(self._inner.halftone(dot_size_range, blend_range))

    def dithering(
        self,
        method: str = "error_diffusion",
        n_colors: int = 2,
        color_mode: str = "grayscale",
        error_diffusion_algorithm: str = "floyd_steinberg",
        bayer_matrix_size: int = 4,
        serpentine: bool = False,
        noise_range: tuple[float, float] = (-0.5, 0.5),
    ) -> Transform:
        """Quantize uint8 or normalized float32 HWC to 2–256 levels per channel.

        Ordered uses a Bayer threshold, random adds seeded noise in palette
        step units, and error_diffusion propagates quantization residuals.
        Grayscale repeats dithered luma across the input channels.
        """
        inner = self._inner.dithering(
            method, n_colors, color_mode, error_diffusion_algorithm,
            bayer_matrix_size, serpentine, noise_range,
        )
        return Transform(inner)

    def ringing_overshoot(
        self,
        blur_range: tuple[int, int] = (7, 15),
        cutoff_range: tuple[float, float] = (math.pi / 4, math.pi / 2),
    ) -> Transform:
        """Create edge ringing with a radial sinc filter on uint8 or float32 HWC.

        Sample an odd kernel size and a cutoff in radians per pixel from
        (0, pi). Use reflected borders and clip the final output.
        """
        return Transform(self._inner.ringing_overshoot(blur_range, cutoff_range))

    def enhance(
        self, mode: str = "edge", alpha_range: tuple[float, float] = (0.5, 1.0)
    ) -> Transform:
        """Blend a local edge or detail preset into uint8 or float32 HWC.

        Alpha is sampled once per image. Zero is identity, one applies
        the full preset, and larger values strengthen the effect.
        Borders reflect without repeating the edge. Output is clipped.
        """
        return Transform(self._inner.enhance(mode, alpha_range))

    def emboss(
        self,
        alpha_range: tuple[float, float] = (0.2, 0.5),
        strength_range: tuple[float, float] = (0.2, 0.7),
    ) -> Transform:
        """Add directional relief to uint8 or normalized float32 HWC.

        Sample alpha and strength once per image. Alpha blends the relief
        with the original; strength scales the diagonal differences.
        The filter uses reflected borders and clips its final output.
        """
        return Transform(self._inner.emboss(alpha_range, strength_range))

    def exposure_matching(
        self,
        target_mean_range: tuple[float, float] = (0.3, 0.5),
        gain_range: tuple[float, float] | None = None,
    ) -> Transform:
        """Match each uint8 or normalized float32 HWC image to a sampled mean.

        The gain is the target divided by the mean of all image channels
        and pixels. Optional gain bounds limit the adjustment. Black images
        stay black, and clipping can prevent reaching the target exactly.
        """
        inner = self._inner.exposure_matching(target_mean_range, gain_range)
        return Transform(inner)

    def fancy_pca(self, alpha: float = 0.1) -> Transform:
        """Add seeded covariance-weighted color offsets to uint8 or float32 HWC.

        Compute PCA in normalized intensity units. Each principal axis
        contributes its variance times a Gaussian sample of standard
        deviation alpha. Grayscale uses its scalar variance.
        """
        return Transform(self._inner.fancy_pca(alpha))

    def colorize(
        self,
        black_range: tuple[tuple[int, int, int], tuple[int, int, int]] = ((0, 0, 0), (0, 0, 0)),
        white_range: tuple[tuple[int, int, int], tuple[int, int, int]] = ((255, 255, 255), (255, 255, 255)),
        mid_range: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
        mid_value_range: tuple[int, int] = (127, 127),
    ) -> Transform:
        """Map one-channel uint8 or normalized float32 HWC to three-channel RGB.

        RGB anchors and the optional midpoint use inclusive 8-bit ranges.
        Each anchor is sampled per image, then interpolated by intensity.
        """
        inner = self._inner.colorize(black_range, white_range, mid_range, mid_value_range)
        return Transform(inner)

    def to_sepia(self, alpha: float = 1.0) -> Transform:
        """Apply a sepia matrix to uint8 or normalized float32 HWC RGB.

        Blend the transformed RGB with the original using alpha in [0, 1],
        then clip. Grayscale is unchanged.
        """
        return Transform(self._inner.to_sepia(alpha))

    def rgb_shift(
        self,
        r_shift_range: tuple[float, float] = (-20, 20),
        g_shift_range: tuple[float, float] = (-20, 20),
        b_shift_range: tuple[float, float] = (-20, 20),
    ) -> Transform:
        """Sample one offset per RGB channel for uint8 or float32 HWC RGB.

        Ranges use 8-bit intensity units in [-255, 255] for both dtypes.
        Float32 offsets are divided by 255, then output is clipped.
        """
        inner = self._inner.rgb_shift(r_shift_range, g_shift_range, b_shift_range)
        return Transform(inner)

    def random_gamma(self, gamma_range: tuple[float, float] = (80, 120)) -> Transform:
        """Power-law correction for uint8 or normalized float32 HWC.

        Sample once per image and divide by 100 for the exponent. A range
        of (100, 100) is identity.
        """
        return Transform(self._inner.random_gamma(gamma_range))

    def random_fog(
        self, fog_coef_range: tuple[float, float] = (0.3, 1.0), alpha_coef: float = 0.08
    ) -> Transform:
        """Blend seeded soft white fog patches into uint8 or float32 HWC RGB.

        Both coefficients are in [0, 1]. Zero intensity or opacity is identity.
        """
        return Transform(self._inner.random_fog(fog_coef_range, alpha_coef))

    def atmospheric_fog(
        self,
        density_range: tuple[float, float] = (1.0, 3.0),
        fog_color: tuple[int, int, int] = (200, 200, 200),
        depth_mode: str = "linear",
    ) -> Transform:
        """Blend fog using exponential attenuation on uint8 or float32 HWC RGB.

        Density is finite and nonnegative. Synthetic depth increases toward
        the top (linear), top-left (diagonal), or corners (radial).
        """
        return Transform(self._inner.atmospheric_fog(density_range, fog_color, depth_mode))

    def spatter(
        self,
        mean_range: tuple[float, float] = (0.65, 0.65),
        std_range: tuple[float, float] = (0.3, 0.3),
        gauss_sigma_range: tuple[float, float] = (2.0, 3.0),
        cutout_threshold_range: tuple[float, float] = (0.68, 0.68),
        intensity_range: tuple[float, float] = (0.6, 0.6),
        mode: str = "rain",
        color: tuple[int, int, int] | None = None,
    ) -> Transform:
        """Add seeded rain highlights or mud occlusion to uint8 or float32 HWC RGB.

        Threshold controls drop coverage, sigma controls edge softness, and
        intensity scales the overlay. Color, when supplied, is uint8 RGB.
        """
        inner = self._inner.spatter(mean_range, std_range, gauss_sigma_range, cutout_threshold_range, intensity_range, mode, color)
        return Transform(inner)

    def random_gravel(
        self,
        gravel_roi: tuple[float, float, float, float] = (0.1, 0.4, 0.9, 0.9),
        number_of_patches: int = 2,
    ) -> Transform:
        """Scatter seeded clusters of gravel on uint8 or float32 HWC RGB.

        The ROI is (left, top, right, bottom) in normalized coordinates.
        Zero patches leaves the image unchanged.
        """
        return Transform(self._inner.random_gravel(gravel_roi, number_of_patches))

    def random_snow(
        self,
        snow_point_range: tuple[float, float] = (0.1, 0.3),
        brightness_coeff: float = 2.5,
        method: str = "bleach",
    ) -> Transform:
        """Simulate snow on uint8 or normalized float32 HWC RGB.

        Bleach scales HSL lightness above the sampled threshold.
        Texture adds a seeded, cool-toned snow layer with more coverage
        at the top of the image and sparse sparkles. A zero snow point
        leaves the image unchanged.
        """
        inner = self._inner.random_snow(snow_point_range, brightness_coeff, method)
        return Transform(inner)

    def random_rain(
        self,
        slant_range: tuple[float, float] = (-10, 10),
        drop_length: int | None = None,
        drop_width: int = 1,
        drop_color: tuple[int, int, int] = (200, 200, 200),
        blur_value: int = 7,
        brightness_coefficient: float = 0.7,
        rain_type: str = "default",
    ) -> Transform:
        """Add seeded semi-transparent streaks to uint8 or float32 HWC RGB.

        Slant is measured from vertical in degrees. Length and width are in
        pixels. Use blur_value=1 to skip box smoothing. Rain type controls
        density and default streak length.
        """
        inner = self._inner.random_rain(
            slant_range, drop_length, drop_width, drop_color, blur_value,
            brightness_coefficient, rain_type,
        )
        return Transform(inner)

    def random_shadow(
        self,
        shadow_roi: tuple[float, float, float, float] = (0, 0.5, 1, 1),
        num_shadows_range: tuple[int, int] = (1, 2),
        shadow_dimension: int = 5,
        shadow_intensity_range: tuple[float, float] = (0.5, 0.5),
    ) -> Transform:
        """Darken random polygons in a normalized ROI on uint8 or float32 HWC.

        The ROI is (left, top, right, bottom). A polygon transmits one minus
        its sampled intensity. Transmittance multiplies where shadows overlap.
        """
        inner = self._inner.random_shadow(
            shadow_roi, num_shadows_range, shadow_dimension, shadow_intensity_range
        )
        return Transform(inner)

    def plasma_shadow(
        self, shadow_intensity_range: tuple[float, float] = (0.3, 0.7),
        plasma_size: int = 256, roughness: float = 3.0,
    ) -> Transform:
        """Darken pixels proportionally to a seeded plasma field.

        Accepts uint8 or normalized float32 HWC. Plasma size sets the field
        resolution (2–1024), and larger roughness preserves more fine detail.
        """
        return Transform(self._inner.plasma_shadow(shadow_intensity_range, plasma_size, roughness))

    def plasma_brightness_contrast(
        self, brightness_range: tuple[float, float] = (-0.3, 0.3), contrast_range: tuple[float, float] = (-0.3, 0.3),
        plasma_size: int = 256, roughness: float = 3.0,
    ) -> Transform:
        """Vary brightness and contrast about midgray using a seeded plasma field.

        Accepts uint8 or normalized float32 HWC. Plasma size sets the field
        resolution (2–1024), and larger roughness preserves more fine detail.
        """
        return Transform(self._inner.plasma_brightness_contrast(brightness_range, contrast_range, plasma_size, roughness))

    def illumination(
        self, mode: str = "linear", intensity_range: tuple[float, float] = (0.01, 0.2),
        effect_type: str = "both",
    ) -> Transform:
        """Apply a seeded smooth gain to uint8 or normalized float32 HWC.

        Linear samples a direction, corner samples a corner, and gaussian
        samples a spotlight center and spread. Intensity in [0, 1] bounds
        the peak gain change. Effect selects brightening, darkening, or
        a random choice of the two.
        """
        return Transform(self._inner.illumination(mode, intensity_range, effect_type))

    def film_grain(
        self,
        intensity_range: tuple[float, float] = (0.1, 0.3),
        grain_size_range: tuple[int, int] = (1, 3),
        *,
        per_channel: bool = False,
    ) -> Transform:
        """Add seeded Gaussian grain to uint8 or normalized float32 HWC.

        Grain size divides the field resolution before bilinear upsampling.
        Grain amplitude is proportional to inverse luma. Set per_channel to
        sample independent fields for each channel. Output is clipped.
        """
        inner = self._inner.film_grain(intensity_range, grain_size_range, per_channel=per_channel)
        return Transform(inner)

    def iso_noise(
        self,
        color_shift_range: tuple[float, float] = (0.01, 0.05),
        intensity_range: tuple[float, float] = (0.1, 0.5),
    ) -> Transform:
        """Add seeded luma and chroma noise to uint8 or normalized float32 HWC RGB.

        Both ranges are in [0, 1]. Intensity scales both components, while
        color shift controls chroma rotation in fractions of a turn. Luma
        noise grows with the square root of luma. Output is clipped.
        """
        return Transform(self._inner.iso_noise(color_shift_range, intensity_range))

    def glass_blur(
        self, sigma: float = 0.7, max_delta: int = 4, iterations: int = 2
    ) -> Transform:
        """Smooth, exchange nearby pixels, then smooth uint8 or float32 HWC again.

        Each pass exchanges whole pixels with partners within max_delta on
        each axis, clipped to the image. Sigma is in [0, 10]; zero skips
        smoothing. Use a positive max_delta and 1–32 passes.
        """
        return Transform(self._inner.glass_blur(sigma, max_delta, iterations))

    def zoom_blur(
        self,
        max_factor_range: tuple[float, float] = (1.0, 1.31),
        step_factor_range: tuple[float, float] = (0.01, 0.03),
    ) -> Transform:
        """Average uint8 or float32 HWC with bilinear zooms about the pixel center.

        Sample a maximum factor and a positive step per image, then average
        the original with factors 1 + n * step up to the maximum. The ranges
        may allow at most 128 zoomed samples per image.
        """
        return Transform(self._inner.zoom_blur(max_factor_range, step_factor_range))

    def stochastic_convolution(
        self,
        kernel_range: tuple[int, int] = (3, 7),
        strength_range: tuple[float, float] = (0.0, 1.0),
        *,
        per_channel: bool = False,
        padding_mode: str = "reflect",
    ) -> Transform:
        """Apply an identity kernel plus Gaussian tap noise to uint8 or float32 HWC.

        The sampled strength divided by the kernel width is the noise standard
        deviation. Kernels are shared across channels by default. Padding modes
        are constant (zero), edge, reflect, and symmetric. Float32 inputs use
        the normalized [0, 1] range. Zero strength gives exact identity.
        """
        inner = self._inner.stochastic_convolution(
            kernel_range, strength_range, per_channel=per_channel, padding_mode=padding_mode
        )
        return Transform(inner)

    def advanced_blur(
        self,
        kernel_range: tuple[int, int] = (3, 7),
        sigma_x_range: tuple[float, float] = (0.2, 1.0),
        sigma_y_range: tuple[float, float] = (0.2, 1.0),
        rotate_range: tuple[float, float] = (-90.0, 90.0),
        beta_range: tuple[float, float] = (0.5, 8.0),
        noise_range: tuple[float, float] = (0.75, 1.25),
    ) -> Transform:
        """Apply a seeded generalized Gaussian filter to uint8 or float32 HWC.

        Sigma controls the spread along each axis. Rotation is in degrees.
        Beta controls the falloff shape. Positive noise factors perturb each
        tap before normalization. Borders reflect without repeating the edge.
        """
        inner = self._inner.advanced_blur(
            kernel_range, sigma_x_range, sigma_y_range, rotate_range, beta_range, noise_range
        )
        return Transform(inner)

    def blur(self, kernel_range: tuple[int, int] = (3, 7)) -> Transform:
        """Average uint8 or float32 HWC pixels using a seeded square box filter.

        Sample an odd size from the inclusive range (odd endpoints in 3–127).
        Reflect neighbors at the border without repeating the edge pixel.
        """
        return Transform(self._inner.blur(kernel_range))

    def median_blur(self) -> Transform:
        """Apply a 3×3 median per channel to uint8 HWC grayscale or RGB.

        Out-of-bounds neighbors use the nearest border pixel.
        """
        return Transform(self._inner.median_blur())

    def planckian_jitter(
        self, temperature_range: tuple[float, float] = (3000, 15000)
    ) -> Transform:
        """Sample a blackbody temperature uniformly in kelvin, from 3000 to 15000.

        Apply linear-sRGB gains relative to a neutral 6500 K blackbody,
        keeping green fixed. Inputs are uint8 or normalized float32 sRGB
        in HWC order. Output is clipped to the input's nominal range.
        """
        return Transform(self._inner.planckian_jitter(temperature_range))

    def vignetting(
        self,
        intensity_range: tuple[float, float] = (0.2, 0.5),
        center_range: tuple[float, float] = (0.3, 0.7),
    ) -> Transform:
        """Darken uint8 or float32 HWC images using seeded radial falloff.

        Each center coordinate is sampled independently as a fraction of its
        pixel axis. Intensity is the fractional loss at the farthest corner.
        """
        return Transform(self._inner.vignetting(intensity_range, center_range))

    def random_tone_curve(self, scale: float = 0.1) -> Transform:
        """Apply one seeded monotone tone curve to all uint8 HWC RGB channels."""
        return Transform(self._inner.random_tone_curve(scale))

    def clahe(
        self, clip_limit: float = 2.0, *, grid: tuple[int, int] = (8, 8)
    ) -> Transform:
        """Equalize local luminance in uint8 HWC grayscale or RGB images.

        Grid is (columns, rows). Clip limit scales the average histogram
        bin size, and neighboring tile mappings are blended bilinearly.
        """
        return Transform(self._inner.clahe(clip_limit, grid=grid))

    def chromatic_aberration(
        self,
        red_range: tuple[float, float] = (0.0, 0.05),
        blue_range: tuple[float, float] = (-0.05, 0.0),
        *,
        interpolation: str = "bilinear",
    ) -> Transform:
        """Radially resample red and blue in uint8 HWC RGB, with edge extension.

        Sample one coefficient from each range per image. Positive coefficients
        sample farther from the center. Green and annotation coordinates are unchanged.
        """
        return Transform(
            self._inner.chromatic_aberration(
                red_range, blue_range, interpolation=interpolation
            )
        )

    def defocus(self, radius_range: tuple[int, int] = (1, 3)) -> Transform:
        """Blur uint8 HWC RGB with a uniform disk and reflected borders.

        Sample one integer radius from the inclusive bounds for each image.
        Radius zero is an identity.
        """
        return Transform(self._inner.defocus(radius_range))

    def channel_dropout(
        self,
        channels_to_drop: Iterable[int] | None = None,
        dropout_ratio: float = 0.5,
        fill: float | str = 0,
    ) -> Transform:
        """Replace a seeded subset of eligible channels in uint8 or float32 images.

        The count is rounded half-up and capped to retain at least one
        channel. Fill is a constant in input intensity units or "mean" for
        each selected channel's spatial mean. Preserves HWC and CHW.
        """
        eligible = None if channels_to_drop is None else list(channels_to_drop)
        return Transform(self._inner.channel_dropout(eligible, dropout_ratio, fill))

    def xy_masking(
        self,
        num_masks_x_range: tuple[int, int] = (0, 0),
        num_masks_y_range: tuple[int, int] = (0, 0),
        mask_x_length_range: tuple[int, int] | tuple[float, float] = (0, 0),
        mask_y_length_range: tuple[int, int] | tuple[float, float] = (0, 0),
        *,
        fill: float = 0,
        fill_mask: int | None = None,
    ) -> Transform:
        """Fill seeded vertical (X) and horizontal (Y) strips.

        Integer lengths are pixels; fractional lengths scale by the
        corresponding axis. Strips may overlap. Structured masks can
        share the fill region, and fully occluded instances are removed.
        Preserves uint8 or float32 and HWC or CHW layout.
        """
        inner = self._inner.xy_masking(
            num_masks_x_range, num_masks_y_range, mask_x_length_range,
            mask_y_length_range, fill=fill, fill_mask=fill_mask,
        )
        return Transform(inner)

    def pixel_dropout(
        self,
        dropout_prob: float = 0.01,
        *,
        per_channel: bool = False,
        drop_value: float | Iterable[float] | None = 0,
        mask_drop_value: int | None = None,
    ) -> Transform:
        """Replace sampled pixels or independent channel elements.

        Fill is a scalar, one value per channel, or None for a seeded
        random fill per channel. Structured mask fill uses the shared
        spatial decisions and requires per_channel=False. Fully occluded
        boxes and associated instance annotations are removed.
        """
        fill = None if drop_value is None else _normalize_fill(drop_value)
        inner = self._inner.pixel_dropout(
            dropout_prob, per_channel=per_channel, drop_value=fill, mask_drop_value=mask_drop_value
        )
        return Transform(inner)

    def coarse_dropout(
        self,
        holes: tuple[int, int] = (1, 4),
        *,
        hole_height: tuple[int, int] = (4, 8),
        hole_width: tuple[int, int] = (4, 8),
        fill: float = 0.0,
    ) -> Transform:
        """Fill random rectangles using inclusive count and pixel size ranges.

        Rectangles can overlap and are clipped in size to fit the image.
        Only image pixels are modified.
        """
        return Transform(
            self._inner.coarse_dropout(
                holes, hole_height=hole_height, hole_width=hole_width, fill=fill
            )
        )

    def motion_blur(
        self,
        length_range: tuple[int, int] = (3, 7),
        *,
        angle_range: tuple[float, float] = (0.0, 180.0),
    ) -> Transform:
        """Blur uint8 HWC RGB along a line with reflected borders.

        Sample an odd length from the inclusive bounds and an angle in
        degrees from the horizontal axis, once per image.
        """
        return Transform(self._inner.motion_blur(length_range, angle_range=angle_range))

    def unsharp_mask(
        self, sigma: float = 1.0, *, amount: float = 1.0, threshold: int = 0
    ) -> Transform:
        """Amplify Gaussian residuals on uint8 HWC RGB above a pixel threshold."""
        return Transform(
            self._inner.unsharp_mask(sigma, amount=amount, threshold=threshold)
        )

    def downscale(
        self,
        scale_range: tuple[float, float] = (0.25, 0.5),
        *,
        down_interpolation: str = "nearest",
        up_interpolation: str = "nearest",
    ) -> Transform:
        """Reduce and restore the resolution of a uint8 HWC RGB image.

        A single scale is sampled per image from the given range in (0, 1].
        The output keeps the input dimensions.
        """
        return Transform(
            self._inner.downscale(
                scale_range,
                down_interpolation=down_interpolation,
                up_interpolation=up_interpolation,
            )
        )

    def salt_and_pepper(
        self, amount: float = 0.05, *, salt_ratio: float = 0.5
    ) -> Transform:
        """Replace each pixel with probability ``amount``, across all channels.

        Of the replaced pixels, ``salt_ratio`` is the probability of white
        (255 for uint8 or 1.0 for float32). The other replacements are black.
        """
        return Transform(self._inner.salt_and_pepper(amount, salt_ratio=salt_ratio))

    def grid_dropout(
        self, unit_size: int, *, ratio: float = 0.5, fill: float = 0.0
    ) -> Transform:
        """Fill one square per grid cell at a shared random offset.

        The hole side is floor(unit_size * ratio), clamped to [1, unit_size - 1].
        The grid starts at the image origin and partial cells are clipped.
        Only image pixels are modified.
        """
        return Transform(self._inner.grid_dropout(unit_size, ratio=ratio, fill=fill))

    def shot_noise(self, intensity: float = 0.2) -> Transform:
        """Add Gaussian noise scaled by intensity * sqrt(max(pixel, 0) + 0.5).

        Call after scaling pixels to float32 [0, 1]. Output is clipped to [0, 1].
        """
        return Transform(self._inner.shot_noise(intensity))

    def jpeg(self, quality: int | Iterable[int]) -> Transform:
        """JPEG encode/decode uint8 gray/RGB pixels; quality bounds are inclusive.

        Uses libjpeg-compatible encoding with 4:2:0 chroma subsampling for RGB;
        grayscale uses one component. Quality is sampled once per image.
        """
        qualities = _normalize_jpeg_quality(quality)
        return Transform(self._inner.jpeg(qualities))

    def rgb(self) -> Transform:
        """Convert decoded grayscale/gray-alpha/RGB/RGBA pixels to RGB.

        Alpha is discarded. Image decoding already produces RGB by default.
        """
        return Transform(self._inner.rgb())

    def to_dtype(self, dtype: str, scale: bool = False) -> Transform:
        """Convert uint8/float32, scaling between [0,255] and [0,1] if requested."""
        return Transform(self._inner.to_dtype(dtype, scale))

    def convert_image_dtype(self, dtype: str) -> Transform:
        return Transform(self._inner.convert_image_dtype(dtype))

    def rotate(self, angle: int) -> Transform:
        return Transform(self._inner.rotate(angle))

    def arbitrary_rotate(
        self,
        angle: float,
        interpolation: str = "nearest",
        expand: bool = False,
        fill: int | Iterable[int] = 0,
        center: Iterable[float] | None = None,
    ) -> Transform:
        return Transform(
            self._inner.arbitrary_rotate(
                angle,
                interpolation,
                expand,
                _normalize_fill(fill),
                None if center is None else list(center),
            )
        )

    def random_affine(
        self,
        degrees: float | Iterable[float],
        translate: Iterable[float] | None = None,
        scale: Iterable[float] | None = None,
        shear: float | Iterable[float] | None = None,
        interpolation: str = "nearest",
        fill: int | Iterable[int] = 0,
        shear_ranges: Iterable[float] | None = None,
    ) -> Transform:
        """Sample an affine transform using Rivet's per-image random stream.

        Degrees may be a scalar maximum or a two-bound interval. Scalar shear
        affects x; a shear pair retains legacy symmetric x/y maxima. Use
        ``shear_ranges`` for two x bounds or four x/y bounds instead."""
        if shear is not None and shear_ranges is not None:
            raise ValueError("pass either shear or shear_ranges, not both")
        if isinstance(shear, (int, float)) and shear < 0:
            raise ValueError("shear must be non-negative")
        return Transform(
            self._inner.random_affine(
                degrees,
                None if translate is None else list(translate),
                None if scale is None else list(scale),
                None
                if shear is None
                else (
                    [float(shear), 0.0]
                    if isinstance(shear, (int, float))
                    else list(shear)
                ),
                interpolation,
                _normalize_fill(fill),
                None if shear_ranges is None else list(shear_ranges),
            )
        )

    def perspective(
        self,
        start_points: Iterable[Iterable[float]],
        end_points: Iterable[Iterable[float]],
        interpolation: str = "bilinear",
        fill: int | Iterable[int] = 0,
    ) -> Transform:
        return Transform(
            self._inner.perspective(
                [list(point) for point in start_points],
                [list(point) for point in end_points],
                interpolation,
                _normalize_fill(fill),
            )
        )

    def random_perspective(
        self,
        distortion_scale: float = 0.5,
        probability: float = 0.5,
        interpolation: str = "bilinear",
        fill: int | Iterable[int] = 0,
    ) -> Transform:
        return Transform(
            self._inner.random_perspective(
                distortion_scale, probability, interpolation, _normalize_fill(fill)
            )
        )

    def elastic_transform(
        self,
        alpha: float,
        sigma: float,
        interpolation: str = "bilinear",
        fill: int | Iterable[int] = 0,
    ) -> Transform:
        return Transform(
            self._inner.elastic_transform(
                alpha, sigma, interpolation, _normalize_fill(fill)
            )
        )

    def normalize(self, mean: list[float], std: list[float]) -> Transform:
        return Transform(self._inner.normalize(mean, std))

    def hwc_to_chw(self) -> Transform:
        return Transform(self._inner.hwc_to_chw())

    def chw_to_hwc(self) -> Transform:
        return Transform(self._inner.chw_to_hwc())

    def random_apply(self, probability: float, nested: Transform) -> Transform:
        return Transform(self._inner.random_apply(probability, nested._inner))

    def random_choice(
        self, choices: Iterable[Transform], p: Iterable[float] | None = None
    ) -> Transform:
        """Choose one branch per image; ``p`` contains relative nonnegative weights.

        Omit ``p`` for the existing uniform sampler. Provided weights must have
        a finite positive sum and are normalized before sampling.
        """
        return Transform(
            self._inner.random_choice(
                [choice._inner for choice in choices], None if p is None else list(p)
            )
        )

    def random_order(self, nested: Transform) -> Transform:
        return Transform(self._inner.random_order(nested._inner))


Compose = Transform
