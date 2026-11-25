"""Reusable image transform composition."""

from __future__ import annotations

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
