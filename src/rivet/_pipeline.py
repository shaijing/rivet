"""Lazy image pipeline construction and execution."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from ._rivet import _ImagePipeline
from ._transforms import Transform
from ._utils import _normalize_fill, _normalize_jpeg_quality

if TYPE_CHECKING:
    from ._loader import DataLoader


class Pipeline:
    """Lazy image input pipeline."""

    def __init__(self, inner: _ImagePipeline, *, as_numpy: bool = True) -> None:
        self._inner = inner
        self.as_numpy = as_numpy

    def decode_image(self) -> Pipeline:
        return Pipeline(self._inner.decode_image(), as_numpy=self.as_numpy)

    def resize(
        self,
        width: int,
        height: int | None = None,
        interpolation: str = "bilinear",
        *,
        max_size: int | None = None,
    ) -> Pipeline:
        """Resize to explicit width/height, or set the short edge to ``width``.

        With no height, preserve aspect ratio; ``max_size`` optionally caps
        the long edge and must exceed the requested short edge."""
        return Pipeline(
            self._inner.resize(width, height, interpolation, max_size),
            as_numpy=self.as_numpy,
        )

    def crop(self, x: int, y: int, width: int, height: int) -> Pipeline:
        return Pipeline(self._inner.crop(x, y, width, height), as_numpy=self.as_numpy)

    def five_crop(self, width: int, height: int) -> Pipeline:
        """Expand each image into top-left, top-right, bottom-left, bottom-right,
        and center crops. ``batch(B)`` yields ``B * 5`` views and repeated labels.
        """
        return Pipeline(self._inner.five_crop(width, height), as_numpy=self.as_numpy)

    def ten_crop(
        self, width: int, height: int, *, vertical_flip: bool = False
    ) -> Pipeline:
        """Emit five original crops followed by five crops of the flipped image.

        The flip is horizontal by default. ``batch(B)`` yields ``B * 10`` views.
        """
        return Pipeline(
            self._inner.ten_crop(width, height, vertical_flip), as_numpy=self.as_numpy
        )

    def center_crop(self, width: int, height: int) -> Pipeline:
        return Pipeline(self._inner.center_crop(width, height), as_numpy=self.as_numpy)

    def pad(
        self,
        padding: int,
        fill: float | Iterable[float] = 0.0,
        padding_mode: str = "constant",
    ) -> Pipeline:
        return Pipeline(
            self._inner.pad(padding, _normalize_fill(fill), padding_mode),
            as_numpy=self.as_numpy,
        )

    def pad_with_sides(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
        fill: float | Iterable[float] = 0.0,
        padding_mode: str = "constant",
    ) -> Pipeline:
        return Pipeline(
            self._inner.pad_with_sides(
                left, top, right, bottom, _normalize_fill(fill), padding_mode
            ),
            as_numpy=self.as_numpy,
        )

    def auto_augment(
        self, policy: str = "imagenet", interpolation: str = "nearest", fill: int = 0
    ) -> Pipeline:
        return Pipeline(
            self._inner.auto_augment(policy, interpolation, fill),
            as_numpy=self.as_numpy,
        )

    def rand_augment(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: str = "nearest",
        fill: int = 0,
    ) -> Pipeline:
        return Pipeline(
            self._inner.rand_augment(
                num_ops, magnitude, num_magnitude_bins, interpolation, fill
            ),
            as_numpy=self.as_numpy,
        )

    def trivial_augment_wide(
        self,
        num_magnitude_bins: int = 31,
        interpolation: str = "nearest",
        fill: int = 0,
    ) -> Pipeline:
        return Pipeline(
            self._inner.trivial_augment_wide(num_magnitude_bins, interpolation, fill),
            as_numpy=self.as_numpy,
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
    ) -> Pipeline:
        return Pipeline(
            self._inner.augmix(
                severity,
                mixture_width,
                chain_depth,
                alpha,
                all_ops,
                interpolation,
                fill,
            ),
            as_numpy=self.as_numpy,
        )

    def to_tensor(self) -> Pipeline:
        """Convert decoded HWC pixels to float32 CHW, scaling uint8 by 255."""
        return Pipeline(self._inner.to_tensor(), as_numpy=self.as_numpy)

    def pil_to_tensor(self) -> Pipeline:
        """Convert decoded HWC pixels to CHW without changing dtype/range."""
        return Pipeline(self._inner.pil_to_tensor(), as_numpy=self.as_numpy)

    def linear_transform(
        self, matrix: Iterable[Iterable[float]], mean: Iterable[float]
    ) -> Pipeline:
        return Pipeline(
            self._inner.linear_transform([list(row) for row in matrix], list(mean)),
            as_numpy=self.as_numpy,
        )

    def saturation(self, factor: float) -> Pipeline:
        return Pipeline(self._inner.saturation(factor), as_numpy=self.as_numpy)

    def gamma(self, gamma: float, gain: float = 1.0) -> Pipeline:
        return Pipeline(self._inner.gamma(gamma, gain), as_numpy=self.as_numpy)

    def affine(
        self,
        angle: float,
        translate: Iterable[float] | None = None,
        scale: float = 1.0,
        shear: float | Iterable[float] | None = None,
        interpolation: str = "nearest",
        fill: int | Iterable[int] = 0,
        center: Iterable[float] | None = None,
    ) -> Pipeline:
        return Pipeline(
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
            ),
            as_numpy=self.as_numpy,
        )

    def erase(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        value: float | Iterable[float] = 0.0,
    ) -> Pipeline:
        values = [float(value)] if isinstance(value, (int, float)) else list(value)
        return Pipeline(
            self._inner.erase(x, y, width, height, values), as_numpy=self.as_numpy
        )

    def resized_crop(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        out_width: int,
        out_height: int,
        interpolation: str = "bilinear",
    ) -> Pipeline:
        return Pipeline(
            self._inner.resized_crop(
                x, y, width, height, out_width, out_height, interpolation
            ),
            as_numpy=self.as_numpy,
        )

    def random_rotation(
        self,
        degrees: float | Iterable[float],
        interpolation: str = "nearest",
        expand: bool = False,
        fill: int | Iterable[int] = 0,
        center: Iterable[float] | None = None,
    ) -> Pipeline:
        if isinstance(degrees, (int, float)):
            if degrees < 0:
                raise ValueError("degrees must be non-negative")
            degrees = [-float(degrees), float(degrees)]
        return Pipeline(
            self._inner.random_rotation(
                list(degrees),
                interpolation,
                expand,
                _normalize_fill(fill),
                None if center is None else list(center),
            ),
            as_numpy=self.as_numpy,
        )

    def random_gaussian_blur(
        self,
        kernel_size: int | Iterable[int],
        sigma: float | Iterable[float] | None = None,
    ) -> Pipeline:
        """Blur with a scalar or (width, height) kernel.

        Sigma is a fixed scalar or [min, max] interval. One sampled sigma
        is shared by both axes; the pair does not specify axis sigmas."""
        return Pipeline(
            self._inner.random_gaussian_blur(
                kernel_size if isinstance(kernel_size, int) else list(kernel_size),
                sigma
                if sigma is None or isinstance(sigma, (int, float))
                else list(sigma),
            ),
            as_numpy=self.as_numpy,
        )

    def random_vertical_flip(self, probability: float = 0.5) -> Pipeline:
        return Pipeline(
            self._inner.random_vertical_flip(probability), as_numpy=self.as_numpy
        )

    def random_invert(self, probability: float = 0.5) -> Pipeline:
        return Pipeline(self._inner.random_invert(probability), as_numpy=self.as_numpy)

    def random_posterize(self, bits: int, probability: float = 0.5) -> Pipeline:
        return Pipeline(
            self._inner.random_posterize(bits, probability), as_numpy=self.as_numpy
        )

    def random_solarize(self, threshold: int, probability: float = 0.5) -> Pipeline:
        return Pipeline(
            self._inner.random_solarize(threshold, probability), as_numpy=self.as_numpy
        )

    def random_adjust_sharpness(
        self, factor: float, probability: float = 0.5
    ) -> Pipeline:
        return Pipeline(
            self._inner.random_adjust_sharpness(factor, probability),
            as_numpy=self.as_numpy,
        )

    def random_autocontrast(self, probability: float = 0.5) -> Pipeline:
        return Pipeline(
            self._inner.random_autocontrast(probability), as_numpy=self.as_numpy
        )

    def random_equalize(self, probability: float = 0.5) -> Pipeline:
        return Pipeline(
            self._inner.random_equalize(probability), as_numpy=self.as_numpy
        )

    def horizontal_flip(self) -> Pipeline:
        return Pipeline(self._inner.horizontal_flip(), as_numpy=self.as_numpy)

    def vertical_flip(self) -> Pipeline:
        return Pipeline(self._inner.vertical_flip(), as_numpy=self.as_numpy)

    def random_crop(
        self,
        width: int,
        height: int,
        padding: int | Sequence[int] = 0,
        *,
        pad_if_needed: bool = False,
        fill: float | Sequence[float] = 0,
        padding_mode: str = "constant",
    ) -> Pipeline:
        """Pad each image, then sample a ``width`` x ``height`` crop.

        Padding accepts one integer, a horizontal/vertical pair, or four
        left/top/right/bottom values. ``pad_if_needed`` adds padding after the
        explicit padding. Modes are constant, edge, reflect, and symmetric;
        constant fill accepts a scalar or RGB channel sequence.

        Stochastic ops are seeded by the pipeline seed: when the pipeline
        shuffles (``.shuffle(seed)``), the same seed reproduces the same
        sample order *and* the same augmentations at any worker count; use
        ``.epoch(epoch)`` for a fresh augmentation stream per epoch. Without
        a shuffle the augmentations are still deterministic.
        Requires decoded uint8 HWC input.
        """
        return Pipeline(
            self._inner.random_crop(
                width, height, padding, pad_if_needed, fill, padding_mode
            ),
            as_numpy=self.as_numpy,
        )

    def random_resized_crop(self, width: int, height: int) -> Pipeline:
        return Pipeline(
            self._inner.random_resized_crop(width, height), as_numpy=self.as_numpy
        )

    def random_resized_crop_with_options(
        self,
        width: int,
        height: int,
        scale: Iterable[float] | None = None,
        ratio: Iterable[float] | None = None,
        interpolation: str = "bilinear",
    ) -> Pipeline:
        return Pipeline(
            self._inner.random_resized_crop_with_options(
                width,
                height,
                None if scale is None else list(scale),
                None if ratio is None else list(ratio),
                interpolation,
            ),
            as_numpy=self.as_numpy,
        )

    def random_horizontal_flip(self, probability: float = 0.5) -> Pipeline:
        """Randomly flip each image horizontally with `probability`
        (one draw per sample from the pipeline seed, see `random_crop`).
        Requires decoded uint8 HWC input.
        """
        return Pipeline(
            self._inner.random_horizontal_flip(probability),
            as_numpy=self.as_numpy,
        )

    def brightness(self, value: float) -> Pipeline:
        """Adjust brightness by a nonnegative factor (1 is identity)."""
        return Pipeline(self._inner.brightness(value), as_numpy=self.as_numpy)

    def contrast(self, value: float) -> Pipeline:
        return Pipeline(self._inner.contrast(value), as_numpy=self.as_numpy)

    def color_jitter(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        hue: float = 0.0,
        saturation: float = 0.0,
    ) -> Pipeline:
        return Pipeline(
            self._inner.color_jitter(brightness, contrast, hue, saturation),
            as_numpy=self.as_numpy,
        )

    def compose(self, sequence: Transform) -> Pipeline:
        return Pipeline(self._inner.compose(sequence._inner), as_numpy=self.as_numpy)

    def random_apply(self, probability: float, sequence: Transform) -> Pipeline:
        return Pipeline(
            self._inner.random_apply(probability, sequence._inner),
            as_numpy=self.as_numpy,
        )

    def random_choice(
        self, choices: Iterable[Transform], p: Iterable[float] | None = None
    ) -> Pipeline:
        """Choose one branch per image; ``p`` contains relative nonnegative weights.

        Omit ``p`` for the existing uniform sampler. Provided weights must have
        a finite positive sum and are normalized before sampling.
        """
        return Pipeline(
            self._inner.random_choice(
                [choice._inner for choice in choices], None if p is None else list(p)
            ),
            as_numpy=self.as_numpy,
        )

    def random_order(self, sequence: Transform) -> Pipeline:
        return Pipeline(
            self._inner.random_order(sequence._inner),
            as_numpy=self.as_numpy,
        )

    def invert(self) -> Pipeline:
        return Pipeline(self._inner.invert(), as_numpy=self.as_numpy)

    def posterize(self, bits: int) -> Pipeline:
        return Pipeline(self._inner.posterize(bits), as_numpy=self.as_numpy)

    def solarize(self, threshold: int) -> Pipeline:
        return Pipeline(self._inner.solarize(threshold), as_numpy=self.as_numpy)

    def autocontrast(self) -> Pipeline:
        return Pipeline(self._inner.autocontrast(), as_numpy=self.as_numpy)

    def equalize(self) -> Pipeline:
        return Pipeline(self._inner.equalize(), as_numpy=self.as_numpy)

    def sharpness(self, amount: float) -> Pipeline:
        return Pipeline(self._inner.sharpness(amount), as_numpy=self.as_numpy)

    def gaussian_blur(
        self,
        sigma: float | Iterable[float],
        kernel_size: int | Iterable[int] | None = None,
    ) -> Pipeline:
        """Blur with scalar or (x, y) sigma and (width, height) kernel size.

        Omitting kernel size retains Rivet's sigma-derived kernel convenience;
        sigma zero acts as identity on that axis."""
        return Pipeline(
            self._inner.gaussian_blur(
                sigma if isinstance(sigma, (int, float)) else list(sigma),
                kernel_size
                if kernel_size is None or isinstance(kernel_size, int)
                else list(kernel_size),
            ),
            as_numpy=self.as_numpy,
        )

    def hue(self, factor: float) -> Pipeline:
        """Shift HSV hue by a factor in [-0.5, 0.5]."""
        return Pipeline(self._inner.hue(factor), as_numpy=self.as_numpy)

    def grayscale(self, num_output_channels: int = 1) -> Pipeline:
        return Pipeline(
            self._inner.grayscale(num_output_channels), as_numpy=self.as_numpy
        )

    def random_grayscale(
        self, probability: float = 0.1, num_output_channels: int = 3
    ) -> Pipeline:
        return Pipeline(
            self._inner.random_grayscale(probability, num_output_channels),
            as_numpy=self.as_numpy,
        )

    def random_erasing(
        self,
        probability: float = 0.5,
        *,
        scale: Iterable[float] | None = None,
        ratio: Iterable[float] | None = None,
        value: float | Iterable[float] | str = 0.0,
    ) -> Pipeline:
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
    ) -> Pipeline:
        return Pipeline(
            self._inner.random_erasing_with_options(
                probability,
                None if scale is None else list(scale),
                None if ratio is None else list(ratio),
                value if isinstance(value, (int, float, str)) else list(value),
            ),
            as_numpy=self.as_numpy,
        )

    def permute_channels(self, permutation: Iterable[int]) -> Pipeline:
        """Reorder channels with a permutation containing each channel once."""
        return Pipeline(
            self._inner.permute_channels(list(permutation)), as_numpy=self.as_numpy
        )

    def random_channel_permutation(self) -> Pipeline:
        """Uniformly shuffle image channels using the pipeline random stream."""
        return Pipeline(
            self._inner.random_channel_permutation(), as_numpy=self.as_numpy
        )

    def random_photometric_distort(
        self,
        brightness: Iterable[float] = (0.875, 1.125),
        contrast: Iterable[float] = (0.5, 1.5),
        saturation: Iterable[float] = (0.5, 1.5),
        hue: Iterable[float] = (-0.05, 0.05),
        probability: float = 0.5,
    ) -> Pipeline:
        """Sample brightness, contrast placement, saturation, hue and channels.

        Each color adjustment and the final channel shuffle uses its own
        probability draw. Contrast runs either before or after saturation/hue.
        """
        return Pipeline(
            self._inner.random_photometric_distort(
                list(brightness),
                list(contrast),
                list(saturation),
                list(hue),
                probability,
            ),
            as_numpy=self.as_numpy,
        )

    def random_resize(
        self, min_size: int, max_size: int, interpolation: str = "bilinear"
    ) -> Pipeline:
        """Resize the short edge to an integer in [min_size, max_size)."""
        return Pipeline(
            self._inner.random_resize(min_size, max_size, interpolation),
            as_numpy=self.as_numpy,
        )

    def random_shortest_size(
        self,
        min_size: int | Iterable[int],
        max_size: int | None = None,
        interpolation: str = "bilinear",
    ) -> Pipeline:
        """Choose a short edge uniformly, optionally capping the long edge."""
        sizes = [min_size] if isinstance(min_size, int) else list(min_size)
        return Pipeline(
            self._inner.random_shortest_size(sizes, max_size, interpolation),
            as_numpy=self.as_numpy,
        )

    def scale_jitter(
        self,
        width: int,
        height: int,
        scale_range: Iterable[float] = (0.1, 2.0),
        interpolation: str = "bilinear",
    ) -> Pipeline:
        """Fit within target width/height, then sample a multiplicative scale."""
        return Pipeline(
            self._inner.scale_jitter(width, height, list(scale_range), interpolation),
            as_numpy=self.as_numpy,
        )

    def random_zoom_out(
        self,
        fill: float | Iterable[float] = 0.0,
        side_range: Iterable[float] = (1.0, 4.0),
        probability: float = 0.5,
    ) -> Pipeline:
        """Place the image randomly on an expanded canvas with native-range fill."""
        return Pipeline(
            self._inner.random_zoom_out(
                _normalize_fill(fill), list(side_range), probability
            ),
            as_numpy=self.as_numpy,
        )

    def gaussian_noise(
        self, mean: float = 0.0, sigma: float = 0.1, clip: bool = True
    ) -> Pipeline:
        """Add Gaussian noise using mean/sigma in the [0, 1] convention.

        Float32 output clips to [0, 1]; uint8 noise is scaled by 255 and clips
        to [0, 255]. With ``clip=False``, uint8 arithmetic wraps on conversion.
        """
        return Pipeline(
            self._inner.gaussian_noise(mean, sigma, clip), as_numpy=self.as_numpy
        )

    def letterbox(
        self,
        width: int,
        height: int,
        *,
        interpolation: str = "bilinear",
        fill: float = 0.0,
        mask_fill: int = 0,
    ) -> Pipeline:
        """Fit uint8 RGB inside a fixed canvas and center it.

        When applied to annotated samples, masks use nearest-neighbor resizing
        and their own padding value. Odd padding goes on the right or bottom.
        """
        return Pipeline(
            self._inner.letterbox(
                width,
                height,
                interpolation=interpolation,
                fill=fill,
                mask_fill=mask_fill,
            ),
            as_numpy=self.as_numpy,
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
    ) -> Pipeline:
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
        return Pipeline(inner, as_numpy=self.as_numpy)

    def multiplicative_noise(
        self,
        multiplier: tuple[float, float] = (0.9, 1.1),
        *,
        per_channel: bool = False,
        elementwise: bool = False,
    ) -> Pipeline:
        """Multiply uint8 or normalized float32 HWC by seeded uniform gains.

        Gain bounds are finite and nonnegative. Set per_channel for separate
        gains by channel and elementwise for separate gains by pixel. The
        default samples one gain per image. Output is clipped.
        """
        inner = self._inner.multiplicative_noise(
            multiplier, per_channel=per_channel, elementwise=elementwise
        )
        return Pipeline(inner, as_numpy=self.as_numpy)

    def random_sun_flare(
        self,
        flare_roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.5),
        angle_range: tuple[float, float] = (0.0, 1.0),
        num_flare_circles_range: tuple[int, int] = (6, 10),
        src_radius: int = 400,
        src_color: tuple[int, int, int] = (255, 255, 255),
        physics_based: bool = False,
    ) -> Pipeline:
        """Add a seeded optical flare to uint8 or normalized float32 HWC RGB.

        Angles are fractions of a turn, radius is in pixels, and color is RGB.
        Set physics_based for diffraction rays and screen blending.
        """
        inner = self._inner.random_sun_flare(flare_roi, angle_range, num_flare_circles_range, src_radius, src_color, physics_based)
        return Pipeline(inner, as_numpy=self.as_numpy)

    def lens_flare(
        self,
        flare_roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.5),
        num_ghosts_range: tuple[int, int] = (3, 7),
        intensity_range: tuple[float, float] = (0.3, 0.7),
        num_rays_range: tuple[int, int] = (4, 8),
        bloom_range: tuple[float, float] = (0.01, 0.05),
    ) -> Pipeline:
        """Add a seeded optical flare to uint8 or normalized float32 HWC RGB.

        Source-position coordinates and bloom radius are normalized.
        Ghosts and starburst rays use screen blending.
        """
        inner = self._inner.lens_flare(flare_roi, num_ghosts_range, intensity_range, num_rays_range, bloom_range)
        return Pipeline(inner, as_numpy=self.as_numpy)

    def random_gravel(
        self,
        gravel_roi: tuple[float, float, float, float] = (0.1, 0.4, 0.9, 0.9),
        number_of_patches: int = 2,
    ) -> Pipeline:
        """Scatter seeded clusters of gravel on uint8 or float32 HWC RGB.

        The ROI is (left, top, right, bottom) in normalized coordinates.
        Zero patches leaves the image unchanged.
        """
        return Pipeline(self._inner.random_gravel(gravel_roi, number_of_patches), as_numpy=self.as_numpy)

    def random_snow(
        self,
        snow_point_range: tuple[float, float] = (0.1, 0.3),
        brightness_coeff: float = 2.5,
        method: str = "bleach",
    ) -> Pipeline:
        """Simulate snow on uint8 or normalized float32 HWC RGB.

        Bleach scales HSL lightness above the sampled threshold.
        Texture adds a seeded, cool-toned snow layer with more coverage
        at the top of the image and sparse sparkles. A zero snow point
        leaves the image unchanged.
        """
        inner = self._inner.random_snow(snow_point_range, brightness_coeff, method)
        return Pipeline(inner, as_numpy=self.as_numpy)

    def random_rain(
        self,
        slant_range: tuple[float, float] = (-10, 10),
        drop_length: int | None = None,
        drop_width: int = 1,
        drop_color: tuple[int, int, int] = (200, 200, 200),
        blur_value: int = 7,
        brightness_coefficient: float = 0.7,
        rain_type: str = "default",
    ) -> Pipeline:
        """Add seeded semi-transparent streaks to uint8 or float32 HWC RGB.

        Slant is measured from vertical in degrees. Length and width are in
        pixels. Use blur_value=1 to skip box smoothing. Rain type controls
        density and default streak length.
        """
        inner = self._inner.random_rain(
            slant_range, drop_length, drop_width, drop_color, blur_value,
            brightness_coefficient, rain_type,
        )
        return Pipeline(inner, as_numpy=self.as_numpy)

    def random_shadow(
        self,
        shadow_roi: tuple[float, float, float, float] = (0, 0.5, 1, 1),
        num_shadows_range: tuple[int, int] = (1, 2),
        shadow_dimension: int = 5,
        shadow_intensity_range: tuple[float, float] = (0.5, 0.5),
    ) -> Pipeline:
        """Darken random polygons in a normalized ROI on uint8 or float32 HWC.

        The ROI is (left, top, right, bottom). A polygon transmits one minus
        its sampled intensity. Transmittance multiplies where shadows overlap.
        """
        inner = self._inner.random_shadow(
            shadow_roi, num_shadows_range, shadow_dimension, shadow_intensity_range
        )
        return Pipeline(inner, as_numpy=self.as_numpy)

    def plasma_shadow(
        self, shadow_intensity_range: tuple[float, float] = (0.3, 0.7),
        plasma_size: int = 256, roughness: float = 3.0,
    ) -> Pipeline:
        """Darken pixels proportionally to a seeded plasma field.

        Accepts uint8 or normalized float32 HWC. Plasma size sets the field
        resolution (2–1024), and larger roughness preserves more fine detail.
        """
        return Pipeline(self._inner.plasma_shadow(shadow_intensity_range, plasma_size, roughness), as_numpy=self.as_numpy)

    def plasma_brightness_contrast(
        self, brightness_range: tuple[float, float] = (-0.3, 0.3), contrast_range: tuple[float, float] = (-0.3, 0.3),
        plasma_size: int = 256, roughness: float = 3.0,
    ) -> Pipeline:
        """Vary brightness and contrast about midgray using a seeded plasma field.

        Accepts uint8 or normalized float32 HWC. Plasma size sets the field
        resolution (2–1024), and larger roughness preserves more fine detail.
        """
        return Pipeline(self._inner.plasma_brightness_contrast(brightness_range, contrast_range, plasma_size, roughness), as_numpy=self.as_numpy)

    def illumination(
        self, mode: str = "linear", intensity_range: tuple[float, float] = (0.01, 0.2),
        effect_type: str = "both",
    ) -> Pipeline:
        """Apply a seeded smooth gain to uint8 or normalized float32 HWC.

        Linear samples a direction, corner samples a corner, and gaussian
        samples a spotlight center and spread. Intensity in [0, 1] bounds
        the peak gain change. Effect selects brightening, darkening, or
        a random choice of the two.
        """
        return Pipeline(self._inner.illumination(mode, intensity_range, effect_type), as_numpy=self.as_numpy)

    def film_grain(
        self,
        intensity_range: tuple[float, float] = (0.1, 0.3),
        grain_size_range: tuple[int, int] = (1, 3),
        *,
        per_channel: bool = False,
    ) -> Pipeline:
        """Add seeded Gaussian grain to uint8 or normalized float32 HWC.

        Grain size divides the field resolution before bilinear upsampling.
        Grain amplitude is proportional to inverse luma. Set per_channel to
        sample independent fields for each channel. Output is clipped.
        """
        inner = self._inner.film_grain(intensity_range, grain_size_range, per_channel=per_channel)
        return Pipeline(inner, as_numpy=self.as_numpy)

    def iso_noise(
        self,
        color_shift_range: tuple[float, float] = (0.01, 0.05),
        intensity_range: tuple[float, float] = (0.1, 0.5),
    ) -> Pipeline:
        """Add seeded luma and chroma noise to uint8 or normalized float32 HWC RGB.

        Both ranges are in [0, 1]. Intensity scales both components, while
        color shift controls chroma rotation in fractions of a turn. Luma
        noise grows with the square root of luma. Output is clipped.
        """
        return Pipeline(self._inner.iso_noise(color_shift_range, intensity_range), as_numpy=self.as_numpy)

    def glass_blur(
        self, sigma: float = 0.7, max_delta: int = 4, iterations: int = 2
    ) -> Pipeline:
        """Smooth, exchange nearby pixels, then smooth uint8 or float32 HWC again.

        Each pass exchanges whole pixels with partners within max_delta on
        each axis, clipped to the image. Sigma is in [0, 10]; zero skips
        smoothing. Use a positive max_delta and 1–32 passes.
        """
        return Pipeline(self._inner.glass_blur(sigma, max_delta, iterations), as_numpy=self.as_numpy)

    def zoom_blur(
        self,
        max_factor_range: tuple[float, float] = (1.0, 1.31),
        step_factor_range: tuple[float, float] = (0.01, 0.03),
    ) -> Pipeline:
        """Average uint8 or float32 HWC with bilinear zooms about the pixel center.

        Sample a maximum factor and a positive step per image, then average
        the original with factors 1 + n * step up to the maximum. The ranges
        may allow at most 128 zoomed samples per image.
        """
        return Pipeline(self._inner.zoom_blur(max_factor_range, step_factor_range), as_numpy=self.as_numpy)

    def stochastic_convolution(
        self,
        kernel_range: tuple[int, int] = (3, 7),
        strength_range: tuple[float, float] = (0.0, 1.0),
        *,
        per_channel: bool = False,
        padding_mode: str = "reflect",
    ) -> Pipeline:
        """Apply an identity kernel plus Gaussian tap noise to uint8 or float32 HWC.

        The sampled strength divided by the kernel width is the noise standard
        deviation. Kernels are shared across channels by default. Padding modes
        are constant (zero), edge, reflect, and symmetric. Float32 inputs use
        the normalized [0, 1] range. Zero strength gives exact identity.
        """
        inner = self._inner.stochastic_convolution(
            kernel_range, strength_range, per_channel=per_channel, padding_mode=padding_mode
        )
        return Pipeline(inner, as_numpy=self.as_numpy)

    def advanced_blur(
        self,
        kernel_range: tuple[int, int] = (3, 7),
        sigma_x_range: tuple[float, float] = (0.2, 1.0),
        sigma_y_range: tuple[float, float] = (0.2, 1.0),
        rotate_range: tuple[float, float] = (-90.0, 90.0),
        beta_range: tuple[float, float] = (0.5, 8.0),
        noise_range: tuple[float, float] = (0.75, 1.25),
    ) -> Pipeline:
        """Apply a seeded generalized Gaussian filter to uint8 or float32 HWC.

        Sigma controls the spread along each axis. Rotation is in degrees.
        Beta controls the falloff shape. Positive noise factors perturb each
        tap before normalization. Borders reflect without repeating the edge.
        """
        inner = self._inner.advanced_blur(
            kernel_range, sigma_x_range, sigma_y_range, rotate_range, beta_range, noise_range
        )
        return Pipeline(inner, as_numpy=self.as_numpy)

    def blur(self, kernel_range: tuple[int, int] = (3, 7)) -> Pipeline:
        """Average uint8 or float32 HWC pixels using a seeded square box filter.

        Sample an odd size from the inclusive range (odd endpoints in 3–127).
        Reflect neighbors at the border without repeating the edge pixel.
        """
        return Pipeline(self._inner.blur(kernel_range), as_numpy=self.as_numpy)

    def median_blur(self) -> Pipeline:
        """Apply a 3×3 median per channel to uint8 HWC grayscale or RGB.

        Out-of-bounds neighbors use the nearest border pixel.
        """
        return Pipeline(self._inner.median_blur(), as_numpy=self.as_numpy)

    def planckian_jitter(
        self, temperature_range: tuple[float, float] = (3000, 15000)
    ) -> Pipeline:
        """Sample a blackbody temperature uniformly in kelvin, from 3000 to 15000.

        Apply linear-sRGB gains relative to a neutral 6500 K blackbody,
        keeping green fixed. Inputs are uint8 or normalized float32 sRGB
        in HWC order. Output is clipped to the input's nominal range.
        """
        return Pipeline(self._inner.planckian_jitter(temperature_range), as_numpy=self.as_numpy)

    def vignetting(
        self,
        intensity_range: tuple[float, float] = (0.2, 0.5),
        center_range: tuple[float, float] = (0.3, 0.7),
    ) -> Pipeline:
        """Darken uint8 or float32 HWC images using seeded radial falloff.

        Each center coordinate is sampled independently as a fraction of its
        pixel axis. Intensity is the fractional loss at the farthest corner.
        """
        return Pipeline(
            self._inner.vignetting(intensity_range, center_range), as_numpy=self.as_numpy
        )

    def random_tone_curve(self, scale: float = 0.1) -> Pipeline:
        """Apply one seeded monotone tone curve to all uint8 HWC RGB channels."""
        return Pipeline(self._inner.random_tone_curve(scale), as_numpy=self.as_numpy)

    def clahe(
        self, clip_limit: float = 2.0, *, grid: tuple[int, int] = (8, 8)
    ) -> Pipeline:
        """Equalize local luminance in uint8 HWC grayscale or RGB images.

        Grid is (columns, rows). Clip limit scales the average histogram
        bin size, and neighboring tile mappings are blended bilinearly.
        """
        return Pipeline(
            self._inner.clahe(clip_limit, grid=grid), as_numpy=self.as_numpy
        )

    def chromatic_aberration(
        self,
        red_range: tuple[float, float] = (0.0, 0.05),
        blue_range: tuple[float, float] = (-0.05, 0.0),
        *,
        interpolation: str = "bilinear",
    ) -> Pipeline:
        """Radially resample red and blue in uint8 HWC RGB, with edge extension.

        Sample one coefficient from each range per image. Positive coefficients
        sample farther from the center. Green and annotation coordinates are unchanged.
        """
        return Pipeline(
            self._inner.chromatic_aberration(
                red_range, blue_range, interpolation=interpolation
            ),
            as_numpy=self.as_numpy,
        )

    def defocus(self, radius_range: tuple[int, int] = (1, 3)) -> Pipeline:
        """Blur uint8 HWC RGB with a uniform disk and reflected borders.

        Sample one integer radius from the inclusive bounds for each image.
        Radius zero is an identity.
        """
        return Pipeline(self._inner.defocus(radius_range), as_numpy=self.as_numpy)

    def coarse_dropout(
        self,
        holes: tuple[int, int] = (1, 4),
        *,
        hole_height: tuple[int, int] = (4, 8),
        hole_width: tuple[int, int] = (4, 8),
        fill: float = 0.0,
    ) -> Pipeline:
        """Fill random rectangles using inclusive count and pixel size ranges.

        Rectangles can overlap and are clipped in size to fit the image.
        Only image pixels are modified.
        """
        return Pipeline(
            self._inner.coarse_dropout(
                holes, hole_height=hole_height, hole_width=hole_width, fill=fill
            ),
            as_numpy=self.as_numpy,
        )

    def motion_blur(
        self,
        length_range: tuple[int, int] = (3, 7),
        *,
        angle_range: tuple[float, float] = (0.0, 180.0),
    ) -> Pipeline:
        """Blur uint8 HWC RGB along a line with reflected borders.

        Sample an odd length from the inclusive bounds and an angle in
        degrees from the horizontal axis, once per image.
        """
        return Pipeline(
            self._inner.motion_blur(length_range, angle_range=angle_range),
            as_numpy=self.as_numpy,
        )

    def unsharp_mask(
        self, sigma: float = 1.0, *, amount: float = 1.0, threshold: int = 0
    ) -> Pipeline:
        """Amplify Gaussian residuals on uint8 HWC RGB above a pixel threshold."""
        return Pipeline(
            self._inner.unsharp_mask(sigma, amount=amount, threshold=threshold),
            as_numpy=self.as_numpy,
        )

    def downscale(
        self,
        scale_range: tuple[float, float] = (0.25, 0.5),
        *,
        down_interpolation: str = "nearest",
        up_interpolation: str = "nearest",
    ) -> Pipeline:
        """Reduce and restore the resolution of a uint8 HWC RGB image.

        A single scale is sampled per image from the given range in (0, 1].
        The output keeps the input dimensions.
        """
        return Pipeline(
            self._inner.downscale(
                scale_range,
                down_interpolation=down_interpolation,
                up_interpolation=up_interpolation,
            ),
            as_numpy=self.as_numpy,
        )

    def salt_and_pepper(
        self, amount: float = 0.05, *, salt_ratio: float = 0.5
    ) -> Pipeline:
        """Replace each pixel with probability ``amount``, across all channels.

        Of the replaced pixels, ``salt_ratio`` is the probability of white
        (255 for uint8 or 1.0 for float32). The other replacements are black.
        """
        return Pipeline(
            self._inner.salt_and_pepper(amount, salt_ratio=salt_ratio),
            as_numpy=self.as_numpy,
        )

    def grid_dropout(
        self, unit_size: int, *, ratio: float = 0.5, fill: float = 0.0
    ) -> Pipeline:
        """Fill one square per grid cell at a shared random offset.

        The hole side is floor(unit_size * ratio), clamped to [1, unit_size - 1].
        The grid starts at the image origin and partial cells are clipped.
        Only image pixels are modified.
        """
        return Pipeline(
            self._inner.grid_dropout(unit_size, ratio=ratio, fill=fill),
            as_numpy=self.as_numpy,
        )

    def shot_noise(self, intensity: float = 0.2) -> Pipeline:
        """Add Gaussian noise scaled by intensity * sqrt(max(pixel, 0) + 0.5).

        Call after scaling pixels to float32 [0, 1]. Output is clipped to [0, 1].
        """
        return Pipeline(self._inner.shot_noise(intensity), as_numpy=self.as_numpy)

    def jpeg(self, quality: int | Iterable[int]) -> Pipeline:
        """JPEG encode/decode uint8 gray/RGB pixels; quality bounds are inclusive.

        Uses libjpeg-compatible encoding with 4:2:0 chroma subsampling for RGB;
        grayscale uses one component. Quality is sampled once per image.
        """
        qualities = _normalize_jpeg_quality(quality)
        return Pipeline(self._inner.jpeg(qualities), as_numpy=self.as_numpy)

    def rgb(self) -> Pipeline:
        """Convert decoded grayscale/gray-alpha/RGB/RGBA pixels to RGB.

        Alpha is discarded. Image decoding already produces RGB by default.
        """
        return Pipeline(self._inner.rgb(), as_numpy=self.as_numpy)

    def to_dtype(self, dtype: str, scale: bool = False) -> Pipeline:
        """Convert uint8/float32, scaling between [0,255] and [0,1] if requested."""
        return Pipeline(self._inner.to_dtype(dtype, scale), as_numpy=self.as_numpy)

    def convert_image_dtype(self, dtype: str) -> Pipeline:
        return Pipeline(self._inner.convert_image_dtype(dtype), as_numpy=self.as_numpy)

    def rotate(self, angle: int) -> Pipeline:
        return Pipeline(self._inner.rotate(angle), as_numpy=self.as_numpy)

    def arbitrary_rotate(
        self,
        angle: float,
        interpolation: str = "nearest",
        expand: bool = False,
        fill: int | Iterable[int] = 0,
        center: Iterable[float] | None = None,
    ) -> Pipeline:
        return Pipeline(
            self._inner.arbitrary_rotate(
                angle,
                interpolation,
                expand,
                _normalize_fill(fill),
                None if center is None else list(center),
            ),
            as_numpy=self.as_numpy,
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
    ) -> Pipeline:
        """Sample an affine transform using Rivet's per-image random stream.

        Degrees may be a scalar maximum or a two-bound interval. Scalar shear
        affects x; a shear pair retains legacy symmetric x/y maxima. Use
        ``shear_ranges`` for two x bounds or four x/y bounds instead."""
        if shear is not None and shear_ranges is not None:
            raise ValueError("pass either shear or shear_ranges, not both")
        if isinstance(shear, (int, float)) and shear < 0:
            raise ValueError("shear must be non-negative")
        return Pipeline(
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
            ),
            as_numpy=self.as_numpy,
        )

    def perspective(
        self,
        start_points: Iterable[Iterable[float]],
        end_points: Iterable[Iterable[float]],
        interpolation: str = "bilinear",
        fill: int | Iterable[int] = 0,
    ) -> Pipeline:
        return Pipeline(
            self._inner.perspective(
                [list(point) for point in start_points],
                [list(point) for point in end_points],
                interpolation,
                _normalize_fill(fill),
            ),
            as_numpy=self.as_numpy,
        )

    def random_perspective(
        self,
        distortion_scale: float = 0.5,
        probability: float = 0.5,
        interpolation: str = "bilinear",
        fill: int | Iterable[int] = 0,
    ) -> Pipeline:
        return Pipeline(
            self._inner.random_perspective(
                distortion_scale, probability, interpolation, _normalize_fill(fill)
            ),
            as_numpy=self.as_numpy,
        )

    def elastic_transform(
        self,
        alpha: float,
        sigma: float,
        interpolation: str = "bilinear",
        fill: int | Iterable[int] = 0,
    ) -> Pipeline:
        return Pipeline(
            self._inner.elastic_transform(
                alpha, sigma, interpolation, _normalize_fill(fill)
            ),
            as_numpy=self.as_numpy,
        )

    def normalize(self, mean: list[float], std: list[float]) -> Pipeline:
        return Pipeline(self._inner.normalize(mean, std), as_numpy=self.as_numpy)

    def hwc_to_chw(self) -> Pipeline:
        return Pipeline(self._inner.hwc_to_chw(), as_numpy=self.as_numpy)

    def chw_to_hwc(self) -> Pipeline:
        return Pipeline(self._inner.chw_to_hwc(), as_numpy=self.as_numpy)

    def skip(self, count: int) -> Pipeline:
        return Pipeline(self._inner.skip(count), as_numpy=self.as_numpy)

    def take(self, count: int) -> Pipeline:
        return Pipeline(self._inner.take(count), as_numpy=self.as_numpy)

    def mixup(self, num_classes: int, alpha: float = 1.0) -> Pipeline:
        """Mix float32 images and emit [B, num_classes] float32 soft labels.

        Place after batch and all image transforms. Pair each item with the
        previous batch item (roll by one). A shared Beta(alpha, alpha)
        coefficient is drawn per output batch, independent of worker count.
        """
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes <= 0
        ):
            raise ValueError("num_classes must be a positive integer")
        return Pipeline(self._inner.batch_mix(alpha, num_classes, False))

    def cutmix(self, num_classes: int, alpha: float = 1.0) -> Pipeline:
        """Cut/paste from the previous item and mix labels by actual patch area.

        Accept uint8 or float32 images. Place after batch and all image
        transforms. Tail batches use their actual size, including size one.
        """
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes <= 0
        ):
            raise ValueError("num_classes must be a positive integer")
        return Pipeline(self._inner.batch_mix(alpha, num_classes, True))

    def batch(self, size: int, *, drop_last: bool = False) -> Pipeline:
        """Group source samples; multi-view crops yield ``size * V`` images.

        ``drop_last`` drops incomplete source batches, keeping whole view groups.
        """
        return Pipeline(
            self._inner.batch(size, drop_last),
            as_numpy=self.as_numpy,
        )

    def seed(self, seed: int) -> Pipeline:
        """Set the pipeline-owned seed for sampling and transforms."""
        return Pipeline(self._inner.seed(seed), as_numpy=self.as_numpy)

    def shuffle(self, seed: int | None = None) -> Pipeline:
        """Deterministically shuffle the sampled window.

        The same seed reproduces the same order at any worker count.
        If ``seed`` is omitted, use the pipeline-owned seed configured by
        :meth:`seed`; ``.epoch(epoch)`` changes both sampler and transform
        namespaces.
        """
        return Pipeline(
            self._inner.shuffle(seed),
            as_numpy=self.as_numpy,
        )

    def epoch(self, epoch: int) -> Pipeline:
        """Set the deterministic augmentation epoch without reordering data."""
        return Pipeline(self._inner.epoch(epoch), as_numpy=self.as_numpy)

    def workers(self, num_workers: int) -> Pipeline:
        """Load samples on a persistent pool of `num_workers` threads.

        Zero (the default) keeps synchronous loading; ordering and batch
        contents are identical for any worker count.
        """
        return Pipeline(
            self._inner.workers(num_workers),
            as_numpy=self.as_numpy,
        )

    def prefetch_batches(self, prefetch: int) -> Pipeline:
        """Prepare `prefetch` future batches ahead (worker pools only).

        The current batch plus `prefetch` more are in flight; ``0`` keeps
        only the current batch running.
        """
        return Pipeline(
            self._inner.prefetch_batches(prefetch),
            as_numpy=self.as_numpy,
        )

    def with_numpy(self, enabled: bool = True) -> Pipeline:
        return Pipeline(self._inner, as_numpy=enabled)

    def execute(self, *, as_numpy: bool | None = None) -> DataLoader:
        from ._loader import DataLoader

        return DataLoader(
            self,
            as_numpy=self.as_numpy if as_numpy is None else as_numpy,
        )

    def explain(self) -> str:
        """Explain declared/optimized IR and placement without reading images.

        This validates and optimizes a separate plan for inspection. Calling
        execute() still compiles its own execution plan.
        """
        return self._inner.explain()
