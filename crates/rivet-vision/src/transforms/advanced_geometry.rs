use super::logical_offset;
use super::resize::InterpolationMode;
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use rivet_core::{CpuStorageRef, DType, Tensor};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

impl Point2 {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArbitraryRotateConfig {
    /// Positive angles rotate counter-clockwise, as in torchvision rotate.
    pub angle: f32,
    pub expand: bool,
    pub interpolation: InterpolationMode,
    pub fill: u8,
}

impl ArbitraryRotateConfig {
    pub const fn new(angle: f32) -> Self {
        Self {
            angle,
            expand: false,
            interpolation: InterpolationMode::Nearest,
            fill: 0,
        }
    }

    pub const fn with_options(
        mut self,
        expand: bool,
        interpolation: InterpolationMode,
        fill: u8,
    ) -> Self {
        self.expand = expand;
        self.interpolation = interpolation;
        self.fill = fill;
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        if !self.angle.is_finite() {
            return Err(invalid_argument("arbitrary rotation angle must be finite"));
        }
        validate_interpolation(self.interpolation, "arbitrary rotation")
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = require_u8_image(sample, "arbitrary rotation")?;
        let dims = sample.image.dims();
        let (height, width, _) = image_dims(dims, axis_order)?;
        let (matrix, output_width, output_height) =
            rotation_inverse(width, height, self.angle, self.expand);
        warp_decoded::<false, _>(
            sample,
            axis_order,
            output_width,
            output_height,
            self.interpolation,
            self.fill,
            "arbitrary rotation",
            move |x, y| {
                let x = x as f64 + 0.5;
                let y = y as f64 + 0.5;
                [
                    matrix[0] * x + matrix[1] * y + matrix[2] - 0.5,
                    matrix[3] * x + matrix[4] * y + matrix[5] - 0.5,
                ]
            },
        )
        .map(ImageSample::Decoded)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomAffineConfig {
    /// Maximum absolute rotation angle in degrees.
    pub degrees: f32,
    /// Maximum translation fractions `(x, y)` of the input width/height.
    pub translate: [f32; 2],
    /// Inclusive scale range.
    pub scale: [f32; 2],
    /// Maximum absolute shear angles `(x, y)` in degrees.
    pub shear: [f32; 2],
    pub interpolation: InterpolationMode,
    pub fill: u8,
}

impl RandomAffineConfig {
    pub const fn new(degrees: f32) -> Self {
        Self {
            degrees,
            translate: [0.0, 0.0],
            scale: [1.0, 1.0],
            shear: [0.0, 0.0],
            interpolation: InterpolationMode::Nearest,
            fill: 0,
        }
    }

    pub const fn with_translate(mut self, x: f32, y: f32) -> Self {
        self.translate = [x, y];
        self
    }

    pub const fn with_scale(mut self, min: f32, max: f32) -> Self {
        self.scale = [min, max];
        self
    }

    pub const fn with_shear(mut self, x: f32, y: f32) -> Self {
        self.shear = [x, y];
        self
    }

    pub const fn with_options(mut self, interpolation: InterpolationMode, fill: u8) -> Self {
        self.interpolation = interpolation;
        self.fill = fill;
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        if !self.degrees.is_finite() || self.degrees < 0.0 {
            return Err(invalid_argument(
                "random affine degrees must be finite and non-negative",
            ));
        }
        if self
            .translate
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
        {
            return Err(invalid_argument(
                "random affine translate fractions must be finite and in [0, 1]",
            ));
        }
        if !self.scale[0].is_finite()
            || !self.scale[1].is_finite()
            || self.scale[0] <= 0.0
            || self.scale[0] > self.scale[1]
        {
            return Err(invalid_argument(
                "random affine scale must satisfy 0 < min <= max",
            ));
        }
        if self
            .shear
            .iter()
            .any(|value| !value.is_finite() || value.abs() >= 89.0)
        {
            return Err(invalid_argument(
                "random affine shear must be finite and have absolute value below 89 degrees",
            ));
        }
        validate_interpolation(self.interpolation, "random affine")
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        rng: &mut RandomStream,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = require_u8_image(sample, "random affine")?;
        let dims = sample.image.dims();
        let (height, width, _) = image_dims(dims, axis_order)?;
        let angle = f64::from(uniform(rng, -self.degrees, self.degrees)).to_radians();
        let tx = (f64::from(uniform(rng, -self.translate[0], self.translate[0])) * width as f64)
            .round_ties_even();
        let ty = (f64::from(uniform(rng, -self.translate[1], self.translate[1])) * height as f64)
            .round_ties_even();
        let scale = f64::from(uniform(rng, self.scale[0], self.scale[1]));
        let sx = f64::from(uniform(rng, -self.shear[0], self.shear[0])).to_radians();
        let sy = f64::from(uniform(rng, -self.shear[1], self.shear[1])).to_radians();
        // torchvision functional._get_inverse_affine_matrix: R * S * SHy * SHx.
        let a = (angle - sy).cos() / sy.cos();
        let b = -(angle - sy).cos() * sx.tan() / sy.cos() - angle.sin();
        let c = (angle - sy).sin() / sy.cos();
        let d = -(angle - sy).sin() * sx.tan() / sy.cos() + angle.cos();
        let (cx, cy) = (width as f64 * 0.5, height as f64 * 0.5);
        let m = [d / scale, -b / scale, -c / scale, a / scale];
        let ox = m[0] * (-cx - tx) + m[1] * (-cy - ty) + cx;
        let oy = m[2] * (-cx - tx) + m[3] * (-cy - ty) + cy;
        warp_decoded::<false, _>(
            sample,
            axis_order,
            width,
            height,
            self.interpolation,
            self.fill,
            "random affine",
            move |x, y| {
                let (x, y) = (x as f64 + 0.5, y as f64 + 0.5);
                [
                    m[0] * x + m[1] * y + ox - 0.5,
                    m[2] * x + m[3] * y + oy - 0.5,
                ]
            },
        )
        .map(ImageSample::Decoded)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerspectiveConfig {
    /// Source and destination quadrilaterals use `(x, y)` pixel coordinates
    /// in top-left, top-right, bottom-right, bottom-left order.
    pub start_points: [Point2; 4],
    pub end_points: [Point2; 4],
    pub interpolation: InterpolationMode,
    pub fill: u8,
}

impl PerspectiveConfig {
    pub const fn new(start_points: [Point2; 4], end_points: [Point2; 4]) -> Self {
        Self {
            start_points,
            end_points,
            interpolation: InterpolationMode::Bilinear,
            fill: 0,
        }
    }

    pub const fn with_options(mut self, interpolation: InterpolationMode, fill: u8) -> Self {
        self.interpolation = interpolation;
        self.fill = fill;
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self
            .start_points
            .iter()
            .chain(self.end_points.iter())
            .any(|point| !point.x.is_finite() || !point.y.is_finite())
        {
            return Err(invalid_argument(
                "perspective points must contain only finite coordinates",
            ));
        }
        validate_interpolation(self.interpolation, "perspective")
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = require_u8_image(sample, "perspective")?;
        warp_perspective(
            sample,
            axis_order,
            &self.start_points,
            &self.end_points,
            self.interpolation,
            self.fill,
            "perspective",
        )
        .map(ImageSample::Decoded)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomPerspectiveConfig {
    pub distortion_scale: f32,
    pub probability: f64,
    pub interpolation: InterpolationMode,
    pub fill: u8,
}

impl RandomPerspectiveConfig {
    pub const fn new(distortion_scale: f32, probability: f64) -> Self {
        Self {
            distortion_scale,
            probability,
            interpolation: InterpolationMode::Bilinear,
            fill: 0,
        }
    }

    pub const fn with_options(mut self, interpolation: InterpolationMode, fill: u8) -> Self {
        self.interpolation = interpolation;
        self.fill = fill;
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        if !self.distortion_scale.is_finite() || !(0.0..=1.0).contains(&self.distortion_scale) {
            return Err(invalid_argument(
                "random perspective distortion_scale must be finite and in [0, 1]",
            ));
        }
        if !self.probability.is_finite() || !(0.0..=1.0).contains(&self.probability) {
            return Err(invalid_argument(
                "random perspective probability must be finite and in [0, 1]",
            ));
        }
        validate_interpolation(self.interpolation, "random perspective")
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        rng: &mut RandomStream,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = sample.into_decoded()?;
        if !rng.gen_bool(self.probability)? {
            return Ok(ImageSample::Decoded(sample));
        }
        let dims = sample.image.dims();
        let (height, width, _) = image_dims(dims, axis_order)?;
        // There is no non-degenerate quadrilateral to sample for a 1-pixel
        // extent. Treat this as an identity transform rather than surfacing a
        // singular homography from a valid tiny image.
        if width < 2 || height < 2 || self.distortion_scale == 0.0 {
            return Ok(ImageSample::Decoded(sample));
        }
        let max_x = width.saturating_sub(1) as f32;
        let max_y = height.saturating_sub(1) as f32;
        let half_x = (self.distortion_scale * (width / 2) as f32) as usize;
        let half_y = (self.distortion_scale * (height / 2) as f32) as usize;
        let corners = [
            Point2::new(0.0, 0.0),
            Point2::new(max_x, 0.0),
            Point2::new(max_x, max_y),
            Point2::new(0.0, max_y),
        ];
        let mut draw = |max: usize| {
            rng.gen_range_usize(0..max + 1)
                .expect("nonempty perspective range") as f32
        };
        let end_points = [
            Point2::new(draw(half_x), draw(half_y)),
            Point2::new(max_x - half_x as f32 + draw(half_x), draw(half_y)),
            Point2::new(
                max_x - half_x as f32 + draw(half_x),
                max_y - half_y as f32 + draw(half_y),
            ),
            Point2::new(draw(half_x), max_y - half_y as f32 + draw(half_y)),
        ];
        warp_perspective(
            sample,
            axis_order,
            &corners,
            &end_points,
            self.interpolation,
            self.fill,
            "random perspective",
        )
        .map(ImageSample::Decoded)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElasticTransformConfig {
    /// torchvision displacement magnitude (pixel offsets are noise * alpha / 2).
    pub alpha: f32,
    /// Gaussian smoothing sigma in pixels; zero keeps white-noise offsets.
    pub sigma: f32,
    pub interpolation: InterpolationMode,
    pub fill: u8,
}

impl ElasticTransformConfig {
    pub const fn new(alpha: f32, sigma: f32) -> Self {
        Self {
            alpha,
            sigma,
            interpolation: InterpolationMode::Bilinear,
            fill: 0,
        }
    }

    pub const fn with_options(mut self, interpolation: InterpolationMode, fill: u8) -> Self {
        self.interpolation = interpolation;
        self.fill = fill;
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(invalid_argument(
                "elastic transform alpha must be finite and non-negative",
            ));
        }
        if !self.sigma.is_finite() || self.sigma < 0.0 {
            return Err(invalid_argument(
                "elastic transform sigma must be finite and non-negative",
            ));
        }
        if !matches!(
            self.interpolation,
            InterpolationMode::Nearest | InterpolationMode::Bilinear
        ) {
            return Err(invalid_argument(
                "elastic transform supports nearest or bilinear interpolation",
            ));
        }
        Ok(())
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        rng: &mut RandomStream,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = require_u8_image(sample, "elastic transform")?;
        let dims = sample.image.dims();
        let (height, width, _) = image_dims(dims, axis_order)?;
        let count = height
            .checked_mul(width)
            .ok_or_else(|| invalid_shape("elastic transform image is too large"))?;
        let mut x_field = (0..count)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect::<Vec<_>>();
        let mut y_field = (0..count)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect::<Vec<_>>();
        let radius = ((8.0 * self.sigma + 1.0) as usize | 1) / 2;
        if self.sigma > 0.0 && (radius >= width || radius >= height) {
            return Err(invalid_argument(
                "elastic transform reflection padding must be smaller than image dimensions",
            ));
        }
        if self.sigma > 0.0 {
            x_field = smooth_field(&x_field, width, height, self.sigma);
            y_field = smooth_field(&y_field, width, height, self.sigma);
        }
        let alpha = self.alpha * 0.5;
        warp_decoded::<true, _>(
            sample,
            axis_order,
            width,
            height,
            self.interpolation,
            self.fill,
            "elastic transform",
            move |x, y| {
                let index = y * width + x;
                [
                    x as f64 + f64::from(alpha * x_field[index]),
                    y as f64 + f64::from(alpha * y_field[index]),
                ]
            },
        )
        .map(ImageSample::Decoded)
    }
}

fn validate_interpolation(mode: InterpolationMode, op_name: &str) -> RivetResult<()> {
    match mode {
        InterpolationMode::Nearest | InterpolationMode::Bilinear | InterpolationMode::Bicubic => {
            Ok(())
        }
        InterpolationMode::Lanczos3 => Err(invalid_argument(format!(
            "{op_name} supports nearest, bilinear or bicubic interpolation"
        ))),
    }
}

fn require_u8_image(sample: ImageSample, op_name: &str) -> RivetResult<DecodedSample> {
    let sample = sample.into_decoded()?;
    if sample.image.dtype() != DType::U8 || sample.image.rank() != 3 {
        return Err(invalid_argument(format!(
            "{op_name} requires a rank-3 uint8 image, got dtype {:?} shape {:?}",
            sample.image.dtype(),
            sample.image.dims()
        )));
    }
    Ok(sample)
}

fn image_dims(dims: &[usize], axis_order: ImageAxisOrder) -> RivetResult<(usize, usize, usize)> {
    if dims.len() != 3 {
        return Err(invalid_shape(format!(
            "advanced geometry requires a rank-3 image, got shape {dims:?}"
        )));
    }
    let (height, width, channels) = match axis_order {
        ImageAxisOrder::Hwc => (dims[0], dims[1], dims[2]),
        ImageAxisOrder::Chw => (dims[1], dims[2], dims[0]),
    };
    if height == 0 || width == 0 || channels == 0 {
        return Err(invalid_shape(
            "advanced geometry does not support zero-sized image dimensions",
        ));
    }
    Ok((height, width, channels))
}

fn warp_perspective(
    sample: DecodedSample,
    axis_order: ImageAxisOrder,
    start_points: &[Point2; 4],
    end_points: &[Point2; 4],
    interpolation: InterpolationMode,
    fill: u8,
    op_name: &str,
) -> RivetResult<DecodedSample> {
    let inverse = homography(end_points, start_points)
        .ok_or_else(|| invalid_argument(format!("{op_name} points define a singular transform")))?;
    let dims = sample.image.dims();
    let (height, width, _) = image_dims(dims, axis_order)?;
    warp_decoded::<false, _>(
        sample,
        axis_order,
        width,
        height,
        interpolation,
        fill,
        op_name,
        move |x, y| {
            let p = apply_homography(inverse, [x as f64 + 0.5, y as f64 + 0.5]);
            [p[0] - 0.5, p[1] - 0.5]
        },
    )
}

fn warp_decoded<const TENSOR: bool, F>(
    sample: DecodedSample,
    axis_order: ImageAxisOrder,
    output_width: usize,
    output_height: usize,
    interpolation: InterpolationMode,
    fill: u8,
    op_name: &str,
    mut map: F,
) -> RivetResult<DecodedSample>
where
    F: FnMut(usize, usize) -> [f64; 2],
{
    if output_width == 0 || output_height == 0 {
        return Err(invalid_shape(format!(
            "{op_name} produced an empty output shape"
        )));
    }
    let input = &sample.image;
    let dims = input.dims().to_vec();
    let (height, width, channels) = image_dims(&dims, axis_order)?;
    let values = input.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::U8(values) = storage else {
            return Err(rivet_core::Error::UnexpectedDType {
                expected: DType::U8,
                actual: input.dtype(),
            });
        };
        let read = |channel: usize, y: usize, x: usize| {
            let coords = match axis_order {
                ImageAxisOrder::Hwc => [y, x, channel],
                ImageAxisOrder::Chw => [channel, y, x],
            };
            values
                .get(logical_offset(layout, &coords)?)
                .copied()
                .ok_or(rivet_core::Error::StorageOutOfBounds)
        };
        let sample_pixel = |channel: usize, source: [f64; 2]| {
            if TENSOR {
                return tensor_sample_value(
                    &read,
                    channel,
                    source[1] as f32,
                    source[0] as f32,
                    width,
                    height,
                    interpolation,
                    fill,
                );
            }
            sample_value(
                &read,
                channel,
                source[1],
                source[0],
                width,
                height,
                interpolation,
                fill,
            )
        };
        let output_dims = match axis_order {
            ImageAxisOrder::Hwc => vec![output_height, output_width, channels],
            ImageAxisOrder::Chw => vec![channels, output_height, output_width],
        };
        Tensor::from_exact_writer::<u8, _, _>(output_dims, input.device(), |write| {
            match axis_order {
                ImageAxisOrder::Hwc => {
                    for y in 0..output_height {
                        for x in 0..output_width {
                            let source = map(x, y);
                            if channels == 3 {
                                write.write_next(sample_pixel(0, source)?)?;
                                write.write_next(sample_pixel(1, source)?)?;
                                write.write_next(sample_pixel(2, source)?)?;
                            } else {
                                for channel in 0..channels {
                                    write.write_next(sample_pixel(channel, source)?)?;
                                }
                            }
                        }
                    }
                }
                ImageAxisOrder::Chw => {
                    for channel in 0..channels {
                        for y in 0..output_height {
                            for x in 0..output_width {
                                write.write_next(sample_pixel(channel, map(x, y))?)?;
                            }
                        }
                    }
                }
            }
            Ok(())
        })
    })?;
    Ok(DecodedSample {
        image: values,
        label: sample.label,
    })
}

fn tensor_sample_value<F>(
    read: &F,
    c: usize,
    y: f32,
    x: f32,
    width: usize,
    height: usize,
    mode: InterpolationMode,
    fill: u8,
) -> rivet_core::Result<u8>
where
    F: Fn(usize, usize, usize) -> rivet_core::Result<u8>,
{
    let inside = |x: isize, y: isize| x >= 0 && y >= 0 && x < width as isize && y < height as isize;
    if mode == InterpolationMode::Nearest {
        let (sx, sy) = (x.round_ties_even() as isize, y.round_ties_even() as isize);
        return if inside(sx, sy) {
            read(c, sy as usize, sx as usize)
        } else {
            Ok(fill)
        };
    }
    let (x0, y0) = (x.floor() as isize, y.floor() as isize);
    let (dx, dy) = (x - x.floor(), y - y.floor());
    let mut sum = 0.0f32;
    let mut mask = 0.0f32;
    for (sx, sy, weight) in [
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x0 + 1, y0, dx * (1.0 - dy)),
        (x0, y0 + 1, (1.0 - dx) * dy),
        (x0 + 1, y0 + 1, dx * dy),
    ] {
        if inside(sx, sy) {
            sum += f32::from(read(c, sy as usize, sx as usize)?) * weight;
            mask += weight;
        }
    }
    Ok((sum * mask + (1.0 - mask) * f32::from(fill))
        .round_ties_even()
        .clamp(0.0, 255.0) as u8)
}

fn sample_value<F>(
    read: &F,
    channel: usize,
    y: f64,
    x: f64,
    width: usize,
    height: usize,
    interpolation: InterpolationMode,
    fill: u8,
) -> rivet_core::Result<u8>
where
    F: Fn(usize, usize, usize) -> rivet_core::Result<u8>,
{
    // Pixel centers are integers here; Pillow's Geometry.c receives edge
    // coordinates and subtracts 0.5 before interpolating.
    if !x.is_finite()
        || !y.is_finite()
        || x < -0.5
        || y < -0.5
        || x >= width as f64 - 0.5
        || y >= height as f64 - 0.5
    {
        return Ok(fill);
    }
    if interpolation == InterpolationMode::Nearest {
        return read(
            channel,
            (y + 0.5).floor() as usize,
            (x + 0.5).floor() as usize,
        );
    }
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let dx = x - x.floor();
    let dy = y - y.floor();
    let clip_x = |x: isize| x.clamp(0, width as isize - 1) as usize;
    let clip_y = |y: isize| y.clamp(0, height as isize - 1) as usize;
    let value = if interpolation == InterpolationMode::Bicubic {
        let cubic = |v: [f64; 4], d: f64| {
            let p1 = v[1];
            let p2 = -v[0] + v[2];
            let p3 = 2.0 * (v[0] - v[1]) + v[2] - v[3];
            let p4 = -v[0] + v[1] - v[2] + v[3];
            p1 + d * (p2 + d * (p3 + d * p4))
        };
        let mut rows = [0.0; 4];
        for (i, row) in rows.iter_mut().enumerate() {
            let sy = clip_y(y0 - 1 + i as isize);
            let mut pixels = [0.0; 4];
            for (j, pixel) in pixels.iter_mut().enumerate() {
                *pixel = f64::from(read(channel, sy, clip_x(x0 - 1 + j as isize))?);
            }
            *row = cubic(pixels, dx);
        }
        cubic(rows, dy)
    } else {
        let row = |y| -> rivet_core::Result<f64> {
            let a = f64::from(read(channel, clip_y(y), clip_x(x0))?);
            let b = f64::from(read(channel, clip_y(y), clip_x(x0 + 1))?);
            Ok(a + (b - a) * dx)
        };
        let a = row(y0)?;
        let b = row(y0 + 1)?;
        a + (b - a) * dy
    };
    Ok(value.clamp(0.0, 255.0) as u8)
}

fn rotation_inverse(
    width: usize,
    height: usize,
    angle: f32,
    expand: bool,
) -> ([f64; 6], usize, usize) {
    // Image.py Image.rotate: reverse transform, rounded trigonometric
    // coefficients, edge-coordinate bounds, then expansion translation.
    let radians = -f64::from(angle.rem_euclid(360.0)).to_radians();
    let round15 = |v: f64| (v * 1e15).round_ties_even() / 1e15;
    let (sin, cos) = (round15(radians.sin()), round15(radians.cos()));
    let (cx, cy) = (width as f64 * 0.5, height as f64 * 0.5);
    let mut m = [cos, sin, 0.0, -sin, cos, 0.0];
    m[2] = -cos * cx - sin * cy + cx;
    m[5] = sin * cx - cos * cy + cy;
    let (mut ow, mut oh) = (width, height);
    if expand {
        let corners = [
            [0.0, 0.0],
            [width as f64, 0.0],
            [width as f64, height as f64],
            [0.0, height as f64],
        ];
        let xs = corners.map(|[x, y]| m[0] * x + m[1] * y + m[2]);
        let ys = corners.map(|[x, y]| m[3] * x + m[4] * y + m[5]);
        ow = (xs.into_iter().fold(f64::NEG_INFINITY, f64::max).ceil()
            - xs.into_iter().fold(f64::INFINITY, f64::min).floor()) as usize;
        oh = (ys.into_iter().fold(f64::NEG_INFINITY, f64::max).ceil()
            - ys.into_iter().fold(f64::INFINITY, f64::min).floor()) as usize;
        // Image.rotate short-circuits right angles through transpose, whose
        // expanded size swaps axes even when width/height parity differs.
        if angle.rem_euclid(180.0) == 90.0 {
            ow = height;
            oh = width;
        }
        let tx = -(ow as f64 - width as f64) * 0.5;
        let ty = -(oh as f64 - height as f64) * 0.5;
        m[2] += m[0] * tx + m[1] * ty;
        m[5] += m[3] * tx + m[4] * ty;
    }
    (m, ow, oh)
}

fn uniform(rng: &mut RandomStream, min: f32, max: f32) -> f32 {
    min + (max - min) * rng.next_f32()
}

fn homography(from: &[Point2; 4], to: &[Point2; 4]) -> Option<[[f64; 3]; 3]> {
    let mut matrix = [[0.0f64; 9]; 8];
    for (index, (source, target)) in from.iter().zip(to).enumerate() {
        let x = f64::from(source.x);
        let y = f64::from(source.y);
        let u = f64::from(target.x);
        let v = f64::from(target.y);
        let row = index * 2;
        matrix[row] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        matrix[row + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    for pivot in 0..8 {
        let Some((best, value)) = (pivot..8)
            .map(|row| (row, matrix[row][pivot].abs()))
            .max_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
        else {
            return None;
        };
        if value < 1e-10 || !value.is_finite() {
            return None;
        }
        matrix.swap(pivot, best);
        let divisor = matrix[pivot][pivot];
        for column in pivot..9 {
            matrix[pivot][column] /= divisor;
        }
        for row in 0..8 {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..9 {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
        }
    }
    // torchvision solves in f64 then quantizes the eight coefficients to
    // float32 before passing them to Pillow's double-coordinate sampler.
    let coefficient = |i: usize| f64::from(matrix[i][8] as f32);
    Some([
        [coefficient(0), coefficient(1), coefficient(2)],
        [coefficient(3), coefficient(4), coefficient(5)],
        [coefficient(6), coefficient(7), 1.0],
    ])
}

fn apply_homography(matrix: [[f64; 3]; 3], point: [f64; 2]) -> [f64; 2] {
    let denominator = matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2];
    if denominator.abs() < 1e-7 || !denominator.is_finite() {
        [f64::NAN, f64::NAN]
    } else {
        [
            (matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2]) / denominator,
            (matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2]) / denominator,
        ]
    }
}

fn reflect_coordinate(i: isize, len: usize) -> usize {
    if i < 0 {
        (-i) as usize
    } else if i >= len as isize {
        2 * len - i as usize - 2
    } else {
        i as usize
    }
}

fn smooth_field(field: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let radius = ((8.0 * sigma + 1.0) as usize | 1) / 2;
    if radius == 0 {
        return field.to_vec();
    }
    let mut kernel = Vec::with_capacity(radius * 2 + 1);
    for offset in 0..=radius * 2 {
        let distance = offset as f32 - radius as f32;
        kernel.push((-distance * distance / (2.0 * sigma * sigma)).exp());
    }
    let norm = kernel.iter().sum::<f32>();
    for value in &mut kernel {
        *value /= norm;
    }
    let mut horizontal = vec![0.0; field.len()];
    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0;
            for (index, weight) in kernel.iter().enumerate() {
                let coordinate = x as isize + index as isize - radius as isize;
                let source_x = reflect_coordinate(coordinate, width);
                value += field[y * width + source_x] * weight;
            }
            horizontal[y * width + x] = value;
        }
    }
    let mut output = vec![0.0; field.len()];
    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0;
            for (index, weight) in kernel.iter().enumerate() {
                let coordinate = y as isize + index as isize - radius as isize;
                let source_y = reflect_coordinate(coordinate, height);
                value += horizontal[source_y * width + x] * weight;
            }
            output[y * width + x] = value;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{
        ArbitraryRotateConfig, ElasticTransformConfig, PerspectiveConfig, Point2,
        RandomAffineConfig, RandomPerspectiveConfig,
    };
    use crate::pipeline::op::SampleContext;
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{DType, Device, Tensor};
    use rivet_data::random::{OpKey, RandomContext};

    #[test]
    fn tensor_displacement_sampling_matches_local_torchvision() {
        // torchvision 0.29 functional.elastic_transform, uint8 [1,3,4],
        // constant displacement (+0.2, -0.3) pixels, fill=37.
        let read = |_: usize, y: usize, x: usize| Ok((y * 4 + x + 1) as u8);
        let mut values = Vec::new();
        for y in 0..3 {
            for x in 0..4 {
                values.push(
                    super::tensor_sample_value(
                        &read,
                        0,
                        y as f32 - 0.3,
                        x as f32 + 0.2,
                        4,
                        3,
                        crate::transforms::InterpolationMode::Bilinear,
                        37,
                    )
                    .unwrap(),
                );
            }
        }
        assert_eq!(values, [12, 12, 13, 18, 4, 5, 6, 12, 8, 9, 10, 14]);
    }

    #[test]
    fn elastic_smoothing_matches_torchvision_reflection_without_renormalizing() {
        let field: Vec<f32> = (0..30).map(|x| x as f32 / 30.0).collect();
        let output = super::smooth_field(&field, 6, 5, 0.5);
        let expected = [
            0.04992329, 0.07614232, 0.2023261, 0.20723744, 0.46666658, 0.9167432,
        ];
        for (i, expected) in [0, 1, 5, 6, 14, 29].into_iter().zip(expected) {
            assert!((output[i] - expected).abs() < 1e-6);
        }
    }

    fn image() -> ImageSample {
        ImageSample::Decoded(DecodedSample {
            image: Tensor::from_vec(vec![1u8, 2, 3, 4], [2, 2, 1], &Device::Cpu).unwrap(),
            label: 8,
        })
    }

    fn corners() -> [Point2; 4] {
        [
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ]
    }

    #[test]
    fn arbitrary_rotation_identity_preserves_shape_dtype_and_materializes() {
        let input = image().into_decoded().unwrap().image;
        let output = ArbitraryRotateConfig::new(0.0)
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: input.clone(),
                    label: 8,
                }),
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dims(), [2, 2, 1]);
        assert_eq!(output.image.dtype(), DType::U8);
        assert_eq!(output.image.to_vec::<u8>().unwrap(), [1, 2, 3, 4]);
        assert!(!output.image.same_storage(&input));
    }

    #[test]
    fn arbitrary_rotation_expand_swaps_rectangular_geometry() {
        let input = Tensor::from_vec(vec![1u8, 2, 3, 4, 5, 6], [2, 3, 1], &Device::Cpu).unwrap();
        let output = ArbitraryRotateConfig::new(90.0)
            .with_options(true, crate::transforms::InterpolationMode::Nearest, 0)
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: input,
                    label: 8,
                }),
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dims(), [3, 2, 1]);
    }

    #[test]
    fn perspective_identity_supports_chw_and_non_contiguous_input() {
        let base = Tensor::from_vec(vec![1u8, 2, 3, 4], [1, 2, 2], &Device::Cpu).unwrap();
        let input = base.permute(&[0, 2, 1]).unwrap();
        assert!(!input.is_contiguous());
        let output = PerspectiveConfig::new(corners(), corners())
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: input.clone(),
                    label: 8,
                }),
                ImageAxisOrder::Chw,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dims(), [1, 2, 2]);
        assert_eq!(output.image.to_vec::<u8>().unwrap(), [1, 3, 2, 4]);
        assert!(!output.image.same_storage(&input));
    }

    #[test]
    fn random_geometry_is_seeded_and_validates_contracts() {
        let left_ctx = SampleContext::with_random(3, RandomContext::new(19).with_epoch(4));
        let right_ctx = SampleContext::with_random(3, RandomContext::new(19).with_epoch(4));
        let mut left_rng = left_ctx.stream(OpKey::from_parts("RandomAffine", 0));
        let mut right_rng = right_ctx.stream(OpKey::from_parts("RandomAffine", 0));
        let left = RandomAffineConfig::new(15.0)
            .with_translate(0.2, 0.2)
            .apply(image(), &mut left_rng, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        let right = RandomAffineConfig::new(15.0)
            .with_translate(0.2, 0.2)
            .apply(image(), &mut right_rng, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(
            left.image.to_vec::<u8>().unwrap(),
            right.image.to_vec::<u8>().unwrap()
        );

        let perspective_ctx = SampleContext::with_random(3, RandomContext::new(19).with_epoch(4));
        let mut perspective_rng = perspective_ctx.stream(OpKey::from_parts("RandomPerspective", 0));
        let perspective = RandomPerspectiveConfig::new(0.5, 1.0)
            .apply(image(), &mut perspective_rng, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(perspective.image.dims(), [2, 2, 1]);

        let ctx = SampleContext::new(0);
        let mut rng = ctx.stream(OpKey::from_parts("ElasticTransform", 0));
        let elastic = ElasticTransformConfig::new(0.0, 0.1)
            .apply(image(), &mut rng, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(elastic.image.to_vec::<u8>().unwrap(), [1, 2, 3, 4]);

        assert!(RandomAffineConfig::new(-1.0).validate().is_err());
        assert!(RandomPerspectiveConfig::new(1.1, 0.5).validate().is_err());
        assert!(ArbitraryRotateConfig::new(f32::NAN).validate().is_err());
        assert!(
            PerspectiveConfig::new(corners(), corners())
                .with_options(crate::transforms::InterpolationMode::Lanczos3, 0)
                .validate()
                .is_err()
        );
    }
}
