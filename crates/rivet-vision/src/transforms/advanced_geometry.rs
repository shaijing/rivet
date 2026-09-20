use super::logical_offset;
use super::resize::InterpolationMode;
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::pipeline::op::SampleContext;
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use rivet_core::{CpuStorageRef, DType, Tensor};

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
    /// Positive angles rotate clockwise in image coordinates (x right, y down).
    pub angle: f32,
    pub expand: bool,
    pub interpolation: InterpolationMode,
    pub fill: u8,
}

impl ArbitraryRotateConfig {
    pub const fn new(angle: f32) -> Self {
        Self {
            angle,
            expand: true,
            interpolation: InterpolationMode::Bilinear,
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
        let angle = self.angle.to_radians();
        let matrix = rotation_matrix(angle);
        let source_center = center(width, height);
        let (output_width, output_height, destination_center) = if self.expand {
            expanded_rotation_shape(width, height, matrix)?
        } else {
            (width, height, source_center)
        };
        let translation = subtract(destination_center, matrix_point(matrix, source_center));
        warp_affine(
            sample,
            axis_order,
            output_width,
            output_height,
            matrix,
            translation,
            self.interpolation,
            self.fill,
            "arbitrary rotation",
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
            interpolation: InterpolationMode::Bilinear,
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
        ctx: &mut SampleContext,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = require_u8_image(sample, "random affine")?;
        let dims = sample.image.dims();
        let (height, width, _) = image_dims(dims, axis_order)?;
        let angle = uniform(ctx, -self.degrees, self.degrees).to_radians();
        let translate_x =
            uniform(ctx, -self.translate[0], self.translate[0]) * width.saturating_sub(1) as f32;
        let translate_y =
            uniform(ctx, -self.translate[1], self.translate[1]) * height.saturating_sub(1) as f32;
        let scale = uniform(ctx, self.scale[0], self.scale[1]);
        let shear_x = uniform(ctx, -self.shear[0], self.shear[0]).to_radians();
        let shear_y = uniform(ctx, -self.shear[1], self.shear[1]).to_radians();
        let matrix = multiply_matrix(
            rotation_matrix(angle),
            multiply_matrix(
                [[scale, 0.0], [0.0, scale]],
                [[1.0, shear_x.tan()], [shear_y.tan(), 1.0]],
            ),
        );
        let center = center(width, height);
        let translation = add(
            subtract(center, matrix_point(matrix, center)),
            [translate_x, translate_y],
        );
        warp_affine(
            sample,
            axis_order,
            width,
            height,
            matrix,
            translation,
            self.interpolation,
            self.fill,
            "random affine",
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
        ctx: &mut SampleContext,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = sample.into_decoded()?;
        if ctx.next_rng_f64() >= self.probability {
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
        let half_x = max_x * self.distortion_scale * 0.5;
        let half_y = max_y * self.distortion_scale * 0.5;
        let corners = [
            Point2::new(0.0, 0.0),
            Point2::new(max_x, 0.0),
            Point2::new(max_x, max_y),
            Point2::new(0.0, max_y),
        ];
        let mut end_points = corners;
        for point in &mut end_points {
            point.x += uniform(ctx, -half_x, half_x);
            point.y += uniform(ctx, -half_y, half_y);
        }
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
    /// Maximum displacement in pixels after smoothing.
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
        validate_interpolation(self.interpolation, "elastic transform")
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        ctx: &mut SampleContext,
        axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = require_u8_image(sample, "elastic transform")?;
        let dims = sample.image.dims();
        let (height, width, _) = image_dims(dims, axis_order)?;
        let count = height
            .checked_mul(width)
            .ok_or_else(|| invalid_shape("elastic transform image is too large"))?;
        let seed_x = ctx.next_rng_u64();
        let seed_y = ctx.next_rng_u64();
        let mut x_field = (0..count)
            .map(|index| hash_unit(seed_x, index))
            .collect::<Vec<_>>();
        let mut y_field = (0..count)
            .map(|index| hash_unit(seed_y, index))
            .collect::<Vec<_>>();
        if self.sigma > 0.0 {
            x_field = smooth_field(&x_field, width, height, self.sigma);
            y_field = smooth_field(&y_field, width, height, self.sigma);
        }
        let alpha = self.alpha;
        warp_decoded(
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
                    x as f32 + alpha * x_field[index],
                    y as f32 + alpha * y_field[index],
                ]
            },
        )
        .map(ImageSample::Decoded)
    }
}

fn validate_interpolation(mode: InterpolationMode, op_name: &str) -> RivetResult<()> {
    match mode {
        InterpolationMode::Nearest | InterpolationMode::Bilinear => Ok(()),
        InterpolationMode::Bicubic | InterpolationMode::Lanczos3 => Err(invalid_argument(format!(
            "{op_name} supports only nearest or bilinear interpolation"
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

fn warp_affine(
    sample: DecodedSample,
    axis_order: ImageAxisOrder,
    output_width: usize,
    output_height: usize,
    matrix: [[f32; 2]; 2],
    translation: [f32; 2],
    interpolation: InterpolationMode,
    fill: u8,
    op_name: &str,
) -> RivetResult<DecodedSample> {
    let inverse = inverse_matrix(matrix)
        .ok_or_else(|| invalid_argument(format!("{op_name} transform matrix is singular")))?;
    warp_decoded(
        sample,
        axis_order,
        output_width,
        output_height,
        interpolation,
        fill,
        op_name,
        move |x, y| {
            let destination = [x as f32, y as f32];
            let relative = subtract(destination, translation);
            matrix_point(inverse, relative)
        },
    )
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
    warp_decoded(
        sample,
        axis_order,
        width,
        height,
        interpolation,
        fill,
        op_name,
        move |x, y| apply_homography(inverse, [x as f32, y as f32]),
    )
}

fn warp_decoded<F>(
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
    F: FnMut(usize, usize) -> [f32; 2],
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
        let mut output = Vec::with_capacity(
            output_width
                .checked_mul(output_height)
                .and_then(|pixels| pixels.checked_mul(channels))
                .ok_or(rivet_core::Error::StorageOutOfBounds)?,
        );
        let mut append_pixel = |channel: usize, y: usize, x: usize| {
            let source = map(x, y);
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
        match axis_order {
            ImageAxisOrder::Hwc => {
                for y in 0..output_height {
                    for x in 0..output_width {
                        for channel in 0..channels {
                            output.push(append_pixel(channel, y, x)?);
                        }
                    }
                }
            }
            ImageAxisOrder::Chw => {
                for channel in 0..channels {
                    for y in 0..output_height {
                        for x in 0..output_width {
                            output.push(append_pixel(channel, y, x)?);
                        }
                    }
                }
            }
        }
        Ok(output)
    })?;
    let output_dims = match axis_order {
        ImageAxisOrder::Hwc => vec![output_height, output_width, channels],
        ImageAxisOrder::Chw => vec![channels, output_height, output_width],
    };
    Ok(DecodedSample {
        image: Tensor::from_vec(values, output_dims, input.device())?,
        label: sample.label,
    })
}

fn sample_value<F>(
    read: &F,
    channel: usize,
    y: f32,
    x: f32,
    width: usize,
    height: usize,
    interpolation: InterpolationMode,
    fill: u8,
) -> rivet_core::Result<u8>
where
    F: Fn(usize, usize, usize) -> rivet_core::Result<u8>,
{
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x > width.saturating_sub(1) as f32
        || y > height.saturating_sub(1) as f32
    {
        return Ok(fill);
    }
    if interpolation == InterpolationMode::Nearest {
        return read(channel, y.round() as usize, x.round() as usize);
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let wx = x - x0 as f32;
    let wy = y - y0 as f32;
    let top =
        f32::from(read(channel, y0, x0)?) * (1.0 - wx) + f32::from(read(channel, y0, x1)?) * wx;
    let bottom =
        f32::from(read(channel, y1, x0)?) * (1.0 - wx) + f32::from(read(channel, y1, x1)?) * wx;
    Ok((top * (1.0 - wy) + bottom * wy).round().clamp(0.0, 255.0) as u8)
}

fn expanded_rotation_shape(
    width: usize,
    height: usize,
    matrix: [[f32; 2]; 2],
) -> RivetResult<(usize, usize, [f32; 2])> {
    let center = center(width, height);
    let corners = [
        [0.0, 0.0],
        [width.saturating_sub(1) as f32, 0.0],
        [
            width.saturating_sub(1) as f32,
            height.saturating_sub(1) as f32,
        ],
        [0.0, height.saturating_sub(1) as f32],
    ];
    let transformed =
        corners.map(|corner| add(matrix_point(matrix, subtract(corner, center)), center));
    let min_x = transformed
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min);
    let max_x = transformed
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let min_y = transformed
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min);
    let max_y = transformed
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max);
    let output_width = usize::try_from((max_x - min_x + 1.0).ceil() as u64)
        .map_err(|_| invalid_shape("arbitrary rotation output width is too large"))?
        .max(1);
    let output_height = usize::try_from((max_y - min_y + 1.0).ceil() as u64)
        .map_err(|_| invalid_shape("arbitrary rotation output height is too large"))?
        .max(1);
    Ok((
        output_width,
        output_height,
        [
            (output_width.saturating_sub(1) as f32) * 0.5,
            (output_height.saturating_sub(1) as f32) * 0.5,
        ],
    ))
}

fn center(width: usize, height: usize) -> [f32; 2] {
    [
        width.saturating_sub(1) as f32 * 0.5,
        height.saturating_sub(1) as f32 * 0.5,
    ]
}

fn rotation_matrix(angle: f32) -> [[f32; 2]; 2] {
    let (sin, cos) = angle.sin_cos();
    [[cos, -sin], [sin, cos]]
}

fn matrix_point(matrix: [[f32; 2]; 2], point: [f32; 2]) -> [f32; 2] {
    [
        matrix[0][0] * point[0] + matrix[0][1] * point[1],
        matrix[1][0] * point[0] + matrix[1][1] * point[1],
    ]
}

fn multiply_matrix(lhs: [[f32; 2]; 2], rhs: [[f32; 2]; 2]) -> [[f32; 2]; 2] {
    [
        [
            lhs[0][0] * rhs[0][0] + lhs[0][1] * rhs[1][0],
            lhs[0][0] * rhs[0][1] + lhs[0][1] * rhs[1][1],
        ],
        [
            lhs[1][0] * rhs[0][0] + lhs[1][1] * rhs[1][0],
            lhs[1][0] * rhs[0][1] + lhs[1][1] * rhs[1][1],
        ],
    ]
}

fn inverse_matrix(matrix: [[f32; 2]; 2]) -> Option<[[f32; 2]; 2]> {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if !determinant.is_finite() || determinant.abs() < 1e-6 {
        return None;
    }
    Some([
        [matrix[1][1] / determinant, -matrix[0][1] / determinant],
        [-matrix[1][0] / determinant, matrix[0][0] / determinant],
    ])
}

fn add(lhs: [f32; 2], rhs: [f32; 2]) -> [f32; 2] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1]]
}

fn subtract(lhs: [f32; 2], rhs: [f32; 2]) -> [f32; 2] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1]]
}

fn uniform(ctx: &mut SampleContext, min: f32, max: f32) -> f32 {
    min + (max - min) * ctx.next_rng_f64() as f32
}

fn homography(from: &[Point2; 4], to: &[Point2; 4]) -> Option<[[f32; 3]; 3]> {
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
    Some([
        [
            matrix[0][8] as f32,
            matrix[1][8] as f32,
            matrix[2][8] as f32,
        ],
        [
            matrix[3][8] as f32,
            matrix[4][8] as f32,
            matrix[5][8] as f32,
        ],
        [matrix[6][8] as f32, matrix[7][8] as f32, 1.0],
    ])
}

fn apply_homography(matrix: [[f32; 3]; 3], point: [f32; 2]) -> [f32; 2] {
    let denominator = matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2];
    if denominator.abs() < 1e-7 || !denominator.is_finite() {
        [f32::NAN, f32::NAN]
    } else {
        [
            (matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2]) / denominator,
            (matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2]) / denominator,
        ]
    }
}

fn hash_unit(seed: u64, index: usize) -> f32 {
    let mut value = seed.wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let value = value ^ (value >> 31);
    ((value >> 11) as f32 * (1.0 / (1u64 << 53) as f32)) * 2.0 - 1.0
}

fn smooth_field(field: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let radius = (sigma * 3.0).ceil() as usize;
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
                let source_x = x
                    .saturating_add(index)
                    .saturating_sub(radius)
                    .min(width - 1);
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
                let source_y = y
                    .saturating_add(index)
                    .saturating_sub(radius)
                    .min(height - 1);
                value += horizontal[source_y * width + x] * weight;
            }
            output[y * width + x] = value;
        }
    }
    let max_abs = output
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    if max_abs > 0.0 {
        for value in &mut output {
            *value /= max_abs;
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
        let mut left = SampleContext::new(3);
        left.global_seed = 19;
        left.epoch = 4;
        let mut right = SampleContext::new(3);
        right.global_seed = 19;
        right.epoch = 4;
        let left = RandomAffineConfig::new(15.0)
            .with_translate(0.2, 0.2)
            .apply(image(), &mut left, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        let right = RandomAffineConfig::new(15.0)
            .with_translate(0.2, 0.2)
            .apply(image(), &mut right, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(
            left.image.to_vec::<u8>().unwrap(),
            right.image.to_vec::<u8>().unwrap()
        );

        let mut perspective_ctx = SampleContext::new(3);
        perspective_ctx.global_seed = 19;
        perspective_ctx.epoch = 4;
        let perspective = RandomPerspectiveConfig::new(0.5, 1.0)
            .apply(image(), &mut perspective_ctx, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(perspective.image.dims(), [2, 2, 1]);

        let mut ctx = SampleContext::new(0);
        let elastic = ElasticTransformConfig::new(0.0, 1.0)
            .apply(image(), &mut ctx, ImageAxisOrder::Hwc)
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(elastic.image.to_vec::<u8>().unwrap(), [1, 2, 3, 4]);

        assert!(RandomAffineConfig::new(-1.0).validate().is_err());
        assert!(RandomPerspectiveConfig::new(1.1, 0.5).validate().is_err());
        assert!(ArbitraryRotateConfig::new(f32::NAN).validate().is_err());
        assert!(
            PerspectiveConfig::new(corners(), corners())
                .with_options(crate::transforms::InterpolationMode::Bicubic, 0)
                .validate()
                .is_err()
        );
    }
}
