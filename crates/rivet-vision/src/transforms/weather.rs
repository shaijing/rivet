use super::{RivetResult, contiguous_values, invalid_argument};
use rivet_core::{CpuStorageRef, DType, Tensor};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RainType {
    Default,
    Drizzle,
    Heavy,
    Torrential,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RandomRainConfig {
    pub slant_range: (f64, f64),
    pub drop_length: Option<u32>,
    pub drop_width: u32,
    pub drop_color: [u8; 3],
    pub blur_value: u32,
    pub brightness_coefficient: f64,
    pub style: RainType,
}

impl RandomRainConfig {
    pub(super) fn valid(&self) -> bool {
        let (lo, hi) = self.slant_range;
        lo >= -90.0
            && lo <= hi
            && hi <= 90.0
            && self.drop_length != Some(0)
            && (1..=127).contains(&self.drop_width)
            && (1..=127).contains(&self.blur_value)
            && self.blur_value % 2 == 1
            && self.brightness_coefficient > 0.0
            && self.brightness_coefficient <= 1.0
    }
}

pub(super) fn random_rain(
    input: &Tensor,
    config: &RandomRainConfig,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    let [height, width, 3] = *input.dims() else {
        return Err(invalid_argument("RandomRain requires HWC RGB input"));
    };
    if height == 0 || width == 0 || !matches!(input.dtype(), DType::U8 | DType::F32) {
        return Err(invalid_argument(
            "RandomRain requires nonempty uint8 or float32 HWC RGB input",
        ));
    }
    let (density, divisor, opacity) = match config.style {
        RainType::Drizzle => (1200, 12, 0.27),
        RainType::Default => (720, 8, 0.40),
        RainType::Heavy => (360, 6, 0.48),
        RainType::Torrential => (180, 4, 0.55),
    };
    let length = config
        .drop_length
        .map_or((height / divisor).max(1), |v| v as usize);
    let length = length.min(width.saturating_add(height));
    let (lo, hi) = config.slant_range;
    let (dx, dy) = (lo + rng.next_f64() * (hi - lo)).to_radians().sin_cos();
    let mut coverage = vec![0.0f32; height * width];
    let radius = (config.drop_width / 2) as i64;
    for _ in 0..(height * width).div_ceil(density) {
        let x0 = rng.gen_range_usize(0..width)? as f64;
        let y0 = rng.gen_range_usize(0..height)? as f64;
        for step in 0..=length {
            let x = (x0 + step as f64 * dx).round() as i64;
            let y = (y0 + step as f64 * dy).round() as i64;
            if y >= height as i64 || x + radius < 0 || x - radius >= width as i64 {
                break;
            }
            // Deposit one horizontal brush stroke per step. Max composition
            // keeps thickness and opacity independent of raster sampling.
            let left = (x - radius).max(0) as usize;
            let right = (x - radius + i64::from(config.drop_width)).clamp(0, width as i64) as usize;
            if left < right {
                coverage[y as usize * width + left..y as usize * width + right].fill(opacity);
            }
        }
    }
    let input = input.contiguous()?;
    let streaks = input.with_cpu_storage(|storage, layout| {
        macro_rules! overlay {
            ($values:expr, $scale:expr, $finish:expr) => {{
                let pixels = contiguous_values($values, layout)?;
                let color = config.drop_color.map(|v| f64::from(v) * $scale);
                let finish = $finish;
                Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                    for (pixel, &alpha) in pixels.chunks_exact(3).zip(&coverage) {
                        let alpha = f64::from(alpha);
                        for (&v, target) in pixel.iter().zip(color) {
                            let v = f64::from(v);
                            out.write_next(finish(
                                (v + alpha * (target - v)) * config.brightness_coefficient,
                            ))?;
                        }
                    }
                    Ok(())
                })
            }};
        }
        match storage {
            CpuStorageRef::U8(values) => overlay!(values, 1.0, |v: f64| v.round() as u8),
            CpuStorageRef::F32(values) => {
                overlay!(values, 1.0 / 255.0, |v: f64| v.clamp(0.0, 1.0) as f32)
            }
            _ => unreachable!(),
        }
    })?;
    if config.blur_value == 1 {
        Ok(streaks)
    } else {
        super::blur::box_blur(&streaks, config.blur_value as usize)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnowMethod {
    Bleach,
    Texture,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomSnowConfig {
    pub snow_point_range: (f64, f64),
    pub brightness_coeff: f64,
    pub method: SnowMethod,
}

impl RandomSnowConfig {
    pub(super) fn valid(&self) -> bool {
        let (low, high) = self.snow_point_range;
        0.0 <= low
            && low <= high
            && high <= 1.0
            && self.brightness_coeff.is_finite()
            && self.brightness_coeff >= 1.0
    }
}

pub(super) fn random_snow(
    input: &Tensor,
    config: &RandomSnowConfig,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    let [height, width, 3] = *input.dims() else {
        return Err(invalid_argument("RandomSnow requires HWC RGB input"));
    };
    if height == 0 || width == 0 || !matches!(input.dtype(), DType::U8 | DType::F32) {
        return Err(invalid_argument(
            "RandomSnow requires nonempty uint8 or float32 HWC RGB input",
        ));
    }
    let (low, high) = config.snow_point_range;
    let point = low + rng.next_f64() * (high - low);
    if point == 0.0 {
        return Ok(input.clone());
    }
    match config.method {
        SnowMethod::Bleach => write_weather_rgb(input, |_, _, rgb| {
            let lo = rgb.into_iter().fold(1.0, f64::min);
            let hi = rgb.into_iter().fold(0.0, f64::max);
            let light = (lo + hi) * 0.5;
            if light <= point || light >= 1.0 {
                return rgb;
            }
            let lit = (light * config.brightness_coeff).min(1.0);
            let chroma = (1.0 - (2.0 * lit - 1.0).abs()) / (1.0 - (2.0 * light - 1.0).abs());
            rgb.map(|v| lit + (v - light) * chroma)
        }),
        SnowMethod::Texture => {
            let noise = Tensor::from_exact_iter(
                (0..height * width).map(|_| {
                    (0.55 + 0.25 * super::noise::normal_pair(rng).0).clamp(0.0, 1.0) as f32
                }),
                [height, width, 1],
                input.device(),
            )?;
            let noise = super::blur::box_blur(&noise, 3)?;
            let values = noise.to_vec::<f32>()?;
            let gain = 1.0 + (config.brightness_coeff - 1.0) * point;
            write_weather_rgb(input, |x, y, rgb| {
                let depth = 1.0 - 0.8 * y as f64 / (height - 1).max(1) as f64;
                let sparkle = if rng.next_f64() < 0.015 { 0.45 } else { 0.0 };
                let alpha = (point * depth * (f64::from(values[y * width + x]) + sparkle)).min(1.0);
                let lit = rgb.map(|v| (v * gain).min(1.0));
                std::array::from_fn(|c| lit[c] + alpha * ([0.89, 0.95, 1.0][c] - lit[c]))
            })
        }
    }
}

fn write_weather_rgb(
    input: &Tensor,
    mut pixel: impl FnMut(usize, usize, [f64; 3]) -> [f64; 3],
) -> RivetResult<Tensor> {
    let width = input.dims()[1];
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| {
        macro_rules! write_pixels {
            ($values:expr, $scale:expr, $finish:expr) => {{
                let values = contiguous_values($values, layout)?;
                let finish = $finish;
                Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                    for (y, row) in values.chunks_exact(width * 3).enumerate() {
                        for (x, rgb) in row.chunks_exact(3).enumerate() {
                            let rgb = [rgb[0], rgb[1], rgb[2]].map(|v| f64::from(v) * $scale);
                            for v in pixel(x, y, rgb) {
                                out.write_next(finish(v.clamp(0.0, 1.0)))?;
                            }
                        }
                    }
                    Ok(())
                })
            }};
        }
        match storage {
            CpuStorageRef::U8(values) => {
                write_pixels!(values, 1.0 / 255.0, |v: f64| (v * 255.0).round() as u8)
            }
            CpuStorageRef::F32(values) => write_pixels!(values, 1.0, |v| v as f32),
            _ => unreachable!(),
        }
    })?)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomGravelConfig {
    pub gravel_roi: (f64, f64, f64, f64),
    pub number_of_patches: u32,
}

impl RandomGravelConfig {
    pub(super) fn valid(&self) -> bool {
        let (left, top, right, bottom) = self.gravel_roi;
        0.0 <= left
            && left < right
            && right <= 1.0
            && 0.0 <= top
            && top < bottom
            && bottom <= 1.0
            && self.number_of_patches <= 128
    }
}

pub(super) fn random_gravel(
    input: &Tensor,
    config: &RandomGravelConfig,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    let [height, width, 3] = *input.dims() else {
        return Err(invalid_argument("RandomGravel requires HWC RGB input"));
    };
    if height == 0 || width == 0 || !matches!(input.dtype(), DType::U8 | DType::F32) {
        return Err(invalid_argument(
            "RandomGravel requires nonempty uint8 or float32 HWC RGB input",
        ));
    }
    let (left, top, right, bottom) = config.gravel_roi;
    let axis = |low: f64, high: f64, length: usize| {
        let extent = (length - 1) as f64;
        (low * extent).ceil() as usize..(high * extent).floor() as usize + 1
    };
    let xs = axis(left, right, width);
    let ys = axis(top, bottom, height);
    if config.number_of_patches == 0 || xs.is_empty() || ys.is_empty() {
        return Ok(input.clone());
    }
    let patch_width = xs.len().div_ceil(4);
    let patch_height = ys.len().div_ceil(4);
    let mut overlay = vec![(0.0_f32, 0.0_f32); xs.len() * ys.len()];
    for _ in 0..config.number_of_patches {
        let px = rng.gen_range_usize(xs.start..xs.end - patch_width + 1)?;
        let py = rng.gen_range_usize(ys.start..ys.end - patch_height + 1)?;
        let particles = (patch_width * patch_height).div_ceil(20).min(1024);
        for _ in 0..particles {
            let x = rng.gen_range_usize(px..px + patch_width)?;
            let y = rng.gen_range_usize(py..py + patch_height)?;
            let radius = rng.gen_range_usize(1..4)?;
            let shade = (0.12 + 0.76 * rng.next_f64()) as f32;
            for gy in y.saturating_sub(radius).max(ys.start)..(y + radius + 1).min(ys.end) {
                for gx in x.saturating_sub(radius).max(xs.start)..(x + radius + 1).min(xs.end) {
                    let dist = gx.abs_diff(x).pow(2) + gy.abs_diff(y).pow(2);
                    if dist <= radius * radius {
                        let alpha = 0.85f32 * (1.0 - dist as f32 / (radius * radius + 1) as f32);
                        let (coverage, premultiplied) =
                            &mut overlay[(gy - ys.start) * xs.len() + gx - xs.start];
                        *premultiplied = shade * alpha + *premultiplied * (1.0 - alpha);
                        *coverage += alpha * (1.0 - *coverage);
                    }
                }
            }
        }
    }
    write_weather_rgb(input, |x, y, rgb| {
        if !xs.contains(&x) || !ys.contains(&y) {
            return rgb;
        }
        let (alpha, shade) = overlay[(y - ys.start) * xs.len() + x - xs.start];
        rgb.map(|v| v * (1.0 - f64::from(alpha)) + f64::from(shade))
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatterMode {
    Rain,
    Mud,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpatterConfig {
    pub mean_range: (f64, f64),
    pub std_range: (f64, f64),
    pub gauss_sigma_range: (f64, f64),
    pub cutout_threshold_range: (f64, f64),
    pub intensity_range: (f64, f64),
    pub mode: SpatterMode,
    pub color: Option<[u8; 3]>,
}

impl SpatterConfig {
    pub(super) fn valid(&self) -> bool {
        [
            self.mean_range,
            self.std_range,
            self.cutout_threshold_range,
            self.intensity_range,
        ]
        .into_iter()
        .all(|(lo, hi)| lo >= 0.0 && lo <= hi && hi <= 1.0)
            && self.gauss_sigma_range.0 >= 0.0
            && self.gauss_sigma_range.0 <= self.gauss_sigma_range.1
            && self.gauss_sigma_range.1 <= 10.0
    }
}

pub(super) fn spatter(
    input: &Tensor,
    config: &SpatterConfig,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    let [height, width, 3] = *input.dims() else {
        return Err(invalid_argument("Spatter requires HWC RGB input"));
    };
    if height == 0 || width == 0 || !matches!(input.dtype(), DType::U8 | DType::F32) {
        return Err(invalid_argument(
            "Spatter requires nonempty uint8 or float32 HWC RGB input",
        ));
    }
    let mut sample = |(lo, hi)| lo + rng.next_f64() * (hi - lo);
    let intensity = sample(config.intensity_range);
    if intensity == 0.0 {
        return Ok(input.clone());
    }
    let mean = sample(config.mean_range);
    let std = sample(config.std_range);
    let threshold = sample(config.cutout_threshold_range);
    let sigma = sample(config.gauss_sigma_range);
    let mut mask: Vec<_> = (0..height * width)
        .map(|_| {
            let liquid = (mean + std * super::noise::normal_pair(rng).0).clamp(0.0, 1.0);
            if liquid > threshold { 1.0f32 } else { 0.0 }
        })
        .collect();
    if sigma > 0.0 {
        smooth_spatter_mask(&mut mask, width, height, sigma);
    }
    let color = config
        .color
        .unwrap_or(match config.mode {
            SpatterMode::Rain => [238, 238, 175],
            SpatterMode::Mud => [63, 42, 20],
        })
        .map(|v| f64::from(v) / 255.0);
    write_weather_rgb(input, |x, y, rgb| {
        let amount = f64::from(mask[y * width + x]).clamp(0.0, 1.0);
        match config.mode {
            SpatterMode::Mud => {
                std::array::from_fn(|c| rgb[c] + intensity * amount * (color[c] - rgb[c]))
            }
            SpatterMode::Rain => {
                let east = mask[y * width + (x + 1).min(width - 1)];
                let south = mask[(y + 1).min(height - 1) * width + x];
                let rim = f64::from(
                    (mask[y * width + x] - east).abs() + (mask[y * width + x] - south).abs(),
                );
                let light = intensity * (0.35 * amount + 0.65 * rim.min(1.0));
                std::array::from_fn(|c| rgb[c] + light * color[c] * (1.0 - rgb[c]))
            }
        }
    })
}

fn smooth_spatter_mask(mask: &mut [f32], width: usize, height: usize, sigma: f64) {
    let radius = (3.0 * sigma).ceil() as i64;
    let mut kernel: Vec<_> = (-radius..=radius)
        .map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp())
        .collect();
    let total = kernel.iter().sum::<f64>();
    for v in &mut kernel {
        *v /= total;
    }
    let reflect = super::blur::reflect101;
    let mut rows = vec![0.0_f32; mask.len()];
    for y in 0..height {
        for x in 0..width {
            rows[y * width + x] = kernel
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    v * f64::from(mask[y * width + reflect(x as i64 + i as i64 - radius, width)])
                })
                .sum::<f64>() as f32;
        }
    }
    for y in 0..height {
        for x in 0..width {
            mask[y * width + x] = kernel
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    v * f64::from(rows[reflect(y as i64 + i as i64 - radius, height) * width + x])
                })
                .sum::<f64>() as f32;
        }
    }
}
