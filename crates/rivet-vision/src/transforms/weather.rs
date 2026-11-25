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
