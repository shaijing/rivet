use crate::errors::{RivetResult, invalid_argument};
use crate::transforms::degradation::weather::write_weather_rgb;
use rivet_core::{DType, Tensor};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomFogConfig {
    pub fog_coef_range: (f64, f64),
    pub alpha_coef: f64,
}

impl RandomFogConfig {
    pub(in crate::transforms) fn valid(&self) -> bool {
        let (lo, hi) = self.fog_coef_range;
        lo >= 0.0 && lo <= hi && hi <= 1.0 && (0.0..=1.0).contains(&self.alpha_coef)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FogDepthMode {
    Linear,
    Diagonal,
    Radial,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtmosphericFogConfig {
    pub density_range: (f64, f64),
    pub fog_color: [u8; 3],
    pub depth_mode: FogDepthMode,
}

impl AtmosphericFogConfig {
    pub(in crate::transforms) fn valid(&self) -> bool {
        let (lo, hi) = self.density_range;
        lo >= 0.0 && lo <= hi && hi.is_finite()
    }
}

fn fog_shape(input: &Tensor, name: &str) -> RivetResult<(usize, usize)> {
    match input.dims() {
        &[h, w, 3] if h > 0 && w > 0 && matches!(input.dtype(), DType::U8 | DType::F32) => {
            Ok((h, w))
        }
        _ => Err(invalid_argument(format!(
            "{name} requires nonempty uint8 or float32 HWC RGB input"
        ))),
    }
}

pub(in crate::transforms) fn random_fog(
    input: &Tensor,
    config: &RandomFogConfig,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    let (height, width) = fog_shape(input, "RandomFog")?;
    let (lo, hi) = config.fog_coef_range;
    let strength = lo + rng.next_f64() * (hi - lo);
    let opacity = strength * config.alpha_coef;
    if opacity == 0.0 {
        return Ok(input.clone());
    }
    let mut coverage = vec![0.0_f64; height * width];
    let base_radius = height.min(width) as f64 * (0.15 + 0.2 * strength);
    let count = 4 + (24.0 * strength).round() as usize;
    for _ in 0..count {
        let cx = rng.next_f64() * (width - 1) as f64;
        let cy = rng.next_f64() * (height - 1) as f64;
        let radius = (base_radius * (0.65 + rng.next_f64())).max(1.0);
        let x0 = (cx - radius).max(0.0) as usize;
        let y0 = (cy - radius).max(0.0) as usize;
        let x1 = ((cx + radius).ceil() as usize).min(width - 1);
        let y1 = ((cy + radius).ceil() as usize).min(height - 1);
        let inverse_radius_sq = radius.powi(2).recip();
        for y in y0..=y1 {
            let dy = (y as f64 - cy).powi(2);
            for x in x0..=x1 {
                let distance = ((x as f64 - cx).powi(2) + dy) * inverse_radius_sq;
                let alpha = opacity * (1.0 - distance).max(0.0).powi(2);
                let accumulated = &mut coverage[y * width + x];
                *accumulated += (1.0 - *accumulated) * alpha;
            }
        }
    }
    write_weather_rgb(input, |x, y, rgb| {
        let alpha = coverage[y * width + x];
        rgb.map(|v| v + (1.0 - v) * alpha)
    })
}

pub(in crate::transforms) fn atmospheric_fog(
    input: &Tensor,
    config: &AtmosphericFogConfig,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    let (height, width) = fog_shape(input, "AtmosphericFog")?;
    let (lo, hi) = config.density_range;
    let density = lo + rng.next_f64() * (hi - lo);
    if density == 0.0 {
        return Ok(input.clone());
    }
    let color = config.fog_color.map(|v| f64::from(v) / 255.0);
    let dx = 1.0 / (width - 1).max(1) as f64;
    let dy = 1.0 / (height - 1).max(1) as f64;
    write_weather_rgb(input, |x, y, rgb| {
        let x = if width == 1 {
            0.5
        } else {
            (x as f64 * dx).min(1.0)
        };
        let y = if height == 1 {
            0.5
        } else {
            (y as f64 * dy).min(1.0)
        };
        let depth = match config.depth_mode {
            FogDepthMode::Linear => 1.0 - y,
            FogDepthMode::Diagonal => 1.0 - (x + y) * 0.5,
            FogDepthMode::Radial => (x - 0.5).hypot(y - 0.5) * std::f64::consts::SQRT_2,
        }
        .min(1.0);
        let alpha = -(-density * depth).exp_m1();
        std::array::from_fn(|c| rgb[c] + alpha * (color[c] - rgb[c]))
    })
}
