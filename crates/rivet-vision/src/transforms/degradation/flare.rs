use crate::errors::{RivetResult, invalid_argument};
use rivet_core::{CpuStorageRef, DType, Tensor};
use rivet_data::random::RandomStream;

type FlareRoi = (f64, f64, f64, f64);

#[derive(Clone, Debug, PartialEq)]
pub struct RandomSunFlareConfig {
    pub flare_roi: (f64, f64, f64, f64),
    pub angle_range: (f64, f64),
    pub num_flare_circles_range: (u32, u32),
    pub src_radius: u32,
    pub src_color: (u8, u8, u8),
    pub physics_based: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LensFlareConfig {
    pub flare_roi: (f64, f64, f64, f64),
    pub num_ghosts_range: (u32, u32),
    pub intensity_range: (f64, f64),
    pub num_rays_range: (u32, u32),
    pub bloom_range: (f64, f64),
}

fn flare_roi_valid((x0, y0, x1, y1): FlareRoi) -> bool {
    0.0 <= x0 && x0 <= x1 && x1 <= 1.0 && 0.0 <= y0 && y0 <= y1 && y1 <= 1.0
}

fn flare_range_valid((lo, hi): (f64, f64)) -> bool {
    0.0 <= lo && lo <= hi && hi <= 1.0
}

fn flare_count_valid((lo, hi): (u32, u32)) -> bool {
    lo <= hi && hi <= 64
}

pub(in crate::transforms) fn random_sun_flare_valid(c: &RandomSunFlareConfig) -> bool {
    flare_roi_valid(c.flare_roi)
        && flare_range_valid(c.angle_range)
        && flare_count_valid(c.num_flare_circles_range)
        && c.src_radius > 0
}

pub(in crate::transforms) fn lens_flare_valid(c: &LensFlareConfig) -> bool {
    flare_roi_valid(c.flare_roi)
        && flare_range_valid(c.intensity_range)
        && flare_range_valid(c.bloom_range)
        && c.bloom_range.0 > 0.0
        && flare_count_valid(c.num_ghosts_range)
        && flare_count_valid(c.num_rays_range)
}

fn flare_sample((lo, hi): (f64, f64), rng: &mut RandomStream) -> f64 {
    lo + rng.next_f64() * (hi - lo)
}

fn flare_count((lo, hi): (u32, u32), rng: &mut RandomStream) -> RivetResult<usize> {
    Ok(rng.gen_range_usize(lo as usize..hi as usize + 1)?)
}

struct FlareCircle {
    x: f64,
    y: f64,
    radius: f64,
    alpha: f64,
    color: [f64; 3],
}

fn flare_source(
    input: &Tensor,
    name: &str,
    (x0, y0, x1, y1): FlareRoi,
    rng: &mut RandomStream,
) -> RivetResult<(f64, f64, f64, f64)> {
    let [h, w, 3] = *input.dims() else {
        return Err(invalid_argument(format!("{name} requires HWC RGB input")));
    };
    if h == 0 || w == 0 || !matches!(input.dtype(), DType::U8 | DType::F32) {
        return Err(invalid_argument(format!(
            "{name} requires nonempty uint8 or float32 HWC RGB input"
        )));
    }
    let (x, y) = ((w - 1) as f64, (h - 1) as f64);
    Ok((
        x * flare_sample((x0, x1), rng),
        y * flare_sample((y0, y1), rng),
        x,
        y,
    ))
}

pub(in crate::transforms) fn random_sun_flare(
    input: &Tensor,
    c: &RandomSunFlareConfig,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    let (x, y, w, h) = flare_source(input, "RandomSunFlare", c.flare_roi, rng)?;
    let diagonal = w.hypot(h).max(1.0);
    let angle = flare_sample(c.angle_range, rng) * std::f64::consts::TAU;
    let (dy, dx) = angle.sin_cos();
    let count = flare_count(c.num_flare_circles_range, rng)?;
    let (r, g, b) = c.src_color;
    let color = [r, g, b].map(|v| f64::from(v) / 255.0);
    let mut circles = vec![FlareCircle {
        x,
        y,
        radius: f64::from(c.src_radius),
        alpha: 1.0,
        color,
    }];
    for _ in 0..count {
        let distance = diagonal * flare_sample((-0.4, 0.8), rng);
        circles.push(FlareCircle {
            x: x + distance * dx,
            y: y + distance * dy,
            radius: (diagonal * flare_sample((0.015, 0.12), rng)).max(0.5),
            alpha: flare_sample((0.05, 0.28), rng),
            color,
        });
    }
    let rays = if c.physics_based { 6 } else { 0 };
    render_flare(input, &circles, rays, angle, c.physics_based)
}

pub(in crate::transforms) fn lens_flare(
    input: &Tensor,
    c: &LensFlareConfig,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    let (x, y, w, h) = flare_source(input, "LensFlare", c.flare_roi, rng)?;
    let strength = flare_sample(c.intensity_range, rng);
    if strength == 0.0 {
        return Ok(input.clone());
    }
    let bloom = (w.hypot(h) * flare_sample(c.bloom_range, rng)).max(0.5);
    let count = flare_count(c.num_ghosts_range, rng)?;
    let rays = flare_count(c.num_rays_range, rng)?;
    let angle = rng.next_f64() * std::f64::consts::TAU;
    let mut circles = vec![FlareCircle {
        x,
        y,
        radius: bloom,
        alpha: strength,
        color: [1.0, 0.95, 0.85],
    }];
    for i in 0..count {
        let offset = (i + 1) as f64 / (count + 1) as f64;
        let distance = 2.0 * offset + flare_sample((0.0, 0.15), rng);
        circles.push(FlareCircle {
            x: x + (w * 0.5 - x) * distance,
            y: y + (h * 0.5 - y) * distance,
            radius: (bloom * (1.2 - offset)).max(0.5),
            alpha: strength * (0.35 - 0.2 * offset),
            color: [
                1.0 - 0.6 * offset,
                0.65 + 0.3 * offset,
                0.45 + 0.55 * offset,
            ],
        });
    }
    render_flare(input, &circles, rays, angle, true)
}

fn render_flare(
    input: &Tensor,
    circles: &[FlareCircle],
    rays: usize,
    angle: f64,
    screen: bool,
) -> RivetResult<Tensor> {
    let source = &circles[0];
    let dirs: Vec<_> = (0..rays)
        .map(|i| (angle + std::f64::consts::TAU * i as f64 / rays as f64).sin_cos())
        .collect();
    let optical = |x: f64, y: f64| {
        let mut transmission = 1.0;
        let mut light = [0.0; 3];
        let mut blend = |alpha: f64, color: [f64; 3]| {
            if screen {
                for (v, c) in light.iter_mut().zip(color) {
                    *v += (1.0 - *v) * alpha * c;
                }
            } else {
                transmission *= 1.0 - alpha;
                for (v, c) in light.iter_mut().zip(color) {
                    *v = *v * (1.0 - alpha) + alpha * c;
                }
            }
        };
        for circle in circles {
            let d = ((x - circle.x).powi(2) + (y - circle.y).powi(2)) / circle.radius.powi(2);
            let weight = if screen {
                (-2.0 * d).exp()
            } else {
                (1.0 - d).max(0.0).powi(2)
            };
            blend(circle.alpha * weight, circle.color);
        }
        let (dx, dy) = (x - source.x, y - source.y);
        for &(sin, cos) in &dirs {
            let along = dx * cos + dy * sin;
            if along > 0.0 {
                let across = (dy * cos - dx * sin) / (0.15 * source.radius).max(0.5);
                blend(
                    source.alpha * 0.2 * (-across * across - along / (3.0 * source.radius)).exp(),
                    source.color,
                );
            }
        }
        (transmission, light)
    };
    let input = input.contiguous()?;
    let width = input.dims()[1];
    Ok(input.with_cpu_storage(|storage, layout| {
        macro_rules! write_flare {
            ($values:expr, $scale:expr, $finish:expr) => {{
                let values = crate::transforms::contiguous_values($values, layout)?;
                let finish = $finish;
                Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                    for (y, row) in values.chunks_exact(width * 3).enumerate() {
                        for (x, pixel) in row.chunks_exact(3).enumerate() {
                            let (transmission, light) = optical(x as f64, y as f64);
                            for (&value, light) in pixel.iter().zip(light) {
                                let value = f64::from(value) * $scale;
                                let value = if screen {
                                    value + (1.0 - value) * light
                                } else {
                                    value * transmission + light
                                };
                                out.write_next(finish(value.clamp(0.0, 1.0)))?;
                            }
                        }
                    }
                    Ok(())
                })
            }};
        }
        match storage {
            CpuStorageRef::U8(values) => {
                write_flare!(values, 1.0 / 255.0, |v: f64| (v * 255.0).round() as u8)
            }
            CpuStorageRef::F32(values) => write_flare!(values, 1.0, |v| v as f32),
            _ => unreachable!(),
        }
    })?)
}
