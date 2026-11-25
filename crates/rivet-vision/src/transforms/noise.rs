use super::{RivetResult, contiguous_values, invalid_argument};
use rivet_core::{CpuStorageRef, DType, Layout, Tensor, WithDType};
use rivet_data::random::RandomStream;

pub(super) fn validate_gaussian_noise(mean: f64, sigma: f64) -> RivetResult<()> {
    if !mean.is_finite() || !sigma.is_finite() || sigma < 0.0 {
        return Err(invalid_argument("invalid GaussianNoise parameters"));
    }
    Ok(())
}

pub(super) fn normal_pair(rng: &mut RandomStream) -> (f64, f64) {
    let radius = (-2.0 * rng.next_f64().max(f64::MIN_POSITIVE).ln()).sqrt();
    let (sine, cosine) = (std::f64::consts::TAU * rng.next_f64()).sin_cos();
    (radius * cosine, radius * sine)
}

fn standard_normal(rng: &mut RandomStream) -> f32 {
    let radius = (-2.0 * rng.next_f64().max(f64::MIN_POSITIVE).ln()).sqrt();
    (radius * (std::f64::consts::TAU * rng.next_f64()).cos()) as f32
}

fn map_pixels<T: WithDType>(
    input: &Tensor,
    layout: &Layout,
    values: &[T],
    mut convert: impl FnMut(T) -> T,
) -> rivet_core::Result<Tensor> {
    if layout.is_contiguous() {
        Tensor::from_exact_iter(
            contiguous_values(values, layout)?
                .iter()
                .copied()
                .map(convert),
            input.dims(),
            input.device(),
        )
    } else {
        Tensor::from_exact_try_iter(
            layout.strided_index().map(|i| {
                values
                    .get(i)
                    .copied()
                    .map(&mut convert)
                    .ok_or(rivet_core::Error::StorageOutOfBounds)
            }),
            input.dims(),
            input.device(),
        )
    }
}

/// Add independent Gaussian noise to each element of a CPU image tensor.
///
/// Supports uint8 and float32, including strided views and tensors with leading
/// batch or frame dimensions. Shape and dtype are preserved. Mean and standard
/// deviation are in normalized units for both dtypes. With clipping enabled,
/// outputs are limited to [0, 1] for float32 or [0, 255] for uint8.
/// Unclipped uint8 output wraps. Sampling uses the supplied Rivet random stream.
pub fn gaussian_noise(
    input: &Tensor,
    mean: f64,
    sigma: f64,
    clip: bool,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    validate_gaussian_noise(mean, sigma)?;
    if !matches!(input.dtype(), DType::U8 | DType::F32) {
        return Err(invalid_argument(
            "GaussianNoise requires uint8 or float32 input",
        ));
    }
    let mut normal = || -> f32 {
        if sigma == 0.0 {
            return 0.0;
        }
        standard_normal(rng)
    };
    Ok(input.with_cpu_storage(|storage, layout| {
        match storage {
            CpuStorageRef::F32(values) => map_pixels(input, layout, values, |value: f32| {
                let noise = mean as f32 + normal() * sigma as f32;
                let result = value + noise;
                if clip { result.clamp(0.0, 1.0) } else { result }
            }),
            CpuStorageRef::U8(values) => map_pixels(input, layout, values, |value: u8| {
                let noise = (mean * 255.0) as f32 + normal() * (sigma * 255.0) as f32;
                // PyTorch converts the noise to int16 *before* addition.
                let noise = if noise.is_finite() {
                    noise.trunc().rem_euclid(65536.0) as u16 as i16
                } else {
                    0
                };
                let result = i16::from(value).wrapping_add(noise);
                if clip {
                    result.clamp(0, 255) as u8
                } else {
                    result as u8
                }
            }),
            _ => unreachable!(),
        }
    })?)
}

pub(super) fn validate_shot_noise(intensity: f32) -> RivetResult<()> {
    if !intensity.is_finite() || intensity < 0.0 {
        return Err(invalid_argument(
            "ShotNoise intensity must be finite and nonnegative",
        ));
    }
    Ok(())
}

/// Add signal-dependent Gaussian noise to a normalized float32 CPU image tensor.
///
/// Each element receives independent standard normal noise scaled by
/// `intensity * sqrt(max(pixel, 0) + 0.5)`. Results are clipped to [0, 1].
/// Shape is preserved for strided views and tensors with batch or frame dimensions.
pub fn shot_noise(input: &Tensor, intensity: f32, rng: &mut RandomStream) -> RivetResult<Tensor> {
    validate_shot_noise(intensity)?;
    if input.dtype() != DType::F32 {
        return Err(invalid_argument("ShotNoise requires float32 input"));
    }
    Ok(input.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::F32(values) = storage else {
            unreachable!()
        };
        if intensity == 0.0 {
            return map_pixels(input, layout, values, |pixel| pixel.clamp(0.0, 1.0));
        }
        map_pixels(input, layout, values, |pixel| {
            let scale = (pixel.max(0.0) + 0.5).sqrt();
            (pixel + intensity * standard_normal(rng) * scale).clamp(0.0, 1.0)
        })
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rivet_core::Device;

    #[test]
    fn shot_noise_scales_each_strided_pixel_and_clips() {
        let input = Tensor::from_vec(vec![-0.25f32, 0.1, 0.5, 1.25], [2, 2], &Device::Cpu)
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let mut rng = RandomStream::from_seed(11);
        let mut reference_rng = rng.clone();
        let expected = [-0.25f32, 0.5, 0.1, 1.25].map(|pixel| {
            let normal = standard_normal(&mut reference_rng);
            (pixel + 0.2 * normal * (pixel.max(0.0) + 0.5).sqrt()).clamp(0.0, 1.0)
        });
        let output = shot_noise(&input, 0.2, &mut rng).unwrap();
        assert_eq!(output.dims(), input.dims());
        assert_eq!(output.to_vec::<f32>().unwrap(), expected);
        let zero = shot_noise(&input, 0.0, &mut rng).unwrap();
        assert_eq!(zero.to_vec::<f32>().unwrap(), [0.0, 0.5, 0.1, 1.0]);
        assert!(shot_noise(&input, f32::INFINITY, &mut rng).is_err());
        assert!(shot_noise(&input, -0.2, &mut rng).is_err());
        let uint8 = Tensor::from_vec(vec![0u8], [1], &Device::Cpu).unwrap();
        assert!(shot_noise(&uint8, 0.2, &mut rng).is_err());
    }

    #[test]
    fn uint8_noise_truncates_before_addition_on_strided_inputs() {
        let input = Tensor::from_vec(vec![0u8, 1, 250, 255], [2, 2], &Device::Cpu)
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let mut rng = RandomStream::from_seed(7);
        for (clip, expected) in [(true, [0, 249, 0, 254]), (false, [255, 249, 0, 254])] {
            let output = gaussian_noise(&input, -1.5 / 255.0, 0.0, clip, &mut rng).unwrap();
            assert_eq!(output.dims(), input.dims());
            assert_eq!(output.to_vec::<u8>().unwrap(), expected);
        }
    }

    #[test]
    fn float_noise_clips_and_preserves_batched_shape() {
        let input = Tensor::from_vec(vec![0.25f32, 0.75], [2, 1, 1, 1], &Device::Cpu).unwrap();
        let mut rng = RandomStream::from_seed(7);
        for (clip, expected) in [(true, [0.75, 1.0]), (false, [0.75, 1.25])] {
            let output = gaussian_noise(&input, 0.5, 0.0, clip, &mut rng).unwrap();
            assert_eq!(output.dims(), input.dims());
            assert_eq!(output.to_vec::<f32>().unwrap(), expected);
        }
        assert!(gaussian_noise(&input, 0.0, -1.0, true, &mut rng).is_err());
        assert!(gaussian_noise(&input, f64::NAN, 0.0, true, &mut rng).is_err());
    }

    #[test]
    fn noise_is_seeded_and_sampled_independently_across_frames() {
        let input = Tensor::from_vec(vec![0.5f32; 16], [2, 1, 2, 4], &Device::Cpu).unwrap();
        let sample = || {
            gaussian_noise(&input, 0.0, 0.1, false, &mut RandomStream::from_seed(7))
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
        };
        let output = sample();
        assert_eq!(output, sample());
        assert_ne!(output[..8], output[8..]);
    }
}

pub(super) fn iso_noise(
    input: &Tensor,
    color_shift: f64,
    intensity: f64,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    if !matches!(input.dims(), [h, w, 3] if *h > 0 && *w > 0)
        || !matches!(input.dtype(), DType::U8 | DType::F32)
    {
        return Err(invalid_argument(
            "ISONoise requires nonempty uint8 or float32 HWC RGB input",
        ));
    }
    if intensity == 0.0 {
        return Ok(input.clone());
    }
    let chroma_scale = std::f64::consts::TAU * intensity * color_shift;
    let mut perturb = |[r, g, b]: [f64; 3]| {
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let (luma_noise, chroma_noise) = normal_pair(rng);
        let delta = 0.1 * intensity * y.clamp(0.0, 1.0).sqrt() * luma_noise;
        if chroma_scale == 0.0 {
            return [r + delta, g + delta, b + delta];
        }
        let (sin, cos) = (chroma_scale * chroma_noise).sin_cos();
        let cb = (b - y) / 1.772;
        let cr = (r - y) / 1.402;
        let cb_new = cb * cos - cr * sin;
        let cr_new = cb * sin + cr * cos;
        let red = 1.402 * cr_new;
        let blue = 1.772 * cb_new;
        let y = y + delta;
        [y + red, y - (0.299 * red + 0.114 * blue) / 0.587, y + blue]
    };
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| {
        macro_rules! write_rgb {
            ($values:expr, $scale:expr, $finish:expr) => {{
                let pixels = contiguous_values($values, layout)?;
                Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                    for p in pixels.chunks_exact(3) {
                        let rgb = [p[0], p[1], p[2]].map(|v| f64::from(v) * $scale);
                        for v in perturb(rgb) {
                            out.write_next($finish(v.clamp(0.0, 1.0)))?;
                        }
                    }
                    Ok(())
                })
            }};
        }
        match storage {
            CpuStorageRef::U8(values) => {
                write_rgb!(values, 1.0 / 255.0, |v: f64| (v * 255.0).round() as u8)
            }
            CpuStorageRef::F32(values) => write_rgb!(values, 1.0, |v| v as f32),
            _ => unreachable!(),
        }
    })?)
}

pub(super) fn film_grain(
    input: &Tensor,
    intensity: f64,
    size: usize,
    per_channel: bool,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    let [h, w, channels] = *input.dims() else {
        return Err(invalid_argument("FilmGrain expects rank-3 HWC input"));
    };
    if h == 0 || w == 0 || channels == 0 || !matches!(input.dtype(), DType::U8 | DType::F32) {
        return Err(invalid_argument(
            "FilmGrain requires nonempty uint8 or float32 HWC input",
        ));
    }
    if intensity == 0.0 {
        return Ok(input.clone());
    }
    let (gh, gw) = ((h / size).max(1), (w / size).max(1));
    let planes = if per_channel { channels } else { 1 };
    let count = gh * gw * planes;
    let grain: Vec<_> = (0..count.div_ceil(2))
        .flat_map(|_| {
            let (a, b) = normal_pair(rng);
            [a, b]
        })
        .take(count)
        .collect();
    let coordinates = |length, coarse: usize| {
        let scale = coarse as f64 / length as f64;
        (0..length).map(move |i| {
            let source = ((i as f64 + 0.5) * scale - 0.5).clamp(0.0, (coarse - 1) as f64);
            let low = source as usize;
            (low, (low + 1).min(coarse - 1), source - low as f64)
        })
    };
    let xs: Vec<_> = coordinates(w, gw).collect();
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| {
        macro_rules! write_grain {
            ($values:expr, $scale:expr, $finish:expr) => {{
                let pixels = contiguous_values($values, layout)?;
                let finish = $finish;
                Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                    for (row, (top, bottom, fy)) in
                        pixels.chunks_exact(w * channels).zip(coordinates(h, gh))
                    {
                        let (top, bottom) = (top * gw * planes, bottom * gw * planes);
                        for (pixel, &(left, right, fx)) in row.chunks_exact(channels).zip(&xs) {
                            let (left, right) = (left * planes, right * planes);
                            let luma = match pixel {
                                &[r, g, b] => {
                                    0.299 * f64::from(r)
                                        + 0.587 * f64::from(g)
                                        + 0.114 * f64::from(b)
                                }
                                _ => {
                                    pixel.iter().map(|&v| f64::from(v)).sum::<f64>()
                                        / channels as f64
                                }
                            };
                            let amplitude = intensity * (1.0 - (luma * $scale).clamp(0.0, 1.0));
                            let interpolate = |c| {
                                let a = grain[top + left + c];
                                let b = grain[bottom + left + c];
                                let upper = a + fx * (grain[top + right + c] - a);
                                let lower = b + fx * (grain[bottom + right + c] - b);
                                upper + fy * (lower - upper)
                            };
                            let shared = if per_channel { 0.0 } else { interpolate(0) };
                            for (c, &value) in pixel.iter().enumerate() {
                                let noise = if per_channel { interpolate(c) } else { shared };
                                let value =
                                    (f64::from(value) * $scale + amplitude * noise).clamp(0.0, 1.0);
                                out.write_next(finish(value))?;
                            }
                        }
                    }
                    Ok(())
                })
            }};
        }
        match storage {
            CpuStorageRef::U8(values) => {
                write_grain!(values, 1.0 / 255.0, |v: f64| (v * 255.0).round() as u8)
            }
            CpuStorageRef::F32(values) => write_grain!(values, 1.0, |v| v as f32),
            _ => unreachable!(),
        }
    })?)
}

pub(super) fn multiplicative_noise(
    input: &Tensor,
    multiplier: (f64, f64),
    per_channel: bool,
    elementwise: bool,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    let (low, high) = multiplier;
    apply_sampled_noise(
        input,
        "MultiplicativeNoise",
        per_channel,
        elementwise,
        multiplier == (1.0, 1.0),
        |_, _| true,
        |_| low + rng.next_f64() * (high - low),
        |value, gain| value * gain,
    )
}

fn apply_sampled_noise(
    input: &Tensor,
    name: &str,
    per_channel: bool,
    elementwise: bool,
    identity: bool,
    active: impl Fn(usize, usize) -> bool,
    mut sample: impl FnMut(usize) -> f64,
    combine: impl Fn(f64, f64) -> f64,
) -> RivetResult<Tensor> {
    let [h, w, channels] = *input.dims() else {
        return Err(invalid_argument(format!("{name} expects rank-3 HWC input")));
    };
    if h == 0 || w == 0 || channels == 0 || !matches!(input.dtype(), DType::U8 | DType::F32) {
        return Err(invalid_argument(format!(
            "{name} requires nonempty uint8 or float32 HWC input"
        )));
    }
    if identity {
        return Ok(input.clone());
    }

    let samples = if per_channel { channels } else { 1 };
    let mut noise = vec![0.0; samples];
    let mut fill = |noise: &mut [f64]| {
        for (c, value) in noise.iter_mut().enumerate() {
            *value = sample(c);
        }
    };
    if !elementwise {
        fill(&mut noise);
    }
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| {
        macro_rules! write_noise {
            ($values:expr, $scale:expr, $finish:expr) => {{
                let pixels = contiguous_values($values, layout)?;
                let finish = $finish;
                Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                    for (y, row) in pixels.chunks_exact(w * channels).enumerate() {
                        for (x, pixel) in row.chunks_exact(channels).enumerate() {
                            if !active(x, y) {
                                for &value in pixel {
                                    out.write_next(value)?;
                                }
                                continue;
                            }
                            if elementwise {
                                fill(&mut noise);
                            }
                            for (&value, &noise) in pixel.iter().zip(noise.iter().cycle()) {
                                let value =
                                    combine(f64::from(value) * $scale, noise).clamp(0.0, 1.0);
                                out.write_next(finish(value))?;
                            }
                        }
                    }
                    Ok(())
                })
            }};
        }
        match storage {
            CpuStorageRef::U8(values) => {
                write_noise!(values, 1.0 / 255.0, |v: f64| (v * 255.0).round() as u8)
            }
            CpuStorageRef::F32(values) => write_noise!(values, 1.0, |v| v as f32),
            _ => unreachable!(),
        }
    })?)
}

/// Parameter ranges sampled once per image, except uniform offsets.
#[derive(Clone, Debug, PartialEq)]
pub enum AdditiveDistribution {
    Uniform(Vec<(f64, f64)>),
    Gaussian {
        mean: (f64, f64),
        std: (f64, f64),
    },
    Laplace {
        mean: (f64, f64),
        scale: (f64, f64),
    },
    Beta {
        alpha: (f64, f64),
        beta: (f64, f64),
        scale: (f64, f64),
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoiseSharing {
    Constant,
    PerPixel,
    Shared,
    Patch {
        count: (u32, u32),
        height: (f64, f64),
        width: (f64, f64),
        per_channel: bool,
    },
}

impl NoiseSharing {
    fn per_channel(self) -> bool {
        !matches!(
            self,
            Self::Shared
                | Self::Patch {
                    per_channel: false,
                    ..
                }
        )
    }

    fn valid(self) -> bool {
        match self {
            Self::Patch {
                count,
                height,
                width,
                ..
            } => {
                (1..=128).contains(&count.0)
                    && count.0 <= count.1
                    && count.1 <= 128
                    && [height, width]
                        .into_iter()
                        .all(|(lo, hi)| lo > 0.0 && lo <= hi && hi <= 1.0)
            }
            _ => true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdditiveNoiseConfig {
    pub distribution: AdditiveDistribution,
    pub sharing: NoiseSharing,
}

impl AdditiveDistribution {
    pub(super) fn valid(&self, sharing: NoiseSharing) -> bool {
        let bounded = |(a, b), min, max| a >= min && a <= b && b <= max;
        let shape = |(a, b): (f64, f64)| a >= 0.0001 && a <= b && b <= 1e6;
        sharing.valid()
            && (match self {
                Self::Uniform(ranges) => {
                    !ranges.is_empty()
                        && (sharing.per_channel() || ranges.len() == 1)
                        && ranges.iter().all(|&range| bounded(range, -1.0, 1.0))
                }
                Self::Gaussian { mean, std } => {
                    bounded(*mean, -1.0, 1.0) && bounded(*std, 0.0, 1.0)
                }
                Self::Laplace { mean, scale } => {
                    bounded(*mean, -1.0, 1.0) && bounded(*scale, 0.0, 1.0)
                }
                Self::Beta { alpha, beta, scale } => {
                    shape(*alpha) && shape(*beta) && bounded(*scale, 0.0, 1.0)
                }
            })
    }
}

fn log_gamma_sample(shape: f64, rng: &mut RandomStream) -> f64 {
    // Marsaglia–Tsang sampler. Work in log space to retain small-shape samples.
    let (shape, adjustment) = if shape < 1.0 {
        (
            shape + 1.0,
            rng.next_f64().max(f64::MIN_POSITIVE).ln() / shape,
        )
    } else {
        (shape, 0.0)
    };
    let d = shape - 1.0 / 3.0;
    let c = (9.0 * d).sqrt().recip();
    loop {
        let z = normal_pair(rng).0;
        let base = 1.0 + c * z;
        if base <= 0.0 {
            continue;
        }
        let v = base.powi(3);
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * z.powi(4) || u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            return d.ln() + v.ln() + adjustment;
        }
    }
}

pub(super) fn additive_noise(
    input: &Tensor,
    distribution: &AdditiveDistribution,
    sharing: NoiseSharing,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    if let AdditiveDistribution::Uniform(ranges) = distribution
        && ranges.len() != 1
        && input.dims().get(2) != Some(&ranges.len())
    {
        return Err(invalid_argument(
            "AdditiveNoise requires one uniform range or one per channel",
        ));
    }
    let mut draw = |(low, high)| low + rng.next_f64() * (high - low);
    let parameters = match distribution {
        AdditiveDistribution::Uniform(_) => [0.0; 3],
        AdditiveDistribution::Gaussian { mean, std } => [draw(*mean), draw(*std), 0.0],
        AdditiveDistribution::Laplace { mean, scale } => [draw(*mean), draw(*scale), 0.0],
        AdditiveDistribution::Beta { alpha, beta, scale } => {
            [draw(*alpha), draw(*beta), draw(*scale)]
        }
    };
    let mut patches = Vec::new();
    if let NoiseSharing::Patch {
        count,
        height,
        width,
        ..
    } = sharing
    {
        let [h, w, _] = *input.dims() else {
            return Err(invalid_argument("AdditiveNoise expects rank-3 HWC input"));
        };
        if h == 0 || w == 0 {
            return Err(invalid_argument(
                "AdditiveNoise expects nonempty spatial dimensions",
            ));
        }
        let count = rng.gen_range_usize(count.0 as usize..count.1 as usize + 1)?;
        let sample_extent = |(lo, hi): (f64, f64), len: usize, rng: &mut RandomStream| {
            (((lo + rng.next_f64() * (hi - lo)) * len as f64).round() as usize).clamp(1, len)
        };
        for _ in 0..count {
            let ph = sample_extent(height, h, rng);
            let pw = sample_extent(width, w, rng);
            let y = rng.gen_range_usize(0..h - ph + 1)?;
            let x = rng.gen_range_usize(0..w - pw + 1)?;
            patches.push((x..x + pw, y..y + ph));
        }
    }
    let [a, b, scale] = parameters;
    apply_sampled_noise(
        input,
        "AdditiveNoise",
        sharing.per_channel(),
        sharing != NoiseSharing::Constant,
        false,
        |x, y| {
            patches.is_empty()
                || patches
                    .iter()
                    .any(|(xs, ys)| xs.contains(&x) && ys.contains(&y))
        },
        |channel| match distribution {
            AdditiveDistribution::Uniform(ranges) => {
                let (low, high) = ranges[if ranges.len() == 1 { 0 } else { channel }];
                low + rng.next_f64() * (high - low)
            }
            AdditiveDistribution::Gaussian { .. } => a + b * normal_pair(rng).0,
            AdditiveDistribution::Laplace { .. } => {
                let u = 2.0 * rng.next_f64() - 1.0;
                a - b * u.signum() * (1.0 - u.abs()).max(f64::MIN_POSITIVE).ln()
            }
            AdditiveDistribution::Beta { .. } => {
                let x = log_gamma_sample(a, rng);
                let y = log_gamma_sample(b, rng);
                scale * (2.0 / (1.0 + (y - x).exp()) - 1.0)
            }
        },
        |value, offset| value + offset,
    )
}

pub(super) fn rgb_shift(
    input: &Tensor,
    ranges: &[(f64, f64); 3],
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    if !matches!(input.dims(), [_, _, 3]) {
        return Err(invalid_argument("RGBShift requires HWC RGB input"));
    }
    apply_sampled_noise(
        input,
        "RGBShift",
        true,
        false,
        ranges.iter().all(|&range| range == (0.0, 0.0)),
        |_, _| true,
        |c| {
            let (lo, hi) = ranges[c];
            (lo + rng.next_f64() * (hi - lo)) / 255.0
        },
        |value, shift| value + shift,
    )
}
