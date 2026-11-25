use super::{RivetResult, invalid_argument};
use rivet_core::{CpuStorageRef, DType, Tensor};
use rivet_data::random::RandomStream;

pub(super) fn tone_curve(
    input: &Tensor,
    scale: f64,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    if scale == 0.0 {
        return Ok(input.clone());
    }
    let (low, high) = super::noise::normal_pair(rng);
    let a = (1.0 / 3.0 + scale * low).clamp(0.0, 1.0);
    let b = (2.0 / 3.0 + scale * high).clamp(0.0, 1.0);
    // Ordered control points guarantee a monotone curve.
    let (a, b) = (a.min(b), a.max(b));
    let lut: [u8; 256] = std::array::from_fn(|i| {
        let t = i as f64 / 255.0;
        let s = 1.0 - t;
        ((3.0 * s * s * t * a + 3.0 * s * t * t * b + t * t * t) * 255.0).round() as u8
    });
    super::with_rgb_pixels(input, "RandomToneCurve", |pixels, _, _| {
        Ok(Tensor::from_exact_iter(
            pixels.iter().map(|&value| lut[usize::from(value)]),
            input.dims(),
            input.device(),
        )?)
    })
}

fn tile_lut(mut histogram: [usize; 256], count: usize, clip_limit: f64) -> [u8; 256] {
    if histogram.contains(&count) {
        return std::array::from_fn(|i| i as u8);
    }
    let limit = ((clip_limit * (count as f64 / 256.0)) as usize).max(1);
    let mut excess = 0;
    for bin in &mut histogram {
        if *bin > limit {
            excess += *bin - limit;
            *bin = limit;
        }
    }
    let (share, remainder) = (excess / 256, excess % 256);
    for bin in &mut histogram {
        *bin += share;
    }
    for i in 0..remainder {
        histogram[i * 256 / remainder] += 1;
    }
    let mut cumulative = 0;
    histogram.map(|bin| {
        cumulative += bin;
        (255.0 * (cumulative as f64 / count as f64)).round() as u8
    })
}

fn tile_axis(length: usize, tiles: usize) -> (Vec<usize>, Vec<(usize, usize, f64)>) {
    let tiles = tiles.min(length);
    let bounds: Vec<_> = (0..=tiles)
        .map(|i| (i as u128 * length as u128 / tiles as u128) as usize)
        .collect();
    let centers: Vec<_> = bounds
        .windows(2)
        .map(|pair| (pair[0] as f64 + (pair[1] - 1) as f64) * 0.5)
        .collect();
    let mut right = 0;
    let weights = (0..length)
        .map(|i| {
            let coordinate = i as f64;
            while right + 1 < tiles && coordinate > centers[right] {
                right += 1;
            }
            let left = right.saturating_sub(1);
            let fraction = if left == right {
                0.0
            } else {
                ((coordinate - centers[left]) / (centers[right] - centers[left])).clamp(0.0, 1.0)
            };
            (left, right, fraction)
        })
        .collect();
    (bounds, weights)
}

pub(super) fn clahe(input: &Tensor, grid: (u32, u32), clip_limit: f64) -> RivetResult<Tensor> {
    let [height, width, channels] = *input.dims() else {
        return Err(invalid_argument("CLAHE expects a rank-3 HWC image"));
    };
    if height == 0 || width == 0 || !matches!(channels, 1 | 3) || input.dtype() != DType::U8 {
        return Err(invalid_argument(
            "CLAHE requires a nonempty uint8 HWC grayscale or RGB image",
        ));
    }
    // Compact a strided input once when necessary. Each tile LUT and each
    // output pixel then read directly from the contiguous source allocation.
    let input = input.contiguous()?;
    let (xs, x_weights) = tile_axis(width, grid.0 as usize);
    let (ys, y_weights) = tile_axis(height, grid.1 as usize);
    let tiles_x = xs.len() - 1;
    Ok(input.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::U8(values) = storage else {
            unreachable!()
        };
        let pixels = super::contiguous_values(values, layout)?;
        let luminance = |pixel: &[u8]| {
            if channels == 1 {
                pixel[0]
            } else {
                let [r, g, b] = [pixel[0], pixel[1], pixel[2]].map(u32::from);
                ((299 * r + 587 * g + 114 * b + 500) / 1000) as u8
            }
        };
        let mut luts = Vec::new();
        for row in ys.windows(2) {
            for col in xs.windows(2) {
                let mut histogram = [0; 256];
                for y in row[0]..row[1] {
                    let start = (y * width + col[0]) * channels;
                    let end = (y * width + col[1]) * channels;
                    for pixel in pixels[start..end].chunks_exact(channels) {
                        histogram[usize::from(luminance(pixel))] += 1;
                    }
                }
                let count = (row[1] - row[0]) * (col[1] - col[0]);
                luts.push(tile_lut(histogram, count, clip_limit));
            }
        }
        Tensor::from_exact_writer::<u8, _, _>(input.dims(), input.device(), |out| {
            for (row, &(top, bottom, fy)) in pixels.chunks_exact(width * channels).zip(&y_weights) {
                let top = top * tiles_x;
                let bottom = bottom * tiles_x;
                for (pixel, &(left, right, fx)) in row.chunks_exact(channels).zip(&x_weights) {
                    let y = luminance(pixel);
                    let bin = usize::from(y);
                    let lerp = |base: usize| {
                        let a = f64::from(luts[base + left][bin]);
                        let b = f64::from(luts[base + right][bin]);
                        a + fx * (b - a)
                    };
                    let above = lerp(top);
                    let mapped = (above + fy * (lerp(bottom) - above)).round() as i16;
                    let offset = mapped - i16::from(y);
                    for &channel in pixel {
                        out.write_next((i16::from(channel) + offset).clamp(0, 255) as u8)?;
                    }
                }
            }
            Ok(())
        })
    })?)
}

pub(super) fn vignette(input: &Tensor, intensity: f64, center: (f64, f64)) -> RivetResult<Tensor> {
    let [height, width, channels] = *input.dims() else {
        return Err(invalid_argument("Vignetting expects a rank-3 HWC image"));
    };
    if height == 0
        || width == 0
        || channels == 0
        || !matches!(input.dtype(), DType::U8 | DType::F32)
    {
        return Err(invalid_argument(
            "Vignetting requires nonempty uint8 or float32 HWC input",
        ));
    }
    if intensity == 0.0 || (height, width) == (1, 1) {
        return Ok(input.clone());
    }
    // Ignore degenerate axes so a single row or column uses the full falloff.
    let axis = |length: usize, center: f64| {
        let extent = length.saturating_sub(1);
        let limit = if extent == 0 {
            0.0
        } else {
            center.max(1.0 - center).powi(2)
        };
        let distances: Vec<_> = (0..length)
            .map(|i| {
                if extent == 0 {
                    0.0
                } else {
                    (i as f64 / extent as f64 - center).powi(2)
                }
            })
            .collect();
        (limit, distances)
    };
    let (x_limit, xs) = axis(width, center.0);
    let (y_limit, ys) = axis(height, center.1);
    let scale = intensity / (x_limit + y_limit);
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| {
        macro_rules! write_pixels {
            ($values:expr, $scale_pixel:expr) => {{
                let pixels = super::contiguous_values($values, layout)?;
                Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                    let scale_pixel = $scale_pixel;
                    for (row, &dy) in pixels.chunks_exact(width * channels).zip(&ys) {
                        for (pixel, &dx) in row.chunks_exact(channels).zip(&xs) {
                            let gain = (1.0 - scale * (dx + dy)).max(0.0);
                            for &value in pixel {
                                out.write_next(scale_pixel(value, gain))?;
                            }
                        }
                    }
                    Ok(())
                })
            }};
        }
        match storage {
            CpuStorageRef::U8(values) => {
                write_pixels!(values, |value, gain: f64| (f64::from(value) * gain).round()
                    as u8)
            }
            CpuStorageRef::F32(values) => {
                write_pixels!(values, |value, gain: f64| value * gain as f32)
            }
            _ => unreachable!(),
        }
    })?)
}

fn blackbody_rgb(kelvin: f64) -> [f64; 3] {
    // Kang et al. (2002), Planckian locus in CIE 1931 xy.
    let t = 1000.0 / kelvin;
    let (x, y) = if kelvin <= 4000.0 {
        let x = ((-0.2661239 * t - 0.2343580) * t + 0.8776956) * t + 0.179910;
        (
            x,
            ((-0.9549476 * x - 1.37418593) * x + 2.09137015) * x - 0.16748867,
        )
    } else {
        let x = ((-3.0258469 * t + 2.1070379) * t + 0.2226347) * t + 0.240390;
        (
            x,
            ((3.0817580 * x - 5.87338670) * x + 3.75112997) * x - 0.37001483,
        )
    };
    // XYZ with Y = 1, converted to linear sRGB.
    let (x, z) = (x / y, (1.0 - x - y) / y);
    [
        3.2404542 * x - 1.5371385 - 0.4985314 * z,
        -0.9692660 * x + 1.8760108 + 0.0415560 * z,
        0.0556434 * x - 0.2040259 + 1.0572252 * z,
    ]
}

pub(super) fn planckian_jitter(input: &Tensor, kelvin: f64) -> RivetResult<Tensor> {
    if !matches!(input.dims(), [h, w, 3] if *h > 0 && *w > 0)
        || !matches!(input.dtype(), DType::U8 | DType::F32)
    {
        return Err(invalid_argument(
            "PlanckianJitter requires nonempty uint8 or float32 HWC RGB input",
        ));
    }
    if kelvin == 6500.0 {
        return Ok(input.clone());
    }
    let [r, g, b] = blackbody_rgb(kelvin);
    let [white_r, white_g, white_b] = blackbody_rgb(6500.0);
    let red = (r / g) / (white_r / white_g);
    let blue = (b / g) / (white_b / white_g);
    let adjust = |value: f64, gain: f64| {
        let value = value.clamp(0.0, 1.0);
        let linear = if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        };
        let linear = (linear * gain).clamp(0.0, 1.0);
        if linear <= 0.0031308 {
            12.92 * linear
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        }
    };
    if input.dtype() == DType::U8 {
        let lut = |gain| {
            std::array::from_fn::<_, 256, _>(|i| {
                (255.0 * adjust(i as f64 / 255.0, gain)).round() as u8
            })
        };
        let (red_lut, blue_lut) = (lut(red), lut(blue));
        return super::with_rgb_pixels(input, "PlanckianJitter", |pixels, _, _| {
            Ok(Tensor::from_exact_writer(
                input.dims(),
                input.device(),
                |out| {
                    for p in pixels.chunks_exact(3) {
                        out.write_next(red_lut[p[0] as usize])?;
                        out.write_next(p[1])?;
                        out.write_next(blue_lut[p[2] as usize])?;
                    }
                    Ok(())
                },
            )?)
        });
    }
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::F32(values) = storage else {
            unreachable!()
        };
        let pixels = super::contiguous_values(values, layout)?;
        Tensor::from_exact_writer(input.dims(), input.device(), |out| {
            for p in pixels.chunks_exact(3) {
                out.write_next(adjust(f64::from(p[0]), red) as f32)?;
                out.write_next(p[1].clamp(0.0, 1.0))?;
                out.write_next(adjust(f64::from(p[2]), blue) as f32)?;
            }
            Ok(())
        })
    })?)
}

pub(super) fn illumination(
    input: &Tensor,
    intensity: (f64, f64),
    mode: super::IlluminationMode,
    effect: super::IlluminationEffect,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    use super::{IlluminationEffect, IlluminationMode};
    let amount = intensity.0 + rng.next_f64() * (intensity.1 - intensity.0);
    let signed = match effect {
        IlluminationEffect::Brighten => amount,
        IlluminationEffect::Darken => -amount,
        IlluminationEffect::Both if rng.next_f64() < 0.5 => -amount,
        IlluminationEffect::Both => amount,
    };
    let angle = rng.next_f64() * std::f64::consts::TAU;
    let (dy, dx) = angle.sin_cos();
    let (cx, cy) = match mode {
        IlluminationMode::Corner => (
            rng.gen_range_usize(0..2)? as f64,
            rng.gen_range_usize(0..2)? as f64,
        ),
        _ => (0.1 + 0.8 * rng.next_f64(), 0.1 + 0.8 * rng.next_f64()),
    };
    let sigma = 0.2 + 0.8 * rng.next_f64();
    apply_light_field(input, "Illumination", signed == 0.0, |x, y| {
        let weight = match mode {
            IlluminationMode::Linear => {
                0.5 + ((x - 0.5) * dx + (y - 0.5) * dy) / (dx.abs() + dy.abs())
            }
            IlluminationMode::Corner => {
                1.0 - ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() * std::f64::consts::FRAC_1_SQRT_2
            }
            IlluminationMode::Gaussian => {
                (-((x - cx).powi(2) + (y - cy).powi(2)) / (2.0 * sigma * sigma)).exp()
            }
        };
        (1.0 + signed * weight.clamp(0.0, 1.0), 0.0)
    })
}

fn apply_light_field(
    input: &Tensor,
    name: &str,
    identity: bool,
    field: impl Fn(f64, f64) -> (f64, f64),
) -> RivetResult<Tensor> {
    let [height, width, channels] = *input.dims() else {
        return Err(invalid_argument(format!("{name} expects HWC input")));
    };
    if height == 0
        || width == 0
        || channels == 0
        || !matches!(input.dtype(), DType::U8 | DType::F32)
    {
        return Err(invalid_argument(format!(
            "{name} requires nonempty uint8 or float32 HWC input"
        )));
    }
    if identity {
        return Ok(input.clone());
    }
    let input = input.contiguous()?;
    let dx = 1.0 / (width - 1).max(1) as f64;
    let dy = 1.0 / (height - 1).max(1) as f64;
    Ok(input.with_cpu_storage(|storage, layout| {
        macro_rules! write_gain {
            ($values:expr, $max:expr, $finish:expr) => {{
                let pixels = super::contiguous_values($values, layout)?;
                let finish = $finish;
                Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                    for (y, row) in pixels.chunks_exact(width * channels).enumerate() {
                        let y = if height == 1 { 0.5 } else { y as f64 * dy };
                        for (x, pixel) in row.chunks_exact(channels).enumerate() {
                            let x = if width == 1 { 0.5 } else { x as f64 * dx };
                            let (gain, offset) = field(x, y);
                            let offset = offset * $max;
                            for &v in pixel {
                                out.write_next(finish(f64::from(v) * gain + offset))?;
                            }
                        }
                    }
                    Ok(())
                })
            }};
        }
        match storage {
            CpuStorageRef::U8(values) => write_gain!(values, 255.0, |v: f64| v.round() as u8),
            CpuStorageRef::F32(values) => {
                write_gain!(values, 1.0, |v: f64| v.clamp(0.0, 1.0) as f32)
            }
            _ => unreachable!(),
        }
    })?)
}

pub(super) fn apply_plasma(
    input: &Tensor,
    name: &str,
    size: usize,
    roughness: f64,
    identity: bool,
    rng: &mut RandomStream,
    affine: impl Fn(f64) -> (f64, f64),
) -> RivetResult<Tensor> {
    if identity {
        return apply_light_field(input, name, true, |_, _| (1.0, 0.0));
    }
    let edge = (size - 1).next_power_of_two();
    let stride = edge + 1;
    let mut field = vec![0.0_f64; stride * stride];
    let mut noise = || 2.0 * rng.next_f64() - 1.0;
    for (x, y) in [(0, 0), (edge, 0), (0, edge), (edge, edge)] {
        field[y * stride + x] = noise();
    }
    let mut step = edge;
    let mut amplitude = 1.0;
    let decay = (-roughness.recip()).exp();
    while step > 1 {
        let half = step / 2;
        for y in (half..edge).step_by(step) {
            for x in (half..edge).step_by(step) {
                let top = (y - half) * stride;
                let bottom = (y + half) * stride;
                field[y * stride + x] = (field[top + x - half]
                    + field[top + x + half]
                    + field[bottom + x - half]
                    + field[bottom + x + half])
                    * 0.25
                    + amplitude * noise();
            }
        }
        for (row, y) in (0..=edge).step_by(half).enumerate() {
            let start = if row % 2 == 0 { half } else { 0 };
            for x in (start..=edge).step_by(step) {
                let neighbors = [
                    x.checked_sub(half).map(|x| y * stride + x),
                    (x + half <= edge).then_some(y * stride + x + half),
                    y.checked_sub(half).map(|y| y * stride + x),
                    (y + half <= edge).then_some((y + half) * stride + x),
                ];
                let (sum, count) = neighbors
                    .into_iter()
                    .flatten()
                    .fold((0.0, 0), |(sum, count), i| (sum + field[i], count + 1));
                field[y * stride + x] = sum / f64::from(count) + amplitude * noise();
            }
        }
        amplitude *= decay;
        step = half;
    }
    let min = field.iter().copied().fold(f64::INFINITY, f64::min);
    let max = field.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (max - min).max(f64::EPSILON);
    apply_light_field(input, name, false, |x, y| {
        let (x, y) = (x * edge as f64, y * edge as f64);
        let (left, top) = (x as usize, y as usize);
        let (right, bottom) = ((left + 1).min(edge), (top + 1).min(edge));
        let (fx, fy) = (x - left as f64, y - top as f64);
        let upper = field[top * stride + left] * (1.0 - fx) + field[top * stride + right] * fx;
        let lower =
            field[bottom * stride + left] * (1.0 - fx) + field[bottom * stride + right] * fx;
        affine(((upper * (1.0 - fy) + lower * fy - min) / span).clamp(0.0, 1.0))
    })
}

pub(super) fn random_shadow(
    input: &Tensor,
    config: &super::RandomShadowConfig,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    let (min, max) = config.num_shadows_range;
    let (low, high) = config.shadow_intensity_range;
    let count = rng.gen_range_usize(min as usize..max as usize + 1)?;
    let (x0, y0, x1, y1) = config.shadow_roi;
    let mut shadows = Vec::with_capacity(count);
    if high > 0.0 {
        for _ in 0..count {
            let gain = 1.0 - (low + rng.next_f64() * (high - low));
            let mut vertices: Vec<_> = (0..config.shadow_dimension)
                .map(|_| {
                    (
                        x0 + rng.next_f64() * (x1 - x0),
                        y0 + rng.next_f64() * (y1 - y0),
                    )
                })
                .collect();
            let (cx, cy) = vertices
                .iter()
                .fold((0.0, 0.0), |(cx, cy), &(x, y)| (cx + x, cy + y));
            let count = f64::from(config.shadow_dimension);
            let (cx, cy) = (cx / count, cy / count);
            vertices.sort_by(|&(ax, ay), &(bx, by)| {
                (ay - cy)
                    .atan2(ax - cx)
                    .total_cmp(&(by - cy).atan2(bx - cx))
            });
            let edges = vertices
                .iter()
                .zip(vertices.iter().cycle().skip(1))
                .filter_map(|(&(ax, ay), &(bx, by))| {
                    (ay != by).then(|| (ax, ay, by, (bx - ax) / (by - ay)))
                })
                .collect::<Vec<_>>();
            shadows.push((gain, edges));
        }
    }
    apply_light_field(input, "RandomShadow", shadows.is_empty(), |x, y| {
        if x < x0 || y < y0 || x > x1 || y > y1 {
            return (1.0, 0.0);
        }
        let mut gain = 1.0;
        for (transmission, edges) in &shadows {
            let inside = edges.iter().fold(false, |inside, &(ax, ay, by, slope)| {
                inside ^ (((ay > y) != (by > y)) && x < ax + (y - ay) * slope)
            });
            if inside {
                gain *= transmission;
            }
        }
        (gain, 0.0)
    })
}
