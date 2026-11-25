use super::{RivetResult, contiguous_values, invalid_argument};
use rivet_core::{CpuStorageRef, DType, Tensor, WithDType};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiffusionAlgorithm {
    FloydSteinberg,
    Jarvis,
    Stucki,
    Atkinson,
    Burkes,
    Sierra,
    Sierra2Row,
    SierraLite,
}

impl DiffusionAlgorithm {
    fn stencil(self) -> (f64, &'static [(isize, usize, u8)]) {
        match self {
            Self::FloydSteinberg => (16.0, &[(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)]),
            Self::Jarvis => (
                48.0,
                &[
                    (1, 0, 7),
                    (2, 0, 5),
                    (-2, 1, 3),
                    (-1, 1, 5),
                    (0, 1, 7),
                    (1, 1, 5),
                    (2, 1, 3),
                    (-2, 2, 1),
                    (-1, 2, 3),
                    (0, 2, 5),
                    (1, 2, 3),
                    (2, 2, 1),
                ],
            ),
            Self::Stucki => (
                42.0,
                &[
                    (1, 0, 8),
                    (2, 0, 4),
                    (-2, 1, 2),
                    (-1, 1, 4),
                    (0, 1, 8),
                    (1, 1, 4),
                    (2, 1, 2),
                    (-2, 2, 1),
                    (-1, 2, 2),
                    (0, 2, 4),
                    (1, 2, 2),
                    (2, 2, 1),
                ],
            ),
            Self::Atkinson => (
                8.0,
                &[
                    (1, 0, 1),
                    (2, 0, 1),
                    (-1, 1, 1),
                    (0, 1, 1),
                    (1, 1, 1),
                    (0, 2, 1),
                ],
            ),
            Self::Burkes => (
                32.0,
                &[
                    (1, 0, 8),
                    (2, 0, 4),
                    (-2, 1, 2),
                    (-1, 1, 4),
                    (0, 1, 8),
                    (1, 1, 4),
                    (2, 1, 2),
                ],
            ),
            Self::Sierra => (
                32.0,
                &[
                    (1, 0, 5),
                    (2, 0, 3),
                    (-2, 1, 2),
                    (-1, 1, 4),
                    (0, 1, 5),
                    (1, 1, 4),
                    (2, 1, 2),
                    (-1, 2, 2),
                    (0, 2, 3),
                    (1, 2, 2),
                ],
            ),
            Self::Sierra2Row => (
                16.0,
                &[
                    (1, 0, 4),
                    (2, 0, 3),
                    (-2, 1, 1),
                    (-1, 1, 2),
                    (0, 1, 3),
                    (1, 1, 2),
                    (2, 1, 1),
                ],
            ),
            Self::SierraLite => (4.0, &[(1, 0, 2), (-1, 1, 1), (0, 1, 1)]),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DitherMethod {
    Random {
        noise_range: (f64, f64),
    },
    Ordered {
        size: u8,
    },
    ErrorDiffusion {
        algorithm: DiffusionAlgorithm,
        serpentine: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DitheringConfig {
    pub method: DitherMethod,
    pub n_colors: u16,
    pub grayscale: bool,
}

impl DitheringConfig {
    pub(super) fn valid(&self) -> bool {
        (2..=256).contains(&self.n_colors)
            && match self.method {
                DitherMethod::Random {
                    noise_range: (lo, hi),
                } => lo >= -1.0 && lo <= hi && hi <= 1.0,
                DitherMethod::Ordered { size } => matches!(size, 2 | 4 | 8 | 16),
                DitherMethod::ErrorDiffusion { .. } => true,
            }
    }

    pub(super) fn apply(&self, input: &Tensor, rng: &mut RandomStream) -> RivetResult<Tensor> {
        if !matches!(input.dims(), [h,w,c] if *h > 0 && *w > 0 && *c > 0)
            || !matches!(input.dtype(), DType::U8 | DType::F32)
        {
            return Err(invalid_argument(
                "Dithering requires nonempty uint8 or float32 HWC input",
            ));
        }
        let input = input.contiguous()?;
        Ok(input.with_cpu_storage(|storage, layout| match storage {
            CpuStorageRef::U8(values) => self.write(
                &input,
                contiguous_values(values, layout)?,
                1.0 / 255.0,
                rng,
                |v| (v * 255.0).round() as u8,
            ),
            CpuStorageRef::F32(values) => {
                self.write(&input, contiguous_values(values, layout)?, 1.0, rng, |v| {
                    v as f32
                })
            }
            _ => unreachable!(),
        })?)
    }

    fn write<T: WithDType + Into<f64>>(
        &self,
        input: &Tensor,
        pixels: &[T],
        scale: f64,
        rng: &mut RandomStream,
        finish: impl Fn(f64) -> T,
    ) -> rivet_core::Result<Tensor> {
        let [height, width, channels] = *input.dims() else {
            unreachable!()
        };
        let bins = f64::from(self.n_colors - 1);
        let quantize = |value: f64| value.round().clamp(0.0, bins);
        let intensity = |pixel: &[T], c| {
            let value = if !self.grayscale {
                <T as Into<f64>>::into(pixel[c])
            } else if let &[r, g, b] = pixel {
                0.299 * r.into() + 0.587 * g.into() + 0.114 * b.into()
            } else {
                pixel.iter().map(|&v| v.into()).sum::<f64>() / channels as f64
            };
            value * scale * bins
        };
        let planes = if self.grayscale { 1 } else { channels };
        let stride = width * planes;
        Tensor::from_exact_writer(input.dims(), input.device(), |out| {
            match self.method {
                DitherMethod::ErrorDiffusion {
                    algorithm,
                    serpentine,
                } => {
                    let (denom, stencil) = algorithm.stencil();
                    let mut errors = [vec![0.0; stride], vec![0.0; stride], vec![0.0; stride]];
                    let mut result = vec![0.0; stride];
                    for (y, row) in pixels.chunks_exact(width * channels).enumerate() {
                        let reverse = serpentine && y % 2 == 1;
                        let direction = if reverse { -1 } else { 1 };
                        let rows = [y % 3, (y + 1) % 3, (y + 2) % 3];
                        for col in 0..width {
                            let x = if reverse { width - 1 - col } else { col };
                            let pixel = &row[x * channels..(x + 1) * channels];
                            for c in 0..planes {
                                let index = x * planes + c;
                                let adjusted = intensity(pixel, c) + errors[rows[0]][index];
                                let level = quantize(adjusted);
                                result[index] = level / bins;
                                let error = (adjusted - level) / denom;
                                for &(dx, dy, numerator) in stencil {
                                    if y + dy < height
                                        && let Some(nx) = x.checked_add_signed(dx * direction)
                                        && nx < width
                                    {
                                        errors[rows[dy]][nx * planes + c] +=
                                            error * f64::from(numerator);
                                    }
                                }
                            }
                        }
                        errors[rows[0]].fill(0.0);
                        for levels in result.chunks_exact(planes) {
                            for &v in levels.iter().cycle().take(channels) {
                                out.write_next(finish(v))?;
                            }
                        }
                    }
                }
                method => {
                    let bayer = match method {
                        DitherMethod::Ordered { size } => bayer_thresholds(size as usize),
                        _ => Vec::new(),
                    };
                    for (y, row) in pixels.chunks_exact(width * channels).enumerate() {
                        for (x, pixel) in row.chunks_exact(channels).enumerate() {
                            let mut last = 0.0;
                            for c in 0..channels {
                                if c < planes {
                                    let bias = match method {
                                        DitherMethod::Random {
                                            noise_range: (lo, hi),
                                        } => lo + rng.next_f64() * (hi - lo),
                                        DitherMethod::Ordered { size } => {
                                            let index = (y & (size as usize - 1)) * size as usize
                                                + (x & (size as usize - 1));
                                            bayer[index]
                                        }
                                        _ => unreachable!(),
                                    };
                                    last = quantize(intensity(pixel, c) + bias) / bins;
                                }
                                out.write_next(finish(last))?;
                            }
                        }
                    }
                }
            }
            Ok(())
        })
    }
}

fn bayer_thresholds(size: usize) -> Vec<f64> {
    let mut ranks = vec![0];
    let mut edge = 1;
    while edge < size {
        let mut expanded = vec![0; 4 * edge * edge];
        for y in 0..edge {
            for x in 0..edge {
                let rank = 4 * ranks[y * edge + x];
                expanded[y * 2 * edge + x] = rank;
                expanded[y * 2 * edge + x + edge] = rank + 2;
                expanded[(y + edge) * 2 * edge + x] = rank + 3;
                expanded[(y + edge) * 2 * edge + x + edge] = rank + 1;
            }
        }
        edge *= 2;
        ranks = expanded;
    }
    ranks
        .into_iter()
        .map(|rank| (f64::from(rank) + 0.5) / (size * size) as f64 - 0.5)
        .collect()
}

pub(super) fn halftone(input: &Tensor, size: usize, blend: f64) -> RivetResult<Tensor> {
    if !matches!(input.dims(), [h,w,c] if *h > 0 && *w > 0 && *c > 0)
        || !matches!(input.dtype(), DType::U8 | DType::F32)
    {
        return Err(invalid_argument(
            "Halftone requires nonempty uint8 or float32 HWC input",
        ));
    }
    if blend == 1.0 {
        return Ok(input.clone());
    }
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| match storage {
        CpuStorageRef::U8(values) => write_halftone(
            &input,
            contiguous_values(values, layout)?,
            size,
            blend,
            255.0,
            |v| v.round() as u8,
        ),
        CpuStorageRef::F32(values) => write_halftone(
            &input,
            contiguous_values(values, layout)?,
            size,
            blend,
            1.0,
            |v| v.clamp(0.0, 1.0) as f32,
        ),
        _ => unreachable!(),
    })?)
}

fn write_halftone<T: WithDType + Into<f64>>(
    input: &Tensor,
    pixels: &[T],
    size: usize,
    blend: f64,
    max: f64,
    finish: impl Fn(f64) -> T,
) -> rivet_core::Result<Tensor> {
    let [height, width, channels] = *input.dims() else {
        unreachable!()
    };
    let columns = width.div_ceil(size);
    let stride = width * channels;
    let mut colors = vec![0.0; columns * channels];
    let mut radii = vec![0.0; columns];

    Tensor::from_exact_writer(input.dims(), input.device(), |out| {
        for y0 in (0..height).step_by(size) {
            let h = (height - y0).min(size);
            for (column, color) in colors.chunks_exact_mut(channels).enumerate() {
                color.fill(0.0);
                let x0 = column * size;
                let w = (width - x0).min(size);
                for y in y0..y0 + h {
                    let start = y * stride + x0 * channels;
                    for pixel in pixels[start..start + w * channels].chunks_exact(channels) {
                        for (sum, &v) in color.iter_mut().zip(pixel) {
                            *sum += v.into();
                        }
                    }
                }
                for v in color.iter_mut() {
                    *v /= (w * h) as f64;
                }
                let luma = match &*color {
                    &[r, g, b] => 0.299 * r + 0.587 * g + 0.114 * b,
                    _ => color.iter().sum::<f64>() / channels as f64,
                };
                radii[column] = (0.5 * h.min(w) as f64 * (luma / max).clamp(0.0, 1.0)).powi(2);
            }
            for y in 0..h {
                let dy = (y as f64 + 0.5 - h as f64 * 0.5).powi(2);
                for (column, color) in colors.chunks_exact(channels).enumerate() {
                    let x0 = column * size;
                    let w = (width - x0).min(size);
                    let start = (y0 + y) * stride + x0 * channels;
                    for (x, pixel) in pixels[start..start + w * channels]
                        .chunks_exact(channels)
                        .enumerate()
                    {
                        let dx = x as f64 + 0.5 - w as f64 * 0.5;
                        let weight = if dx * dx + dy <= radii[column] {
                            1.0 - blend
                        } else {
                            0.0
                        };
                        for (&value, &dot) in pixel.iter().zip(color) {
                            out.write_next(finish(value.into() * blend + dot * weight))?;
                        }
                    }
                }
            }
        }
        Ok(())
    })
}
