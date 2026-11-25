use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::ImageSample;
use crate::transforms::with_rgb_pixels;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianBlurConfig {
    pub sigma: f32,
    /// Standard deviations in torchvision's (horizontal, vertical) order.
    pub axis_sigma: Option<[f32; 2]>,
    pub kernel_size: Option<u32>,
    /// Kernel dimensions in torchvision's (width, height) order.
    pub kernel_dimensions: Option<[u32; 2]>,
}

impl GaussianBlurConfig {
    pub const fn new(sigma: f32) -> Self {
        Self {
            sigma,
            axis_sigma: None,
            kernel_size: None,
            kernel_dimensions: None,
        }
    }

    pub const fn with_kernel_size(mut self, kernel_size: u32) -> Self {
        self.kernel_size = Some(kernel_size);
        self
    }

    pub const fn with_axis_sigmas(mut self, sigma: [f32; 2]) -> Self {
        self.axis_sigma = Some(sigma);
        self
    }

    pub const fn with_kernel_dimensions(mut self, dimensions: [u32; 2]) -> Self {
        self.kernel_dimensions = Some(dimensions);
        self
    }

    pub fn validate(&self) -> RivetResult<()> {
        if self.kernel_size.is_some_and(|k| k == 0 || k % 2 == 0)
            || self
                .kernel_dimensions
                .is_some_and(|ks| ks.iter().any(|k| *k == 0 || *k % 2 == 0))
        {
            return Err(invalid_argument(
                "gaussian_blur kernel size must be positive and odd",
            ));
        }
        let sigmas = self.axis_sigma.unwrap_or([self.sigma; 2]);
        if sigmas.iter().all(|s| s.is_finite() && *s >= 0.0) {
            Ok(())
        } else {
            Err(invalid_argument(
                "gaussian_blur sigma must be finite and non-negative",
            ))
        }
    }

    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = sample.into_decoded()?;
        let image = with_rgb_pixels(&sample.image, "gaussian_blur", |pixels, width, height| {
            let sigmas = self.axis_sigma.unwrap_or([self.sigma; 2]);
            if sigmas == [0.0, 0.0] {
                return Ok(rivet_core::Tensor::from_exact_iter(
                    pixels.iter().copied(),
                    [height, width, 3],
                    sample.image.device(),
                )?);
            }
            let default_kernel = |sigma: f32| {
                ((3.0 * sigma).ceil() as usize)
                    .checked_mul(2)
                    .and_then(|size| size.checked_add(1))
                    .ok_or_else(|| invalid_argument("gaussian_blur kernel size overflows usize"))
            };
            let dimensions = self.kernel_dimensions.unwrap_or_else(|| {
                let size = self.kernel_size.unwrap_or(0);
                [size, size]
            });
            let kernel_x = if dimensions[0] > 0 {
                dimensions[0] as usize
            } else {
                default_kernel(sigmas[0])?
            };
            let kernel_y = if dimensions[1] > 0 {
                dimensions[1] as usize
            } else {
                default_kernel(sigmas[1])?
            };
            let radius_x = kernel_x / 2;
            let radius_y = kernel_y / 2;
            if radius_x >= width || radius_y >= height {
                return Err(invalid_argument(
                    "gaussian_blur reflection padding must be smaller than image dimensions",
                ));
            }
            let weights = |kernel: usize, sigma: f32| -> Vec<f32> {
                if sigma == 0.0 {
                    let mut w = vec![0.0; kernel];
                    w[kernel / 2] = 1.0;
                    return w;
                }
                let weights: Vec<f32> = (0..kernel)
                    .map(|i| {
                        let x = (i as f32 - (kernel / 2) as f32) / sigma;
                        (-0.5 * x * x).exp()
                    })
                    .collect();
                let total: f32 = weights.iter().sum();
                weights.into_iter().map(|w| w / total).collect()
            };
            let weights_x = weights(kernel_x, sigmas[0]);
            let weights_y = weights(kernel_y, sigmas[1]);
            let reflect = |i: isize, len: usize| {
                if i < 0 {
                    (-i) as usize
                } else if i >= len as isize {
                    2 * len - i as usize - 2
                } else {
                    i as usize
                }
            };
            let image = rivet_core::Tensor::from_exact_writer::<u8, _, _>(
                [height, width, 3],
                &rivet_core::Device::Cpu,
                |out| {
                    for y in 0..height {
                        for x in 0..width {
                            let mut sums = [0.0f32; 3];
                            for (ky, wy) in weights_y.iter().enumerate() {
                                let sy =
                                    reflect(y as isize + ky as isize - radius_y as isize, height);
                                for (kx, wx) in weights_x.iter().enumerate() {
                                    let sx = reflect(
                                        x as isize + kx as isize - radius_x as isize,
                                        width,
                                    );
                                    let offset = (sy * width + sx) * 3;
                                    let weight = wy * wx;
                                    sums[0] += f32::from(pixels[offset]) * weight;
                                    sums[1] += f32::from(pixels[offset + 1]) * weight;
                                    sums[2] += f32::from(pixels[offset + 2]) * weight;
                                }
                            }
                            out.write_next(sums[0].round_ties_even().clamp(0.0, 255.0) as u8)?;
                            out.write_next(sums[1].round_ties_even().clamp(0.0, 255.0) as u8)?;
                            out.write_next(sums[2].round_ties_even().clamp(0.0, 255.0) as u8)?;
                        }
                    }
                    Ok(())
                },
            )?;
            Ok(image)
        })?;
        Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
            image,
            label: sample.label,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::GaussianBlurConfig;
    use crate::sample::image::{DecodedSample, ImageSample};
    use rivet_core::{DType, Device, Tensor};

    #[test]
    fn gaussian_blur_materializes_and_preserves_shape_dtype() {
        let input = Tensor::from_vec(vec![127u8; 7 * 7 * 3], [7, 7, 3], &Device::Cpu).unwrap();
        let output = GaussianBlurConfig::new(1.0)
            .apply(ImageSample::Decoded(DecodedSample {
                image: input.clone(),
                label: 2,
            }))
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dims(), [7, 7, 3]);
        assert_eq!(output.image.dtype(), DType::U8);
        assert!(!output.image.same_storage(&input));
    }

    #[test]
    fn gaussian_blur_rejects_invalid_sigma() {
        assert!(GaussianBlurConfig::new(-1.0).validate().is_err());
        assert!(GaussianBlurConfig::new(f32::NAN).validate().is_err());
    }
}

pub(super) fn motion_blur(
    input: &rivet_core::Tensor,
    length: u32,
    angle: f64,
) -> RivetResult<rivet_core::Tensor> {
    let (dy, dx) = angle.to_radians().sin_cos();
    let radius = i64::from(length / 2);
    let taps = (-radius..=radius)
        .map(|i| {
            let i = i as f64;
            ((i * dy).round() as i64, (i * dx).round() as i64)
        })
        .collect::<Vec<_>>();
    convolve_uniform(input, "MotionBlur", &taps)
}

pub(super) fn defocus(input: &rivet_core::Tensor, radius: u32) -> RivetResult<rivet_core::Tensor> {
    let radius = i64::from(radius);
    let taps = (-radius..=radius)
        .flat_map(|y| (-radius..=radius).map(move |x| (y, x)))
        .filter(|&(y, x)| y * y + x * x <= radius * radius)
        .collect::<Vec<_>>();
    convolve_uniform(input, "Defocus", &taps)
}

fn convolve_uniform(
    input: &rivet_core::Tensor,
    name: &str,
    taps: &[(i64, i64)],
) -> RivetResult<rivet_core::Tensor> {
    let count = taps.len() as u64;
    with_rgb_pixels(input, name, |pixels, width, height| {
        Ok(rivet_core::Tensor::from_exact_writer::<u8, _, _>(
            [height, width, 3],
            input.device(),
            |out| {
                for y in 0..height {
                    for x in 0..width {
                        let mut sums = [0u64; 3];
                        for &(dy, dx) in taps {
                            let sy = reflect101(y as i64 + dy, height);
                            let sx = reflect101(x as i64 + dx, width);
                            let offset = (sy * width + sx) * 3;
                            sums[0] += u64::from(pixels[offset]);
                            sums[1] += u64::from(pixels[offset + 1]);
                            sums[2] += u64::from(pixels[offset + 2]);
                        }
                        for sum in sums {
                            out.write_next(((sum + count / 2) / count) as u8)?;
                        }
                    }
                }
                Ok(())
            },
        )?)
    })
}

pub(super) fn unsharp_mask(
    sample: crate::sample::image::DecodedSample,
    sigma: f32,
    amount: f32,
    threshold: u8,
) -> RivetResult<rivet_core::Tensor> {
    let input = &sample.image;
    let [height, width, _] = *input.dims() else {
        return Err(invalid_argument("UnsharpMask requires a rank-3 image"));
    };
    if amount == 0.0 {
        return Ok(input.clone());
    }
    // Bound the radius for the existing reflect-padding Gaussian kernel.
    let radius = |len: usize| {
        ((3.0 * sigma).ceil() as u32).min((len.saturating_sub(1) as u32).min(u32::MAX / 2))
    };
    let blurred = GaussianBlurConfig::new(sigma)
        .with_kernel_dimensions([2 * radius(width) + 1, 2 * radius(height) + 1])
        .apply(ImageSample::Decoded(sample.clone()))?
        .into_decoded()?;
    with_rgb_pixels(input, "UnsharpMask", |pixels, _, _| {
        with_rgb_pixels(&blurred.image, "UnsharpMask", |smooth, _, _| {
            Ok(rivet_core::Tensor::from_exact_iter(
                pixels.iter().zip(smooth).map(|(&pixel, &blurred)| {
                    let detail = i16::from(pixel) - i16::from(blurred);
                    if detail.unsigned_abs() > u16::from(threshold) {
                        (f32::from(pixel) + amount * f32::from(detail))
                            .round()
                            .clamp(0.0, 255.0) as u8
                    } else {
                        pixel
                    }
                }),
                input.dims(),
                input.device(),
            )?)
        })
    })
}

pub(super) fn median_blur_3x3(input: &rivet_core::Tensor) -> RivetResult<rivet_core::Tensor> {
    use rivet_core::{CpuStorageRef, DType, Tensor};

    let [height, width, channels] = *input.dims() else {
        return Err(invalid_argument("MedianBlur expects a rank-3 HWC image"));
    };
    if height == 0 || width == 0 || !matches!(channels, 1 | 3) || input.dtype() != DType::U8 {
        return Err(invalid_argument(
            "MedianBlur requires nonempty uint8 HWC grayscale or RGB input",
        ));
    }
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::U8(values) = storage else {
            unreachable!()
        };
        let pixels = super::contiguous_values(values, layout)?;
        let stride = width * channels;
        Tensor::from_exact_writer(input.dims(), input.device(), |out| {
            for y in 0..height {
                let rows = [y.saturating_sub(1), y, (y + 1).min(height - 1)].map(|y| y * stride);
                for x in 0..width {
                    let cols =
                        [x.saturating_sub(1), x, (x + 1).min(width - 1)].map(|x| x * channels);
                    for c in 0..channels {
                        let [a, b, d] = rows.map(|row| row + c);
                        let [l, m, r] = cols;
                        let mut window = [
                            pixels[a + l],
                            pixels[a + m],
                            pixels[a + r],
                            pixels[b + l],
                            pixels[b + m],
                            pixels[b + r],
                            pixels[d + l],
                            pixels[d + m],
                            pixels[d + r],
                        ];
                        let (_, median, _) = window.select_nth_unstable(4);
                        out.write_next(*median)?;
                    }
                }
            }
            Ok(())
        })
    })?)
}

pub(super) fn reflect101(index: i64, len: usize) -> usize {
    let len = len as i64;
    if (0..len).contains(&index) {
        return index as usize;
    }
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len - 1);
    let index = index.rem_euclid(period);
    index.min(period - index) as usize
}

pub(super) fn box_blur(input: &rivet_core::Tensor, size: usize) -> RivetResult<rivet_core::Tensor> {
    let weights = vec![1.0 / (size * size) as f64; size * size];
    convolve_square(input, "Blur", size, &weights, super::PaddingMode::Reflect)
}

fn convolution_shape(input: &rivet_core::Tensor, name: &str) -> RivetResult<[usize; 3]> {
    use rivet_core::DType;
    let [height, width, channels] = *input.dims() else {
        return Err(invalid_argument(format!("{name} expects rank-3 HWC input")));
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
    Ok([height, width, channels])
}

fn convolve_square(
    input: &rivet_core::Tensor,
    name: &str,
    size: usize,
    weights: &[f64],
    padding: super::PaddingMode,
) -> RivetResult<rivet_core::Tensor> {
    use rivet_core::CpuStorageRef;

    let [height, width, channels] = convolution_shape(input, name)?;
    let coordinate = |i: i64, length: usize| {
        use super::PaddingMode::*;
        let len = length as i64;
        match padding {
            Reflect => Some(reflect101(i, length)),
            Edge => Some(i.clamp(0, len - 1) as usize),
            Symmetric => {
                let index = i.rem_euclid(2 * len);
                Some(index.min(2 * len - 1 - index) as usize)
            }
            Constant => (0..len).contains(&i).then_some(i as usize),
        }
    };
    let axis_offsets = |length, stride| {
        let radius = (size / 2) as i64;
        (0..length)
            .flat_map(|i| {
                (-radius..=radius)
                    .map(move |d| coordinate(i as i64 + d, length).map(|offset| offset * stride))
            })
            .collect::<Vec<_>>()
    };
    let rows = axis_offsets(height, width * channels);
    let cols = axis_offsets(width, channels);
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| match storage {
        CpuStorageRef::U8(values) => write_square(
            &input,
            super::contiguous_values(values, layout)?,
            &rows,
            &cols,
            channels,
            size,
            weights,
            |sum| sum.round().clamp(0.0, 255.0) as u8,
        ),
        CpuStorageRef::F32(values) => write_square(
            &input,
            super::contiguous_values(values, layout)?,
            &rows,
            &cols,
            channels,
            size,
            weights,
            |sum| {
                if matches!(
                    name,
                    "StochasticConvolution" | "Emboss" | "Enhance" | "RingingOvershoot"
                ) {
                    sum.clamp(0.0, 1.0) as f32
                } else {
                    sum as f32
                }
            },
        ),
        _ => unreachable!(),
    })?)
}

fn write_square<T: rivet_core::WithDType + Into<f64>>(
    input: &rivet_core::Tensor,
    pixels: &[T],
    rows: &[Option<usize>],
    cols: &[Option<usize>],
    channels: usize,
    size: usize,
    weights: &[f64],
    finish: impl Fn(f64) -> T,
) -> rivet_core::Result<rivet_core::Tensor> {
    let area = size * size;
    let channel_stride = if weights.len() == area { 0 } else { area };
    rivet_core::Tensor::from_exact_writer(input.dims(), input.device(), |out| {
        for ys in rows.chunks_exact(size) {
            for xs in cols.chunks_exact(size) {
                for c in 0..channels {
                    let mut sum = 0.0;
                    let start = c * channel_stride;
                    for (&y, weights) in ys
                        .iter()
                        .zip(weights[start..start + area].chunks_exact(size))
                    {
                        if let Some(y) = y {
                            for (&x, &weight) in xs.iter().zip(weights) {
                                if let Some(x) = x {
                                    sum += pixels[y + x + c].into() * weight;
                                }
                            }
                        }
                    }
                    out.write_next(finish(sum))?;
                }
            }
        }
        Ok(())
    })
}

pub(super) fn advanced_blur(
    input: &rivet_core::Tensor,
    size: usize,
    sigma: (f64, f64),
    angle: f64,
    beta: f64,
    noise: (f64, f64),
    rng: &mut rivet_data::random::RandomStream,
) -> RivetResult<rivet_core::Tensor> {
    let radius = (size / 2) as i32;
    let (sin, cos) = angle.to_radians().sin_cos();
    let mut weights = Vec::with_capacity(size * size);
    let noise_min = noise.0 / noise.1;
    for y in -radius..=radius {
        for x in -radius..=radius {
            let x = f64::from(x);
            let y = f64::from(y);
            let u = (cos * x + sin * y) / sigma.0;
            let v = (cos * y - sin * x) / sigma.1;
            let multiplier = noise_min + rng.next_f64() * (1.0 - noise_min);
            weights.push((-0.5 * (u * u + v * v).powf(beta)).exp() * multiplier);
        }
    }
    let sum: f64 = weights.iter().sum();
    if sum == 0.0 {
        return Ok(input.clone());
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    convolve_square(
        input,
        "AdvancedBlur",
        size,
        &weights,
        super::PaddingMode::Reflect,
    )
}

pub(super) fn stochastic_convolution(
    input: &rivet_core::Tensor,
    size: usize,
    strength: f64,
    per_channel: bool,
    padding: super::PaddingMode,
    rng: &mut rivet_data::random::RandomStream,
) -> RivetResult<rivet_core::Tensor> {
    let name = "StochasticConvolution";
    let [_, _, channels] = convolution_shape(input, name)?;
    if strength == 0.0 {
        return Ok(input.clone());
    }
    let area = size * size;
    let kernel_count = if per_channel { channels } else { 1 };
    let count = area
        .checked_mul(kernel_count)
        .ok_or_else(|| invalid_argument("convolution kernel is too large"))?;
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(count)
        .map_err(|_| invalid_argument("convolution kernel is too large"))?;
    let sigma = strength / size as f64;
    for _ in 0..count.div_ceil(2) {
        let (a, b) = super::noise::normal_pair(rng);
        weights.push(a * sigma);
        if weights.len() < count {
            weights.push(b * sigma);
        }
    }
    for kernel in weights.chunks_exact_mut(area) {
        kernel[area / 2] += 1.0;
    }
    convolve_square(input, name, size, &weights, padding)
}

pub(super) fn zoom_blur(
    input: &rivet_core::Tensor,
    max_factor: f64,
    step: f64,
) -> RivetResult<rivet_core::Tensor> {
    use rivet_core::CpuStorageRef;

    let [height, width, channels] = convolution_shape(input, "ZoomBlur")?;
    let scales: Vec<_> = (1..=128)
        .map(|i| 1.0 + f64::from(i) * step)
        .take_while(|&factor| factor <= max_factor)
        .filter(|&factor| factor > 1.0)
        .map(f64::recip)
        .collect();
    if scales.is_empty() || (height, width) == (1, 1) {
        return Ok(input.clone());
    }

    let axis = |len: usize, stride: usize| {
        let center = (len - 1) as f64 * 0.5;
        (0..len)
            .flat_map(|i| {
                scales.iter().map(move |&scale| {
                    let source =
                        (center + (i as f64 - center) * scale).clamp(0.0, (len - 1) as f64);
                    let low = source as usize;
                    (
                        low * stride,
                        (low + 1).min(len - 1) * stride,
                        source - low as f64,
                    )
                })
            })
            .collect::<Vec<_>>()
    };
    let rows = axis(height, width * channels);
    let cols = axis(width, channels);
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| match storage {
        CpuStorageRef::U8(values) => write_zoom(
            &input,
            super::contiguous_values(values, layout)?,
            &rows,
            &cols,
            scales.len(),
            |value| value.round() as u8,
        ),
        CpuStorageRef::F32(values) => write_zoom(
            &input,
            super::contiguous_values(values, layout)?,
            &rows,
            &cols,
            scales.len(),
            |value| value as f32,
        ),
        _ => unreachable!(),
    })?)
}

fn write_zoom<T: rivet_core::WithDType + Into<f64>>(
    input: &rivet_core::Tensor,
    pixels: &[T],
    rows: &[(usize, usize, f64)],
    cols: &[(usize, usize, f64)],
    samples: usize,
    finish: impl Fn(f64) -> T,
) -> rivet_core::Result<rivet_core::Tensor> {
    let channels = input.dims()[2];
    let mean = 1.0 / (1 + samples) as f64;
    let mut pixel = 0;
    rivet_core::Tensor::from_exact_writer(input.dims(), input.device(), |out| {
        for ys in rows.chunks_exact(samples) {
            for xs in cols.chunks_exact(samples) {
                for c in 0..channels {
                    let mut sum = pixels[pixel + c].into();
                    for (&(top, bottom, wy), &(left, right, wx)) in ys.iter().zip(xs) {
                        let a = pixels[top + left + c].into();
                        let b = pixels[top + right + c].into();
                        let d = pixels[bottom + left + c].into();
                        let e = pixels[bottom + right + c].into();
                        let upper = a + wx * (b - a);
                        let lower = d + wx * (e - d);
                        sum += upper + wy * (lower - upper);
                    }
                    out.write_next(finish(sum * mean))?;
                }
                pixel += channels;
            }
        }
        Ok(())
    })
}

pub(super) fn glass_blur(
    input: &rivet_core::Tensor,
    sigma: f64,
    max_delta: usize,
    iterations: u32,
    rng: &mut rivet_data::random::RandomStream,
) -> RivetResult<rivet_core::Tensor> {
    use rivet_core::CpuStorageRef;

    let name = "GlassBlur";
    let [height, width, channels] = convolution_shape(input, name)?;
    if (height, width) == (1, 1) {
        return Ok(input.clone());
    }
    let radius = (3.0 * sigma).ceil() as i32;
    let size = (2 * radius + 1) as usize;
    let mut weights = Vec::new();
    if sigma > 0.0 {
        let axis: Vec<_> = (-radius..=radius)
            .map(|d| (-0.5 * (f64::from(d) / sigma).powi(2)).exp())
            .collect();
        let total = axis.iter().sum::<f64>().powi(2);
        weights.extend(
            axis.iter()
                .flat_map(|&y| axis.iter().map(move |&x| x * y / total)),
        );
    }
    let smooth = |image: &rivet_core::Tensor| {
        if weights.is_empty() {
            Ok(image.clone())
        } else {
            convolve_square(image, name, size, &weights, super::PaddingMode::Reflect)
        }
    };
    let image = smooth(input)?.contiguous()?;
    let mut offsets: Vec<_> = (0..height * width).map(|i| i * channels).collect();
    let neighborhood = |i: usize, len: usize| {
        i.saturating_sub(max_delta)..i.saturating_add(max_delta).min(len - 1) + 1
    };
    for _ in 0..iterations {
        for y in 0..height {
            for x in 0..width {
                let other_y = rng.gen_range_usize(neighborhood(y, height))?;
                let other_x = rng.gen_range_usize(neighborhood(x, width))?;
                offsets.swap(y * width + x, other_y * width + other_x);
            }
        }
    }
    let shuffled = image.with_cpu_storage(|storage, layout| {
        macro_rules! permute {
            ($values:expr) => {{
                let values = super::contiguous_values($values, layout)?;
                rivet_core::Tensor::from_exact_writer(image.dims(), image.device(), |out| {
                    for &offset in &offsets {
                        for &value in &values[offset..offset + channels] {
                            out.write_next(value)?;
                        }
                    }
                    Ok(())
                })
            }};
        }
        match storage {
            CpuStorageRef::U8(values) => permute!(values),
            CpuStorageRef::F32(values) => permute!(values),
            _ => unreachable!(),
        }
    })?;
    smooth(&shuffled)
}

pub(super) fn emboss(
    input: &rivet_core::Tensor,
    alpha: f64,
    strength: f64,
) -> RivetResult<rivet_core::Tensor> {
    let gain = alpha * strength;
    if gain == 0.0 {
        convolution_shape(input, "Emboss")?;
        return Ok(input.clone());
    }
    // Opposite taps create diagonal relief. A unit DC gain preserves flats.
    let weights = [
        -gain,
        -gain * 0.5,
        0.0,
        -gain * 0.5,
        1.0,
        gain * 0.5,
        0.0,
        gain * 0.5,
        gain,
    ];
    convolve_square(input, "Emboss", 3, &weights, super::PaddingMode::Reflect)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnhanceMode {
    Edge,
    Detail,
}

pub(super) fn enhance(
    input: &rivet_core::Tensor,
    mode: EnhanceMode,
    alpha: f64,
) -> RivetResult<rivet_core::Tensor> {
    if alpha == 0.0 {
        convolution_shape(input, "Enhance")?;
        return Ok(input.clone());
    }
    // Pillow-inspired stencils with unit DC gain; fold the identity blend
    // into the center tap so clipping and rounding happen only once.
    let (corner, side) = match mode {
        EnhanceMode::Edge => (-0.5 * alpha, -0.5 * alpha),
        EnhanceMode::Detail => (0.0, -alpha / 6.0),
    };
    let center = 1.0 - 4.0 * (corner + side);
    let weights = [
        corner, side, corner, side, center, side, corner, side, corner,
    ];
    convolve_square(input, "Enhance", 3, &weights, super::PaddingMode::Reflect)
}

pub(super) fn ringing_overshoot(
    input: &rivet_core::Tensor,
    size: usize,
    cutoff: f64,
) -> RivetResult<rivet_core::Tensor> {
    let half = (size / 2) as i32;
    let mut weights = Vec::with_capacity(size * size);
    for y in -half..=half {
        for x in -half..=half {
            let radius = f64::from(x * x + y * y).sqrt();
            let z = cutoff * radius;
            // The radial impulse response of a circular low-pass filter,
            // scaled by its center value before normalizing the finite grid.
            let tap = if z < 1e-7 { 1.0 } else { 2.0 * libm::j1(z) / z };
            weights.push(tap);
        }
    }
    let sum = weights.iter().sum::<f64>();
    if !sum.is_finite() || sum.abs() < 1e-12 {
        return Err(invalid_argument("RingingOvershoot kernel has zero DC gain"));
    }
    for tap in &mut weights {
        *tap /= sum;
    }
    convolve_square(
        input,
        "RingingOvershoot",
        size,
        &weights,
        super::PaddingMode::Reflect,
    )
}
