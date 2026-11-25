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
