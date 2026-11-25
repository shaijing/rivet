use super::{RivetResult, contiguous_values, invalid_argument};
use rivet_core::{CpuStorageRef, DType, Tensor, WithDType};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SuperpixelsConfig {
    pub p_replace_range: (f64, f64),
    pub n_segments_range: (u32, u32),
    pub max_size: Option<u32>,
}

impl SuperpixelsConfig {
    pub(super) fn valid(&self) -> bool {
        let (lo, hi) = self.p_replace_range;
        let (min, max) = self.n_segments_range;
        lo >= 0.0
            && lo <= hi
            && hi <= 1.0
            && min > 0
            && min <= max
            && max <= 4096
            && self.max_size != Some(0)
    }

    pub(super) fn apply(&self, input: &Tensor, rng: &mut RandomStream) -> RivetResult<Tensor> {
        if !matches!(input.dims(), [h,w,c] if *h>0 && *w>0 && matches!(c,1|3))
            || !matches!(input.dtype(), DType::U8 | DType::F32)
        {
            return Err(invalid_argument(
                "Superpixels requires nonempty uint8 or float32 HWC grayscale or RGB input",
            ));
        }
        let (lo, hi) = self.p_replace_range;
        let probability = lo + rng.next_f64() * (hi - lo);
        if probability == 0.0 {
            return Ok(input.clone());
        }
        let (lo, hi) = self.n_segments_range;
        let segments = rng.gen_range_usize(lo as usize..hi as usize + 1)?;
        let input = input.contiguous()?;
        Ok(input.with_cpu_storage(|storage, layout| match storage {
            CpuStorageRef::U8(v) => write(
                &input,
                contiguous_values(v, layout)?,
                1.0 / 255.0,
                segments,
                self.max_size,
                probability,
                rng,
                |v| v.round() as u8,
            ),
            CpuStorageRef::F32(v) => write(
                &input,
                contiguous_values(v, layout)?,
                1.0,
                segments,
                self.max_size,
                probability,
                rng,
                |v| v as f32,
            ),
            _ => unreachable!(),
        })?)
    }
}

fn segment(colors: &[[f64; 3]], h: usize, w: usize, target: usize) -> (Vec<usize>, usize) {
    let step = (h as f64 * w as f64 / target as f64).sqrt().max(1.0);
    let (rows, cols) = (
        ((h as f64 / step).round() as usize).clamp(1, h),
        ((w as f64 / step).round() as usize).clamp(1, w),
    );
    let (dy, dx) = (h as f64 / rows as f64, w as f64 / cols as f64);
    let mut centers = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let (y, x) = (
                ((r as f64 + 0.5) * dy) as usize,
                ((c as f64 + 0.5) * dx) as usize,
            );
            let [r, g, b] = colors[y * w + x];
            centers.push([x as f64, y as f64, r, g, b]);
        }
    }
    let mut labels = vec![0; h * w];
    let mut distances = vec![f64::INFINITY; h * w];
    let mut sums = vec![([0.0; 5], 0usize); centers.len()];
    let distance = |center: [f64; 5], x: usize, y: usize, color: [f64; 3]| {
        ((x as f64 - center[0]) / dx).powi(2) * 0.04
            + ((y as f64 - center[1]) / dy).powi(2) * 0.04
            + (color[0] - center[2]).powi(2)
            + (color[1] - center[3]).powi(2)
            + (color[2] - center[4]).powi(2)
    };
    for _ in 0..8 {
        for (i, &color) in colors.iter().enumerate() {
            let (y, x) = (i / w, i % w);
            let home = (y * rows / h) * cols + x * cols / w;
            labels[i] = home;
            distances[i] = distance(centers[home], x, y, color);
        }
        for (i, &center) in centers.iter().enumerate() {
            let top = (center[1] - dy).floor().max(0.0) as usize;
            let left = (center[0] - dx).floor().max(0.0) as usize;
            let bottom = ((center[1] + dy).ceil() as usize).min(h - 1);
            let right = ((center[0] + dx).ceil() as usize).min(w - 1);
            for y in top..=bottom {
                for x in left..=right {
                    let p = y * w + x;
                    let d = distance(center, x, y, colors[p]);
                    if d < distances[p] {
                        distances[p] = d;
                        labels[p] = i;
                    }
                }
            }
        }
        sums.fill(([0.0; 5], 0));
        for (i, &label) in labels.iter().enumerate() {
            let (sum, count) = &mut sums[label];
            let [r, g, b] = colors[i];
            for (s, v) in sum
                .iter_mut()
                .zip([(i % w) as f64, (i / w) as f64, r, g, b])
            {
                *s += v;
            }
            *count += 1;
        }
        let mut delta = 0.0f64;
        for (center, (sum, n)) in centers.iter_mut().zip(&sums) {
            if *n == 0 {
                continue;
            }
            for (v, total) in center.iter_mut().zip(sum) {
                let next = *total / *n as f64;
                delta = delta.max((*v - next).abs());
                *v = next;
            }
        }
        if delta < 1e-6 {
            break;
        }
    }
    // Split disconnected islands so every replacement region is connected.
    let mut connected = vec![usize::MAX; labels.len()];
    let mut queue = Vec::new();
    let mut count = 0;
    for i in 0..labels.len() {
        if connected[i] != usize::MAX {
            continue;
        }
        connected[i] = count;
        queue.clear();
        queue.push(i);
        let mut cursor = 0;
        while cursor < queue.len() {
            let p = queue[cursor];
            cursor += 1;
            for next in [
                (p % w > 0).then(|| p - 1),
                (p % w + 1 < w).then(|| p + 1),
                (p >= w).then(|| p - w),
                (p + w < labels.len()).then(|| p + w),
            ]
            .into_iter()
            .flatten()
            {
                if connected[next] == usize::MAX && labels[next] == labels[i] {
                    connected[next] = count;
                    queue.push(next);
                }
            }
        }
        count += 1;
    }
    (connected, count)
}

fn write<T: WithDType + Into<f64>>(
    input: &Tensor,
    pixels: &[T],
    scale: f64,
    segments: usize,
    max_size: Option<u32>,
    probability: f64,
    rng: &mut RandomStream,
    cast: impl Fn(f64) -> T,
) -> rivet_core::Result<Tensor> {
    let [h, w, c] = *input.dims() else {
        unreachable!()
    };
    let max = h.max(w);
    let limit = max_size.map_or(max, |n| max.min(n as usize));
    let (sh, sw) = ((h * limit / max).max(1), (w * limit / max).max(1));
    let mut colors = Vec::with_capacity(sh * sw);
    for y in 0..sh {
        for x in 0..sw {
            let p = (y * h / sh * w + x * w / sw) * c;
            let r = pixels[p].into() * scale;
            colors.push(if c == 1 {
                [r; 3]
            } else {
                [
                    r,
                    pixels[p + 1].into() * scale,
                    pixels[p + 2].into() * scale,
                ]
            });
        }
    }
    let (labels, n) = segment(&colors, sh, sw, segments);
    let mut sums = vec![([0.0; 3], 0usize); n];
    let label = |y, x| labels[y * sh / h * sw + x * sw / w];
    for (y, row) in pixels.chunks_exact(w * c).enumerate() {
        for (x, px) in row.chunks_exact(c).enumerate() {
            let (sum, count) = &mut sums[label(y, x)];
            for (s, &v) in sum.iter_mut().zip(px) {
                *s += v.into();
            }
            *count += 1;
        }
    }
    let replacements: Vec<_> = sums
        .into_iter()
        .map(|(sum, count)| {
            (count > 0 && rng.next_f64() < probability).then(|| sum.map(|v| cast(v / count as f64)))
        })
        .collect();
    Tensor::from_exact_writer(input.dims(), input.device(), |out| {
        for (y, row) in pixels.chunks_exact(w * c).enumerate() {
            for (x, px) in row.chunks_exact(c).enumerate() {
                let values = replacements[label(y, x)].as_ref().map_or(px, |v| &v[..c]);
                out.extend_from_slice(values)?;
            }
        }
        Ok(())
    })
}
