use crate::errors::{RivetResult, invalid_argument};
use crate::transforms::contiguous_values;
use rivet_core::{CpuStorageRef, DType, Tensor, WithDType};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactElement {
    Text,
    Rectangle,
    Arrow,
    Line,
    Callout,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnnotationArtifactsConfig {
    pub elements: Vec<(ArtifactElement, f64)>,
    pub count_range: (u32, u32),
    pub thickness_range: (u32, u32),
    pub corner_prob: f64,
    pub black_white_prob: f64,
}

#[derive(Clone, Copy)]
enum Mark {
    Color([u8; 3]),
    Copy(usize),
}

struct Raster {
    width: usize,
    height: usize,
    marks: Vec<Option<Mark>>,
}

impl Raster {
    fn put(&mut self, x: i64, y: i64, mark: Mark) {
        if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
            self.marks[y as usize * self.width + x as usize] = Some(mark);
        }
    }

    fn line(
        &mut self,
        [mut x, mut y]: [i64; 2],
        [end_x, end_y]: [i64; 2],
        thickness: i64,
        color: [u8; 3],
        style: usize,
    ) {
        let (dx, dy) = ((end_x - x).abs(), -(end_y - y).abs());
        let (sx, sy) = ((end_x - x).signum(), (end_y - y).signum());
        let mut error = dx + dy;
        let mut n = 0;
        loop {
            if style == 0 || (style == 1 && n % 8 < 5) || (style == 2 && n % 4 == 0) {
                for oy in 0..thickness {
                    for ox in 0..thickness {
                        self.put(
                            x + ox - thickness / 2,
                            y + oy - thickness / 2,
                            Mark::Color(color),
                        );
                    }
                }
            }
            if x == end_x && y == end_y {
                break;
            }
            let double = 2 * error;
            if double >= dy {
                error += dy;
                x += sx;
            }
            if double <= dx {
                error += dx;
                y += sy;
            }
            n += 1;
        }
    }

    fn rect(&mut self, x: i64, y: i64, w: i64, h: i64, t: i64, color: [u8; 3]) {
        let corners = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
        for i in 0..4 {
            self.line(corners[i], corners[(i + 1) % 4], t, color, 0);
        }
    }
}

// Five rows of three bitmap bits for 0–9 and A–Z.
const GLYPHS: [u32; 36] = [
    0x75757, 0x26227, 0x71747, 0x71717, 0x55711, 0x74717, 0x74757, 0x71111, 0x75757, 0x75717,
    0x25755, 0x65656, 0x34443, 0x65556, 0x74647, 0x74644, 0x34553, 0x55755, 0x72227, 0x11153,
    0x55655, 0x44447, 0x57755, 0x57775, 0x25552, 0x75744, 0x25573, 0x65655, 0x34716, 0x72222,
    0x55557, 0x55552, 0x55775, 0x55255, 0x55222, 0x71247,
];

impl AnnotationArtifactsConfig {
    pub(in crate::transforms) fn valid(&self) -> bool {
        let weights = self.elements.iter().map(|&(_, w)| w).sum::<f64>();
        !self.elements.is_empty()
            && weights.is_finite()
            && weights > 0.0
            && self
                .elements
                .iter()
                .all(|&(_, w)| w >= 0.0 && w.is_finite())
            && self.count_range.0 <= self.count_range.1
            && self.count_range.1 <= 64
            && self.thickness_range.0 > 0
            && self.thickness_range.0 <= self.thickness_range.1
            && self.thickness_range.1 <= 64
            && (0.0..=1.0).contains(&self.corner_prob)
            && (0.0..=1.0).contains(&self.black_white_prob)
    }

    pub(in crate::transforms) fn apply(
        &self,
        input: &Tensor,
        rng: &mut RandomStream,
    ) -> RivetResult<Tensor> {
        let [h, w, c] = *input.dims() else {
            return Err(invalid_argument("AnnotationArtifacts expects HWC input"));
        };
        if h == 0 || w == 0 || c == 0 || !matches!(input.dtype(), DType::U8 | DType::F32) {
            return Err(invalid_argument(
                "AnnotationArtifacts requires nonempty uint8 or float32 HWC input",
            ));
        }
        let (lo, hi) = self.count_range;
        let count = rng.gen_range_usize(lo as usize..hi as usize + 1)?;
        if count == 0 {
            return Ok(input.clone());
        }
        let mut raster = Raster {
            width: w,
            height: h,
            marks: vec![None; h * w],
        };
        let total = self.elements.iter().map(|&(_, w)| w).sum::<f64>();
        let limit = h.min(w) as f64;
        let fits = |v: i64, n: usize| v.clamp(0, n as i64 - 1);
        for _ in 0..count {
            let mut ticket = rng.next_f64() * total;
            let mut element = self.elements.last().expect("validated weights").0;
            for &(kind, weight) in &self.elements {
                if ticket < weight {
                    element = kind;
                    break;
                }
                ticket -= weight;
            }
            let (lo, hi) = self.thickness_range;
            let thick = rng.gen_range_usize(lo as usize..hi as usize + 1)? as i64;
            let color = if rng.next_f64() < self.black_white_prob {
                if rng.next_f64() < 0.5 {
                    [0; 3]
                } else {
                    [255; 3]
                }
            } else {
                [255, 0, 0]
            };
            let (mut x, mut y) = (
                rng.gen_range_usize(0..w)? as i64,
                rng.gen_range_usize(0..h)? as i64,
            );
            if rng.next_f64() < self.corner_prob {
                x = if x < w as i64 / 2 {
                    x / 3
                } else {
                    w as i64 - 1 - (w as i64 - 1 - x) / 3
                };
                y = if y < h as i64 / 2 {
                    y / 3
                } else {
                    h as i64 - 1 - (h as i64 - 1 - y) / 3
                };
            }
            let len = (limit * (0.1 + 0.7 * rng.next_f64())).round().max(1.0);
            let style = rng.gen_range_usize(0..3)?;
            match element {
                ArtifactElement::Line | ArtifactElement::Arrow => {
                    let angle = if element == ArtifactElement::Line {
                        rng.gen_range_usize(0..4)? as f64 * std::f64::consts::FRAC_PI_2
                    } else {
                        rng.next_f64() * std::f64::consts::TAU
                    };
                    let end = [
                        fits(x + (len * angle.cos()).round() as i64, w),
                        fits(y + (len * angle.sin()).round() as i64, h),
                    ];
                    raster.line([x, y], end, thick, color, style);
                    if element == ArtifactElement::Arrow {
                        for tip in [angle - 0.5, angle + 0.5] {
                            let tail = [
                                fits(end[0] - (len * 0.3 * tip.cos()).round() as i64, w),
                                fits(end[1] - (len * 0.3 * tip.sin()).round() as i64, h),
                            ];
                            raster.line(end, tail, thick, color, 0);
                        }
                    }
                }
                ArtifactElement::Rectangle | ArtifactElement::Callout => {
                    let rw = ((0.1 + 0.25 * rng.next_f64()) * w as f64).round().max(1.0) as i64;
                    let rh = ((0.1 + 0.25 * rng.next_f64()) * h as f64).round().max(1.0) as i64;
                    x = x.min((w as i64 - 1 - rw).max(0));
                    y = y.min((h as i64 - 1 - rh).max(0));
                    let (right, bottom) = (fits(x + rw, w), fits(y + rh, h));
                    if element == ArtifactElement::Callout {
                        let (sx, sy) = (
                            rng.gen_range_usize(0..w)? as i64,
                            rng.gen_range_usize(0..h)? as i64,
                        );
                        for py in y + 1..bottom {
                            for px in x + 1..right {
                                let spx = fits(sx + (px - x) / 2, w);
                                let spy = fits(sy + (py - y) / 2, h);
                                raster.put(px, py, Mark::Copy(spy as usize * w + spx as usize));
                            }
                        }
                        raster.line([sx, sy], [x, y], thick, color, style);
                    }
                    raster.rect(x, y, right - x, bottom - y, thick, color);
                }
                ArtifactElement::Text => {
                    let count = rng.gen_range_usize(1..6)?;
                    let scale = rng.gen_range_usize(1..3)? as i64;
                    let max_x = (w as i64 - count as i64 * 4 * scale).max(0);
                    let max_y = (h as i64 - 5 * scale).max(0);
                    x = x.min(max_x);
                    y = y.min(max_y);
                    for n in 0..count {
                        let glyph = GLYPHS[rng.gen_range_usize(0..GLYPHS.len())?];
                        for gy in 0..5 {
                            for gx in 0..3 {
                                if glyph & (1 << ((4 - gy) * 4 + 2 - gx)) != 0 {
                                    for dy in 0..scale {
                                        for dx in 0..scale {
                                            raster.put(
                                                x + (n as i64 * 4 + gx) * scale + dx,
                                                y + gy * scale + dy,
                                                Mark::Color(color),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let input = input.contiguous()?;
        Ok(input.with_cpu_storage(|storage, layout| match storage {
            CpuStorageRef::U8(v) => write(
                &input,
                contiguous_values(v, layout)?,
                &raster.marks,
                c,
                |v| v,
            ),
            CpuStorageRef::F32(v) => write(
                &input,
                contiguous_values(v, layout)?,
                &raster.marks,
                c,
                |v| f32::from(v) / 255.0,
            ),
            _ => unreachable!(),
        })?)
    }
}

fn write<T: WithDType>(
    input: &Tensor,
    pixels: &[T],
    marks: &[Option<Mark>],
    channels: usize,
    cast: impl Fn(u8) -> T,
) -> rivet_core::Result<Tensor> {
    Tensor::from_exact_writer(input.dims(), input.device(), |out| {
        let mut next = 0;
        for (index, &mark) in marks.iter().enumerate() {
            let Some(mark) = mark else {
                continue;
            };
            if next < index {
                out.extend_from_slice(&pixels[next * channels..index * channels])?;
            }
            match mark {
                Mark::Copy(source) => {
                    out.extend_from_slice(&pixels[source * channels..(source + 1) * channels])?
                }
                Mark::Color(color) => {
                    for c in 0..channels {
                        out.write_next(cast(if color[0] == color[1] {
                            color[0]
                        } else {
                            color.get(c).copied().unwrap_or(0)
                        }))?;
                    }
                }
            }
            next = index + 1;
        }
        out.extend_from_slice(&pixels[next * channels..])
    })
}
