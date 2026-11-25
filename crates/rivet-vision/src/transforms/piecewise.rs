use super::{InterpolationMode, remap::SpatialRemap};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PiecewiseAffineConfig {
    pub scale_range: (f64, f64),
    pub nb_rows_range: (u32, u32),
    pub nb_cols_range: (u32, u32),
    pub absolute_scale: bool,
    pub interpolation: InterpolationMode,
    pub fill: u8,
    pub fill_mask: i64,
}

impl PiecewiseAffineConfig {
    pub(super) fn valid(&self) -> bool {
        let (lo, hi) = self.scale_range;
        lo >= 0.0
            && lo <= hi
            && hi.is_finite()
            && [self.nb_rows_range, self.nb_cols_range]
                .into_iter()
                .all(|(lo, hi)| lo >= 2 && lo <= hi && hi <= 128)
            && matches!(
                self.interpolation,
                InterpolationMode::Nearest | InterpolationMode::Bilinear
            )
    }

    pub(crate) fn sample(
        &self,
        h: usize,
        w: usize,
        rng: &mut RandomStream,
    ) -> crate::errors::RivetResult<TriangleGrid> {
        let (lo, hi) = self.scale_range;
        let sigma = lo + rng.next_f64() * (hi - lo);
        let mut count = |(lo, hi): (u32, u32)| rng.gen_range_usize(lo as usize..hi as usize + 1);
        let rows = count(self.nb_rows_range)?;
        let cols = count(self.nb_cols_range)?;
        let spread = if self.absolute_scale {
            [sigma; 2]
        } else {
            [sigma * w as f64, sigma * h as f64]
        };
        Ok(TriangleGrid::new(
            h,
            w,
            rows,
            cols,
            |_| {
                let (x, y) = super::noise::normal_pair(rng);
                [
                    if x == 0.0 { 0.0 } else { x * spread[0] },
                    if y == 0.0 { 0.0 } else { y * spread[1] },
                ]
            },
            sigma == 0.0,
        ))
    }
}

type Point = [f64; 2];

pub(crate) struct TriangleGrid {
    rows: usize,
    cols: usize,
    step: Point,
    controls: Vec<Point>,
    identity: bool,
}

impl TriangleGrid {
    pub(crate) fn new(
        h: usize,
        w: usize,
        rows: usize,
        cols: usize,
        mut offset: impl FnMut(Point) -> Point,
        identity: bool,
    ) -> Self {
        let step = [
            w.saturating_sub(1) as f64 / (cols - 1) as f64,
            h.saturating_sub(1) as f64 / (rows - 1) as f64,
        ];
        let identity = identity || step.contains(&0.0);
        let mut controls = Vec::new();
        if !identity {
            controls.reserve(rows * cols);
            for y in 0..rows {
                for x in 0..cols {
                    let p = [x as f64 * step[0], y as f64 * step[1]];
                    let delta = offset(p);
                    // Bounded jitter keeps triangle Jacobians positive.
                    controls.push(std::array::from_fn(|i| {
                        p[i] + delta[i].clamp(-0.2 * step[i], 0.2 * step[i])
                    }));
                }
            }
        }
        Self {
            rows,
            cols,
            step,
            controls,
            identity,
        }
    }

    fn cell(&self, p: Point) -> (usize, usize) {
        (
            ((p[0] / self.step[0]).floor() as usize).min(self.cols - 2),
            ((p[1] / self.step[1]).floor() as usize).min(self.rows - 2),
        )
    }

    fn triangle(&self, x: usize, y: usize, upper: bool) -> ([Point; 3], [Point; 3]) {
        let vertices = if upper {
            [(x, y), (x + 1, y), (x + 1, y + 1)]
        } else {
            [(x, y), (x, y + 1), (x + 1, y + 1)]
        };
        (
            vertices.map(|(x, y)| [x as f64 * self.step[0], y as f64 * self.step[1]]),
            vertices.map(|(x, y)| self.controls[y * self.cols + x]),
        )
    }
}

fn barycentric([a, b, c]: [Point; 3], p: Point) -> [f64; 3] {
    let (ux, uy) = (b[0] - a[0], b[1] - a[1]);
    let (vx, vy) = (c[0] - a[0], c[1] - a[1]);
    let (px, py) = (p[0] - a[0], p[1] - a[1]);
    let det = ux * vy - uy * vx;
    let u = (px * vy - py * vx) / det;
    let v = (ux * py - uy * px) / det;
    [1.0 - u - v, u, v]
}

fn blend(points: [Point; 3], b: [f64; 3]) -> Point {
    std::array::from_fn(|i| points[0][i] * b[0] + points[1][i] * b[1] + points[2][i] * b[2])
}

impl SpatialRemap for TriangleGrid {
    fn source(&self, p: Point) -> Point {
        if self.identity {
            return p;
        }
        let (x, y) = self.cell(p);
        let upper = p[0] / self.step[0] - x as f64 >= p[1] / self.step[1] - y as f64;
        let (regular, warped) = self.triangle(x, y, upper);
        blend(warped, barycentric(regular, p))
    }

    fn destination(&self, p: Point) -> Point {
        if self.identity {
            return p;
        }
        let (x, y) = self.cell(p);
        let mut best = (f64::INFINITY, p);
        for yy in y.saturating_sub(1)..=(y + 1).min(self.rows - 2) {
            for xx in x.saturating_sub(1)..=(x + 1).min(self.cols - 2) {
                for upper in [false, true] {
                    let (regular, warped) = self.triangle(xx, yy, upper);
                    let weights = barycentric(warped, p);
                    let outside = weights.iter().map(|v| (-v).max(0.0)).sum::<f64>();
                    if outside < best.0 {
                        best = (outside, blend(regular, weights));
                    }
                }
            }
        }
        best.1
    }

    fn is_identity(&self) -> bool {
        self.identity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_triangles_roundtrip_pixel_centers() {
        let grid = TriangleGrid::new(
            9,
            13,
            4,
            5,
            |[x, y]| [(x * 13.0 + y).sin() * 10.0, (y * 7.0 + x).cos() * 10.0],
            false,
        );
        for y in 0..9 {
            for x in 0..13 {
                let p = [x as f64, y as f64];
                let recovered = grid.destination(grid.source(p));
                for (a, b) in recovered.into_iter().zip(p) {
                    assert!((a - b).abs() < 1e-9);
                }
            }
        }
    }
}
