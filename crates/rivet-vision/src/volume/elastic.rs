use super::{VolumeBorder, VolumeInterpolation, VolumeSample, warp};
use crate::{
    RandomStream,
    errors::{RivetResult, invalid_argument},
};
use rivet_core::Tensor;

#[derive(Clone, Debug)]
pub struct ElasticTransform3DConfig {
    pub displacement_range: [f64; 2],
    pub control_grid_shape: [usize; 2],
    pub interpolation: VolumeInterpolation,
    pub border: VolumeBorder,
    pub fill: Vec<f32>,
    pub fill_mask: i64,
}

struct ElasticField {
    coefficients: [Vec<[f64; 3]>; 3],
    grid: [usize; 2],
    span: [f64; 3],
}

impl ElasticField {
    fn displacement(&self, point: [f64; 3]) -> [f64; 3] {
        let normalized: [f64; 3] = std::array::from_fn(|i| {
            if self.span[i] == 0.0 {
                0.0
            } else {
                (point[i] / self.span[i]).clamp(0.0, 1.0)
            }
        });
        let mut delta = [0.0; 3];
        for (plane, [row_axis, col_axis]) in [[1, 0], [2, 0], [2, 1]].into_iter().enumerate() {
            let (row, rw) = spline_basis(normalized[row_axis], self.grid[0]);
            let (col, cw) = spline_basis(normalized[col_axis], self.grid[1]);
            for (r, wr) in rw.into_iter().enumerate() {
                for (c, wc) in cw.into_iter().enumerate() {
                    let coefficient = self.coefficients[plane][(row + r) * self.grid[1] + col + c];
                    for i in 0..3 {
                        delta[i] += coefficient[i] * wr * wc;
                    }
                }
            }
        }
        delta
    }
}

fn spline_basis(position: f64, count: usize) -> (usize, [f64; 4]) {
    let scale = position * (count - 3) as f64;
    let start = (scale as usize).min(count - 4);
    let t = scale - start as f64;
    let u = 1.0 - t;
    let t2 = t * t;
    (
        start,
        [
            u * u * u / 6.0,
            (4.0 - 6.0 * t2 + 3.0 * t2 * t) / 6.0,
            (1.0 + 3.0 * t + 3.0 * t2 * u) / 6.0,
            t2 * t / 6.0,
        ],
    )
}

impl VolumeSample {
    pub fn elastic_transform(
        mut self,
        config: &ElasticTransform3DConfig,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        warp::validate_fill(&self, &config.fill, config.fill_mask, config.border)?;
        let [lo, hi] = config.displacement_range;
        let [rows, cols] = config.control_grid_shape;
        if rows < 4
            || cols < 4
            || lo < 0.0
            || lo > hi
            || !lo.is_finite()
            || !hi.is_finite()
            || 2.0 * hi * ((rows - 3) as f64).hypot((cols - 3) as f64) >= 0.75
        {
            return Err(invalid_argument(
                "ElasticTransform3D requires a cubic grid and bounded displacement",
            ));
        }
        let [d, h, w] = self.spatial_shape()?;
        let span = [w - 1, h - 1, d - 1].map(|n| n as f64);
        let shortest = span.into_iter().filter(|&v| v > 0.0).reduce(f64::min);
        let Some(shortest) = shortest.filter(|_| hi > 0.0) else {
            return Ok(self);
        };
        let radius = shortest * (lo + (hi - lo) * rng.next_f64());
        if radius == 0.0 {
            return Ok(self);
        }
        let size = rows
            .checked_mul(cols)
            .ok_or_else(|| invalid_argument("elastic control grid is too large"))?;
        let mut field = ElasticField {
            coefficients: std::array::from_fn(|_| Vec::new()),
            grid: config.control_grid_shape,
            span,
        };
        // Scale each component so each control vector has norm at most radius.
        // Divide by three here to average the planes when evaluating the field.
        let extent = radius / (3.0 * 3.0_f64.sqrt());
        for plane in &mut field.coefficients {
            plane
                .try_reserve_exact(size)
                .map_err(|_| invalid_argument("elastic control grid is too large"))?;
            plane.extend((0..size).map(|_| {
                span.map(|s| {
                    if s == 0.0 {
                        0.0
                    } else {
                        (2.0 * rng.next_f64() - 1.0) * extent
                    }
                })
            }));
        }
        let inverse = |point: [f64; 3]| {
            let delta = field.displacement(point);
            std::array::from_fn(|i| point[i] + delta[i])
        };
        self.volume = warp::warp(
            &self.volume,
            config.interpolation,
            config.border,
            &config.fill,
            None,
            inverse,
        )?;
        if let Some(mask) = &self.mask {
            self.mask = Some(warp::warp(
                mask,
                VolumeInterpolation::Nearest,
                config.border,
                &[],
                Some(config.fill_mask),
                inverse,
            )?);
        }
        if let Some(points) = &self.keypoints {
            let mut coords = points.to_vec::<f32>()?;
            for point in coords.chunks_exact_mut(3) {
                let source = [
                    f64::from(point[0]),
                    f64::from(point[1]),
                    f64::from(point[2]),
                ];
                let mut dest = source;
                // Invert source = destination + displacement(destination).
                for _ in 0..96 {
                    let delta = field.displacement(dest);
                    let next = std::array::from_fn(|i| source[i] - delta[i]);
                    let converged = (0..3).all(|i| (next[i] - dest[i]).abs() < 1e-8);
                    dest = next;
                    if converged {
                        break;
                    }
                }
                for i in 0..3 {
                    point[i] = dest[i] as f32;
                }
            }
            self.keypoints = Some(Tensor::from_vec(coords, points.dims(), points.device())?);
        }
        Ok(self)
    }
}
