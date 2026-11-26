use crate::transforms::{InterpolationMode, spatial::remap::SpatialRemap};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThinPlateSplineConfig {
    pub scale_range: (f64, f64),
    pub num_control_points: u32,
    pub interpolation: InterpolationMode,
    pub fill: u8,
    pub fill_mask: i64,
}

impl ThinPlateSplineConfig {
    pub(in crate::transforms) fn valid(&self) -> bool {
        let (lo, hi) = self.scale_range;
        lo >= 0.0
            && lo <= hi
            && hi <= 1.0
            && (2..=12).contains(&self.num_control_points)
            && matches!(
                self.interpolation,
                InterpolationMode::Nearest | InterpolationMode::Bilinear
            )
    }

    pub(crate) fn sample(&self, h: usize, w: usize, rng: &mut RandomStream) -> SplineMap {
        let extent = [w.saturating_sub(1) as f64, h.saturating_sub(1) as f64];
        let (lo, hi) = self.scale_range;
        let scale = lo + rng.next_f64() * (hi - lo);
        if scale == 0.0 || extent.contains(&0.0) {
            return SplineMap {
                extent,
                controls: Vec::new(),
                weights: Vec::new(),
            };
        }
        let side = self.num_control_points as usize;
        let controls: Vec<_> = (0..side)
            .flat_map(|y| {
                (0..side).map(move |x| [x as f64 / (side - 1) as f64, y as f64 / (side - 1) as f64])
            })
            .collect();
        let displacement: Vec<_> = controls
            .iter()
            .map(|_| {
                let (x, y) = crate::transforms::degradation::noise::normal_pair(rng);
                [x * scale, y * scale]
            })
            .collect();
        let Some(mut weights) = fit(&controls, &displacement) else {
            return SplineMap {
                extent,
                controls: Vec::new(),
                weights: Vec::new(),
            };
        };
        let n = controls.len();
        // On [0,1]^2, each partial derivative of r² log(r²) has
        // magnitude < 5. Bound the displacement Jacobian's row sums.
        let bound = (0..2)
            .map(|i| {
                10.0 * weights[..n].iter().map(|w| w[i].abs()).sum::<f64>()
                    + weights[n + 1][i].abs()
                    + weights[n + 2][i].abs()
            })
            .fold(0.0, f64::max);
        let gain = if bound > 0.65 { 0.65 / bound } else { 1.0 };
        for w in &mut weights {
            w[0] *= gain;
            w[1] *= gain;
        }
        SplineMap {
            extent,
            controls,
            weights,
        }
    }
}

type Point = [f64; 2];

fn kernel(a: Point, b: Point) -> f64 {
    let r = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
    if r == 0.0 { 0.0 } else { r * r.ln() }
}

fn fit(controls: &[Point], displacement: &[Point]) -> Option<Vec<Point>> {
    let n = controls.len();
    let dim = n + 3;
    let mut system = vec![vec![0.0; dim + 2]; dim];
    for (i, &[x, y]) in controls.iter().enumerate() {
        for (j, &other) in controls.iter().enumerate() {
            system[i][j] = kernel([x, y], other);
        }
        for (j, v) in [1.0, x, y].into_iter().enumerate() {
            system[i][n + j] = v;
            system[n + j][i] = v;
        }
        system[i][dim..].copy_from_slice(&displacement[i]);
    }
    for k in 0..dim {
        let pivot = (k..dim).max_by(|&a, &b| system[a][k].abs().total_cmp(&system[b][k].abs()))?;
        if system[pivot][k].abs() < 1e-12 {
            return None;
        }
        system.swap(k, pivot);
        let value = system[k][k];
        for v in &mut system[k][k..] {
            *v /= value;
        }
        let (first, rest) = system.split_at_mut(k + 1);
        let row = &first[k];
        for dest in rest {
            let scale = dest[k];
            for (v, pivot) in dest[k..].iter_mut().zip(&row[k..]) {
                *v -= scale * pivot;
            }
        }
    }
    let mut weights = vec![[0.0; 2]; dim];
    for i in (0..dim).rev() {
        for channel in 0..2 {
            weights[i][channel] = system[i][dim + channel]
                - (i + 1..dim)
                    .map(|j| system[i][j] * weights[j][channel])
                    .sum::<f64>();
        }
    }
    weights
        .iter()
        .flatten()
        .all(|v| v.is_finite())
        .then_some(weights)
}

pub(crate) struct SplineMap {
    extent: Point,
    controls: Vec<Point>,
    weights: Vec<Point>,
}

impl SplineMap {
    fn displacement(&self, p: Point) -> Point {
        let p = p.map(|v| v.clamp(0.0, 1.0));
        let n = self.controls.len();
        let mut sum = std::array::from_fn(|i| {
            self.weights[n][i] + p[0] * self.weights[n + 1][i] + p[1] * self.weights[n + 2][i]
        });
        for (&control, weight) in self.controls.iter().zip(&self.weights) {
            let basis = kernel(control, p);
            for i in 0..2 {
                sum[i] += basis * weight[i];
            }
        }
        sum
    }
}

impl SpatialRemap for SplineMap {
    fn source(&self, p: Point) -> Point {
        if self.is_identity() {
            return p;
        }
        let d = self.displacement([p[0] / self.extent[0], p[1] / self.extent[1]]);
        std::array::from_fn(|i| p[i] + d[i] * self.extent[i])
    }
    fn destination(&self, p: Point) -> Point {
        if self.is_identity() {
            return p;
        }
        let source = [p[0] / self.extent[0], p[1] / self.extent[1]];
        let mut output = source;
        for _ in 0..64 {
            let d = self.displacement(output);
            let next = [source[0] - d[0], source[1] - d[1]];
            let error = (next[0] - output[0]).abs().max((next[1] - output[1]).abs());
            output = next;
            if error < 1e-12 {
                break;
            }
        }
        std::array::from_fn(|i| output[i] * self.extent[i])
    }
    fn is_identity(&self) -> bool {
        self.weights.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spline_interpolates_controls_and_rejects_duplicates() {
        let control = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let delta = [[0.0, 0.1], [0.2, 0.0], [0.0, -0.2], [0.4, 0.1]];
        let map = SplineMap {
            extent: [1.0; 2],
            controls: control.to_vec(),
            weights: fit(&control, &delta).unwrap(),
        };
        for (p, d) in control.into_iter().zip(delta) {
            for (actual, expected) in map.displacement(p).into_iter().zip(d) {
                assert!((actual - expected).abs() < 1e-10);
            }
        }
        assert!(fit(&[[0.0; 2]; 4], &delta).is_none());
    }

    #[test]
    fn strong_spline_roundtrips_points_even_near_the_edges() {
        let config = ThinPlateSplineConfig {
            scale_range: (1.0, 1.0),
            num_control_points: 5,
            interpolation: InterpolationMode::Bilinear,
            fill: 0,
            fill_mask: 0,
        };
        let map = config.sample(17, 23, &mut RandomStream::from_seed(47));
        for p in [[0., 0.], [22., 16.], [5., 11.], [2., 16.], [20., 0.]] {
            for (actual, expected) in map.destination(map.source(p)).into_iter().zip(p) {
                assert!((actual - expected).abs() < 1e-8);
            }
        }
    }
}
