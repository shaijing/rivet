//! Pixel-center coordinates in the same input and output canvas.
//! Backward maps drive pixels and masks; forward maps move keypoints.
use crate::errors::RivetResult;
use crate::sample::image::{ImageAxisOrder, ImageSample};
use crate::transforms::InterpolationMode;
use rivet_data::random::RandomStream;

pub(crate) trait SpatialRemap {
    fn source(&self, output: [f64; 2]) -> [f64; 2];
    fn destination(&self, source: [f64; 2]) -> [f64; 2];
    fn is_identity(&self) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpticalDistortionMode {
    Camera,
    Fisheye,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpticalDistortionConfig {
    pub distort_range: (f64, f64),
    pub mode: OpticalDistortionMode,
    pub interpolation: InterpolationMode,
    pub fill: u8,
    pub fill_mask: i64,
}

pub(crate) struct OpticalMap {
    center: [f64; 2],
    radius: f64,
    k: f64,
    camera: bool,
}

impl OpticalDistortionConfig {
    pub(in crate::transforms) fn valid(&self) -> bool {
        let (lo, hi) = self.distort_range;
        let min = if self.mode == OpticalDistortionMode::Camera {
            -0.12
        } else {
            -0.3
        };
        lo >= min
            && lo <= hi
            && hi <= 1.0
            && matches!(
                self.interpolation,
                InterpolationMode::Nearest | InterpolationMode::Bilinear
            )
    }

    pub(crate) fn sample(&self, h: usize, w: usize, rng: &mut RandomStream) -> OpticalMap {
        let (lo, hi) = self.distort_range;
        let center = [
            w.saturating_sub(1) as f64 * 0.5,
            h.saturating_sub(1) as f64 * 0.5,
        ];
        OpticalMap {
            center,
            radius: (w as f64).hypot(h as f64) * 0.5,
            k: lo + rng.next_f64() * (hi - lo),
            camera: self.mode == OpticalDistortionMode::Camera,
        }
    }
}

impl OpticalMap {
    fn radial(&self, r: f64) -> f64 {
        let terms = |r: f64| r.powi(3) + if self.camera { r.powi(5) } else { 0.0 };
        if r <= 1.0 {
            r + self.k * terms(r)
        } else {
            let slope = 1.0 + self.k * if self.camera { 8.0 } else { 3.0 };
            1.0 + self.k * terms(1.0) + slope * (r - 1.0)
        }
    }

    fn radius_at(&self, [x, y]: [f64; 2]) -> f64 {
        (x - self.center[0]).hypot(y - self.center[1]) / self.radius
    }

    fn resize_delta(&self, p: [f64; 2], ratio: f64) -> [f64; 2] {
        std::array::from_fn(|i| self.center[i] + (p[i] - self.center[i]) * ratio)
    }
}

impl SpatialRemap for OpticalMap {
    fn source(&self, p: [f64; 2]) -> [f64; 2] {
        if self.k == 0.0 {
            return p;
        }
        let (dx, dy) = (p[0] - self.center[0], p[1] - self.center[1]);
        let radial_sq = (dx * dx + dy * dy) / (self.radius * self.radius);
        let ratio = if radial_sq <= 1.0 {
            1.0 + self.k * radial_sq * if self.camera { 1.0 + radial_sq } else { 1.0 }
        } else {
            let r = radial_sq.sqrt();
            self.radial(r) / r
        };
        self.resize_delta(p, ratio)
    }

    fn destination(&self, p: [f64; 2]) -> [f64; 2] {
        let source = self.radius_at(p);
        if source == 0.0 || self.k == 0.0 {
            return p;
        }
        let edge = self.radial(1.0);
        let radius = if source > edge {
            1.0 + (source - edge) / (self.radial(2.0) - edge)
        } else {
            let (mut lo, mut hi) = (0.0, 1.0);
            for _ in 0..48 {
                let mid = (lo + hi) * 0.5;
                if self.radial(mid) < source {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            (lo + hi) * 0.5
        };
        self.resize_delta(p, radius / source)
    }

    fn is_identity(&self) -> bool {
        self.k == 0.0
    }
}

pub(crate) fn remap_image(
    sample: ImageSample,
    axis: ImageAxisOrder,
    size: (usize, usize),
    interpolation: InterpolationMode,
    fill: u8,
    map: impl FnMut(usize, usize) -> [f64; 2],
) -> RivetResult<ImageSample> {
    let (h, w) = size;
    super::advanced_geometry::warp_decoded::<false, _>(
        sample.into_decoded()?,
        axis,
        w,
        h,
        interpolation,
        fill,
        None,
        "spatial remap",
        map,
    )
    .map(ImageSample::Decoded)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridDistortionConfig {
    pub num_steps: u32,
    pub distort_range: (f64, f64),
    pub normalized: bool,
    pub interpolation: InterpolationMode,
    pub fill: u8,
    pub fill_mask: i64,
}

impl GridDistortionConfig {
    pub(in crate::transforms) fn valid(&self) -> bool {
        let (lo, hi) = self.distort_range;
        (1..=1024).contains(&self.num_steps)
            && lo > -1.0
            && lo <= hi
            && hi <= 1.0
            && matches!(
                self.interpolation,
                InterpolationMode::Nearest | InterpolationMode::Bilinear
            )
    }

    pub(crate) fn sample(&self, h: usize, w: usize, rng: &mut RandomStream) -> GridMap {
        let mut axis = |length: usize| {
            let span = length.saturating_sub(1) as f64;
            let cells = (self.num_steps as usize).min(length.saturating_sub(1).max(1));
            let step = span / cells as f64;
            let mut knots = Vec::with_capacity(cells + 1);
            knots.push(0.0);
            let (lo, hi) = self.distort_range;
            for i in 0..cells {
                let factor = (1.0 + lo) + rng.next_f64() * (hi - lo);
                knots.push(knots[i] + step * factor);
            }
            if self.normalized && span > 0.0 {
                let scale = span / knots[cells];
                for knot in &mut knots[1..cells] {
                    *knot *= scale;
                }
                knots[cells] = span;
            }
            GridAxis { step, knots }
        };
        GridMap {
            x: axis(w),
            y: axis(h),
            identity: self.distort_range == (0.0, 0.0),
        }
    }
}

struct GridAxis {
    step: f64,
    knots: Vec<f64>,
}

impl GridAxis {
    fn backward(&self, x: f64) -> f64 {
        if self.step == 0.0 {
            return x;
        }
        let u = x / self.step;
        let i = (u.floor() as usize).min(self.knots.len() - 2);
        self.knots[i] + (u - i as f64) * (self.knots[i + 1] - self.knots[i])
    }

    fn forward(&self, x: f64) -> f64 {
        if self.step == 0.0 {
            return x;
        }
        let i = self
            .knots
            .partition_point(|&k| k <= x)
            .saturating_sub(1)
            .min(self.knots.len() - 2);
        (i as f64 + (x - self.knots[i]) / (self.knots[i + 1] - self.knots[i])) * self.step
    }
}

pub(crate) struct GridMap {
    x: GridAxis,
    y: GridAxis,
    identity: bool,
}

impl SpatialRemap for GridMap {
    fn source(&self, [x, y]: [f64; 2]) -> [f64; 2] {
        [self.x.backward(x), self.y.backward(y)]
    }
    fn destination(&self, [x, y]: [f64; 2]) -> [f64; 2] {
        [self.x.forward(x), self.y.forward(y)]
    }
    fn is_identity(&self) -> bool {
        self.identity
    }
}
