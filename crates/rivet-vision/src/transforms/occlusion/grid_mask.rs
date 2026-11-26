use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::ImageAxisOrder;
use crate::transforms::occlusion::pixel_mask::apply;
use rivet_core::{DType, Tensor};
use rivet_data::random::RandomStream;

#[derive(Clone, Debug, PartialEq)]
pub struct GridMaskConfig {
    pub num_grid_range: (u32, u32),
    pub line_width_range: (f64, f64),
    pub rotation_range: (f64, f64),
    pub fill: f32,
    pub fill_mask: Option<i64>,
}

#[derive(Clone, Copy)]
pub(crate) struct GridLines {
    height: usize,
    width: usize,
    center: (f64, f64),
    fraction: f64,
    sin: f64,
    cos: f64,
}

impl GridLines {
    pub(crate) fn covers(&self, y: usize, x: usize) -> bool {
        if self.fraction == 0.0 {
            return false;
        }
        if self.fraction == 1.0 {
            return true;
        }
        let (x, y) = (x as f64 - self.center.0, y as f64 - self.center.1);
        let u = self.cos * x + self.sin * y;
        let v = self.cos * y - self.sin * x;
        let near_line = |p: f64| (p - p.round()).abs() < 0.5 * self.fraction;
        near_line(u) || near_line(v)
    }
}

impl GridMaskConfig {
    pub(in crate::transforms) fn valid(&self) -> bool {
        let (lo, hi) = self.num_grid_range;
        let (a, b) = self.line_width_range;
        let (min_angle, max_angle) = self.rotation_range;
        lo > 0
            && lo <= hi
            && a >= 0.0
            && a <= b
            && b <= 1.0
            && min_angle >= -std::f64::consts::TAU
            && min_angle <= max_angle
            && max_angle <= std::f64::consts::TAU
            && self.fill.is_finite()
    }

    pub(crate) fn sample(
        &self,
        h: usize,
        w: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<GridLines> {
        if h == 0 || w == 0 {
            return Err(invalid_argument(
                "GridMask requires nonempty spatial dimensions",
            ));
        }
        let (lo, hi) = self.num_grid_range;
        let count = lo as usize + rng.gen_range_usize(0..(hi - lo) as usize + 1)?;
        let mut uniform = |(lo, hi)| lo + rng.next_f64() * (hi - lo);
        let fraction = uniform(self.line_width_range);
        let (sin, cos) = uniform(self.rotation_range).sin_cos();
        let side = h.min(w);
        let frequency = count.min(side) as f64 / side as f64;
        Ok(GridLines {
            height: h,
            width: w,
            center: ((w - 1) as f64 * 0.5, (h - 1) as f64 * 0.5),
            fraction,
            sin: sin * frequency,
            cos: cos * frequency,
        })
    }

    pub(crate) fn apply(
        &self,
        input: &Tensor,
        axis: ImageAxisOrder,
        mask: GridLines,
    ) -> RivetResult<Tensor> {
        if !matches!(input.dtype(), DType::U8 | DType::F32) {
            return Err(invalid_argument("GridMask requires uint8 or float32 input"));
        }
        if mask.fraction == 0.0 {
            return Ok(input.clone());
        }
        apply(
            input,
            axis,
            mask.height,
            mask.width,
            [self.fill; 2],
            |y, x| mask.covers(y, x).then_some(false),
        )
    }
}
