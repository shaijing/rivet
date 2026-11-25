use super::StructuredSample;
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use rivet_core::{DType, Tensor};
use rivet_data::random::RandomStream;

#[derive(Clone, Debug)]
pub struct GuidedCoarseDropoutConfig {
    /// Shared [H,W] or per-source-sample [N,H,W] binary regions.
    pub region: Tensor,
    pub protected_bbox_labels: Vec<i64>,
    pub protection_margin: f64,
    pub num_holes_range: (u32, u32),
    pub hole_height_range: (f64, f64),
    pub hole_width_range: (f64, f64),
    pub fill: f32,
    pub fill_mask: Option<i64>,
    pub min_visibility: f64,
}

impl GuidedCoarseDropoutConfig {
    pub fn validate(&self) -> RivetResult<()> {
        let (lo, hi) = self.num_holes_range;
        if self.region.dtype() != DType::U8
            || !matches!(self.region.rank(), 2 | 3)
            || !(0.0..=1.0).contains(&self.min_visibility)
            || lo > hi
            || hi > 1024
            || !self.fill.is_finite()
            || self.protection_margin < 0.0
            || !self.protection_margin.is_finite()
            || ![self.hole_width_range, self.hole_height_range]
                .into_iter()
                .all(|(a, b)| a > 0.0 && a <= b && b <= 1.0)
        {
            return Err(invalid_argument(
                "invalid GuidedCoarseDropout configuration",
            ));
        }
        Ok(())
    }
}

impl StructuredSample {
    pub fn guided_coarse_dropout(
        mut self,
        config: &GuidedCoarseDropoutConfig,
        index: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        config.validate()?;
        let (h, w) = self.size()?;
        let guide = if config.region.rank() == 3 {
            config.region.narrow(0, index, 1)?.squeeze(0)?
        } else {
            config.region.clone()
        };
        if guide.dims() != [h, w] {
            return Err(invalid_shape(
                "GuidedCoarseDropout region does not match the image dimensions",
            ));
        }
        let mut eligible = guide.to_vec::<u8>()?;
        if eligible.iter().any(|&v| v > 1) {
            return Err(invalid_argument(
                "GuidedCoarseDropout region must be binary",
            ));
        }
        if !config.protected_bbox_labels.is_empty()
            && let (Some(boxes), Some(labels)) = (self.xyxy()?, &self.labels)
        {
            for (b, label) in boxes
                .to_vec::<f32>()?
                .chunks_exact(4)
                .zip(labels.to_vec::<i64>()?)
            {
                if !config.protected_bbox_labels.contains(&label) || b[0] >= b[2] || b[1] >= b[3] {
                    continue;
                }
                let (l, t, r, b) = (
                    f64::from(b[0]),
                    f64::from(b[1]),
                    f64::from(b[2]),
                    f64::from(b[3]),
                );
                let (dx, dy) = (
                    (r - l) * config.protection_margin,
                    (b - t) * config.protection_margin,
                );
                let (left, right) = (
                    ((l - dx).floor() as usize).min(w),
                    ((r + dx).ceil() as usize).min(w),
                );
                let (top, bottom) = (
                    ((t - dy).floor() as usize).min(h),
                    ((b + dy).ceil() as usize).min(h),
                );
                for y in top..bottom {
                    eligible[y * w + left..y * w + right].fill(0);
                }
            }
        }
        let available = eligible.iter().filter(|&&v| v == 1).count();
        let (lo, hi) = config.num_holes_range;
        let count = rng.gen_range_usize(lo as usize..hi as usize + 1)?;
        if count == 0 || available == 0 {
            return Ok(self);
        }
        let sample_size = |(lo, hi): (f64, f64), len: usize, rng: &mut RandomStream| {
            ((lo + rng.next_f64() * (hi - lo)) * len as f64)
                .round()
                .max(1.0) as usize
        };
        let mut holes = Vec::with_capacity(count);
        for _ in 0..count {
            let rank = rng.gen_range_usize(0..available)?;
            holes.push((
                rank,
                sample_size(config.hole_width_range, w, rng),
                sample_size(config.hole_height_range, h, rng),
            ));
        }
        holes.sort_unstable_by_key(|hole| hole.0);
        let mut centers = eligible
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v == 1).then_some(i));
        let mut last = None;
        for (rank, _, _) in &mut holes {
            let position = match last {
                Some((previous, position)) if previous == *rank => position,
                Some((previous, _)) => centers
                    .nth(*rank - previous - 1)
                    .expect("counted eligible pixels"),
                None => centers.nth(*rank).expect("counted eligible pixels"),
            };
            last = Some((*rank, position));
            *rank = position;
        }
        for (position, cw, ch) in holes {
            let (x, y) = (position % w, position / w);
            let (left, top) = (x.saturating_sub(cw / 2), y.saturating_sub(ch / 2));
            let (right, bottom) = ((x + cw.div_ceil(2)).min(w), (y + ch.div_ceil(2)).min(h));
            for y in top..bottom {
                for x in left..right {
                    let state = &mut eligible[y * w + x];
                    if *state == 1 {
                        *state = 2;
                    }
                }
            }
        }
        let dropped = |y: usize, x: usize| eligible[y * w + x] == 2;
        self.image = crate::transforms::replace_spatial_pixels(
            &self.image,
            self.axis_order,
            h,
            w,
            [config.fill; 2],
            |y, x| dropped(y, x).then_some(false),
        )?;
        super::geometry::apply_occlusion_with_visibility(
            self,
            h,
            w,
            config.fill_mask,
            config.min_visibility,
            dropped,
        )
    }
}
