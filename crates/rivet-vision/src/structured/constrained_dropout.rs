use super::StructuredSample;
use crate::errors::{RivetResult, invalid_argument};
use rivet_data::random::RandomStream;

#[derive(Clone, Debug, PartialEq)]
pub struct ConstrainedCoarseDropoutConfig {
    pub num_holes_range: (u32, u32),
    pub hole_height_range: (f64, f64),
    pub hole_width_range: (f64, f64),
    pub mask_indices: Option<Vec<i64>>,
    pub bbox_labels: Option<Vec<i64>>,
    pub fill: f32,
    pub fill_mask: Option<i64>,
}

impl ConstrainedCoarseDropoutConfig {
    pub fn validate(&self) -> RivetResult<()> {
        let (lo, hi) = self.num_holes_range;
        if lo > hi
            || hi > 1024
            || !self.fill.is_finite()
            || ![self.hole_width_range, self.hole_height_range]
                .into_iter()
                .all(|(a, b)| a > 0.0 && a <= b && b <= 1.0)
        {
            return Err(invalid_argument(
                "invalid ConstrainedCoarseDropout configuration",
            ));
        }
        Ok(())
    }
}

impl StructuredSample {
    pub fn constrained_coarse_dropout(
        mut self,
        config: &ConstrainedCoarseDropoutConfig,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        config.validate()?;
        let (height, width) = self.size()?;
        let (lo, hi) = config.num_holes_range;
        let count = rng.gen_range_usize(lo as usize..hi as usize + 1)?;
        if count == 0 {
            return Ok(self);
        }
        let mut region = vec![false; height * width];
        let mut paint = |x: usize, y: usize, w: usize, h: usize| {
            for row in y..(y + h).min(height) {
                region[row * width + x..row * width + (x + w).min(width)].fill(true);
            }
        };
        let extent = |(lo, hi): (f64, f64), len: f64, rng: &mut RandomStream| {
            ((lo + rng.next_f64() * (hi - lo)) * len).round().max(1.0) as usize
        };
        if let (Some(ids), Some(mask)) = (&config.mask_indices, &self.masks) {
            let (_, pixels) = super::mask_dropout::object_pixels(mask, height, width)?;
            for (plane_idx, plane) in pixels.chunks_exact(height * width).enumerate() {
                let instance = mask.rank() == 3;
                if instance && !ids.contains(&(plane_idx as i64)) {
                    continue;
                }
                let mut visited = vec![false; plane.len()];
                let mut component = Vec::new();
                for i in 0..plane.len() {
                    let label = plane[i];
                    if visited[i] || label == 0 || (!instance && !ids.contains(&label)) {
                        continue;
                    }
                    component.clear();
                    component.push(i);
                    visited[i] = true;
                    let mut cursor = 0;
                    while cursor < component.len() {
                        let p = component[cursor];
                        cursor += 1;
                        let (y, x) = (p / width, p % width);
                        for other in [
                            (x > 0).then(|| p - 1),
                            (x + 1 < width).then(|| p + 1),
                            (y > 0).then(|| p - width),
                            (y + 1 < height).then(|| p + width),
                        ]
                        .into_iter()
                        .flatten()
                        {
                            if !visited[other]
                                && if instance {
                                    plane[other] != 0
                                } else {
                                    plane[other] == label
                                }
                            {
                                visited[other] = true;
                                component.push(other);
                            }
                        }
                    }
                    let side = (component.len() as f64).sqrt();
                    for _ in 0..count {
                        let center = component[rng.gen_range_usize(0..component.len())?];
                        let cw = extent(config.hole_width_range, side, rng);
                        let ch = extent(config.hole_height_range, side, rng);
                        paint(
                            (center % width).saturating_sub(cw / 2),
                            (center / width).saturating_sub(ch / 2),
                            cw,
                            ch,
                        );
                    }
                }
            }
        } else if let (Some(ids), Some(labels), Some(boxes)) =
            (&config.bbox_labels, &self.labels, self.xyxy()?)
        {
            let boxes = boxes.to_vec::<f32>()?;
            for (b, label) in boxes.chunks_exact(4).zip(labels.to_vec::<i64>()?) {
                if !ids.contains(&label) {
                    continue;
                }
                let (l, t, r, b) = (
                    (b[0].floor() as usize).min(width),
                    (b[1].floor() as usize).min(height),
                    (b[2].ceil() as usize).min(width),
                    (b[3].ceil() as usize).min(height),
                );
                if l >= r || t >= b {
                    continue;
                }
                for _ in 0..count {
                    let cw = extent(config.hole_width_range, (r - l) as f64, rng).min(r - l);
                    let ch = extent(config.hole_height_range, (b - t) as f64, rng).min(b - t);
                    paint(
                        rng.gen_range_usize(l..r - cw + 1)?,
                        rng.gen_range_usize(t..b - ch + 1)?,
                        cw,
                        ch,
                    );
                }
            }
        }
        if !region.contains(&true) {
            return Ok(self);
        }
        let covered = |y: usize, x: usize| region[y * width + x];
        self.image = crate::transforms::replace_spatial_pixels(
            &self.image,
            self.axis_order,
            height,
            width,
            [config.fill; 2],
            |y, x| covered(y, x).then_some(false),
        )?;
        super::geometry::apply_occlusion(self, height, width, config.fill_mask, covered)
    }
}
