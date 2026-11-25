use super::StructuredSample;
use crate::errors::{RivetResult, invalid_argument};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomCropNearBBoxConfig {
    pub box_index: usize,
    pub reference_box: Option<[f64; 4]>,
    /// Maximum vertical and horizontal shifts, as fractions of box size.
    pub max_part_shift: (f64, f64),
}

impl RandomCropNearBBoxConfig {
    pub fn validate(&self) -> RivetResult<()> {
        let (y, x) = self.max_part_shift;
        if !(0.0..=1.0).contains(&x)
            || !(0.0..=1.0).contains(&y)
            || self.reference_box.is_some_and(|[l, t, r, b]| {
                ![l, t, r, b].iter().all(|v| v.is_finite()) || l >= r || t >= b
            })
        {
            return Err(invalid_argument("invalid RandomCropNearBBox configuration"));
        }
        Ok(())
    }
}

impl StructuredSample {
    pub fn random_crop_near_bbox(
        self,
        config: &RandomCropNearBBoxConfig,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        config.validate()?;
        let (height, width) = self.size()?;
        let boxes = self.xyxy()?.map(|b| b.to_vec::<f32>()).transpose()?;
        let reference = config.reference_box.or_else(|| {
            let b = boxes.as_ref()?.chunks_exact(4).nth(config.box_index)?;
            Some([b[0], b[1], b[2], b[3]].map(f64::from))
        });
        let Some([l, t, r, b]) = reference else {
            return Ok(self);
        };
        let draw = |lo: f64, hi: f64, limit: usize, shift: f64, rng: &mut RandomStream| {
            let limit = limit as f64;
            let (lo, hi) = (lo.clamp(0.0, limit), hi.clamp(0.0, limit));
            if lo >= hi {
                return None;
            }
            let extent = (hi - lo) * shift;
            let start = (lo + (2.0 * rng.next_f64() - 1.0) * extent)
                .floor()
                .clamp(0.0, limit) as usize;
            let end = (hi + (2.0 * rng.next_f64() - 1.0) * extent)
                .ceil()
                .clamp(0.0, limit) as usize;
            Some(if start < end {
                (start, end)
            } else {
                (lo.floor() as usize, hi.ceil() as usize)
            })
        };
        let Some((x0, x1)) = draw(l, r, width, config.max_part_shift.1, rng) else {
            return Ok(self);
        };
        let Some((y0, y1)) = draw(t, b, height, config.max_part_shift.0, rng) else {
            return Ok(self);
        };
        super::geometry::crop_visible_instances(self, boxes.as_deref(), [x0, y0, x1, y1])
    }
}
