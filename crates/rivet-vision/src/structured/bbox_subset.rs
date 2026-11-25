use super::{BBoxSafeRandomCropConfig, StructuredSample};
use crate::errors::{RivetResult, invalid_argument};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBoxSubsetSafeRandomCropConfig {
    pub erosion_rate: f64,
    pub subset_fraction_range: (f64, f64),
    pub aspect_ratio_range: (f64, f64),
}

impl BBoxSubsetSafeRandomCropConfig {
    pub fn validate(&self) -> RivetResult<()> {
        let (lo, hi) = self.subset_fraction_range;
        let (min, max) = self.aspect_ratio_range;
        if !((0.0..=1.0).contains(&self.erosion_rate)
            && 0.0 < lo
            && lo <= hi
            && hi <= 1.0
            && 0.0 < min
            && min <= max
            && max.is_finite())
        {
            return Err(invalid_argument(
                "invalid BBoxSubsetSafeRandomCrop configuration",
            ));
        }
        Ok(())
    }
}

impl StructuredSample {
    pub fn bbox_subset_safe_random_crop(
        self,
        config: &BBoxSubsetSafeRandomCropConfig,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        config.validate()?;
        self.safe_crop(
            &BBoxSafeRandomCropConfig {
                erosion_rate: config.erosion_rate,
                output: None,
            },
            Some(config),
            rng,
        )
    }
}

pub(super) fn sample_aspect_crop(
    [x0, y0, x1, y1]: [f64; 4],
    width: usize,
    height: usize,
    (min_ratio, max_ratio): (f64, f64),
    rng: &mut RandomStream,
) -> RivetResult<[usize; 4]> {
    let left = (x0.floor() as usize).min(width - 1);
    let right = (x1.ceil() as usize).max(left + 1).min(width);
    let top = (y0.floor() as usize).min(height - 1);
    let bottom = (y1.ceil() as usize).max(top + 1).min(height);
    let min_w = (right - left).max(((bottom - top) as f64 / max_ratio).next_down().ceil() as usize);
    let max_w = width.min((height as f64 / min_ratio).next_up().floor() as usize);
    let heights = |w: usize| {
        let low = (bottom - top).max((min_ratio * w as f64).next_down().ceil() as usize);
        let high = height.min((max_ratio * w as f64).next_up().floor() as usize);
        (low <= high).then_some((low, high))
    };
    // Count feasible widths explicitly: discrete height bounds can leave
    // gaps even when the continuous aspect interval is nonempty.
    let feasible = (min_w..=max_w).filter(|&w| heights(w).is_some());
    let n = feasible.clone().count();
    if n == 0 {
        return Ok([0, 0, width, height]);
    }
    let w = feasible
        .clone()
        .nth(rng.gen_range_usize(0..n)?)
        .expect("counted feasible widths");
    let (lo, hi) = heights(w).expect("feasible width has a height");
    let h = rng.gen_range_usize(lo..hi + 1)?;
    let x = rng.gen_range_usize(right.saturating_sub(w)..left.min(width - w) + 1)?;
    let y = rng.gen_range_usize(bottom.saturating_sub(h)..top.min(height - h) + 1)?;
    Ok([x, y, x + w, y + h])
}
