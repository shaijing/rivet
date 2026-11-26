use crate::errors::{RivetResult, invalid_argument};
use crate::transforms::InterpolationMode;
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomSizedCropConfig {
    pub width: u32,
    pub height: u32,
    pub min_max_height: (u32, u32),
    pub w2h_ratio: f64,
    pub interpolation: InterpolationMode,
}

impl RandomSizedCropConfig {
    pub(in crate::transforms) fn valid(&self) -> bool {
        let (lo, hi) = self.min_max_height;
        self.width > 0
            && self.height > 0
            && lo > 0
            && lo <= hi
            && self.w2h_ratio.is_finite()
            && self.w2h_ratio > 0.0
    }

    pub(crate) fn sample(
        &self,
        height: usize,
        width: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<[u32; 4]> {
        if height == 0 || width == 0 || height > u32::MAX as usize || width > u32::MAX as usize {
            return Err(invalid_argument(
                "RandomSizedCrop requires nonempty axes that fit u32",
            ));
        }
        let (lo, hi) = self.min_max_height;
        let lo = lo as usize;
        let max_height = (hi as usize).min(height);
        let crop_width = |h| ((h as f64 * self.w2h_ratio).floor() as usize).max(1);
        if lo > max_height || crop_width(lo) > width {
            return Err(invalid_argument(
                "RandomSizedCrop height and aspect ratio do not fit the image",
            ));
        }
        // Width is monotone in height. Find the last feasible integer height.
        let mut left = lo;
        let mut right = max_height;
        while left < right {
            let mid = left + (right - left).div_ceil(2);
            if crop_width(mid) <= width {
                left = mid;
            } else {
                right = mid - 1;
            }
        }
        let h = lo + rng.gen_range_usize(0..left - lo + 1)?;
        let w = crop_width(h);
        let x = rng.gen_range_usize(0..width - w + 1)?;
        let y = rng.gen_range_usize(0..height - h + 1)?;
        Ok([x as u32, y as u32, w as u32, h as u32])
    }
}
