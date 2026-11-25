use super::{BBoxSubsetSafeRandomCropConfig, StructuredSample, bbox_subset::sample_aspect_crop};
use crate::errors::{RivetResult, invalid_argument};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBoxSafeRandomCropConfig {
    pub erosion_rate: f64,
    pub output: Option<(u32, u32, crate::transforms::InterpolationMode)>,
}

impl BBoxSafeRandomCropConfig {
    pub fn validate(&self) -> RivetResult<()> {
        if !(0.0..=1.0).contains(&self.erosion_rate)
            || self.output.is_some_and(|(w, h, _)| w == 0 || h == 0)
        {
            return Err(invalid_argument("invalid BBoxSafeRandomCrop configuration"));
        }
        Ok(())
    }
}

impl StructuredSample {
    pub fn bbox_safe_random_crop(
        self,
        config: &BBoxSafeRandomCropConfig,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        config.validate()?;
        self.safe_crop(config, None, rng)
    }

    pub(super) fn safe_crop(
        self,
        config: &BBoxSafeRandomCropConfig,
        subset: Option<&BBoxSubsetSafeRandomCropConfig>,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        let (height, width) = self.size()?;
        let boxes = self.xyxy()?.map(|b| b.to_vec::<f32>()).transpose()?;
        let candidates: Vec<_> = boxes
            .as_deref()
            .unwrap_or_default()
            .chunks_exact(4)
            .enumerate()
            .filter_map(|(index, b)| {
                let x0 = f64::from(b[0]).clamp(0.0, width as f64);
                let x1 = f64::from(b[2]).clamp(0.0, width as f64);
                let y0 = f64::from(b[1]).clamp(0.0, height as f64);
                let y1 = f64::from(b[3]).clamp(0.0, height as f64);
                (x0 < x1 && y0 < y1).then_some((index, [x0, y0, x1, y1]))
            })
            .collect();
        let chosen = if let Some(subset) = subset.filter(|_| !candidates.is_empty()) {
            let n = candidates.len();
            let (lo, hi) = subset.subset_fraction_range;
            let lo = ((lo * n as f64).ceil() as usize).clamp(1, n);
            let hi = ((hi * n as f64).ceil() as usize).clamp(lo, n);
            let count = rng.gen_range_usize(lo..hi + 1)?;
            let mut indices: Vec<_> = (0..n).collect();
            for i in 0..count {
                let j = rng.gen_range_usize(i..n)?;
                indices.swap(i, j);
            }
            indices.truncate(count);
            Some(indices)
        } else {
            None
        };
        let count = chosen.as_ref().map_or(candidates.len(), Vec::len);
        let mut union = [width as f64, height as f64, 0.0, 0.0];
        for i in 0..count {
            let (_, [x0, y0, x1, y1]) = candidates[chosen.as_ref().map_or(i, |indices| indices[i])];
            let ex = (x1 - x0) * config.erosion_rate * 0.5;
            let ey = (y1 - y0) * config.erosion_rate * 0.5;
            union[0] = union[0].min(x0 + ex);
            union[1] = union[1].min(y0 + ey);
            union[2] = union[2].max(x1 - ex);
            union[3] = union[3].max(y1 - ey);
        }
        let mut draw_axis = |low: f64, high: f64, limit: usize| -> RivetResult<(usize, usize)> {
            if candidates.is_empty() {
                let min = ((1.0 - config.erosion_rate) * limit as f64).ceil() as usize;
                let extent = rng.gen_range_usize(min.max(1)..limit + 1)?;
                let start = rng.gen_range_usize(0..limit - extent + 1)?;
                return Ok((start, start + extent));
            }
            let left = (low.floor() as usize).min(limit - 1);
            let right = (high.ceil() as usize).max(left + 1).min(limit);
            Ok((
                rng.gen_range_usize(0..left + 1)?,
                rng.gen_range_usize(right..limit + 1)?,
            ))
        };
        let [left, top, right, bottom] = if let Some(subset) = subset.filter(|_| count != 0) {
            sample_aspect_crop(union, width, height, subset.aspect_ratio_range, rng)?
        } else {
            let (left, right) = draw_axis(union[0], union[2], width)?;
            let (top, bottom) = draw_axis(union[1], union[3], height)?;
            [left, top, right, bottom]
        };
        let sample = if boxes.is_some() {
            let keep: Vec<_> = candidates
                .iter()
                .filter_map(|&(index, [x0, y0, x1, y1])| {
                    (x0 < right as f64 && y0 < bottom as f64 && x1 > left as f64 && y1 > top as f64)
                        .then_some(index)
                })
                .collect();
            self.retain_instances(&keep)?
        } else {
            self
        };
        let cropped = super::geometry::crop(
            sample,
            left as u32,
            top as u32,
            (right - left) as u32,
            (bottom - top) as u32,
        )?;
        if let Some((w, h, interpolation)) = config.output {
            super::geometry::resize(cropped, w, h, interpolation)
        } else {
            Ok(cropped)
        }
    }
}
