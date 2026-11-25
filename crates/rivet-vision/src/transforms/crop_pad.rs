use super::{InterpolationMode, PadConfig, PaddingMode, RivetResult, invalid_argument};
use rivet_data::random::RandomStream;

#[derive(Clone, Debug, PartialEq)]
pub enum CropPadAmounts {
    Pixels([(i32, i32); 4]),
    Fractions([(f64, f64); 4]),
    PixelChoices(Vec<i32>),
    FractionChoices(Vec<f64>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct CropAndPadConfig {
    pub amounts: CropPadAmounts,
    pub keep_size: bool,
    pub sample_independently: bool,
    pub padding_mode: PaddingMode,
    pub fill: Vec<f32>,
    pub fill_mask: i64,
    pub interpolation: InterpolationMode,
}

impl CropAndPadConfig {
    pub(super) fn valid(&self) -> bool {
        let ranges = match &self.amounts {
            CropPadAmounts::Pixels(v) => v.iter().all(|(lo, hi)| lo <= hi),
            CropPadAmounts::Fractions(v) => v
                .iter()
                .all(|&(lo, hi)| lo >= -1.0 && lo <= hi && hi <= 1.0),
            CropPadAmounts::PixelChoices(v) => !v.is_empty(),
            CropPadAmounts::FractionChoices(v) => {
                !v.is_empty() && v.iter().all(|v| (-1.0..=1.0).contains(v))
            }
        };
        ranges && matches!(self.fill.len(), 1 | 3) && self.fill.iter().all(|v| v.is_finite())
    }

    pub(crate) fn sample(
        &self,
        h: usize,
        w: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<([u32; 4], PadConfig)> {
        if h == 0 || w == 0 || h > u32::MAX as usize || w > u32::MAX as usize {
            return Err(invalid_argument(
                "CropAndPad requires nonempty spatial dimensions that fit u32",
            ));
        }
        let shared = (!self.sample_independently).then(|| rng.next_u64());
        let sizes = [h, w, h, w];
        let mut sides = [0_i64; 4];
        for (i, value) in sides.iter_mut().enumerate() {
            // Reuse one uniform draw across sides, while retaining side-specific bounds.
            let draw = shared.unwrap_or_else(|| rng.next_u64());
            let pick = |n: u64| ((u128::from(draw) * u128::from(n)) >> 64) as u64;
            let uniform = (draw >> 11) as f64 / ((1u64 << 53) as f64);
            *value = match &self.amounts {
                CropPadAmounts::Pixels(ranges) => {
                    let (lo, hi) = ranges[i];
                    let (lo, hi) = (i64::from(lo), i64::from(hi));
                    lo + pick((hi - lo + 1) as u64) as i64
                }
                CropPadAmounts::Fractions(ranges) => {
                    let (lo, hi) = ranges[i];
                    ((lo + uniform * (hi - lo)) * sizes[i] as f64).round() as i64
                }
                CropPadAmounts::PixelChoices(v) => i64::from(v[pick(v.len() as u64) as usize]),
                CropPadAmounts::FractionChoices(v) => {
                    (v[pick(v.len() as u64) as usize] * sizes[i] as f64).round() as i64
                }
            }
        }
        let [top, right, bottom, left] = sides;
        let crop_axis = |len: usize, start: i64, end: i64| {
            let first = (-start).max(0).min(len as i64 - 1) as usize;
            let last = (-end).max(0).min((len - first - 1) as i64) as usize;
            (first as u32, (len - first - last) as u32)
        };
        let (x, cw) = crop_axis(w, left, right);
        let (y, ch) = crop_axis(h, top, bottom);
        let [pt, pr, pb, pl] = sides.map(|v| v.max(0) as u32);
        if cw.checked_add(pl).and_then(|v| v.checked_add(pr)).is_none()
            || ch.checked_add(pt).and_then(|v| v.checked_add(pb)).is_none()
        {
            return Err(invalid_argument("CropAndPad output dimensions exceed u32"));
        }
        let pad = PadConfig::with_sides(pl, pt, pr, pb).with_mode(self.padding_mode);
        let pad = match self.fill.as_slice() {
            &[v] => pad.with_fill(v),
            &[r, g, b] => pad.with_fill_channels([r, g, b]),
            _ => unreachable!("validated padding fill"),
        };
        Ok(([x, y, cw, ch], pad))
    }
}
