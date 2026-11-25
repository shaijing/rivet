use super::{InterpolationMode, RivetResult, invalid_argument};
use rivet_data::random::RandomStream;

#[derive(Clone, Debug, PartialEq)]
pub enum LongestSizeLimit {
    Longest(Vec<u32>),
    Axes {
        height: Option<u32>,
        width: Option<u32>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct LongestMaxSizeConfig {
    pub limit: LongestSizeLimit,
    pub interpolation: InterpolationMode,
}

impl LongestMaxSizeConfig {
    pub(super) fn valid(&self) -> bool {
        match &self.limit {
            LongestSizeLimit::Longest(sizes) => !sizes.is_empty() && !sizes.contains(&0),
            LongestSizeLimit::Axes { height, width } => {
                (height.is_some() || width.is_some()) && *height != Some(0) && *width != Some(0)
            }
        }
    }

    pub(crate) fn random(&self) -> bool {
        matches!(&self.limit, LongestSizeLimit::Longest(sizes) if sizes.len() > 1)
    }

    pub(crate) fn dimensions(&self, w: usize, h: usize, choice: usize) -> RivetResult<(u32, u32)> {
        let (w, h) = (w as u64, h as u64);
        if w == 0 || h == 0 || w > u64::from(u32::MAX) || h > u64::from(u32::MAX) {
            return Err(invalid_argument(
                "LongestMaxSize requires positive input axes that fit u32",
            ));
        }
        let (numerator, denominator) = match &self.limit {
            LongestSizeLimit::Longest(sizes) => (u64::from(sizes[choice]), w.max(h)),
            LongestSizeLimit::Axes {
                width: Some(a),
                height: Some(b),
            } => {
                let (a, b) = (u64::from(*a), u64::from(*b));
                if a * h <= b * w { (a, w) } else { (b, h) }
            }
            LongestSizeLimit::Axes {
                width: Some(a),
                height: None,
            } => (u64::from(*a), w),
            LongestSizeLimit::Axes {
                width: None,
                height: Some(b),
            } => (u64::from(*b), h),
            LongestSizeLimit::Axes { .. } => {
                return Err(invalid_argument(
                    "LongestMaxSize requires at least one bound",
                ));
            }
        };
        let size = |axis: u64| {
            let product = axis * numerator;
            let rounded =
                product / denominator + u64::from(product % denominator >= denominator.div_ceil(2));
            u32::try_from(rounded.max(1))
                .map_err(|_| invalid_argument("LongestMaxSize output axis exceeds u32"))
        };
        Ok((size(w)?, size(h)?))
    }

    pub(crate) fn sample(
        &self,
        w: usize,
        h: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<(u32, u32)> {
        let choice = if let LongestSizeLimit::Longest(sizes) = &self.limit
            && sizes.len() > 1
        {
            rng.gen_range_usize(0..sizes.len())?
        } else {
            0
        };
        self.dimensions(w, h, choice)
    }
}
