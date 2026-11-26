use crate::errors::{RivetResult, invalid_argument};
use crate::transforms::{PadConfig, PaddingMode};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadPosition {
    Center,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Random,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadIfNeededConfig {
    pub min_height: Option<u32>,
    pub min_width: Option<u32>,
    pub pad_height_divisor: Option<u32>,
    pub pad_width_divisor: Option<u32>,
    pub position: PadPosition,
    pub padding_mode: PaddingMode,
    pub fill: f32,
    pub fill_channels: Option<[f32; 3]>,
    pub fill_mask: i64,
}

impl PadIfNeededConfig {
    pub(in crate::transforms) fn valid(&self) -> bool {
        [
            self.min_height,
            self.min_width,
            self.pad_height_divisor,
            self.pad_width_divisor,
        ]
        .into_iter()
        .flatten()
        .all(|v| v != 0)
            && self.fill.is_finite()
            && self
                .fill_channels
                .is_none_or(|v| v.iter().all(|v| v.is_finite()))
    }

    pub(crate) fn output_axis(
        length: usize,
        min: Option<u32>,
        divisor: Option<u32>,
    ) -> RivetResult<usize> {
        let length = length.max(min.unwrap_or(0) as usize);
        let divisor = divisor.unwrap_or(1) as usize;
        length
            .div_ceil(divisor)
            .checked_mul(divisor)
            .filter(|&v| v <= u32::MAX as usize)
            .ok_or_else(|| invalid_argument("PadIfNeeded output dimensions exceed u32"))
    }

    pub(crate) fn sample(
        &self,
        height: usize,
        width: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<PadConfig> {
        let dh = Self::output_axis(height, self.min_height, self.pad_height_divisor)? - height;
        let dw = Self::output_axis(width, self.min_width, self.pad_width_divisor)? - width;
        let (left, top) = match self.position {
            PadPosition::Center => (dw / 2, dh / 2),
            PadPosition::TopLeft => (0, 0),
            PadPosition::TopRight => (dw, 0),
            PadPosition::BottomLeft => (0, dh),
            PadPosition::BottomRight => (dw, dh),
            PadPosition::Random => (
                rng.gen_range_usize(0..dw + 1)?,
                rng.gen_range_usize(0..dh + 1)?,
            ),
        };
        let pad = PadConfig::with_sides(
            left as u32,
            top as u32,
            (dw - left) as u32,
            (dh - top) as u32,
        )
        .with_fill(self.fill)
        .with_mode(self.padding_mode);
        Ok(self
            .fill_channels
            .map_or(pad, |f| pad.with_fill_channels(f)))
    }
}
