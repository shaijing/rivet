use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::transforms::{InterpolationMode, PadConfig, ResizeConfig};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LetterBoxConfig {
    pub width: u32,
    pub height: u32,
    pub interpolation: InterpolationMode,
    pub fill: f32,
    pub mask_fill: i64,
}

impl LetterBoxConfig {
    pub fn validate(&self) -> RivetResult<()> {
        if self.width == 0 || self.height == 0 || !self.fill.is_finite() {
            return Err(invalid_argument("invalid LetterBox parameters"));
        }
        Ok(())
    }

    pub(crate) fn resolve(
        &self,
        width: usize,
        height: usize,
    ) -> RivetResult<(ResizeConfig, PadConfig)> {
        let width =
            u32::try_from(width).map_err(|_| invalid_shape("LetterBox input width exceeds u32"))?;
        let height = u32::try_from(height)
            .map_err(|_| invalid_shape("LetterBox input height exceeds u32"))?;
        if width == 0 || height == 0 {
            return Err(invalid_shape("LetterBox requires a nonempty image"));
        }
        let (out_w, out_h) = (u64::from(self.width), u64::from(self.height));
        let (width, height) = (u64::from(width), u64::from(height));
        let (w, h) = if out_w * height <= out_h * width {
            (out_w, ((height * out_w + width / 2) / width).max(1))
        } else {
            (((width * out_h + height / 2) / height).max(1), out_h)
        };
        let (w, h) = (w as u32, h as u32);
        let (horizontal, vertical) = (self.width - w, self.height - h);
        let (left, top) = (horizontal / 2, vertical / 2);
        Ok((
            ResizeConfig::with_interpolation(w, h, self.interpolation),
            PadConfig::with_sides(left, top, horizontal - left, vertical - top)
                .with_fill(self.fill),
        ))
    }
}
