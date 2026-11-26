use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use rivet_core::{DType, Tensor};

#[derive(Clone, Debug)]
pub struct VolumeSample {
    pub volume: Tensor,
    pub mask: Option<Tensor>,
    pub keypoints: Option<Tensor>,
    pub label: i64,
}

impl VolumeSample {
    pub fn spatial_shape(&self) -> RivetResult<[usize; 3]> {
        match self.volume.dims() {
            &[d, h, w] if d > 0 && h > 0 && w > 0 => Ok([d, h, w]),
            &[d, h, w, c] if d > 0 && h > 0 && w > 0 && c > 0 => Ok([d, h, w]),
            _ => Err(invalid_shape("volume must be nonempty DHW or DHWC")),
        }
    }

    pub fn validate(&self) -> RivetResult<()> {
        let size = self.spatial_shape()?;
        if !matches!(self.volume.dtype(), DType::U8 | DType::F32) {
            return Err(invalid_argument("volume requires uint8 or float32 pixels"));
        }
        if self.mask.as_ref().is_some_and(|mask| {
            mask.dims() != size || !matches!(mask.dtype(), DType::U8 | DType::I64)
        }) {
            return Err(invalid_shape(
                "volume mask requires matching DHW uint8 or int64 data",
            ));
        }
        if let Some(points) = &self.keypoints {
            if !matches!(points.dims(), [_, 3])
                || points.dtype() != DType::F32
                || points.to_vec::<f32>()?.iter().any(|v| !v.is_finite())
            {
                return Err(invalid_argument(
                    "volume keypoints require finite float32 [N,3] coordinates",
                ));
            }
        }
        Ok(())
    }
}
