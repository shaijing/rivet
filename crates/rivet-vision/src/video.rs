//! Videos are decoded rank-4 THWC/TCHW sequences; timestamps follow frames.
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use rivet_core::{DType, Tensor};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoLayout {
    Thwc,
    Tchw,
}
impl VideoLayout {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Thwc => "THWC",
            Self::Tchw => "TCHW",
        }
    }
}
#[derive(Clone, Debug)]
pub struct VideoSample {
    pub frames: Tensor,
    pub timestamps: Option<Tensor>,
    pub label: i64,
    pub layout: VideoLayout,
}
impl VideoSample {
    pub fn validate(&self) -> RivetResult<()> {
        if self.frames.rank() != 4
            || self.frames.dims().contains(&0)
            || !matches!(self.frames.dtype(), DType::U8 | DType::F32)
        {
            return Err(invalid_shape(
                "video requires nonempty rank-4 uint8/float32 THWC or TCHW frames",
            ));
        }
        if let Some(t) = &self.timestamps {
            if t.dims() != [self.frames.dims()[0]] || t.dtype() != DType::F64 {
                return Err(invalid_shape("timestamps require float64 [T]"));
            }
            let values = t.to_vec::<f64>()?;
            if values.iter().any(|v| !v.is_finite()) || values.windows(2).any(|p| p[1] < p[0]) {
                return Err(invalid_argument(
                    "timestamps must be finite and nondecreasing",
                ));
            }
        }
        Ok(())
    }
    pub fn uniform_temporal_subsample(mut self, num_samples: usize) -> RivetResult<Self> {
        self.validate()?;
        self.frames = uniform_temporal_subsample(&self.frames, num_samples, 0)?;
        if let Some(t) = self.timestamps {
            self.timestamps = Some(uniform_temporal_subsample(&t, num_samples, 0)?);
        }
        Ok(self)
    }
}
/// Uniform inclusive endpoints, truncating linspace positions toward zero.
/// Oversampling repeats frames; a single requested sample selects frame zero.
pub fn uniform_temporal_subsample(
    input: &Tensor,
    num_samples: usize,
    axis: usize,
) -> RivetResult<Tensor> {
    if num_samples == 0 || axis >= input.rank() || input.dims()[axis] == 0 {
        return Err(invalid_argument(
            "temporal axis must be nonempty and num_samples positive",
        ));
    }
    let t = input.dims()[axis];
    let mut dims = input.dims().to_vec();
    dims[axis] = num_samples;
    dims.iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| invalid_shape("video sample size overflow"))?;
    let views = (0..num_samples)
        .map(|i| {
            let index = if num_samples == 1 {
                0
            } else {
                // Match CPU torch.linspace's default float32 step and mirrored
                // endpoint evaluation, including fused rounding before truncation.
                let end = (t - 1) as f32;
                let step = end / (num_samples - 1) as f32;
                let position = if i < num_samples / 2 {
                    step * (i as f32)
                } else {
                    (-step).mul_add((num_samples - i - 1) as f32, end)
                };
                (position as usize).min(t - 1)
            };
            input.narrow(axis, index, 1)
        })
        .collect::<rivet_core::Result<Vec<_>>>()?;
    let refs: Vec<_> = views.iter().collect();
    Ok(Tensor::cat(&refs, axis)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rivet_core::Device;
    #[test]
    fn temporal_linspace_rounding_and_strided_axis() {
        let tensor = Tensor::from_vec(vec![0u8, 1, 2], [3, 1, 1, 1], &Device::Cpu).unwrap();
        assert_eq!(
            uniform_temporal_subsample(&tensor, 15, 0)
                .unwrap()
                .to_vec::<u8>()
                .unwrap(),
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2]
        );
        let x = Tensor::from_vec((0..12).collect::<Vec<i64>>(), [3, 4], &Device::Cpu)
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        assert_eq!(
            uniform_temporal_subsample(&x, 2, 1)
                .unwrap()
                .to_vec::<i64>()
                .unwrap(),
            [0, 8, 1, 9, 2, 10, 3, 11]
        );
        assert!(uniform_temporal_subsample(&x, 0, 1).is_err());
        assert!(uniform_temporal_subsample(&x, 2, 2).is_err());
    }
    #[test]
    fn video_timestamps_follow_repeated_frame_indices() {
        let frames = Tensor::from_vec(vec![10u8, 20, 30], [3, 1, 1, 1], &Device::Cpu).unwrap();
        let timestamps = Tensor::from_vec(vec![0f64, 0.4, 0.9], [3], &Device::Cpu).unwrap();
        let video = VideoSample {
            frames,
            timestamps: Some(timestamps),
            label: 5,
            layout: VideoLayout::Thwc,
        }
        .uniform_temporal_subsample(5)
        .unwrap();
        assert_eq!(video.frames.to_vec::<u8>().unwrap(), [10, 10, 20, 20, 30]);
        assert_eq!(
            video.timestamps.unwrap().to_vec::<f64>().unwrap(),
            [0., 0., 0.4, 0.4, 0.9]
        );
        assert_eq!(video.label, 5);
    }
}
