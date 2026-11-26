use super::VolumeSample;
use crate::errors::RivetResult;

impl VolumeSample {
    pub fn apply_callback(
        mut self,
        config: &crate::api::CallbackConfig,
        rng: &mut crate::RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        if let Some(seed) = config.sample(rng)? {
            self.volume = config.apply("volume", &self.volume, seed)?;
            if let Some(mask) = &mut self.mask {
                *mask = config.apply("mask", mask, seed)?;
            }
            if let Some(points) = &mut self.keypoints {
                *points = config.apply("keypoints", points, seed)?;
            }
            self.validate()?;
        }
        Ok(self)
    }
}
