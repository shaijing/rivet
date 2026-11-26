use super::VolumeSample;
use crate::errors::{RivetResult, invalid_argument};

impl VolumeSample {
    pub fn rician_noise(
        mut self,
        std_range: [f64; 2],
        per_channel: bool,
        rng: &mut crate::RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        let channels = self.volume.dims().get(3).copied().unwrap_or(1);
        self.volume = crate::transforms::degradation::noise::rician_noise(
            &self.volume,
            channels,
            std_range,
            per_channel,
            rng,
        )?;
        Ok(self)
    }
}

impl VolumeSample {
    pub fn he_stain(
        mut self,
        config: crate::transforms::HEStainConfig,
        rng: &mut crate::RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        if self.volume.rank() != 4 {
            return Err(invalid_argument("HEStain requires DHWC RGB volume"));
        }
        self.volume = config.apply(&self.volume, rng)?;
        Ok(self)
    }
}
