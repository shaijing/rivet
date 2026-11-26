//! Random Gaussian blur parameter sampling.

use super::GaussianBlurConfig;
use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::ImageSample;
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomGaussianBlurConfig {
    pub kernel_size: u32,
    pub kernel_dimensions: Option<[u32; 2]>,
    pub sigma: [f32; 2],
}
impl RandomGaussianBlurConfig {
    pub const fn new(kernel_size: u32, sigma: [f32; 2]) -> Self {
        Self {
            kernel_size,
            kernel_dimensions: None,
            sigma,
        }
    }
    pub const fn with_kernel_dimensions(mut self, dimensions: [u32; 2]) -> Self {
        self.kernel_dimensions = Some(dimensions);
        self
    }
    pub fn validate(&self) -> RivetResult<()> {
        if self.sigma.iter().any(|v| !v.is_finite() || *v <= 0.0) || self.sigma[0] > self.sigma[1] {
            return Err(invalid_argument(
                "random gaussian sigma must be a positive ordered range",
            ));
        }
        let config = GaussianBlurConfig::new(self.sigma[0]);
        let config = if let Some(dimensions) = self.kernel_dimensions {
            config.with_kernel_dimensions(dimensions)
        } else {
            config.with_kernel_size(self.kernel_size)
        };
        config.validate()
    }
    pub fn apply(&self, sample: ImageSample, rng: &mut RandomStream) -> RivetResult<ImageSample> {
        self.validate()?;
        let sigma = self.sigma[0] + (self.sigma[1] - self.sigma[0]) * rng.next_f32();
        let config = GaussianBlurConfig::new(sigma);
        let config = if let Some(dimensions) = self.kernel_dimensions {
            config.with_kernel_dimensions(dimensions)
        } else {
            config.with_kernel_size(self.kernel_size)
        };
        config.apply(sample)
    }
}
