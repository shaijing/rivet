//! Random parameter policies whose execution stays in the native kernels.
use super::{ArbitraryRotateConfig, GaussianBlurConfig, InterpolationMode};
use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::{ImageAxisOrder, ImageSample};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomRotationConfig {
    pub degrees: [f32; 2],
    pub interpolation: InterpolationMode,
    pub expand: bool,
    pub fill: u8,
}
impl RandomRotationConfig {
    pub const fn new(degrees: [f32; 2]) -> Self {
        Self {
            degrees,
            interpolation: InterpolationMode::Nearest,
            expand: false,
            fill: 0,
        }
    }
    pub const fn with_options(
        mut self,
        interpolation: InterpolationMode,
        expand: bool,
        fill: u8,
    ) -> Self {
        self.interpolation = interpolation;
        self.expand = expand;
        self.fill = fill;
        self
    }
    pub fn validate(&self) -> RivetResult<()> {
        if self.degrees.iter().any(|v| !v.is_finite()) || self.degrees[0] > self.degrees[1] {
            return Err(invalid_argument(
                "random rotation degrees must be a finite ordered range",
            ));
        }
        ArbitraryRotateConfig::new(0.0)
            .with_options(self.expand, self.interpolation, self.fill)
            .validate()
    }
    pub fn apply(
        &self,
        sample: ImageSample,
        rng: &mut RandomStream,
        axis: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let angle = self.degrees[0] + (self.degrees[1] - self.degrees[0]) * rng.next_f32();
        ArbitraryRotateConfig::new(angle)
            .with_options(self.expand, self.interpolation, self.fill)
            .apply(sample, axis)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomGaussianBlurConfig {
    pub kernel_size: u32,
    pub sigma: [f32; 2],
}
impl RandomGaussianBlurConfig {
    pub const fn new(kernel_size: u32, sigma: [f32; 2]) -> Self {
        Self { kernel_size, sigma }
    }
    pub fn validate(&self) -> RivetResult<()> {
        if self.sigma.iter().any(|v| !v.is_finite() || *v <= 0.0) || self.sigma[0] > self.sigma[1] {
            return Err(invalid_argument(
                "random gaussian sigma must be a positive ordered range",
            ));
        }
        GaussianBlurConfig::new(self.sigma[0])
            .with_kernel_size(self.kernel_size)
            .validate()
    }
    pub fn apply(&self, sample: ImageSample, rng: &mut RandomStream) -> RivetResult<ImageSample> {
        self.validate()?;
        let sigma = self.sigma[0] + (self.sigma[1] - self.sigma[0]) * rng.next_f32();
        GaussianBlurConfig::new(sigma)
            .with_kernel_size(self.kernel_size)
            .apply(sample)
    }
}
