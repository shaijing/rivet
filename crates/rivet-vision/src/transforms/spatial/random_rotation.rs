//! Random parameter policies whose execution stays in the native kernels.
use super::{ArbitraryRotateConfig, InterpolationMode};
use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::{ImageAxisOrder, ImageSample};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomRotationConfig {
    pub degrees: [f32; 2],
    pub interpolation: InterpolationMode,
    pub expand: bool,
    pub fill: u8,
    pub fill_channels: Option<[u8; 3]>,
    pub center: Option<[f32; 2]>,
}
impl RandomRotationConfig {
    pub const fn new(degrees: [f32; 2]) -> Self {
        Self {
            degrees,
            interpolation: InterpolationMode::Nearest,
            expand: false,
            fill: 0,
            fill_channels: None,
            center: None,
        }
    }
    pub const fn with_center(mut self, center: [f32; 2]) -> Self {
        self.center = Some(center);
        self
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
        self.fill_channels = None;
        self
    }
    pub const fn with_fill_channels(mut self, fill: [u8; 3]) -> Self {
        self.fill_channels = Some(fill);
        self
    }
    pub fn validate(&self) -> RivetResult<()> {
        if self.degrees.iter().any(|v| !v.is_finite()) || self.degrees[0] > self.degrees[1] {
            return Err(invalid_argument(
                "random rotation degrees must be a finite ordered range",
            ));
        }
        if self
            .center
            .is_some_and(|center| center.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid_argument("rotation center must be finite"));
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
        let mut config = ArbitraryRotateConfig::new(angle).with_options(
            self.expand,
            self.interpolation,
            self.fill,
        );
        if let Some(fill) = self.fill_channels {
            config = config.with_fill_channels(fill);
        }
        if let Some(center) = self.center {
            config = config.with_center(center[0], center[1]);
        }
        config.apply(sample, axis)
    }
}
