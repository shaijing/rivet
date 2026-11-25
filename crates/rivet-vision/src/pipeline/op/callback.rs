use crate::errors::{RivetResult, invalid_argument};
use rivet_core::Tensor;
use rivet_data::random::RandomStream;
use std::sync::Arc;

pub type TargetCallback = dyn Fn(&str, &Tensor, u64) -> RivetResult<Tensor> + Send + Sync;

#[derive(Clone)]
pub struct CallbackConfig {
    pub probability: f64,
    pub callback: Arc<TargetCallback>,
}

impl CallbackConfig {
    pub fn validate(&self) -> RivetResult<()> {
        if !(0.0..=1.0).contains(&self.probability) {
            return Err(invalid_argument("Lambda probability must be in [0, 1]"));
        }
        Ok(())
    }

    pub(crate) fn sample(&self, rng: &mut RandomStream) -> RivetResult<Option<u64>> {
        Ok(rng.gen_bool(self.probability)?.then(|| rng.next_u64()))
    }

    pub(crate) fn apply(&self, name: &str, input: &Tensor, seed: u64) -> RivetResult<Tensor> {
        let output = (self.callback)(name, input, seed)?;
        if output.dims() != input.dims()
            || output.dtype() != input.dtype()
            || !output.device().same_device(input.device())
        {
            return Err(invalid_argument(format!(
                "Lambda {name} must preserve shape, dtype and device"
            )));
        }
        Ok(output)
    }
}
