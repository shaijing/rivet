use rivet_data::random::{OpKey, RandomContext, RandomStream, SampleKey};

/// Semantic random context for one logical sample.
///
/// The context does not own a shared mutable RNG stream. Each stochastic
/// operator derives an independent local stream from its semantic `OpKey`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleContext {
    pub sample_key: SampleKey,
    pub random: RandomContext,
}

impl SampleContext {
    pub fn new(sample_index: usize) -> Self {
        Self {
            sample_key: SampleKey::from_index(sample_index),
            random: RandomContext::new(0),
        }
    }

    pub const fn with_random(sample_index: usize, random: RandomContext) -> Self {
        Self {
            sample_key: SampleKey::from_index(sample_index),
            random,
        }
    }

    pub const fn sample_index(self) -> u64 {
        self.sample_key.as_u64()
    }

    pub fn stream(&self, op_key: OpKey) -> RandomStream {
        self.random.stream(self.sample_key, op_key)
    }
}
