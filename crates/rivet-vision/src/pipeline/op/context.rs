pub struct SampleContext {
    pub sample_index: usize,
    pub epoch: u64,
    pub global_seed: u64,
    rng_state: Option<u64>,
}

impl SampleContext {
    pub fn new(sample_index: usize) -> Self {
        Self {
            sample_index,
            epoch: 0,
            global_seed: 0,
            rng_state: None,
        }
    }

    pub fn sample_seed(&self) -> u64 {
        let mut seed = self.global_seed ^ 0x9E37_79B9_7F4A_7C15;
        seed = mix_seed(seed ^ self.epoch);
        mix_seed(seed ^ self.sample_index as u64)
    }

    /// Draw the next deterministic u64 for a stochastic op on this sample.
    ///
    /// The stream is seeded from `(global_seed, epoch, sample_index)`, so
    /// every sample gets a reproducible, distinct draw sequence regardless
    /// of worker count or scheduling; each stochastic op consumes one or
    /// more draws in pipeline order.
    pub fn next_rng_u64(&mut self) -> u64 {
        if self.rng_state.is_none() {
            self.rng_state = Some(self.sample_seed());
        }
        splitmix64(self.rng_state.as_mut().unwrap())
    }
}

/// SplitMix64 stream step: advance the state and return a mixed output.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn mix_seed(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}
