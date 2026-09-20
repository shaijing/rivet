use std::ops::Range;

use crate::errors::{DataResult, invalid_argument};

#[derive(Clone, Debug)]
pub struct RandomStream {
    state: u64,
}

impl RandomStream {
    pub const fn from_seed(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        super::mix_u64(self.state)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    pub fn next_f32(&mut self) -> f32 {
        const SCALE: f32 = 1.0 / ((1u32 << 24) as f32);
        ((self.next_u64() >> 40) as u32) as f32 * SCALE
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn gen_bool(&mut self, probability: f64) -> DataResult<bool> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(invalid_argument(
                "random probability must be finite and in [0, 1]",
            ));
        }
        Ok(self.next_f64() < probability)
    }

    pub fn gen_range_usize(&mut self, range: Range<usize>) -> DataResult<usize> {
        if range.start >= range.end {
            return Err(invalid_argument("random range must be non-empty"));
        }
        let width = (range.end - range.start) as u64;
        Ok(range.start + self.gen_below(width)? as usize)
    }

    fn gen_below(&mut self, upper: u64) -> DataResult<u64> {
        if upper == 0 {
            return Err(invalid_argument("random upper bound must be non-zero"));
        }
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return Ok(value % upper);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RandomStream;

    #[test]
    fn range_sampling_is_valid_and_rejects_empty_ranges() {
        let mut stream = RandomStream::from_seed(7);
        for _ in 0..100 {
            assert!(matches!(stream.gen_range_usize(3..8), Ok(value) if (3..8).contains(&value)));
        }
        assert!(stream.gen_range_usize(2..2).is_err());
    }

    #[test]
    fn float_generation_is_in_half_open_unit_interval() {
        let mut stream = RandomStream::from_seed(11);
        for _ in 0..1000 {
            assert!((0.0..1.0).contains(&stream.next_f32()));
            assert!((0.0..1.0).contains(&stream.next_f64()));
        }
    }
}
