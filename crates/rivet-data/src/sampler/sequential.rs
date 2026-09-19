#[derive(Clone)]
pub enum SamplerPlan {
    Sequential { start: usize, end: usize },
    Permutation { indices: Vec<usize> },
}

pub struct IndexSampler {
    plan: SamplerPlan,
    position: usize,
}

impl IndexSampler {
    pub fn new(plan: SamplerPlan, position: usize) -> Self {
        let len = plan.len();
        Self {
            plan,
            position: position.min(len),
        }
    }

    pub fn next_indices(&mut self, batch_size: usize) -> Option<Vec<usize>> {
        if self.position >= self.plan.len() {
            return None;
        }

        let end = (self.position + batch_size).min(self.plan.len());
        let indices = self.plan.slice(self.position, end);
        self.position = end;
        Some(indices)
    }
}

impl SamplerPlan {
    pub fn len(&self) -> usize {
        match self {
            Self::Sequential { start, end } => end.saturating_sub(*start),
            Self::Permutation { indices } => indices.len(),
        }
    }

    fn slice(&self, start: usize, end: usize) -> Vec<usize> {
        match self {
            Self::Sequential {
                start: plan_start, ..
            } => (*plan_start + start..*plan_start + end).collect(),
            Self::Permutation { indices } => indices[start..end].to_vec(),
        }
    }
}

/// Deterministic Fisher-Yates permutation of `0..len` seeded by `seed`:
/// the same seed always yields the same order, independent of worker
/// count or scheduling.
pub fn permute(len: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..len).collect();
    let mut state = seed;
    for i in (1..len).rev() {
        state = splitmix64(state);
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
    indices
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::{IndexSampler, SamplerPlan, permute};

    #[test]
    fn sequential_sampler_yields_ranges() {
        let mut sampler = IndexSampler::new(SamplerPlan::Sequential { start: 2, end: 7 }, 0);

        assert_eq!(sampler.next_indices(3), Some(vec![2, 3, 4]));
        assert_eq!(sampler.next_indices(3), Some(vec![5, 6]));
        assert_eq!(sampler.next_indices(3), None);
    }

    #[test]
    fn permutation_is_deterministic_and_complete() {
        let a = permute(100, 42);
        let b = permute(100, 42);
        let c = permute(100, 43);
        assert_eq!(a, b, "same seed must reproduce the same order");
        assert_ne!(a, c, "different seeds must differ");
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            (0..100).collect::<Vec<_>>(),
            "must be a permutation"
        );
    }
}
