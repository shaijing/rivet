#[derive(Clone)]
pub enum SamplerPlan {
    Sequential {
        start: usize,
        end: usize,
    },
    #[allow(dead_code)]
    Permutation {
        indices: Vec<usize>,
    },
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

#[cfg(test)]
mod tests {
    use super::{IndexSampler, SamplerPlan};

    #[test]
    fn sequential_sampler_yields_ranges() {
        let mut sampler = IndexSampler::new(SamplerPlan::Sequential { start: 2, end: 7 }, 0);

        assert_eq!(sampler.next_indices(3), Some(vec![2, 3, 4]));
        assert_eq!(sampler.next_indices(3), Some(vec![5, 6]));
        assert_eq!(sampler.next_indices(3), None);
    }
}
