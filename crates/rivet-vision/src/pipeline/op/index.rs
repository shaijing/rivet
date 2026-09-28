use crate::errors::RivetResult;
use crate::sampler::{SamplerPlan, permute};
use rivet_data::random::{RandomContext, combine};

#[derive(Clone)]
pub enum IndexOp {
    Skip {
        count: usize,
    },
    Take {
        count: usize,
    },
    /// Deterministically shuffle the sequence selected by preceding operations.
    /// Later Skip/Take select from this permutation in declaration order.
    Shuffle {
        seed: u64,
    },
}

impl IndexOp {
    pub fn apply_range(&self, start: &mut usize, end: &mut usize) {
        match self {
            Self::Skip { count } => {
                *start = start.saturating_add(*count).min(*end);
            }
            Self::Take { count } => {
                *end = start.saturating_add(*count).min(*end);
            }
            Self::Shuffle { .. } => {}
        }
    }
}

pub fn compile_sampler(
    len: usize,
    index_ops: &[IndexOp],
    random: RandomContext,
) -> RivetResult<SamplerPlan> {
    let mut plan = SamplerPlan::Sequential { start: 0, end: len };
    let mut shuffle_ordinal = 0u64;
    for op in index_ops {
        match (&mut plan, op) {
            (
                SamplerPlan::Sequential { start, end },
                IndexOp::Skip { .. } | IndexOp::Take { .. },
            ) => {
                op.apply_range(start, end);
            }
            (SamplerPlan::Permutation { indices }, IndexOp::Skip { count }) => {
                indices.drain(..(*count).min(indices.len()));
            }
            (SamplerPlan::Permutation { indices }, IndexOp::Take { count }) => {
                indices.truncate(*count);
            }
            (_, IndexOp::Shuffle { seed }) => {
                // Preserve the V1 seed for the first shuffle, including the
                // builder's global-seed override. Additional shuffles use
                // distinct deterministic namespaces and their declared seed.
                let seed = if shuffle_ordinal == 0 {
                    random.sampler_seed()
                } else {
                    combine(random.sampler_seed(), combine(shuffle_ordinal, *seed))
                };
                shuffle_ordinal += 1;
                let mut order = permute(plan.len(), seed);
                let indices = match &plan {
                    SamplerPlan::Sequential { start, .. } => {
                        for index in &mut order {
                            *index += start;
                        }
                        order
                    }
                    SamplerPlan::Permutation { indices } => {
                        order.into_iter().map(|index| indices[index]).collect()
                    }
                };
                plan = SamplerPlan::Permutation { indices };
            }
        }
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::IndexSampler;

    fn selected(len: usize, ops: &[IndexOp], random: RandomContext) -> Vec<usize> {
        let plan = compile_sampler(len, ops, random).unwrap();
        IndexSampler::new(plan, 0)
            .next_indices(usize::MAX)
            .unwrap_or_default()
    }

    #[test]
    fn sampling_preserves_declared_slice_and_shuffle_order() {
        let random = RandomContext::new(11);
        let full = permute(20, random.sampler_seed());
        let shuffled_first = selected(
            20,
            &[IndexOp::Shuffle { seed: 11 }, IndexOp::Take { count: 4 }],
            random,
        );
        assert_eq!(shuffled_first, full[..4]);
        let taken_first = selected(
            20,
            &[IndexOp::Take { count: 4 }, IndexOp::Shuffle { seed: 11 }],
            random,
        );
        assert_eq!(taken_first, permute(4, random.sampler_seed()));
        assert_ne!(shuffled_first, taken_first);
        assert_eq!(
            selected(
                20,
                &[
                    IndexOp::Shuffle { seed: 11 },
                    IndexOp::Skip { count: 3 },
                    IndexOp::Take { count: 4 }
                ],
                random
            ),
            full[3..7]
        );
    }

    #[test]
    fn sequential_slices_saturate_without_allocation() {
        let plan = compile_sampler(
            usize::MAX,
            &[
                IndexOp::Skip { count: 10 },
                IndexOp::Take { count: usize::MAX },
            ],
            RandomContext::new(0),
        )
        .unwrap();
        assert!(matches!(
            plan,
            SamplerPlan::Sequential {
                start: 10,
                end: usize::MAX
            }
        ));
        let plan = compile_sampler(
            20,
            &[
                IndexOp::Skip { count: usize::MAX },
                IndexOp::Take { count: usize::MAX },
            ],
            RandomContext::new(0),
        )
        .unwrap();
        assert!(matches!(
            plan,
            SamplerPlan::Sequential { start: 20, end: 20 }
        ));
    }

    #[test]
    fn repeated_shuffles_compose_and_separate_seed_and_epoch() {
        let random = RandomContext::new(11).with_epoch(2);
        let first = permute(20, random.sampler_seed());
        let second = permute(7, combine(random.sampler_seed(), combine(1, 22)));
        let expected = second
            .into_iter()
            .map(|index| first[index + 3])
            .collect::<Vec<_>>();
        let ops = [
            IndexOp::Shuffle { seed: 11 },
            IndexOp::Skip { count: 3 },
            IndexOp::Take { count: 7 },
            IndexOp::Shuffle { seed: 22 },
        ];
        assert_eq!(selected(20, &ops, random), expected);
        assert_eq!(selected(20, &ops, random), selected(20, &ops, random));
        assert_ne!(
            selected(20, &ops, random),
            selected(20, &ops, random.with_epoch(3))
        );
        let other = [
            ops[0].clone(),
            ops[1].clone(),
            ops[2].clone(),
            IndexOp::Shuffle { seed: 23 },
        ];
        assert_ne!(selected(20, &ops, random), selected(20, &other, random));
    }
}
