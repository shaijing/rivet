use crate::errors::{RivetResult, invalid_pipeline};
use crate::sampler::{SamplerPlan, permute};
use rivet_data::random::RandomContext;

#[derive(Clone)]
pub enum IndexOp {
    Skip {
        count: usize,
    },
    Take {
        count: usize,
    },
    /// Deterministically shuffle the selected index window with a seed.
    /// Skip/Take apply first, then the window is permuted, so a fixed
    /// `(seed, epoch)` always reproduces the same order regardless of
    /// worker count.
    Shuffle {
        seed: u64,
    },
}

impl IndexOp {
    pub fn apply_range(&self, start: &mut usize, end: &mut usize) {
        match self {
            Self::Skip { count } => {
                *start = (*start + *count).min(*end);
            }
            Self::Take { count } => {
                *end = (*start + *count).min(*end);
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
    let mut start = 0usize;
    let mut end = len;
    let mut shuffle_seed: Option<u64> = None;

    for op in index_ops {
        match op {
            IndexOp::Shuffle { seed } => {
                if shuffle_seed.is_some() {
                    return Err(invalid_pipeline(
                        "multiple shuffle ops are not supported; combine seeds outside the pipeline",
                    ));
                }
                shuffle_seed = Some(*seed);
            }
            _ => op.apply_range(&mut start, &mut end),
        }
    }

    match shuffle_seed {
        None => Ok(SamplerPlan::Sequential { start, end }),
        Some(_) => {
            let window_len = end - start;
            let mut indices = permute(window_len, random.sampler_seed());
            for index in &mut indices {
                *index += start;
            }
            Ok(SamplerPlan::Permutation { indices })
        }
    }
}
