use crate::errors::{RivetResult, invalid_argument};
use rivet_data::random::RandomStream;

/// Validate relative weights and precompute a normalized cumulative distribution.
/// None preserves the existing uniform integer sampler and its random stream.
pub(crate) fn choice_cumulative_weights(
    count: usize,
    weights: Option<&[f64]>,
) -> RivetResult<Option<Vec<f64>>> {
    if count == 0 {
        return Err(invalid_argument(
            "random_choice requires at least one choice",
        ));
    }
    let Some(weights) = weights else {
        return Ok(None);
    };
    if weights.len() != count {
        return Err(invalid_argument(
            "random_choice weights must match the number of choices",
        ));
    }
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(invalid_argument(
            "random_choice weights must be finite and non-negative",
        ));
    }
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(invalid_argument(
            "random_choice weights must have a finite positive sum",
        ));
    }
    let mut sum = 0.0;
    let mut cumulative: Vec<f64> = weights
        .iter()
        .map(|weight| {
            sum += weight / total;
            sum
        })
        .collect();
    // Avoid rounding leaving a draw outside the distribution. Preserve trailing
    // zero-weight branches by saturating at the final positive weight.
    let last_positive = weights.iter().rposition(|weight| *weight > 0.0).unwrap();
    cumulative[last_positive..].fill(1.0);
    Ok(Some(cumulative))
}

pub(crate) fn choose_branch(
    rng: &mut RandomStream,
    count: usize,
    cumulative: Option<&[f64]>,
) -> RivetResult<usize> {
    match cumulative {
        None => Ok(rng.gen_range_usize(0..count)?),
        Some(cumulative) => {
            let draw = rng.next_f64();
            Ok(cumulative
                .partition_point(|&upper| upper <= draw)
                .min(count - 1))
        }
    }
}

#[cfg(test)]
mod choice_weight_tests {
    use super::choice_cumulative_weights;

    #[test]
    fn uniform_choice_keeps_original_sampler() {
        assert_eq!(choice_cumulative_weights(3, None).unwrap(), None);
    }

    #[test]
    fn normalizes_weights_and_preserves_zero_probability_branches() {
        assert_eq!(
            choice_cumulative_weights(4, Some(&[0.0, 1.0, 3.0, 0.0])).unwrap(),
            Some(vec![0.0, 0.25, 1.0, 1.0])
        );
    }

    #[test]
    fn rejects_invalid_choice_weights() {
        assert!(choice_cumulative_weights(0, None).is_err());
        for weights in [
            vec![],
            vec![1.0],
            vec![0.0, 0.0],
            vec![-1.0, 2.0],
            vec![f64::NAN, 1.0],
            vec![f64::INFINITY, 1.0],
            vec![f64::MAX, f64::MAX],
        ] {
            assert!(choice_cumulative_weights(2, Some(&weights)).is_err());
        }
    }
}
