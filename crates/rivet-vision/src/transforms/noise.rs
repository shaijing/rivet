use super::{RivetResult, contiguous_values, invalid_argument};
use rivet_core::{CpuStorageRef, DType, Layout, Tensor, WithDType};
use rivet_data::random::RandomStream;

pub(super) fn validate_gaussian_noise(mean: f64, sigma: f64) -> RivetResult<()> {
    if !mean.is_finite() || !sigma.is_finite() || sigma < 0.0 {
        return Err(invalid_argument("invalid GaussianNoise parameters"));
    }
    Ok(())
}

fn standard_normal(rng: &mut RandomStream) -> f32 {
    let radius = (-2.0 * rng.next_f64().max(f64::MIN_POSITIVE).ln()).sqrt();
    (radius * (std::f64::consts::TAU * rng.next_f64()).cos()) as f32
}

fn map_pixels<T: WithDType>(
    input: &Tensor,
    layout: &Layout,
    values: &[T],
    mut convert: impl FnMut(T) -> T,
) -> rivet_core::Result<Tensor> {
    if layout.is_contiguous() {
        Tensor::from_exact_iter(
            contiguous_values(values, layout)?
                .iter()
                .copied()
                .map(convert),
            input.dims(),
            input.device(),
        )
    } else {
        Tensor::from_exact_try_iter(
            layout.strided_index().map(|i| {
                values
                    .get(i)
                    .copied()
                    .map(&mut convert)
                    .ok_or(rivet_core::Error::StorageOutOfBounds)
            }),
            input.dims(),
            input.device(),
        )
    }
}

/// Add independent Gaussian noise to each element of a CPU image tensor.
///
/// Supports uint8 and float32, including strided views and tensors with leading
/// batch or frame dimensions. Shape and dtype are preserved. Mean and standard
/// deviation are in normalized units for both dtypes. With clipping enabled,
/// outputs are limited to [0, 1] for float32 or [0, 255] for uint8.
/// Unclipped uint8 output wraps. Sampling uses the supplied Rivet random stream.
pub fn gaussian_noise(
    input: &Tensor,
    mean: f64,
    sigma: f64,
    clip: bool,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    validate_gaussian_noise(mean, sigma)?;
    if !matches!(input.dtype(), DType::U8 | DType::F32) {
        return Err(invalid_argument(
            "GaussianNoise requires uint8 or float32 input",
        ));
    }
    let mut normal = || -> f32 {
        if sigma == 0.0 {
            return 0.0;
        }
        standard_normal(rng)
    };
    Ok(input.with_cpu_storage(|storage, layout| {
        match storage {
            CpuStorageRef::F32(values) => map_pixels(input, layout, values, |value: f32| {
                let noise = mean as f32 + normal() * sigma as f32;
                let result = value + noise;
                if clip { result.clamp(0.0, 1.0) } else { result }
            }),
            CpuStorageRef::U8(values) => map_pixels(input, layout, values, |value: u8| {
                let noise = (mean * 255.0) as f32 + normal() * (sigma * 255.0) as f32;
                // PyTorch converts the noise to int16 *before* addition.
                let noise = if noise.is_finite() {
                    noise.trunc().rem_euclid(65536.0) as u16 as i16
                } else {
                    0
                };
                let result = i16::from(value).wrapping_add(noise);
                if clip {
                    result.clamp(0, 255) as u8
                } else {
                    result as u8
                }
            }),
            _ => unreachable!(),
        }
    })?)
}

pub(super) fn validate_shot_noise(intensity: f32) -> RivetResult<()> {
    if !intensity.is_finite() || intensity < 0.0 {
        return Err(invalid_argument(
            "ShotNoise intensity must be finite and nonnegative",
        ));
    }
    Ok(())
}

/// Add signal-dependent Gaussian noise to a normalized float32 CPU image tensor.
///
/// Each element receives independent standard normal noise scaled by
/// `intensity * sqrt(max(pixel, 0) + 0.5)`. Results are clipped to [0, 1].
/// Shape is preserved for strided views and tensors with batch or frame dimensions.
pub fn shot_noise(input: &Tensor, intensity: f32, rng: &mut RandomStream) -> RivetResult<Tensor> {
    validate_shot_noise(intensity)?;
    if input.dtype() != DType::F32 {
        return Err(invalid_argument("ShotNoise requires float32 input"));
    }
    Ok(input.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::F32(values) = storage else {
            unreachable!()
        };
        if intensity == 0.0 {
            return map_pixels(input, layout, values, |pixel| pixel.clamp(0.0, 1.0));
        }
        map_pixels(input, layout, values, |pixel| {
            let scale = (pixel.max(0.0) + 0.5).sqrt();
            (pixel + intensity * standard_normal(rng) * scale).clamp(0.0, 1.0)
        })
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rivet_core::Device;

    #[test]
    fn shot_noise_scales_each_strided_pixel_and_clips() {
        let input = Tensor::from_vec(vec![-0.25f32, 0.1, 0.5, 1.25], [2, 2], &Device::Cpu)
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let mut rng = RandomStream::from_seed(11);
        let mut reference_rng = rng.clone();
        let expected = [-0.25f32, 0.5, 0.1, 1.25].map(|pixel| {
            let normal = standard_normal(&mut reference_rng);
            (pixel + 0.2 * normal * (pixel.max(0.0) + 0.5).sqrt()).clamp(0.0, 1.0)
        });
        let output = shot_noise(&input, 0.2, &mut rng).unwrap();
        assert_eq!(output.dims(), input.dims());
        assert_eq!(output.to_vec::<f32>().unwrap(), expected);
        let zero = shot_noise(&input, 0.0, &mut rng).unwrap();
        assert_eq!(zero.to_vec::<f32>().unwrap(), [0.0, 0.5, 0.1, 1.0]);
        assert!(shot_noise(&input, f32::INFINITY, &mut rng).is_err());
        assert!(shot_noise(&input, -0.2, &mut rng).is_err());
        let uint8 = Tensor::from_vec(vec![0u8], [1], &Device::Cpu).unwrap();
        assert!(shot_noise(&uint8, 0.2, &mut rng).is_err());
    }

    #[test]
    fn uint8_noise_truncates_before_addition_on_strided_inputs() {
        let input = Tensor::from_vec(vec![0u8, 1, 250, 255], [2, 2], &Device::Cpu)
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let mut rng = RandomStream::from_seed(7);
        for (clip, expected) in [(true, [0, 249, 0, 254]), (false, [255, 249, 0, 254])] {
            let output = gaussian_noise(&input, -1.5 / 255.0, 0.0, clip, &mut rng).unwrap();
            assert_eq!(output.dims(), input.dims());
            assert_eq!(output.to_vec::<u8>().unwrap(), expected);
        }
    }

    #[test]
    fn float_noise_clips_and_preserves_batched_shape() {
        let input = Tensor::from_vec(vec![0.25f32, 0.75], [2, 1, 1, 1], &Device::Cpu).unwrap();
        let mut rng = RandomStream::from_seed(7);
        for (clip, expected) in [(true, [0.75, 1.0]), (false, [0.75, 1.25])] {
            let output = gaussian_noise(&input, 0.5, 0.0, clip, &mut rng).unwrap();
            assert_eq!(output.dims(), input.dims());
            assert_eq!(output.to_vec::<f32>().unwrap(), expected);
        }
        assert!(gaussian_noise(&input, 0.0, -1.0, true, &mut rng).is_err());
        assert!(gaussian_noise(&input, f64::NAN, 0.0, true, &mut rng).is_err());
    }

    #[test]
    fn noise_is_seeded_and_sampled_independently_across_frames() {
        let input = Tensor::from_vec(vec![0.5f32; 16], [2, 1, 2, 4], &Device::Cpu).unwrap();
        let sample = || {
            gaussian_noise(&input, 0.0, 0.1, false, &mut RandomStream::from_seed(7))
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
        };
        let output = sample();
        assert_eq!(output, sample());
        assert_ne!(output[..8], output[8..]);
    }
}
