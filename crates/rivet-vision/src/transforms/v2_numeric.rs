use super::{RivetResult, contiguous_values, invalid_argument};
use rivet_core::{CpuStorageRef, DType, Tensor};
use rivet_data::random::RandomStream;

fn wrapping_u8(value: f32) -> u8 {
    if value.is_finite() {
        value.trunc().rem_euclid(256.0) as u8
    } else {
        0
    }
}

pub(super) fn cast_unscaled(input: &Tensor, dtype: DType) -> RivetResult<Tensor> {
    if !matches!(input.dtype(), DType::U8 | DType::F32) || !matches!(dtype, DType::U8 | DType::F32)
    {
        return Err(invalid_argument("ToDtype supports uint8 and float32 only"));
    }
    if input.dtype() == dtype {
        return Ok(input.clone());
    }
    if dtype == DType::F32 {
        return Ok(input.to_dtype(dtype)?);
    }
    Ok(input.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::F32(values) = storage else {
            unreachable!()
        };
        if layout.is_contiguous() {
            Tensor::from_exact_iter(
                contiguous_values(values, layout)?
                    .iter()
                    .copied()
                    .map(wrapping_u8),
                input.dims(),
                input.device(),
            )
        } else {
            Tensor::from_exact_try_iter(
                layout.strided_index().map(|i| {
                    values
                        .get(i)
                        .copied()
                        .map(wrapping_u8)
                        .ok_or(rivet_core::Error::StorageOutOfBounds)
                }),
                input.dims(),
                input.device(),
            )
        }
    })?)
}

pub(super) fn gaussian_noise(
    input: &Tensor,
    mean: f64,
    sigma: f64,
    clip: bool,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    if !matches!(input.dtype(), DType::U8 | DType::F32) {
        return Err(invalid_argument(
            "GaussianNoise requires uint8 or float32 input",
        ));
    }
    let mut normal = || -> f32 {
        if sigma == 0.0 {
            return 0.0;
        }
        let radius = (-2.0 * rng.next_f64().max(f64::MIN_POSITIVE).ln()).sqrt();
        (radius * (std::f64::consts::TAU * rng.next_f64()).cos()) as f32
    };
    Ok(input.with_cpu_storage(|storage, layout| {
        macro_rules! write_values {
            ($values:expr, $convert:expr) => {{
                let values = $values;
                let mut convert = $convert;
                if layout.is_contiguous() {
                    Tensor::from_exact_iter(
                        contiguous_values(values, layout)?
                            .iter()
                            .copied()
                            .map(&mut convert),
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
            }};
        }
        match storage {
            CpuStorageRef::F32(values) => write_values!(values, |value: f32| {
                let noise = mean as f32 + normal() * sigma as f32;
                let result = value + noise;
                if clip { result.clamp(0.0, 1.0) } else { result }
            }),
            CpuStorageRef::U8(values) => write_values!(values, |value: u8| {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rivet_core::Device;

    #[test]
    fn unscaled_cast_truncates_wraps_and_reads_strides() {
        let input = Tensor::from_vec(
            vec![-1.5f32, 256.9, 511.1, f32::NAN, 5., 6.],
            [2, 3],
            &Device::Cpu,
        )
        .unwrap()
        .transpose(0, 1)
        .unwrap();
        let out = cast_unscaled(&input, DType::U8).unwrap();
        assert_eq!(out.to_vec::<u8>().unwrap(), [255, 0, 0, 5, 255, 6]);
        assert!(cast_unscaled(&out, DType::U8).unwrap().same_storage(&out));
    }
}
