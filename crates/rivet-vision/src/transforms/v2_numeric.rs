use super::{RivetResult, contiguous_values, invalid_argument};
use rivet_core::{CpuStorageRef, DType, Tensor};

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
