use super::contiguous_values;
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{ImageAxisOrder, ImageSample};
use rivet_core::{CpuStorageRef, DType, Tensor};

/// Semantic image dtype conversion.
///
/// U8 values are interpreted in `[0, 255]` and converted to F32 in `[0, 1]`;
/// F32 values are multiplied by `256 - 1e-3`, truncated and wrapped to U8,
/// matching torchvision convert_image_dtype. Other dtype pairs are rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvertImageDtypeConfig {
    pub dtype: DType,
}

impl ConvertImageDtypeConfig {
    pub const fn new(dtype: DType) -> Self {
        Self { dtype }
    }

    pub fn validate(&self) -> RivetResult<()> {
        if matches!(self.dtype, DType::U8 | DType::F32) {
            Ok(())
        } else {
            Err(invalid_argument(format!(
                "image dtype conversion supports uint8 and float32, got {:?}",
                self.dtype
            )))
        }
    }

    pub fn apply(
        &self,
        sample: ImageSample,
        _axis_order: ImageAxisOrder,
    ) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = sample.into_decoded()?;
        let output = self.convert_tensor(&sample.image)?;

        Ok(ImageSample::Decoded(crate::sample::image::DecodedSample {
            image: output,
            label: sample.label,
        }))
    }

    /// Apply dtype conversion to a stacked rank-4 image batch.
    ///
    /// Dtype conversion is intentionally a batch-stage operation: it is
    /// deterministic, has no per-sample policy, and can read the stacked
    /// storage once instead of allocating one converted tensor per sample.
    pub fn apply_batch(&self, input: Tensor, _axis_order: ImageAxisOrder) -> RivetResult<Tensor> {
        self.validate()?;
        if input.rank() != 4 {
            return Err(invalid_shape(format!(
                "batch dtype conversion requires a rank-4 image batch, got shape {:?}",
                input.dims()
            )));
        }
        self.convert_tensor(&input)
    }

    /// Apply a dtype conversion whose target dtype and rank were checked by
    /// pipeline compilation.
    pub(crate) fn apply_batch_trusted(
        &self,
        input: Tensor,
        _axis_order: ImageAxisOrder,
    ) -> RivetResult<Tensor> {
        debug_assert!(matches!(self.dtype, DType::U8 | DType::F32));
        debug_assert_eq!(input.rank(), 4);
        self.convert_tensor(&input)
    }

    fn convert_tensor(&self, input: &Tensor) -> RivetResult<Tensor> {
        if input.dtype() == self.dtype {
            return Ok(input.clone());
        }
        let dims = input.dims().to_vec();
        match (input.dtype(), self.dtype) {
            (DType::U8, DType::F32) => {
                let output = input.with_cpu_storage(|storage, layout| {
                    let CpuStorageRef::U8(values) = storage else {
                        return Err(rivet_core::Error::UnexpectedDType {
                            expected: DType::U8,
                            actual: input.dtype(),
                        });
                    };
                    if layout.is_contiguous() {
                        return Tensor::from_exact_iter(
                            contiguous_values(values, layout)?
                                .iter()
                                .map(|&value| value as f32 / 255.0),
                            dims.clone(),
                            input.device(),
                        );
                    }
                    let output = layout.strided_index().map(|index| {
                        values
                            .get(index)
                            .copied()
                            .map(|value| value as f32 / 255.0)
                            .ok_or(rivet_core::Error::StorageOutOfBounds)
                    });
                    Tensor::from_exact_try_iter(output, dims.clone(), input.device())
                })?;
                Ok(output)
            }
            (DType::F32, DType::U8) => {
                let output = input.with_cpu_storage(|storage, layout| {
                    let CpuStorageRef::F32(values) = storage else {
                        return Err(rivet_core::Error::UnexpectedDType {
                            expected: DType::F32,
                            actual: input.dtype(),
                        });
                    };
                    let convert = |value: f32| {
                        let scaled = value * (256.0 - 1e-3);
                        if scaled.is_finite() {
                            scaled.trunc().rem_euclid(256.0) as u8
                        } else {
                            0
                        }
                    };
                    if layout.is_contiguous() {
                        return Tensor::from_exact_iter(
                            contiguous_values(values, layout)?
                                .iter()
                                .copied()
                                .map(convert),
                            dims.clone(),
                            input.device(),
                        );
                    }
                    let output = layout.strided_index().map(|index| {
                        let value = values
                            .get(index)
                            .copied()
                            .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                        Ok(convert(value))
                    });
                    Tensor::from_exact_try_iter(output, dims.clone(), input.device())
                })?;
                Ok(output)
            }
            (actual, target) => Err(invalid_argument(format!(
                "image dtype conversion does not support {:?} -> {:?}",
                actual, target
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ConvertImageDtypeConfig;
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{DType, Device, Tensor};

    #[test]
    fn semantic_u8_f32_conversion_matches_torchvision() {
        let input = Tensor::from_vec(vec![0u8, 128, 255], [1, 3, 1], &Device::Cpu).unwrap();
        let output = ConvertImageDtypeConfig::new(DType::F32)
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: input.clone(),
                    label: 2,
                }),
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.dtype(), DType::F32);
        assert_eq!(output.image.dims(), [1, 3, 1]);
        assert_eq!(
            output.image.to_vec::<f32>().unwrap(),
            [0.0, 128.0 / 255.0, 1.0]
        );
        assert!(!output.image.same_storage(&input));

        let input =
            Tensor::from_vec(vec![-1.0f32, 0.5, 2.0, f32::NAN], [1, 2, 2], &Device::Cpu).unwrap();
        let output = ConvertImageDtypeConfig::new(DType::U8)
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: input,
                    label: 0,
                }),
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert_eq!(output.image.to_vec::<u8>().unwrap(), [1, 127, 255, 0]);
    }

    #[test]
    fn contiguous_offset_and_strided_conversions_match_exactly() {
        let u8_base = Tensor::from_vec(
            (0..2 * 4 * 4 * 3).map(|v| v as u8).collect::<Vec<_>>(),
            [2, 4, 4, 3],
            &Device::Cpu,
        )
        .unwrap()
        .get(1)
        .unwrap();
        let floats = [
            -1.0f32,
            -0.5,
            0.0,
            0.0039,
            0.5,
            0.999,
            1.0,
            2.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1234.5,
        ];
        let f32_base = Tensor::from_vec(floats.repeat(8), [2, 4, 4, 3], &Device::Cpu)
            .unwrap()
            .get(1)
            .unwrap();
        for (input, target) in [(u8_base, DType::F32), (f32_base, DType::U8)] {
            let config = ConvertImageDtypeConfig::new(target);
            for view in [input.clone(), input.permute(&[1, 0, 2]).unwrap()] {
                let strided = view.permute(&[1, 0, 2]).unwrap();
                let packed = match strided.dtype() {
                    DType::U8 => Tensor::from_vec(
                        strided.to_vec::<u8>().unwrap(),
                        strided.dims(),
                        &Device::Cpu,
                    )
                    .unwrap(),
                    DType::F32 => Tensor::from_vec(
                        strided.to_vec::<f32>().unwrap(),
                        strided.dims(),
                        &Device::Cpu,
                    )
                    .unwrap(),
                    _ => unreachable!(),
                };
                let apply = |image| {
                    config
                        .apply(
                            ImageSample::Decoded(DecodedSample { image, label: 4 }),
                            ImageAxisOrder::Hwc,
                        )
                        .unwrap()
                        .into_decoded()
                        .unwrap()
                };
                let expected = apply(strided);
                let actual = apply(packed);
                match target {
                    DType::U8 => assert_eq!(
                        actual.image.to_vec::<u8>().unwrap(),
                        expected.image.to_vec::<u8>().unwrap()
                    ),
                    DType::F32 => assert_eq!(
                        actual.image.to_vec::<f32>().unwrap(),
                        expected.image.to_vec::<f32>().unwrap()
                    ),
                    _ => unreachable!(),
                }
            }
        }
    }

    #[test]
    fn same_dtype_preserves_the_view() {
        let input = Tensor::from_vec(vec![1u8, 2, 3, 4], [2, 2, 1], &Device::Cpu).unwrap();
        let output = ConvertImageDtypeConfig::new(DType::U8)
            .apply(
                ImageSample::Decoded(DecodedSample {
                    image: input.clone(),
                    label: 0,
                }),
                ImageAxisOrder::Hwc,
            )
            .unwrap()
            .into_decoded()
            .unwrap();
        assert!(output.image.same_storage(&input));
    }

    #[test]
    fn batch_conversion_matches_unfused_sample_conversion() {
        let input = Tensor::from_vec(
            (0..24).map(|value| value as u8).collect::<Vec<_>>(),
            [2, 2, 2, 3],
            &Device::Cpu,
        )
        .unwrap();
        let config = ConvertImageDtypeConfig::new(DType::F32);

        let expected = (0..2)
            .map(|index| {
                let sample = input.narrow(0, index, 1).unwrap().squeeze(0).unwrap();
                config
                    .apply(
                        ImageSample::Decoded(DecodedSample {
                            image: sample,
                            label: index as i64,
                        }),
                        ImageAxisOrder::Hwc,
                    )
                    .unwrap()
                    .into_decoded()
                    .unwrap()
                    .image
            })
            .collect::<Vec<_>>();
        let expected = Tensor::stack(&expected.iter().collect::<Vec<_>>(), 0).unwrap();
        let actual = config.apply_batch(input, ImageAxisOrder::Hwc).unwrap();

        assert_eq!(actual.dims(), [2, 2, 2, 3]);
        assert_eq!(actual.dtype(), DType::F32);
        assert_eq!(
            actual.to_vec::<f32>().unwrap(),
            expected.to_vec::<f32>().unwrap()
        );

        let input = Tensor::from_vec(
            vec![-1.0f32, 0.5, 2.0, f32::NAN],
            [2, 1, 2, 1],
            &Device::Cpu,
        )
        .unwrap();
        let config = ConvertImageDtypeConfig::new(DType::U8);
        let expected = (0..2)
            .map(|index| {
                let sample = input.narrow(0, index, 1).unwrap().squeeze(0).unwrap();
                config
                    .apply(
                        ImageSample::Decoded(DecodedSample {
                            image: sample,
                            label: index as i64,
                        }),
                        ImageAxisOrder::Hwc,
                    )
                    .unwrap()
                    .into_decoded()
                    .unwrap()
                    .image
            })
            .collect::<Vec<_>>();
        let expected = Tensor::stack(&expected.iter().collect::<Vec<_>>(), 0).unwrap();
        let actual = config.apply_batch(input, ImageAxisOrder::Hwc).unwrap();

        assert_eq!(actual.dtype(), DType::U8);
        assert_eq!(
            actual.to_vec::<u8>().unwrap(),
            expected.to_vec::<u8>().unwrap()
        );
        assert_eq!(actual.to_vec::<u8>().unwrap(), [1, 127, 255, 0]);
    }

    #[test]
    fn batch_conversion_preserves_same_dtype_storage() {
        let input = Tensor::from_vec(vec![1u8, 2, 3, 4], [1, 2, 2, 1], &Device::Cpu).unwrap();
        let output = ConvertImageDtypeConfig::new(DType::U8)
            .apply_batch(input.clone(), ImageAxisOrder::Hwc)
            .unwrap();

        assert!(output.same_storage(&input));
    }
}
