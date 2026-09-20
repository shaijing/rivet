use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{ImageAxisOrder, ImageSample};
use rivet_core::{CpuStorageRef, DType, Tensor};

/// Semantic image dtype conversion.
///
/// U8 values are interpreted in `[0, 255]` and converted to F32 in `[0, 1]`;
/// F32 values are clamped to `[0, 1]`, scaled to `[0, 255]`, and rounded when
/// converted to U8. Other dtype pairs are intentionally rejected.
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

    fn convert_tensor(&self, input: &Tensor) -> RivetResult<Tensor> {
        if input.dtype() == self.dtype {
            return Ok(input.clone());
        }
        let dims = input.dims().to_vec();
        match (input.dtype(), self.dtype) {
            (DType::U8, DType::F32) => {
                let values = input.with_cpu_storage(|storage, layout| {
                    let CpuStorageRef::U8(values) = storage else {
                        return Err(rivet_core::Error::UnexpectedDType {
                            expected: DType::U8,
                            actual: input.dtype(),
                        });
                    };
                    logical_values(values, layout).map(|values| {
                        values
                            .into_iter()
                            .map(|value| value as f32 / 255.0)
                            .collect()
                    })
                })?;
                Ok(Tensor::from_vec(values, dims, input.device())?)
            }
            (DType::F32, DType::U8) => {
                let values = input.with_cpu_storage(|storage, layout| {
                    let CpuStorageRef::F32(values) = storage else {
                        return Err(rivet_core::Error::UnexpectedDType {
                            expected: DType::F32,
                            actual: input.dtype(),
                        });
                    };
                    logical_values(values, layout).map(|values| {
                        values
                            .into_iter()
                            .map(|value| {
                                let value = if value.is_nan() {
                                    0.0
                                } else {
                                    value.clamp(0.0, 1.0)
                                };
                                (value * 255.0).round() as u8
                            })
                            .collect()
                    })
                })?;
                Ok(Tensor::from_vec(values, dims, input.device())?)
            }
            (actual, target) => Err(invalid_argument(format!(
                "image dtype conversion does not support {:?} -> {:?}",
                actual, target
            ))),
        }
    }
}

fn logical_values<T: Copy>(
    values: &[T],
    layout: &rivet_core::Layout,
) -> rivet_core::Result<Vec<T>> {
    layout
        .strided_index()
        .map(|index| {
            values
                .get(index)
                .copied()
                .ok_or(rivet_core::Error::StorageOutOfBounds)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::ConvertImageDtypeConfig;
    use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
    use rivet_core::{DType, Device, Tensor};

    #[test]
    fn semantic_u8_f32_conversion_scales_and_clamps() {
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
        assert_eq!(output.image.to_vec::<u8>().unwrap(), [0, 128, 255, 0]);
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
        assert_eq!(actual.to_vec::<u8>().unwrap(), [0, 128, 255, 0]);
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
