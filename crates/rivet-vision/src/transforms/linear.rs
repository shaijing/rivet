//! Float32 image whitening in torchvision's CHW flattening order.
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use rivet_core::{DType, Tensor};

#[derive(Clone, Debug)]
pub struct LinearTransformationConfig {
    pub transformation_matrix: Tensor,
    pub mean_vector: Tensor,
}
impl LinearTransformationConfig {
    pub fn new(transformation_matrix: Tensor, mean_vector: Tensor) -> Self {
        Self {
            transformation_matrix,
            mean_vector,
        }
    }
    pub fn validate(&self) -> RivetResult<()> {
        let matrix = &self.transformation_matrix;
        let mean = &self.mean_vector;
        if matrix.rank() != 2
            || mean.rank() != 1
            || matrix.dims()[0] == 0
            || matrix.dims()[0] != matrix.dims()[1]
            || mean.dims()[0] != matrix.dims()[0]
        {
            return Err(invalid_shape(
                "linear transformation requires a nonempty square matrix and a matching mean vector",
            ));
        }
        if matrix.dtype() != DType::F32 || mean.dtype() != DType::F32 {
            return Err(invalid_argument(
                "linear transformation requires float32 matrix and mean",
            ));
        }
        Ok(())
    }
    pub fn apply(&self, sample: ImageSample, axis: ImageAxisOrder) -> RivetResult<ImageSample> {
        self.validate()?;
        let sample = sample.into_decoded()?;
        let input = &sample.image;
        if input.rank() != 3 || input.dtype() != DType::F32 {
            return Err(invalid_argument(
                "linear transformation requires a rank-3 float32 image",
            ));
        }
        let chw = match axis {
            ImageAxisOrder::Hwc => input.permute(&[2, 0, 1])?.contiguous()?,
            ImageAxisOrder::Chw => input.contiguous()?,
        };
        let shape = chw.dims();
        let n = chw.elem_count();
        if n != self.mean_vector.dims()[0] {
            return Err(invalid_shape(
                "linear transformation matrix dimension must equal C*H*W",
            ));
        }
        let flat = chw.reshape([1, n])?;
        let centered = flat.sub(&self.mean_vector.broadcast_as([1, n])?)?;
        // torchvision multiplies row vectors by the supplied matrix (no transpose).
        let output = centered
            .matmul(&self.transformation_matrix)?
            .reshape(shape)?;
        let image = match axis {
            ImageAxisOrder::Hwc => output.permute(&[1, 2, 0])?.contiguous()?,
            ImageAxisOrder::Chw => output,
        };
        Ok(ImageSample::Decoded(DecodedSample {
            image,
            label: sample.label,
        }))
    }
}
