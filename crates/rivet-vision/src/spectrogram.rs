//! Augment precomputed real spectrograms in FT or CFT layout.
use crate::RandomStream;
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use rivet_core::{DType, Tensor};

#[derive(Clone, Copy, Debug)]
pub enum SpectrogramAxis {
    Frequency,
    Time,
}

#[derive(Clone, Debug)]
pub struct SpectrogramSample {
    pub spectrogram: Tensor,
    pub label: i64,
}

impl SpectrogramSample {
    pub fn validate(&self) -> RivetResult<()> {
        if !matches!(self.spectrogram.dims(), [_, _] | [_, _, _])
            || self.spectrogram.dims().contains(&0)
        {
            return Err(invalid_shape(
                "spectrogram must have nonempty FT or CFT shape",
            ));
        }
        if !matches!(self.spectrogram.dtype(), DType::U8 | DType::F32) {
            return Err(invalid_argument(
                "spectrogram requires uint8 or float32 data",
            ));
        }
        Ok(())
    }

    /// Zero one band across all channels. Band width is uniform on
    /// 0..=min(max_length, axis_length), and its start is uniform among valid starts.
    pub fn mask(
        mut self,
        axis: SpectrogramAxis,
        max_length: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        if max_length == 0 {
            return Err(invalid_argument("spectrogram mask length must be positive"));
        }
        let axis = self.spectrogram.rank()
            - match axis {
                SpectrogramAxis::Frequency => 2,
                SpectrogramAxis::Time => 1,
            };
        let length = self.spectrogram.dims()[axis];
        let max_length = max_length.min(length);
        let width = rng.gen_range_usize(0..max_length + 1)?;
        if width == 0 {
            return Ok(self);
        }
        let start = rng.gen_range_usize(0..length - width + 1)?;
        let mut shape = self.spectrogram.dims().to_vec();
        shape[axis] = width;
        let zeros = Tensor::zeros(
            shape.as_slice(),
            self.spectrogram.dtype(),
            self.spectrogram.device(),
        )?;
        self.spectrogram = self.spectrogram.slice_scatter(&zeros, axis, start)?;
        Ok(self)
    }

    pub fn time_reverse(mut self) -> RivetResult<Self> {
        self.validate()?;
        self.spectrogram = self.spectrogram.flip(&[self.spectrogram.rank() - 1])?;
        Ok(self)
    }
}
