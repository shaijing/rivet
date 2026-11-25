use super::{RivetResult, invalid_argument};
use crate::sample::image::ImageAxisOrder;
use rivet_core::{CpuStorageRef, DType, Tensor, WithDType};
use rivet_data::random::RandomStream;

pub(super) fn apply(
    input: &Tensor,
    axis: ImageAxisOrder,
    height: usize,
    width: usize,
    fills: [f32; 2],
    mut replacement: impl FnMut(usize, usize) -> Option<bool>,
) -> RivetResult<Tensor> {
    if !matches!(input.dtype(), DType::U8 | DType::F32) {
        return Err(invalid_argument(
            "pixel masks require uint8 or float32 input",
        ));
    }
    // CHW needs the same decision for each channel plane. HWC samples once
    // per pixel while writing the final image directly.
    let mask = match axis {
        ImageAxisOrder::Hwc => Vec::new(),
        ImageAxisOrder::Chw => (0..height)
            .flat_map(|y| (0..width).map(move |x| (y, x)))
            .map(|(y, x)| replacement(y, x))
            .collect::<Vec<_>>(),
    };
    let mut replace = |y, x| match axis {
        ImageAxisOrder::Hwc => replacement(y, x),
        ImageAxisOrder::Chw => mask[y * width + x],
    };
    Ok(input.with_cpu_storage(|storage, layout| {
        let [s0, s1, s2] = *layout.stride() else {
            return Err(rivet_core::Error::StorageOutOfBounds);
        };
        let start = layout.start_offset();
        fn write<T: WithDType>(
            input: &Tensor,
            values: &[T],
            axis: ImageAxisOrder,
            strides: [usize; 3],
            start: usize,
            mut replace: impl FnMut(usize, usize) -> Option<bool>,
            fills: [T; 2],
        ) -> rivet_core::Result<Tensor> {
            let [s0, s1, s2] = strides;
            let [d0, d1, d2] = *input.dims() else {
                return Err(rivet_core::Error::StorageOutOfBounds);
            };
            Tensor::from_exact_writer::<T, _, _>(input.dims(), input.device(), |out| {
                for i in 0..d0 {
                    let plane = start + i * s0;
                    for j in 0..d1 {
                        let row = plane + j * s1;
                        match axis {
                            ImageAxisOrder::Hwc => {
                                let fill = replace(i, j).map(|index| fills[usize::from(index)]);
                                for k in 0..d2 {
                                    out.write_next(fill.unwrap_or_else(|| values[row + k * s2]))?;
                                }
                            }
                            ImageAxisOrder::Chw => {
                                for k in 0..d2 {
                                    let value = replace(j, k)
                                        .map(|index| fills[usize::from(index)])
                                        .unwrap_or_else(|| values[row + k * s2]);
                                    out.write_next(value)?;
                                }
                            }
                        }
                    }
                }
                Ok(())
            })
        }
        match storage {
            CpuStorageRef::U8(values) => write(
                input,
                values,
                axis,
                [s0, s1, s2],
                start,
                &mut replace,
                fills.map(|fill| fill.trunc().rem_euclid(256.0) as u8),
            ),
            CpuStorageRef::F32(values) => write(
                input,
                values,
                axis,
                [s0, s1, s2],
                start,
                &mut replace,
                fills,
            ),
            _ => unreachable!(),
        }
    })?)
}

pub(super) fn salt_and_pepper(
    input: &Tensor,
    axis: ImageAxisOrder,
    size: (usize, usize),
    amount: f64,
    salt_ratio: f64,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    let white = if input.dtype() == DType::U8 {
        255.0
    } else {
        1.0
    };
    let salt = amount * salt_ratio;
    apply(input, axis, size.0, size.1, [0.0, white], |_, _| {
        let draw = rng.next_f64();
        (draw < amount).then_some(draw < salt)
    })
}

pub(super) fn grid_dropout(
    input: &Tensor,
    axis: ImageAxisOrder,
    size: (usize, usize),
    unit_size: usize,
    ratio: f64,
    fill: f32,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    let hole = ((unit_size as f64 * ratio).floor() as usize).clamp(1, unit_size - 1);
    let offsets = 0..unit_size - hole + 1;
    let y_offset = rng.gen_range_usize(offsets.clone())?;
    let x_offset = rng.gen_range_usize(offsets)?;
    let inside = |length, offset| {
        (0..length)
            .map(|pos| (offset..offset + hole).contains(&(pos % unit_size)))
            .collect::<Vec<_>>()
    };
    let rows = inside(size.0, y_offset);
    let cols = inside(size.1, x_offset);
    apply(input, axis, size.0, size.1, [fill; 2], |y, x| {
        (rows[y] && cols[x]).then_some(false)
    })
}

pub(super) fn coarse_dropout(
    input: &Tensor,
    axis: ImageAxisOrder,
    size: (usize, usize),
    holes: (u32, u32),
    hole_height: (u32, u32),
    hole_width: (u32, u32),
    fill: f32,
    rng: &mut RandomStream,
) -> RivetResult<Tensor> {
    let (height, width) = size;
    if height == 0 || width == 0 {
        return Err(invalid_argument("CoarseDropout requires a nonempty image"));
    }
    let sample = |rng: &mut RandomStream, bounds: (u32, u32), limit: usize| {
        let low = (bounds.0 as usize).min(limit);
        let high = (bounds.1 as usize).min(limit);
        rng.gen_range_usize(low..high + 1)
    };
    let count = sample(rng, holes, usize::MAX - 1)?;
    if count == 0 {
        return Ok(input.clone());
    }
    let mut rectangles = Vec::new();
    for _ in 0..count {
        let h = sample(rng, hole_height, height)?;
        let w = sample(rng, hole_width, width)?;
        let y = rng.gen_range_usize(0..height - h + 1)?;
        let x = rng.gen_range_usize(0..width - w + 1)?;
        rectangles.push((y..y + h, x..x + w));
    }
    apply(input, axis, height, width, [fill; 2], |y, x| {
        rectangles
            .iter()
            .any(|(rows, cols)| rows.contains(&y) && cols.contains(&x))
            .then_some(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rivet_core::Device;

    #[test]
    fn pixel_masks_preserve_layout_and_storage_offsets() {
        let hwc = Tensor::from_vec((0..60u8).collect(), [4, 5, 3], &Device::Cpu)
            .unwrap()
            .narrow(0, 1, 3)
            .unwrap()
            .narrow(1, 1, 4)
            .unwrap();
        let chw = hwc.permute(&[2, 0, 1]).unwrap();
        let mut rng = RandomStream::from_seed(7);
        let hwc_output = salt_and_pepper(
            &hwc,
            ImageAxisOrder::Hwc,
            (3, 4),
            0.5,
            0.4,
            &mut rng.clone(),
        )
        .unwrap();
        let chw_output =
            salt_and_pepper(&chw, ImageAxisOrder::Chw, (3, 4), 0.5, 0.4, &mut rng).unwrap();
        let hwc_values = hwc_output.to_vec::<u8>().unwrap();
        assert_eq!(
            hwc_values,
            chw_output
                .permute(&[1, 2, 0])
                .unwrap()
                .to_vec::<u8>()
                .unwrap(),
        );
        let original = hwc.to_vec::<u8>().unwrap();
        assert!(
            hwc_values
                .chunks_exact(3)
                .zip(original.chunks_exact(3))
                .all(|(out, original)| out == original || out == [0; 3] || out == [255; 3])
        );
        assert_ne!(hwc_values, original);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PixelDropoutConfig {
    pub dropout_prob: f64,
    pub per_channel: bool,
    pub drop_value: Option<Vec<f32>>,
    pub mask_drop_value: Option<i64>,
}

impl PixelDropoutConfig {
    pub(super) fn valid(&self) -> bool {
        (0.0..=1.0).contains(&self.dropout_prob)
            && (!self.per_channel || self.mask_drop_value.is_none())
            && self
                .drop_value
                .as_ref()
                .is_none_or(|fill| !fill.is_empty() && fill.iter().all(|v| v.is_finite()))
    }

    pub(crate) fn apply(
        &self,
        input: &Tensor,
        axis: ImageAxisOrder,
        rng: &mut RandomStream,
        record: Option<&mut Vec<bool>>,
    ) -> RivetResult<Tensor> {
        if input.rank() != 3 || !matches!(input.dtype(), DType::U8 | DType::F32) {
            return Err(invalid_argument(
                "PixelDropout requires rank-3 uint8 or float32 input",
            ));
        }
        let hwc = match axis {
            ImageAxisOrder::Hwc => input.clone(),
            ImageAxisOrder::Chw => input.permute(&[1, 2, 0])?,
        };
        let [height, width, channels] = *hwc.dims() else {
            unreachable!()
        };
        if height == 0
            || width == 0
            || channels == 0
            || self
                .drop_value
                .as_ref()
                .is_some_and(|fill| fill.len() != 1 && fill.len() != channels)
        {
            return Err(invalid_argument(
                "PixelDropout requires nonempty input and a scalar or one fill per channel",
            ));
        }
        if self.dropout_prob == 0.0 {
            if let Some(mask) = record {
                mask.resize(height * width, false);
            }
            return Ok(input.clone());
        }
        let fill = self.drop_value.clone().unwrap_or_else(|| {
            (0..channels)
                .map(|_| {
                    if input.dtype() == DType::U8 {
                        (rng.next_u32() & 255) as f32
                    } else {
                        rng.next_f32()
                    }
                })
                .collect()
        });
        let mut record = record;
        let result = hwc.with_cpu_storage(|storage, layout| {
            let [sy, sx, sc] = *layout.stride() else {
                unreachable!()
            };
            let start = layout.start_offset();
            macro_rules! write_dropout {
                ($values:expr, $convert:expr) => {{
                    let fill: Vec<_> = fill.iter().copied().map($convert).collect();
                    Tensor::from_exact_writer(hwc.dims(), hwc.device(), |out| {
                        for y in 0..height {
                            for x in 0..width {
                                let base = start + y * sy + x * sx;
                                let shared =
                                    !self.per_channel && rng.next_f64() < self.dropout_prob;
                                if let Some(mask) = record.as_deref_mut() {
                                    mask.push(shared);
                                }
                                for (c, &value) in fill.iter().cycle().take(channels).enumerate() {
                                    let drop = if self.per_channel {
                                        rng.next_f64() < self.dropout_prob
                                    } else {
                                        shared
                                    };
                                    out.write_next(if drop {
                                        value
                                    } else {
                                        $values[base + c * sc]
                                    })?;
                                }
                            }
                        }
                        Ok(())
                    })
                }};
            }
            match storage {
                CpuStorageRef::U8(values) => write_dropout!(values, |v: f32| v.round() as u8),
                CpuStorageRef::F32(values) => write_dropout!(values, |v| v),
                _ => unreachable!(),
            }
        })?;
        Ok(match axis {
            ImageAxisOrder::Hwc => result,
            ImageAxisOrder::Chw => result.permute(&[2, 0, 1])?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChannelDropoutFill {
    Constant(f32),
    Mean,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChannelDropoutConfig {
    pub channels_to_drop: Option<Vec<usize>>,
    pub dropout_ratio: f64,
    pub fill: ChannelDropoutFill,
}

impl ChannelDropoutConfig {
    pub(super) fn valid(&self) -> bool {
        (0.0..=1.0).contains(&self.dropout_ratio)
            && self.channels_to_drop.as_ref().is_none_or(|channels| {
                !channels.is_empty()
                    && channels
                        .iter()
                        .copied()
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        == channels.len()
            })
            && match self.fill {
                ChannelDropoutFill::Constant(v) => (0.0..=255.0).contains(&v),
                ChannelDropoutFill::Mean => true,
            }
    }

    pub(super) fn apply(
        &self,
        input: &Tensor,
        axis: ImageAxisOrder,
        rng: &mut RandomStream,
    ) -> RivetResult<Tensor> {
        if input.rank() != 3 || !matches!(input.dtype(), DType::U8 | DType::F32) {
            return Err(invalid_argument(
                "ChannelDropout requires rank-3 uint8 or float32 input",
            ));
        }
        if input.dtype() == DType::F32
            && matches!(self.fill, ChannelDropoutFill::Constant(v) if v > 1.0)
        {
            return Err(invalid_argument(
                "ChannelDropout float32 fill must be in [0, 1]",
            ));
        }
        let hwc = match axis {
            ImageAxisOrder::Hwc => input.clone(),
            ImageAxisOrder::Chw => input.permute(&[1, 2, 0])?,
        };
        let [height, width, channels] = *hwc.dims() else {
            unreachable!()
        };
        let mut eligible = self
            .channels_to_drop
            .clone()
            .unwrap_or_else(|| (0..channels).collect());
        if height == 0 || width == 0 || channels == 0 || eligible.iter().any(|&i| i >= channels) {
            return Err(invalid_argument(
                "ChannelDropout requires nonempty input and valid channel indices",
            ));
        }
        // A value immediately below a half caused by binary multiplication
        // still represents a half-up tie for decimal ratios such as 0.35.
        let scaled = eligible.len() as f64 * self.dropout_ratio;
        let count = (scaled.next_up().round() as usize).min(channels - 1);
        if count == 0 {
            return Ok(input.clone());
        }
        for i in 0..count {
            let j = rng.gen_range_usize(i..eligible.len())?;
            eligible.swap(i, j);
        }
        let selected = &eligible[..count];
        let result = hwc.with_cpu_storage(|storage, layout| {
            let [sy, sx, sc] = *layout.stride() else {
                unreachable!()
            };
            let start = layout.start_offset();
            macro_rules! write_channels {
                ($values:expr, $cast:expr) => {{
                    let mut fill = vec![None; channels];
                    let cast = $cast;
                    for &c in selected {
                        let value = match self.fill {
                            ChannelDropoutFill::Constant(v) => f64::from(v),
                            ChannelDropoutFill::Mean => {
                                let mut sum = 0.0;
                                for y in 0..height {
                                    for x in 0..width {
                                        sum += f64::from($values[start + y * sy + x * sx + c * sc]);
                                    }
                                }
                                sum / (height * width) as f64
                            }
                        };
                        fill[c] = Some(cast(value));
                    }
                    Tensor::from_exact_writer(hwc.dims(), hwc.device(), |out| {
                        for y in 0..height {
                            for x in 0..width {
                                let base = start + y * sy + x * sx;
                                for (c, &value) in fill.iter().enumerate() {
                                    out.write_next(
                                        value.unwrap_or_else(|| $values[base + c * sc]),
                                    )?;
                                }
                            }
                        }
                        Ok(())
                    })
                }};
            }
            match storage {
                CpuStorageRef::U8(values) => write_channels!(values, |v: f64| v.round() as u8),
                CpuStorageRef::F32(values) => write_channels!(values, |v| v as f32),
                _ => unreachable!(),
            }
        })?;
        Ok(match axis {
            ImageAxisOrder::Hwc => result,
            ImageAxisOrder::Chw => result.permute(&[2, 0, 1])?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StripLength {
    Pixels(u32, u32),
    Fraction(f64, f64),
}

impl StripLength {
    fn valid(self) -> bool {
        match self {
            Self::Pixels(lo, hi) => lo <= hi,
            Self::Fraction(lo, hi) => lo >= 0.0 && lo <= hi && hi <= 1.0,
        }
    }
    fn bounds(self, length: usize) -> (usize, usize) {
        match self {
            Self::Pixels(lo, hi) => ((lo as usize).min(length), (hi as usize).min(length)),
            Self::Fraction(lo, hi) => (
                (lo * length as f64).round() as usize,
                (hi * length as f64).round() as usize,
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct XyMaskingConfig {
    pub num_masks_x_range: (u32, u32),
    pub num_masks_y_range: (u32, u32),
    pub mask_x_length: StripLength,
    pub mask_y_length: StripLength,
    pub fill: f32,
    pub fill_mask: Option<i64>,
}

pub(crate) struct StripMask {
    cols: Vec<bool>,
    rows: Vec<bool>,
}

impl StripMask {
    pub(crate) fn covers(&self, y: usize, x: usize) -> bool {
        self.rows[y] || self.cols[x]
    }
}

impl XyMaskingConfig {
    pub(super) fn valid(&self) -> bool {
        [self.num_masks_x_range, self.num_masks_y_range]
            .into_iter()
            .all(|(lo, hi)| lo <= hi && hi <= 1024)
            && self.mask_x_length.valid()
            && self.mask_y_length.valid()
            && self.fill.is_finite()
    }

    pub(crate) fn sample(
        &self,
        h: usize,
        w: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<StripMask> {
        let mut axis =
            |length: usize, (lo, hi): (u32, u32), size: StripLength| -> RivetResult<Vec<bool>> {
                let mut mask = vec![false; length];
                let count = rng.gen_range_usize(lo as usize..hi as usize + 1)?;
                let (lo, hi) = size.bounds(length);
                for _ in 0..count {
                    let size = rng.gen_range_usize(lo..hi + 1)?;
                    if size != 0 {
                        let start = rng.gen_range_usize(0..length - size + 1)?;
                        mask[start..start + size].fill(true);
                    }
                }
                Ok(mask)
            };
        Ok(StripMask {
            cols: axis(w, self.num_masks_x_range, self.mask_x_length)?,
            rows: axis(h, self.num_masks_y_range, self.mask_y_length)?,
        })
    }

    pub(crate) fn apply(
        &self,
        input: &Tensor,
        axis: ImageAxisOrder,
        mask: &StripMask,
    ) -> RivetResult<Tensor> {
        if !matches!(input.dtype(), DType::U8 | DType::F32) {
            return Err(invalid_argument(
                "XYMasking requires uint8 or float32 input",
            ));
        }
        if !mask.rows.contains(&true) && !mask.cols.contains(&true) {
            return Ok(input.clone());
        }
        apply(
            input,
            axis,
            mask.rows.len(),
            mask.cols.len(),
            [self.fill; 2],
            |y, x| mask.covers(y, x).then_some(false),
        )
    }
}
