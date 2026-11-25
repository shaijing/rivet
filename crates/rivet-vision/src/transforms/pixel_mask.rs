use super::{RivetResult, invalid_argument};
use crate::sample::image::ImageAxisOrder;
use rivet_core::{CpuStorageRef, DType, Tensor, WithDType};
use rivet_data::random::RandomStream;

fn apply(
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
