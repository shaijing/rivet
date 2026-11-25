use super::{RivetResult, invalid_argument};
use crate::sample::image::ImageAxisOrder;
use rivet_core::{CpuStorageRef, Tensor, WithDType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MorphologicalOperation {
    Erosion,
    Dilation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MorphologicalConfig {
    /// Elliptical kernel dimensions in width, height order.
    pub scale: (u32, u32),
    pub operation: MorphologicalOperation,
}

impl MorphologicalConfig {
    pub(super) fn valid(&self) -> bool {
        (1..=127).contains(&self.scale.0) && (1..=127).contains(&self.scale.1)
    }

    pub(crate) fn apply(&self, input: &Tensor, axis: ImageAxisOrder) -> RivetResult<Tensor> {
        let ([height, width, channels], [ay, ax, ac]) = match (input.dims(), axis) {
            (&[h, w], _) => ([h, w, 1], [0, 1, 2]),
            (&[h, w, c], ImageAxisOrder::Hwc) => ([h, w, c], [0, 1, 2]),
            (&[c, h, w], ImageAxisOrder::Chw) => ([h, w, c], [1, 2, 0]),
            _ => {
                return Err(invalid_argument(
                    "Morphological expects HW, HWC or CHW input",
                ));
            }
        };
        if height == 0 || width == 0 || channels == 0 || self.scale == (1, 1) {
            return Ok(input.clone());
        }
        let (kw, kh) = (self.scale.0 as isize, self.scale.1 as isize);
        let (rx, ry) = (kw / 2, kh / 2);
        let mut taps = Vec::new();
        for y in 0..kh {
            let dy = y - ry;
            let dx = if ry == 0 {
                rx
            } else {
                (rx as f64 * (1.0 - (dy as f64 / ry as f64).powi(2)).sqrt()).round() as isize
            };
            taps.extend(((-dx).max(-rx)..=dx.min(kw - rx - 1)).map(|x| (dy, x)));
        }
        let dilate = self.operation == MorphologicalOperation::Dilation;
        Ok(input.with_cpu_storage(|storage, layout| {
            let (sy, sx) = (layout.stride()[ay], layout.stride()[ax]);
            let sc = layout.stride().get(ac).copied().unwrap_or(0);
            let offset = layout.start_offset();
            fn write<T: WithDType + PartialOrd, const DILATE: bool>(
                input: &Tensor,
                values: &[T],
                size: [usize; 3],
                strides: [usize; 3],
                offset: usize,
                axis: ImageAxisOrder,
                taps: &[(isize, isize)],
            ) -> rivet_core::Result<Tensor> {
                let [h, w, c] = size;
                let [sy, sx, sc] = strides;
                let pixel = |y: usize, x: usize, c: usize| {
                    let base = offset + c * sc;
                    let mut best = values[base + y * sy + x * sx];
                    for &(dy, dx) in taps {
                        if let (Some(y), Some(x)) =
                            (y.checked_add_signed(dy), x.checked_add_signed(dx))
                            && y < h
                            && x < w
                        {
                            let v = values[base + y * sy + x * sx];
                            if if DILATE { v > best } else { v < best } {
                                best = v;
                            }
                        }
                    }
                    best
                };
                Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                    if axis == ImageAxisOrder::Chw && input.rank() == 3 {
                        for c in 0..c {
                            for y in 0..h {
                                for x in 0..w {
                                    out.write_next(pixel(y, x, c))?;
                                }
                            }
                        }
                    } else {
                        for y in 0..h {
                            for x in 0..w {
                                for c in 0..c {
                                    out.write_next(pixel(y, x, c))?;
                                }
                            }
                        }
                    }
                    Ok(())
                })
            }
            macro_rules! run {
                ($values:expr) => {{
                    if dilate {
                        write::<_, true>(
                            input,
                            $values,
                            [height, width, channels],
                            [sy, sx, sc],
                            offset,
                            axis,
                            &taps,
                        )
                    } else {
                        write::<_, false>(
                            input,
                            $values,
                            [height, width, channels],
                            [sy, sx, sc],
                            offset,
                            axis,
                            &taps,
                        )
                    }
                }};
            }
            match storage {
                CpuStorageRef::U8(v) => run!(v),
                CpuStorageRef::F32(v) => run!(v),
                CpuStorageRef::I64(v) => run!(v),
                other => Err(rivet_core::Error::UnexpectedDType {
                    expected: rivet_core::DType::U8,
                    actual: other.dtype(),
                }),
            }
        })?)
    }
}
