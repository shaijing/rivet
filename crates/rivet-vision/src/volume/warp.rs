use super::{IntoVolumeFloat, VolumeInterpolation, VolumeSample};
use crate::RandomStream;
use crate::errors::{RivetResult, invalid_argument};
use rivet_core::{CpuStorageRef, DType, Tensor, WithDType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolumeBorder {
    Constant,
    Replicate,
}

#[derive(Clone, Debug)]
pub struct Affine3DConfig {
    pub rotate_range: [[f64; 2]; 3],
    pub scale_range: [[f64; 2]; 3],
    pub translate_percent_range: [[f64; 2]; 3],
    pub interpolation: VolumeInterpolation,
    pub border: VolumeBorder,
    pub fill: Vec<f32>,
    pub fill_mask: i64,
}

impl VolumeSample {
    pub fn affine(mut self, config: &Affine3DConfig, rng: &mut RandomStream) -> RivetResult<Self> {
        self.validate()?;
        validate_fill(&self, &config.fill, config.fill_mask, config.border)?;
        for ranges in [
            config.rotate_range,
            config.scale_range,
            config.translate_percent_range,
        ] {
            if ranges
                .iter()
                .any(|[lo, hi]| !lo.is_finite() || !hi.is_finite() || lo > hi)
            {
                return Err(invalid_argument(
                    "Affine3D ranges must be finite and ordered",
                ));
            }
        }
        if config.scale_range.iter().any(|r| r[0] <= 0.0) {
            return Err(invalid_argument("Affine3D scale must be positive"));
        }
        let [d, h, w] = self.spatial_shape()?;
        let size = [w as f64, h as f64, d as f64];
        let center = size.map(|v| (v - 1.0) * 0.5);
        let mut sample = |ranges: [[f64; 2]; 3]| {
            ranges.map(|[lo, hi]| {
                if lo == hi {
                    return lo;
                }
                let t = rng.next_f64();
                lo * (1.0 - t) + hi * t
            })
        };
        let angles = sample(config.rotate_range).map(|v| v.rem_euclid(360.0).to_radians());
        let scales = sample(config.scale_range);
        let percent = sample(config.translate_percent_range);
        let shift: [f64; 3] = std::array::from_fn(|i| percent[i] * size[i]);
        if (0..3).any(|i| {
            !shift[i].is_finite()
                || !scales[i].recip().is_finite()
                || !(scales[i] * size[i]).is_finite()
        }) {
            return Err(invalid_argument(
                "Affine3D sampled transform exceeds numeric limits",
            ));
        }
        if angles == [0.0; 3] && scales == [1.0; 3] && shift == [0.0; 3] {
            return Ok(self);
        }
        let [(sx, cx), (sy, cy), (sz, cz)] = angles.map(f64::sin_cos);
        // Rz * Ry * Rx is orthogonal, so its inverse is its transpose.
        let rotation = [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ];
        let inverse = |p: [f64; 3]| {
            let q: [f64; 3] = std::array::from_fn(|i| p[i] - center[i] - shift[i]);
            std::array::from_fn(|i| {
                center[i] + (0..3).map(|j| rotation[j][i] * q[j]).sum::<f64>() / scales[i]
            })
        };
        self.volume = warp(
            &self.volume,
            config.interpolation,
            config.border,
            &config.fill,
            None,
            inverse,
        )?;
        if let Some(mask) = &self.mask {
            self.mask = Some(warp(
                mask,
                VolumeInterpolation::Nearest,
                config.border,
                &[],
                Some(config.fill_mask),
                inverse,
            )?);
        }
        if let Some(points) = &self.keypoints {
            let mut coords = points.to_vec::<f32>()?;
            for point in coords.chunks_exact_mut(3) {
                let q: [f64; 3] =
                    std::array::from_fn(|i| (f64::from(point[i]) - center[i]) * scales[i]);
                for i in 0..3 {
                    point[i] =
                        (center[i] + shift[i] + (0..3).map(|j| rotation[i][j] * q[j]).sum::<f64>())
                            as f32;
                }
            }
            if coords.iter().any(|v| !v.is_finite()) {
                return Err(invalid_argument("Affine3D keypoints exceed float32 limits"));
            }
            self.keypoints = Some(Tensor::from_vec(coords, points.dims(), points.device())?);
        }
        Ok(self)
    }
}

fn validate_fill(
    sample: &VolumeSample,
    fill: &[f32],
    mask_fill: i64,
    border: VolumeBorder,
) -> RivetResult<()> {
    let channels = sample.volume.dims().get(3).copied().unwrap_or(1);
    if (fill.len() != 1 && fill.len() != channels) || fill.iter().any(|v| !v.is_finite()) {
        return Err(invalid_argument(
            "3D warp fill requires a finite scalar or one value per channel",
        ));
    }
    if border == VolumeBorder::Constant
        && sample.mask.as_ref().is_some_and(|m| m.dtype() == DType::U8)
        && u8::try_from(mask_fill).is_err()
    {
        return Err(invalid_argument(
            "3D warp uint8 mask fill must be in [0, 255]",
        ));
    }
    Ok(())
}

fn warp(
    input: &Tensor,
    interpolation: VolumeInterpolation,
    border: VolumeBorder,
    fill: &[f32],
    mask_fill: Option<i64>,
    inverse: impl Fn([f64; 3]) -> [f64; 3],
) -> RivetResult<Tensor> {
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| {
        let start = layout.start_offset();
        let channels = input.dims().get(3).copied().unwrap_or(1);
        let channel_fill = |c| fill[if fill.len() == 1 { 0 } else { c }];
        macro_rules! run {
            ($pixels:expr, $fill:expr, $cast:expr) => {{
                let fills: Vec<_> = (0..channels).map($fill).collect();
                write_warp(
                    &input,
                    &$pixels[start..],
                    &fills,
                    interpolation,
                    border,
                    &inverse,
                    $cast,
                )
            }};
        }
        match storage {
            CpuStorageRef::U8(v) => run!(
                v,
                |c| mask_fill.map_or_else(|| channel_fill(c).round() as u8, |m| m as u8),
                |x: f64| x.round() as u8
            ),
            CpuStorageRef::F32(v) => run!(v, channel_fill, |x: f64| x as f32),
            CpuStorageRef::I64(v) => {
                run!(v, |_| mask_fill.expect("validated mask fill"), |x: f64| x
                    as i64)
            }
            _ => unreachable!("validated volume warp dtype"),
        }
    })?)
}

fn write_warp<T: WithDType + IntoVolumeFloat>(
    input: &Tensor,
    values: &[T],
    fill: &[T],
    mode: VolumeInterpolation,
    border: VolumeBorder,
    inverse: impl Fn([f64; 3]) -> [f64; 3],
    cast: impl Fn(f64) -> T,
) -> rivet_core::Result<Tensor> {
    let (d, h, w) = (input.dims()[0], input.dims()[1], input.dims()[2]);
    let c = fill.len();
    let upper = [w - 1, h - 1, d - 1].map(|v| v as f64);
    let offset = |p: [f64; 3]| {
        if p.iter().any(|v| v.is_nan()) {
            return None;
        }
        let p = if border == VolumeBorder::Replicate {
            std::array::from_fn(|i| p[i].clamp(0.0, upper[i]))
        } else if (0..3).any(|i| !(0.0..=upper[i]).contains(&p[i])) {
            return None;
        } else {
            p
        };
        Some(((p[2] as usize * h + p[1] as usize) * w + p[0] as usize) * c)
    };
    Tensor::from_exact_writer(input.dims(), input.device(), |out| {
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let p = inverse([x as f64, y as f64, z as f64]);
                    if mode == VolumeInterpolation::Nearest {
                        let p = p.map(|v| (v + 0.5).floor());
                        out.extend_from_slice(offset(p).map_or(fill, |i| &values[i..i + c]))?;
                    } else {
                        let base = p.map(f64::floor);
                        let f: [f64; 3] = std::array::from_fn(|i| p[i] - base[i]);
                        if p.iter().any(|v| !v.is_finite()) {
                            let samples = offset(p).map_or(fill, |i| &values[i..i + c]);
                            out.extend_from_slice(samples)?;
                            continue;
                        }
                        let taps: [(Option<usize>, f64); 8] = std::array::from_fn(|i| {
                            let pos =
                                std::array::from_fn(|axis| base[axis] + ((i >> axis) & 1) as f64);
                            let weight = (0..3)
                                .map(|axis| {
                                    if (i >> axis) & 1 == 0 {
                                        1.0 - f[axis]
                                    } else {
                                        f[axis]
                                    }
                                })
                                .product();
                            (offset(pos), weight)
                        });
                        for (ch, &background) in fill.iter().enumerate() {
                            let sum = taps
                                .iter()
                                .filter(|(_, w)| *w != 0.0)
                                .map(|&(i, w)| {
                                    i.map_or(background, |i| values[i + ch]).as_f64() * w
                                })
                                .sum();
                            out.write_next(cast(sum))?;
                        }
                    }
                }
            }
        }
        Ok(())
    })
}
