use super::{VolumeSample, volume_axis_mask};
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use rivet_core::{CpuStorageRef, Tensor, WithDType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolumeInterpolation {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resize3DConfig {
    pub size: [usize; 3],
    pub interpolation: VolumeInterpolation,
}

#[derive(Clone, Copy)]
pub(super) struct AxisSample {
    nearest: usize,
    pub(super) low: usize,
    pub(super) high: usize,
    pub(super) fraction: f64,
}

pub(super) fn axis_samples(old: usize, new: usize) -> Vec<AxisSample> {
    let scale = old as f64 / new as f64;
    (0..new)
        .map(|i| {
            let p = (i as f64 + 0.5) * scale;
            let nearest = (p as usize).min(old - 1);
            let coordinate = (p - 0.5).clamp(0.0, (old - 1) as f64);
            let low = coordinate as usize;
            AxisSample {
                nearest,
                low,
                high: (low + 1).min(old - 1),
                fraction: coordinate - low as f64,
            }
        })
        .collect()
}

fn resample(
    input: &Tensor,
    old: [usize; 3],
    new: [usize; 3],
    mode: VolumeInterpolation,
) -> RivetResult<Tensor> {
    let [zd, yd, xd] = std::array::from_fn(|i| axis_samples(old[i], new[i]));
    let axes = [zd.as_slice(), yd.as_slice(), xd.as_slice()];
    let channels = input.dims().get(3).copied().unwrap_or(1);
    let mut shape = new.to_vec();
    if input.rank() == 4 {
        shape.push(channels);
    }
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| {
        let start = layout.start_offset();
        let read = |z: usize, y: usize, x: usize, c: usize| {
            start + ((z * old[1] + y) * old[2] + x) * channels + c
        };
        macro_rules! run {
            ($values:expr,$finish:expr) => {{
                let finish = $finish;
                write(
                    &input,
                    &shape,
                    axes,
                    channels,
                    mode,
                    |z, y, x, c| $values[read(z, y, x, c)],
                    finish,
                )
            }};
        }
        match storage {
            CpuStorageRef::U8(v) => run!(v, |p: f64| p.round() as u8),
            CpuStorageRef::F32(v) => run!(v, |p: f64| p as f32),
            CpuStorageRef::I64(v) => run!(v, |p: f64| p as i64),
            _ => unreachable!("validated volume dtype"),
        }
    })?)
}

fn write<T: WithDType + IntoVolumeFloat>(
    input: &Tensor,
    shape: &[usize],
    [zs, ys, xs]: [&[AxisSample]; 3],
    channels: usize,
    mode: VolumeInterpolation,
    read: impl Fn(usize, usize, usize, usize) -> T,
    finish: impl Fn(f64) -> T,
) -> rivet_core::Result<Tensor> {
    Tensor::from_exact_writer(shape, input.device(), |out| {
        if mode == VolumeInterpolation::Nearest {
            for z in zs {
                for y in ys {
                    for x in xs {
                        for c in 0..channels {
                            out.write_next(read(z.nearest, y.nearest, x.nearest, c))?;
                        }
                    }
                }
            }
        } else {
            for z in zs {
                for y in ys {
                    for x in xs {
                        for c in 0..channels {
                            let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
                            let along_x = |z, y| {
                                lerp(
                                    read(z, y, x.low, c).as_f64(),
                                    read(z, y, x.high, c).as_f64(),
                                    x.fraction,
                                )
                            };
                            let along_y =
                                |z| lerp(along_x(z, y.low), along_x(z, y.high), y.fraction);
                            out.write_next(finish(lerp(
                                along_y(z.low),
                                along_y(z.high),
                                z.fraction,
                            )))?;
                        }
                    }
                }
            }
        }
        Ok(())
    })
}

impl VolumeSample {
    pub fn resize(mut self, config: Resize3DConfig) -> RivetResult<Self> {
        self.validate()?;
        let old = self.spatial_shape()?;
        let new = config.size;
        if new.contains(&0) {
            return Err(invalid_shape("Resize3D axes must be positive"));
        }
        if new == old {
            return Ok(self);
        }
        self.volume = resample(&self.volume, old, new, config.interpolation)?;
        if let Some(mask) = &self.mask {
            self.mask = Some(resample(mask, old, new, VolumeInterpolation::Nearest)?);
        }
        if let Some(points) = &self.keypoints {
            let scale = [
                new[2] as f64 / old[2] as f64,
                new[1] as f64 / old[1] as f64,
                new[0] as f64 / old[0] as f64,
            ];
            let coords = points.to_vec::<f32>()?;
            self.keypoints = Some(Tensor::from_exact_iter(
                coords
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| ((f64::from(v) + 0.5) * scale[i % 3] - 0.5) as f32),
                points.dims(),
                points.device(),
            )?);
        }
        Ok(self)
    }
}

pub(super) trait IntoVolumeFloat {
    fn as_f64(self) -> f64;
}
impl IntoVolumeFloat for u8 {
    fn as_f64(self) -> f64 {
        f64::from(self)
    }
}
impl IntoVolumeFloat for f32 {
    fn as_f64(self) -> f64 {
        f64::from(self)
    }
}
impl IntoVolumeFloat for i64 {
    fn as_f64(self) -> f64 {
        self as f64
    }
}

#[derive(Clone, Debug)]
pub struct Anisotropy3DConfig {
    pub axes: Vec<usize>,
    pub num_axes_range: [usize; 2],
    pub downscale_factor_range: [f64; 2],
    pub antialias: bool,
}

impl VolumeSample {
    /// Simulate reduced acquisition resolution while preserving the label grid.
    pub fn anisotropy(
        mut self,
        config: &Anisotropy3DConfig,
        rng: &mut rivet_data::random::RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        volume_axis_mask(&config.axes)?;
        let [min, max] = config.num_axes_range;
        let [low, high] = config.downscale_factor_range;
        if min == 0
            || min > max
            || max > config.axes.len()
            || !low.is_finite()
            || !high.is_finite()
            || low <= 1.0
            || high < low
        {
            return Err(invalid_argument(
                "invalid Anisotropy3D axis count or downscale factor range",
            ));
        }
        let shape = self.spatial_shape()?;
        let count = rng.gen_range_usize(min..max + 1)?;
        let mut reduced = shape;
        let mut axes = config.axes.clone();
        for index in 0..count {
            let selected = rng.gen_range_usize(index..axes.len())?;
            axes.swap(index, selected);
            let axis = axes[index];
            let factor = low + (high - low) * rng.next_f64();
            reduced[axis] = ((shape[axis] as f64 / factor).round() as usize).max(1);
        }
        if reduced != shape {
            let down = if config.antialias {
                area_downsample(&self.volume, shape, reduced)?
            } else {
                resample(&self.volume, shape, reduced, VolumeInterpolation::Linear)?
            };
            self.volume = resample(&down, reduced, shape, VolumeInterpolation::Linear)?;
        }
        Ok(self)
    }
}

fn area_downsample(input: &Tensor, old: [usize; 3], new: [usize; 3]) -> RivetResult<Tensor> {
    let weights: [Vec<Vec<(usize, f64)>>; 3] = std::array::from_fn(|axis| {
        let ratio = old[axis] as f64 / new[axis] as f64;
        (0..new[axis])
            .map(|i| {
                let left = i as f64 * ratio;
                let right = (i + 1) as f64 * ratio;
                (left.floor() as usize..(right.ceil() as usize).min(old[axis]))
                    .map(|j| (j, (right.min((j + 1) as f64) - left.max(j as f64)) / ratio))
                    .collect()
            })
            .collect()
    });
    let mut dims = input.dims().to_vec();
    dims[..3].copy_from_slice(&new);
    let channels = dims.get(3).copied().unwrap_or(1);
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| {
        macro_rules! sample {
            ($data:expr, $cast:expr) => {
                Tensor::from_exact_writer(dims.as_slice(), input.device(), |out| {
                    for zs in &weights[0] {
                        for ys in &weights[1] {
                            for xs in &weights[2] {
                                for c in 0..channels {
                                    let mut sum = 0.0;
                                    for &(z, wz) in zs {
                                        for &(y, wy) in ys {
                                            let row = layout.start_offset()
                                                + (z * old[1] + y) * old[2] * channels;
                                            let wyz = wy * wz;
                                            for &(x, wx) in xs {
                                                sum += f64::from($data[row + x * channels + c])
                                                    * wx
                                                    * wyz;
                                            }
                                        }
                                    }
                                    out.write_next($cast(sum))?;
                                }
                            }
                        }
                    }
                    Ok(())
                })
            };
        }
        match storage {
            CpuStorageRef::U8(v) => sample!(v, |v: f64| v.round() as u8),
            CpuStorageRef::F32(v) => sample!(v, |v: f64| v as f32),
            _ => unreachable!("validated volume dtype"),
        }
    })?)
}
