//! Decoded volumes use DHW or DHWC. Masks are DHW class/instance IDs,
//! and keypoints are stored as [N, 3] in X, Y, Z pixel-center coordinates.
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use rivet_core::{CpuStorageRef, DType, Tensor, WithDType};

#[derive(Clone, Debug)]
pub struct VolumeSample {
    pub volume: Tensor,
    pub mask: Option<Tensor>,
    pub keypoints: Option<Tensor>,
    pub label: i64,
}

impl VolumeSample {
    pub fn spatial_shape(&self) -> RivetResult<[usize; 3]> {
        match self.volume.dims() {
            &[d, h, w] if d > 0 && h > 0 && w > 0 => Ok([d, h, w]),
            &[d, h, w, c] if d > 0 && h > 0 && w > 0 && c > 0 => Ok([d, h, w]),
            _ => Err(invalid_shape("volume must be nonempty DHW or DHWC")),
        }
    }

    pub fn validate(&self) -> RivetResult<()> {
        let size = self.spatial_shape()?;
        if !matches!(self.volume.dtype(), DType::U8 | DType::F32) {
            return Err(invalid_argument("volume requires uint8 or float32 pixels"));
        }
        if self.mask.as_ref().is_some_and(|mask| {
            mask.dims() != size || !matches!(mask.dtype(), DType::U8 | DType::I64)
        }) {
            return Err(invalid_shape(
                "volume mask requires matching DHW uint8 or int64 data",
            ));
        }
        if let Some(points) = &self.keypoints {
            if !matches!(points.dims(), [_, 3])
                || points.dtype() != DType::F32
                || points.to_vec::<f32>()?.iter().any(|v| !v.is_finite())
            {
                return Err(invalid_argument(
                    "volume keypoints require finite float32 [N,3] coordinates",
                ));
            }
        }
        Ok(())
    }

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
struct AxisSample {
    nearest: usize,
    low: usize,
    high: usize,
    fraction: f64,
}

fn axis_samples(old: usize, new: usize) -> Vec<AxisSample> {
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

trait IntoVolumeFloat {
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
