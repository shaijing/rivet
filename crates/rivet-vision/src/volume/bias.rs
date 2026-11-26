use super::{
    VolumeSample,
    resample::{AxisSample, axis_samples},
};
use crate::errors::{RivetResult, invalid_argument};
use crate::{RandomStream, transforms::degradation::noise::normal_pair};
use rivet_core::{CpuStorageRef, DType, Tensor};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiasFieldConfig {
    pub std_range: [f64; 2],
    pub scale_range: [f64; 2],
    pub per_channel: bool,
}

impl BiasFieldConfig {
    pub fn validate(self) -> RivetResult<()> {
        let [std0, std1] = self.std_range;
        let [scale0, scale1] = self.scale_range;
        if !(0.0..=1.0).contains(&std0)
            || !(std0..=1.0).contains(&std1)
            || scale0 <= 0.0
            || !(scale0..=1.0).contains(&scale1)
        {
            return Err(invalid_argument(
                "BiasField requires ordered std in [0, 1] and scale in (0, 1]",
            ));
        }
        Ok(())
    }

    // Singleton dimensions let the same kernel handle HW/HWC images and DHW/DHWC volumes.
    pub(crate) fn apply(
        self,
        input: &Tensor,
        shape: [usize; 3],
        channels: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<Tensor> {
        self.validate()?;
        if !matches!(input.dtype(), DType::U8 | DType::F32) || channels == 0 || shape.contains(&0) {
            return Err(invalid_argument(
                "BiasField requires nonempty uint8 or float32 pixels",
            ));
        }
        let [low, high] = self.std_range;
        let std = low + (high - low) * rng.next_f64();
        if std == 0.0 {
            return Ok(input.clone());
        }
        let [low, high] = self.scale_range;
        let scale = low + (high - low) * rng.next_f64();
        let coarse = shape.map(|n| ((n as f64 * scale) as usize).max(2).min(n));
        let fields = if self.per_channel { channels } else { 1 };
        let count = coarse.iter().product::<usize>() * fields;
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(count)
            .map_err(|_| invalid_argument("BiasField coarse grid exceeds allocation limits"))?;
        while coefficients.len() < count {
            let (a, b) = normal_pair(rng);
            coefficients.push((a * std) as f32);
            if coefficients.len() < count {
                coefficients.push((b * std) as f32);
            }
        }
        let [zs, ys, xs] = std::array::from_fn(|i| axis_samples(coarse[i], shape[i]));
        let input = input.contiguous()?;
        Ok(input.with_cpu_storage(|storage, layout| {
            let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
            let gain = |z: &AxisSample, y: &AxisSample, x: &AxisSample, f| {
                let read = |zz, yy, xx| {
                    let index: usize = ((zz * coarse[1] + yy) * coarse[2] + xx) * fields + f;
                    f64::from(coefficients[index])
                };
                let along_x = |zz, yy| lerp(read(zz, yy, x.low), read(zz, yy, x.high), x.fraction);
                let along_y = |zz| lerp(along_x(zz, y.low), along_x(zz, y.high), y.fraction);
                lerp(along_y(z.low), along_y(z.high), z.fraction).exp()
            };
            macro_rules! write {
                ($pixels:expr, $cast:expr) => {{
                    let mut pixels = $pixels[layout.start_offset()..].iter();
                    Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                        for z in &zs {
                            for y in &ys {
                                for x in &xs {
                                    let shared = if fields == 1 { gain(z, y, x, 0) } else { 0.0 };
                                    for c in 0..channels {
                                        let factor = if fields == 1 {
                                            shared
                                        } else {
                                            gain(z, y, x, c)
                                        };
                                        let value = f64::from(
                                            *pixels.next().expect("validated spatial shape"),
                                        );
                                        out.write_next($cast(value * factor))?;
                                    }
                                }
                            }
                        }
                        Ok(())
                    })
                }};
            }
            match storage {
                CpuStorageRef::U8(v) => write!(v, |v: f64| v.round() as u8),
                CpuStorageRef::F32(v) => write!(v, |v: f64| v.clamp(0.0, 1.0) as f32),
                _ => unreachable!("validated BiasField dtype"),
            }
        })?)
    }
}

impl VolumeSample {
    pub fn bias_field(
        mut self,
        config: BiasFieldConfig,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        let channels = self.volume.dims().get(3).copied().unwrap_or(1);
        self.volume = config.apply(&self.volume, self.spatial_shape()?, channels, rng)?;
        Ok(self)
    }
}
