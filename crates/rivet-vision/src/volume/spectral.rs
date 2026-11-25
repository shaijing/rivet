use super::{IntoVolumeFloat, VolumeSample};
use crate::errors::{RivetResult, invalid_argument};
use crate::{RandomStream, transforms::frequency::Fft3};
use rivet_core::{CpuStorageRef, DType, Tensor, WithDType};
use rustfft::num_complex::Complex64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GibbsRingingConfig {
    pub retained_fraction_range: [f64; 2],
}

impl GibbsRingingConfig {
    pub fn validate(self) -> RivetResult<()> {
        let [lo, hi] = self.retained_fraction_range;
        if !(0.0..=1.0).contains(&lo) || !(lo..=1.0).contains(&hi) {
            return Err(invalid_argument(
                "GibbsRinging requires an ordered retained fraction in [0, 1]",
            ));
        }
        Ok(())
    }

    pub(crate) fn apply(
        self,
        input: &Tensor,
        shape: [usize; 3],
        channels: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<Tensor> {
        self.validate()?;
        if !matches!(input.dtype(), DType::U8 | DType::F32) || shape.contains(&0) || channels == 0 {
            return Err(invalid_argument(
                "GibbsRinging requires nonempty uint8 or float32 pixels",
            ));
        }
        let [lo, hi] = self.retained_fraction_range;
        let fraction = lo + (hi - lo) * rng.next_f64();
        let cutoff = shape.map(|n| (fraction * n as f64 * 0.5).floor() as usize);
        if (0..3).all(|i| cutoff[i] == shape[i] / 2) {
            return Ok(input.clone());
        }
        spectral_filter(input, shape, channels, |_, values| {
            let [d, h, w] = shape;
            for (z, plane) in values.chunks_exact_mut(h * w).enumerate() {
                for (y, row) in plane.chunks_exact_mut(w).enumerate() {
                    for (x, value) in row.iter_mut().enumerate() {
                        if z.min(d - z) > cutoff[0]
                            || y.min(h - y) > cutoff[1]
                            || x.min(w - x) > cutoff[2]
                        {
                            *value = Complex64::default();
                        }
                    }
                }
            }
        })
    }
}

impl VolumeSample {
    pub fn gibbs_ringing(
        mut self,
        config: GibbsRingingConfig,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        let c = self.volume.dims().get(3).copied().unwrap_or(1);
        self.volume = config.apply(&self.volume, self.spatial_shape()?, c, rng)?;
        Ok(self)
    }
}

// Process one channel at a time with a shared FFT plan and complex work buffer.
fn spectral_filter(
    input: &Tensor,
    shape: [usize; 3],
    channels: usize,
    filter: impl FnMut(usize, &mut [Complex64]),
) -> RivetResult<Tensor> {
    let input = input.contiguous()?;
    let count = input.elem_count();
    input.with_cpu_storage(|storage, layout| {
        let start = layout.start_offset();
        Ok(match storage {
            CpuStorageRef::U8(v) => reconstruct(
                &input,
                &v[start..start + count],
                shape,
                channels,
                filter,
                |v| v.round() as u8,
            ),
            CpuStorageRef::F32(v) => reconstruct(
                &input,
                &v[start..start + count],
                shape,
                channels,
                filter,
                |v| v.clamp(0.0, 1.0) as f32,
            ),
            _ => unreachable!("validated spectral dtype"),
        })
    })?
}

fn reconstruct<T: WithDType + IntoVolumeFloat>(
    input: &Tensor,
    pixels: &[T],
    shape: [usize; 3],
    channels: usize,
    mut filter: impl FnMut(usize, &mut [Complex64]),
    cast: impl Fn(f64) -> T,
) -> RivetResult<Tensor> {
    let mut fft = Fft3::new(shape);
    let mut values = vec![Complex64::default(); pixels.len() / channels];
    let mut output = pixels.to_vec();
    for c in 0..channels {
        for (value, pixel) in values.iter_mut().zip(pixels[c..].iter().step_by(channels)) {
            let real = pixel.as_f64();
            if !real.is_finite() {
                return Err(invalid_argument("spectral transforms require finite input"));
            }
            *value = Complex64::new(real, 0.0);
        }
        fft.transform(&mut values, false);
        filter(c, &mut values);
        fft.transform(&mut values, true);
        for (pixel, value) in output[c..].iter_mut().step_by(channels).zip(&values) {
            *pixel = cast(value.re);
        }
    }
    Ok(Tensor::from_vec(output, input.dims(), input.device())?)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GhostingArtifactConfig {
    pub num_ghosts_range: [usize; 2],
    pub intensity_range: [f64; 2],
    pub axis: usize,
    pub restore_range: [f64; 2],
}

impl VolumeSample {
    pub fn ghosting_artifact(
        mut self,
        config: GhostingArtifactConfig,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        let [min, max] = config.num_ghosts_range;
        let valid_fraction =
            |[lo, hi]: [f64; 2]| (0.0..=1.0).contains(&lo) && (lo..=1.0).contains(&hi);
        if min < 2
            || min > max
            || max == usize::MAX
            || config.axis >= 3
            || !valid_fraction(config.intensity_range)
            || !valid_fraction(config.restore_range)
        {
            return Err(invalid_argument(
                "invalid GhostingArtifact axis, period, intensity or restore range",
            ));
        }
        let length = self.spatial_shape()?[config.axis];
        let period = if min == max {
            min
        } else {
            rng.gen_range_usize(min..max + 1)?
        };
        let mut sample = |[lo, hi]: [f64; 2]| lo + (hi - lo) * rng.next_f64();
        let intensity = sample(config.intensity_range);
        let protected = (sample(config.restore_range) * length as f64 * 0.5).floor() as usize;
        if intensity == 0.0 || protected / period == (length / 2) / period {
            return Ok(self);
        }
        let affected: Vec<_> = (0..length)
            .filter(|&bin| {
                let signed = bin.min(length - bin);
                signed > protected && signed % period == 0
            })
            .collect();
        let axes = [0, 1, 2, 3];
        let mut order = axes[..self.volume.rank()].to_vec();
        order.swap(config.axis, 2);
        let input = self.volume.permute(&order)?.contiguous()?;
        let channels = input.dims().get(3).copied().unwrap_or(1);
        let output = input.with_cpu_storage(|storage, layout| {
            let start = layout.start_offset();
            let end = start + input.elem_count();
            match storage {
                CpuStorageRef::U8(v) => ghost_lines(
                    &input,
                    &v[start..end],
                    length,
                    channels,
                    &affected,
                    intensity,
                    |v| v.round() as u8,
                ),
                CpuStorageRef::F32(v) => ghost_lines(
                    &input,
                    &v[start..end],
                    length,
                    channels,
                    &affected,
                    intensity,
                    |v| v.clamp(0.0, 1.0) as f32,
                ),
                _ => unreachable!("validated GhostingArtifact dtype"),
            }
        })?;
        self.volume = output.permute(&order)?;
        Ok(self)
    }
}

fn ghost_lines<T: WithDType + IntoVolumeFloat>(
    input: &Tensor,
    pixels: &[T],
    length: usize,
    channels: usize,
    affected: &[usize],
    intensity: f64,
    cast: impl Fn(f64) -> T,
) -> rivet_core::Result<Tensor> {
    let mut planner = rustfft::FftPlanner::new();
    let forward = planner.plan_fft_forward(length);
    let inverse = planner.plan_fft_inverse(length);
    let scratch_len = forward
        .get_inplace_scratch_len()
        .max(inverse.get_inplace_scratch_len());
    let mut scratch = vec![Complex64::default(); scratch_len];
    let mut spectrum = vec![Complex64::default(); length];
    let mut row = Vec::with_capacity(length * channels);
    Tensor::from_exact_writer(input.dims(), input.device(), |out| {
        for line in pixels.chunks_exact(length * channels) {
            row.clear();
            row.extend_from_slice(line);
            for c in 0..channels {
                for (bin, pixel) in spectrum.iter_mut().zip(line[c..].iter().step_by(channels)) {
                    *bin = Complex64::new(pixel.as_f64(), 0.0);
                }
                forward.process_with_scratch(&mut spectrum, &mut scratch);
                for &bin in affected {
                    spectrum[bin] *= 1.0 - intensity;
                }
                inverse.process_with_scratch(&mut spectrum, &mut scratch);
                for (pixel, bin) in row[c..].iter_mut().step_by(channels).zip(&spectrum) {
                    *pixel = cast(bin.re.abs() / length as f64);
                }
            }
            out.extend_from_slice(&row)?;
        }
        Ok(())
    })
}
