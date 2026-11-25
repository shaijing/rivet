use super::{Affine3DConfig, IntoVolumeFloat, VolumeBorder, VolumeInterpolation, VolumeSample};
use crate::errors::{RivetResult, invalid_argument};
use crate::{RandomStream, transforms::frequency::Fft3};
use rivet_core::{CpuStorageRef, DType, Tensor};
use rustfft::num_complex::Complex64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionArtifactConfig {
    pub num_events_range: [usize; 2],
    pub rotate_range: [f64; 2],
    pub translate_percent_range: [f64; 2],
    pub axis: usize,
    pub interpolation: VolumeInterpolation,
}

impl VolumeSample {
    pub fn motion_artifact(
        mut self,
        config: MotionArtifactConfig,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        let shape = self.spatial_shape()?;
        let [min, max] = config.num_events_range;
        let [r0, r1] = config.rotate_range;
        let [t0, t1] = config.translate_percent_range;
        if config.axis >= 3
            || shape.iter().any(|&n| n < 2)
            || min > max
            || max >= shape[config.axis]
            || !(-180.0..=180.0).contains(&r0)
            || !(r0..=180.0).contains(&r1)
            || !(-1.0..=1.0).contains(&t0)
            || !(t0..=1.0).contains(&t1)
        {
            return Err(invalid_argument(
                "invalid MotionArtifact ranges, acquisition axis or volume dimensions",
            ));
        }
        let events = if min == max {
            min
        } else {
            rng.gen_range_usize(min..max + 1)?
        };
        if events == 0
            || (config.rotate_range == [0.0; 2] && config.translate_percent_range == [0.0; 2])
        {
            return Ok(self);
        }
        let axis_len = shape[config.axis];
        let mut boundaries: Vec<_> = (1..axis_len).collect();
        for i in 0..events {
            let j = rng.gen_range_usize(i..boundaries.len())?;
            boundaries.swap(i, j);
        }
        boundaries.truncate(events);
        boundaries.sort_unstable();
        boundaries.push(axis_len);

        let mut fft = Fft3::new(shape);
        let channels = self.volume.dims().get(3).copied().unwrap_or(1);
        let input = self.volume.contiguous()?;
        let count = input.elem_count();
        let voxels = count / channels;
        let mut combined = vec![Complex64::default(); count];
        let mut work = vec![Complex64::default(); voxels];
        // Resample in floating point before assembling k-space, rounding only the final image.
        let base = match input.dtype() {
            DType::U8 => input.to_dtype(DType::F32)?,
            _ => input.clone(),
        };
        let mut start = 0;
        for (segment, stop) in boundaries.into_iter().enumerate() {
            let moved = if segment == 0 {
                base.clone()
            } else {
                let pose = VolumeSample {
                    volume: base.clone(),
                    mask: None,
                    keypoints: None,
                    label: 0,
                };
                pose.affine(
                    &Affine3DConfig {
                        rotate_range: [config.rotate_range; 3],
                        scale_range: [[1.0; 2]; 3],
                        translate_percent_range: [config.translate_percent_range; 3],
                        interpolation: config.interpolation,
                        border: VolumeBorder::Constant,
                        fill: vec![0.0],
                        fill_mask: 0,
                    },
                    rng,
                )?
                .volume
            };
            moved.with_cpu_storage(|storage, layout| {
                let CpuStorageRef::F32(data) = storage else {
                    unreachable!("motion uses floating-point intermediate poses");
                };
                let pixels = &data[layout.start_offset()..][..count];
                let stride = shape[config.axis + 1..].iter().product::<usize>();
                for (channel, spectrum) in combined.chunks_exact_mut(voxels).enumerate() {
                    for (value, &pixel) in work
                        .iter_mut()
                        .zip(pixels[channel..].iter().step_by(channels))
                    {
                        *value = Complex64::new(pixel.as_f64(), 0.0);
                    }
                    fft.transform(&mut work, false);
                    for plane in start..stop {
                        // Convert the centered acquisition index into an unshifted FFT bin.
                        let bin = (plane + axis_len.div_ceil(2)) % axis_len;
                        for block in (bin * stride..voxels).step_by(axis_len * stride) {
                            spectrum[block..block + stride]
                                .copy_from_slice(&work[block..block + stride]);
                        }
                    }
                }
                Ok(())
            })?;
            start = stop;
        }
        for spectrum in combined.chunks_exact_mut(voxels) {
            fft.transform(spectrum, true);
        }
        macro_rules! reconstruct {
            ($cast:expr) => {
                Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                    for i in 0..voxels {
                        for spectrum in combined.chunks_exact(voxels) {
                            out.write_next($cast(spectrum[i].norm()))?;
                        }
                    }
                    Ok(())
                })
            };
        }
        self.volume = match input.dtype() {
            DType::U8 => reconstruct!(|v: f64| v.round() as u8)?,
            _ => reconstruct!(|v: f64| v.clamp(0.0, 1.0) as f32)?,
        };
        Ok(self)
    }
}
