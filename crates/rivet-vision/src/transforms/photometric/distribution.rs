use crate::errors::{RivetResult, invalid_argument};
use crate::transforms::{ReferenceImages, contiguous_values};
use rivet_core::{CpuStorageRef, Tensor, WithDType};
use rivet_data::random::RandomStream;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributionMode {
    Pca,
    Standard,
    MinMax,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PixelDistributionAdaptationConfig {
    references: Arc<Vec<Statistics>>,
    pub mode: DistributionMode,
    pub blend_range: (f64, f64),
}

#[derive(Clone, Debug, PartialEq)]
struct Statistics {
    channels: usize,
    mean: [f64; 3],
    min: [f64; 3],
    max: [f64; 3],
    covariance: [[f64; 3]; 3],
}

impl PixelDistributionAdaptationConfig {
    pub fn new(
        references: ReferenceImages,
        mode: DistributionMode,
        blend_range: (f64, f64),
    ) -> RivetResult<Self> {
        let stats = references
            .images()
            .iter()
            .map(statistics)
            .collect::<RivetResult<_>>()?;
        let config = Self {
            references: Arc::new(stats),
            mode,
            blend_range,
        };
        if !config.valid() {
            return Err(invalid_argument(
                "invalid PixelDistributionAdaptation blend range",
            ));
        }
        Ok(config)
    }
    pub(in crate::transforms) fn valid(&self) -> bool {
        let (lo, hi) = self.blend_range;
        lo >= 0.0 && lo <= hi && hi <= 1.0
    }
    pub(crate) fn channels(&self) -> usize {
        self.references[0].channels
    }

    pub(in crate::transforms) fn apply(
        &self,
        input: &Tensor,
        rng: &mut RandomStream,
    ) -> RivetResult<Tensor> {
        let channels = channels(input)?;
        if input.rank() != 3 || channels != self.channels() {
            return Err(invalid_argument(
                "PixelDistributionAdaptation requires HWC input with matching reference channels",
            ));
        }
        let target = &self.references[rng.gen_range_usize(0..self.references.len())?];
        let (lo, hi) = self.blend_range;
        let blend = lo + rng.next_f64() * (hi - lo);
        if blend == 0.0 {
            return Ok(input.clone());
        }
        let source = statistics(input)?;
        let (gain, s0, t0) = affine(&source, target, self.mode);
        let input = input.contiguous()?;
        Ok(input.with_cpu_storage(|storage, layout| match storage {
            CpuStorageRef::U8(v) => write(
                &input,
                contiguous_values(v, layout)?,
                gain,
                s0,
                t0,
                blend,
                channels,
                1.0 / 255.0,
                |v| (v * 255.0).round() as u8,
            ),
            CpuStorageRef::F32(v) => write(
                &input,
                contiguous_values(v, layout)?,
                gain,
                s0,
                t0,
                blend,
                channels,
                1.0,
                |v| v as f32,
            ),
            _ => unreachable!(),
        })?)
    }
}

fn channels(input: &Tensor) -> RivetResult<usize> {
    match input.dims() {
        &[h, w] | &[h, w, 1] if h > 0 && w > 0 => Ok(1),
        &[h, w, 3] if h > 0 && w > 0 => Ok(3),
        _ => Err(invalid_argument(
            "PixelDistributionAdaptation requires nonempty grayscale or RGB images",
        )),
    }
}

fn statistics(input: &Tensor) -> RivetResult<Statistics> {
    let channels = channels(input)?;
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| match storage {
        CpuStorageRef::U8(v) => Ok(measure(
            contiguous_values(v, layout)?,
            channels,
            1.0 / 255.0,
        )),
        CpuStorageRef::F32(v) => Ok(measure(contiguous_values(v, layout)?, channels, 1.0)),
        other => Err(rivet_core::Error::UnexpectedDType {
            expected: rivet_core::DType::F32,
            actual: other.dtype(),
        }),
    })?)
}

fn measure<T: Copy + Into<f64>>(data: &[T], channels: usize, scale: f64) -> Statistics {
    let mut stats = Statistics {
        channels,
        mean: [0.; 3],
        min: [f64::INFINITY; 3],
        max: [f64::NEG_INFINITY; 3],
        covariance: [[0.; 3]; 3],
    };
    let mut count = 0.0;
    for pixel in data.chunks_exact(channels) {
        count += 1.0;
        let mut delta = [0.; 3];
        let mut post = [0.; 3];
        for (c, &v) in pixel.iter().enumerate() {
            let v = v.into() * scale;
            delta[c] = v - stats.mean[c];
            stats.mean[c] += delta[c] / count;
            post[c] = v - stats.mean[c];
            stats.min[c] = stats.min[c].min(v);
            stats.max[c] = stats.max[c].max(v);
        }
        for i in 0..channels {
            for j in i..channels {
                stats.covariance[i][j] += delta[i] * post[j];
            }
        }
    }
    for i in 0..channels {
        for j in i..channels {
            stats.covariance[i][j] /= count;
            stats.covariance[j][i] = stats.covariance[i][j];
        }
    }
    stats
}

fn affine(
    source: &Statistics,
    target: &Statistics,
    mode: DistributionMode,
) -> ([[f64; 3]; 3], [f64; 3], [f64; 3]) {
    let mut gain = [[0.; 3]; 3];
    let centers = match mode {
        DistributionMode::MinMax => {
            for i in 0..source.channels {
                let denom = source.max[i] - source.min[i];
                if denom > 1e-12 {
                    gain[i][i] = (target.max[i] - target.min[i]) / denom;
                }
            }
            (source.min, target.min)
        }
        DistributionMode::Standard => {
            for i in 0..source.channels {
                let denom = source.covariance[i][i];
                if denom > 1e-12 {
                    gain[i][i] = (target.covariance[i][i].max(0.0) / denom).sqrt();
                }
            }
            (source.mean, target.mean)
        }
        DistributionMode::Pca => {
            let basis = |cov| {
                let (values, axes) = super::pca::diagonalize(cov);
                let mut order = [0, 1, 2];
                order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
                order.map(|i| {
                    let axis = axes.map(|row| row[i]);
                    let largest = axis
                        .iter()
                        .copied()
                        .max_by(|a, b| a.abs().total_cmp(&b.abs()))
                        .expect("three axes");
                    (values[i].max(0.0), axis.map(|v| v * largest.signum()))
                })
            };
            for ((src, u), (dst, v)) in basis(source.covariance)
                .into_iter()
                .zip(basis(target.covariance))
            {
                if src <= 1e-12 {
                    continue;
                }
                let ratio = (dst / src).sqrt();
                for i in 0..source.channels {
                    for j in 0..source.channels {
                        gain[i][j] += ratio * v[i] * u[j];
                    }
                }
            }
            (source.mean, target.mean)
        }
    };
    (gain, centers.0, centers.1)
}

fn write<T: WithDType + Into<f64>>(
    input: &Tensor,
    data: &[T],
    gain: [[f64; 3]; 3],
    s0: [f64; 3],
    t0: [f64; 3],
    blend: f64,
    channels: usize,
    scale: f64,
    cast: impl Fn(f64) -> T,
) -> rivet_core::Result<Tensor> {
    Tensor::from_exact_writer(input.dims(), input.device(), |out| {
        for pixel in data.chunks_exact(channels) {
            let mut src = [0.0; 3];
            for (i, &v) in pixel.iter().enumerate() {
                src[i] = v.into() * scale;
            }
            for c in 0..channels {
                let mapped = t0[c]
                    + (0..channels)
                        .map(|j| gain[c][j] * (src[j] - s0[j]))
                        .sum::<f64>();
                out.write_next(cast((src[c] + blend * (mapped - src[c])).clamp(0.0, 1.0)))?;
            }
        }
        Ok(())
    })
}
