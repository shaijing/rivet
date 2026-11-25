use super::{RivetResult, contiguous_values, invalid_argument};
use rivet_core::{CpuStorageRef, Tensor, WithDType};
use rivet_data::random::RandomStream;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ReferenceImages(Arc<Vec<Tensor>>);

impl PartialEq for ReferenceImages {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl ReferenceImages {
    pub fn new(images: Vec<Tensor>) -> RivetResult<Self> {
        if images.is_empty() {
            return Err(invalid_argument("at least one reference image is required"));
        }
        let channels = reference_shape(&images[0])?;
        for image in &images[1..] {
            if reference_shape(image)? != channels {
                return Err(invalid_argument(
                    "reference images must share a channel count",
                ));
            }
        }
        Ok(Self(Arc::new(images)))
    }

    pub(crate) fn images(&self) -> &[Tensor] {
        &self.0
    }
}

fn reference_shape(input: &Tensor) -> RivetResult<usize> {
    match input.dims() {
        &[h, w] if h > 0 && w > 0 => {}
        &[h, w, c] if h > 0 && w > 0 && c > 0 => return Ok(c),
        _ => {
            return Err(invalid_argument(
                "reference image must be nonempty HW or HWC",
            ));
        }
    }
    Ok(1)
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistogramMatchingConfig {
    references: ReferenceImages,
    histograms: Arc<Vec<Vec<Vec<u64>>>>,
    pub blend_range: (f64, f64),
}

impl HistogramMatchingConfig {
    pub fn new(references: ReferenceImages, blend_range: (f64, f64)) -> RivetResult<Self> {
        let histograms = references
            .images()
            .iter()
            .map(histogram)
            .collect::<RivetResult<_>>()?;
        let config = Self {
            references,
            histograms: Arc::new(histograms),
            blend_range,
        };
        if !config.valid() {
            return Err(invalid_argument("invalid HistogramMatching blend range"));
        }
        Ok(config)
    }

    pub(crate) fn channels(&self) -> usize {
        self.histograms[0].len()
    }

    pub(super) fn valid(&self) -> bool {
        let (lo, hi) = self.blend_range;
        lo >= 0.0 && lo <= hi && hi <= 1.0
    }

    pub(super) fn apply(&self, input: &Tensor, rng: &mut RandomStream) -> RivetResult<Tensor> {
        let &[h, w, channels] = input.dims() else {
            return Err(invalid_argument("HistogramMatching expects HWC input"));
        };
        if h == 0 || w == 0 || channels != self.channels() {
            return Err(invalid_argument(
                "HistogramMatching image and references must share a channel count",
            ));
        }
        let index = rng.gen_range_usize(0..self.histograms.len())?;
        let (lo, hi) = self.blend_range;
        let blend = lo + rng.next_f64() * (hi - lo);
        if blend == 0.0 {
            return Ok(input.clone());
        }
        let source = histogram(input)?;
        let reference = &self.histograms[index];
        let tables: Vec<_> = source
            .iter()
            .zip(reference)
            .map(|(a, b)| build_table(a, b))
            .collect();
        let input = input.contiguous()?;
        Ok(input.with_cpu_storage(|storage, layout| match storage {
            CpuStorageRef::U8(values) => write(
                &input,
                contiguous_values(values, layout)?,
                &tables,
                blend,
                1.0 / 255.0,
                |p| (p * 255.0).round() as u8,
            ),
            CpuStorageRef::F32(values) => write(
                &input,
                contiguous_values(values, layout)?,
                &tables,
                blend,
                1.0,
                |p| p as f32,
            ),
            _ => unreachable!(),
        })?)
    }
}

const FLOAT_BINS: usize = 1024;

fn bin(v: f64, n: usize) -> usize {
    (v.clamp(0.0, 1.0) * (n - 1) as f64).round() as usize
}

fn histogram(input: &Tensor) -> RivetResult<Vec<Vec<u64>>> {
    let channels = reference_shape(input)?;
    let input = input.contiguous()?;
    input
        .with_cpu_storage(|storage, layout| {
            let bins = match storage {
                CpuStorageRef::U8(_) => 256,
                CpuStorageRef::F32(_) => FLOAT_BINS,
                other => {
                    return Err(rivet_core::Error::UnexpectedDType {
                        expected: rivet_core::DType::U8,
                        actual: other.dtype(),
                    });
                }
            };
            let mut counts = vec![vec![0; bins]; channels];
            macro_rules! count {
                ($values:expr,$index:expr) => {{
                    let values = contiguous_values($values, layout)?;
                    let index = $index;
                    for pixel in values.chunks_exact(channels) {
                        for (hist, &v) in counts.iter_mut().zip(pixel) {
                            hist[index(v)] += 1;
                        }
                    }
                }};
            }
            match storage {
                CpuStorageRef::U8(v) => count!(v, usize::from),
                CpuStorageRef::F32(v) => count!(v, |x: f32| bin(f64::from(x), bins)),
                _ => unreachable!(),
            }
            Ok(counts)
        })
        .map_err(Into::into)
}

fn build_table(source: &[u64], reference: &[u64]) -> Vec<f64> {
    let total = source.iter().sum::<u64>() as f64;
    let ref_total = reference.iter().sum::<u64>() as f64;
    let mut count = 0;
    let knots: Vec<_> = reference
        .iter()
        .enumerate()
        .filter_map(|(i, &n)| {
            if n == 0 {
                return None;
            }
            count += n;
            Some((
                count as f64 / ref_total,
                i as f64 / (reference.len() - 1) as f64,
            ))
        })
        .collect();
    let mut cumulative = 0;
    let mut upper = 0;
    source
        .iter()
        .map(|&n| {
            cumulative += n;
            let q = cumulative as f64 / total;
            while upper + 1 < knots.len() && knots[upper].0 < q {
                upper += 1;
            }
            let (qb, b) = knots[upper];
            if upper == 0 {
                b
            } else {
                let (qa, a) = knots[upper - 1];
                a + (b - a) * ((q - qa) / (qb - qa)).clamp(0.0, 1.0)
            }
        })
        .collect()
}

fn write<T: WithDType + Into<f64>>(
    input: &Tensor,
    values: &[T],
    tables: &[Vec<f64>],
    blend: f64,
    scale: f64,
    cast: impl Fn(f64) -> T,
) -> rivet_core::Result<Tensor> {
    Tensor::from_exact_writer(input.dims(), input.device(), |out| {
        for pixel in values.chunks_exact(tables.len()) {
            for (&v, table) in pixel.iter().zip(tables) {
                let v = v.into() * scale;
                let matched = table[bin(v, table.len())];
                out.write_next(cast(v + (matched - v) * blend))?;
            }
        }
        Ok(())
    })
}
