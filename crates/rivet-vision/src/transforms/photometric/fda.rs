use crate::errors::{RivetResult, invalid_argument};
use crate::transforms::{ReferenceImages, contiguous_values, degradation::frequency::Fft2};
use rivet_core::{CpuStorageRef, DType, Tensor, WithDType};
use rivet_data::random::RandomStream;
use rustfft::num_complex::Complex64;

#[derive(Clone, Debug, PartialEq)]
pub struct FdaConfig {
    references: ReferenceImages,
    pub beta_range: (f64, f64),
}

impl FdaConfig {
    pub fn new(references: ReferenceImages, beta_range: (f64, f64)) -> RivetResult<Self> {
        let config = Self {
            references,
            beta_range,
        };
        if !config.valid()
            || config
                .references
                .images()
                .iter()
                .any(|t| !matches!(t.dtype(), DType::U8 | DType::F32))
        {
            return Err(invalid_argument(
                "FDA requires uint8 or float32 references and beta in [0, 0.5]",
            ));
        }
        Ok(config)
    }

    pub(in crate::transforms) fn valid(&self) -> bool {
        let (lo, hi) = self.beta_range;
        lo >= 0.0 && lo <= hi && hi <= 0.5
    }

    pub(crate) fn channels(&self) -> usize {
        self.references.images()[0]
            .dims()
            .get(2)
            .copied()
            .unwrap_or(1)
    }

    pub(in crate::transforms) fn apply(
        &self,
        input: &Tensor,
        rng: &mut RandomStream,
    ) -> RivetResult<Tensor> {
        let &[h, w, c] = input.dims() else {
            return Err(invalid_argument("FDA expects HWC input"));
        };
        if h == 0
            || w == 0
            || c != self.channels()
            || !matches!(input.dtype(), DType::U8 | DType::F32)
        {
            return Err(invalid_argument(
                "FDA requires nonempty uint8 or float32 HWC with matching reference channels",
            ));
        }
        let reference =
            &self.references.images()[rng.gen_range_usize(0..self.references.images().len())?];
        let (lo, hi) = self.beta_range;
        let beta = lo + rng.next_f64() * (hi - lo);
        if beta == 0.0 {
            return Ok(input.clone());
        }
        let [rh, rw] = [reference.dims()[0], reference.dims()[1]];
        let input = input.contiguous()?;
        let reference = reference.contiguous()?;
        let result = reference.with_cpu_storage(|storage, layout| match storage {
            CpuStorageRef::U8(v) => Ok(adapt(
                &input,
                contiguous_values(v, layout)?,
                1.0 / 255.0,
                [rh, rw],
                beta,
            )),
            CpuStorageRef::F32(v) => Ok(adapt(
                &input,
                contiguous_values(v, layout)?,
                1.0,
                [rh, rw],
                beta,
            )),
            _ => unreachable!(),
        })?;
        result
    }
}

fn adapt<T: Copy + Into<f64>>(
    input: &Tensor,
    reference: &[T],
    rs: f64,
    [rh, rw]: [usize; 2],
    beta: f64,
) -> RivetResult<Tensor> {
    let &[h, w, c] = input.dims() else {
        unreachable!();
    };
    let position = |i: usize, from: usize, to: usize| {
        let p = ((i as f64 + 0.5) * from as f64 / to as f64 - 0.5).clamp(0.0, (from - 1) as f64);
        let lo = p as usize;
        (lo, (lo + 1).min(from - 1), p - lo as f64)
    };
    let ys: Vec<_> = (0..h).map(|i| position(i, rh, h)).collect();
    let xs: Vec<_> = (0..w).map(|i| position(i, rw, w)).collect();
    let target = |x: usize, y: usize, ch: usize| {
        let (l, r, fx) = xs[x];
        let (t, b, fy) = ys[y];
        let value = |x, y| <T as Into<f64>>::into(reference[(y * rw + x) * c + ch]) * rs;
        let (a, b0) = (value(l, t), value(l, b));
        let upper = a + (value(r, t) - a) * fx;
        let lower = b0 + (value(r, b) - b0) * fx;
        upper + (lower - upper) * fy
    };
    Ok(input.with_cpu_storage(|storage, layout| match storage {
        CpuStorageRef::U8(v) => write(
            input,
            contiguous_values(v, layout)?,
            1.0 / 255.0,
            beta,
            target,
            |v| (v * 255.0).round() as u8,
        ),
        CpuStorageRef::F32(v) => write(
            input,
            contiguous_values(v, layout)?,
            1.0,
            beta,
            target,
            |v| v as f32,
        ),
        _ => unreachable!(),
    })?)
}

fn write<T: WithDType + Into<f64>>(
    input: &Tensor,
    pixels: &[T],
    scale: f64,
    beta: f64,
    reference: impl Fn(usize, usize, usize) -> f64,
    cast: impl Fn(f64) -> T,
) -> rivet_core::Result<Tensor> {
    let &[h, w, c] = input.dims() else {
        unreachable!();
    };
    let radius = (h.min(w) as f64 * beta).floor() as usize;
    let mut fft = Fft2::new(w, h);
    let mut target = vec![Complex64::default(); w * h];
    let mut planes = Vec::with_capacity(c);
    for ch in 0..c {
        let mut source: Vec<_> = pixels
            .chunks_exact(c)
            .map(|pixel| Complex64::new(pixel[ch].into() * scale, 0.0))
            .collect();
        for (i, v) in target.iter_mut().enumerate() {
            *v = Complex64::new(reference(i % w, i / w, ch), 0.0);
        }
        fft.transform(&mut source, false);
        fft.transform(&mut target, false);
        for y in 0..h {
            if y.min(h - y) > radius {
                continue;
            }
            for x in 0..w {
                if x.min(w - x) > radius {
                    continue;
                }
                let i = y * w + x;
                let from = source[i].norm();
                let to = target[i].norm();
                source[i] = if from > 1e-12 {
                    source[i] * (to / from)
                } else {
                    Complex64::new(to, 0.0)
                };
            }
        }
        fft.transform(&mut source, true);
        planes.push(source);
    }
    Tensor::from_exact_writer(input.dims(), input.device(), |out| {
        for i in 0..h * w {
            for plane in &planes {
                out.write_next(cast(plane[i].re.clamp(0.0, 1.0)))?;
            }
        }
        Ok(())
    })
}
