//! Terminal batch augmentation. Randomness is keyed by output batch ordinal.
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{ImageAxisOrder, ImageBatch};
use rivet_core::{CpuStorageRef, DType, Device, Tensor};
use rivet_data::random::RandomStream;

#[derive(Clone, Debug)]
pub struct BatchMixConfig {
    pub cutmix: bool,
    pub alpha: f64,
    pub num_classes: Option<usize>,
}
impl BatchMixConfig {
    pub fn name(&self) -> &'static str {
        if self.cutmix { "CutMix" } else { "MixUp" }
    }
    pub fn validate(&self) -> RivetResult<()> {
        if !self.alpha.is_finite() || self.alpha <= 0.0 || self.num_classes == Some(0) {
            return Err(invalid_argument(
                "batch mixing requires finite alpha > 0 and num_classes > 0",
            ));
        }
        Ok(())
    }
    pub fn apply(&self, batch: ImageBatch, rng: &mut RandomStream) -> RivetResult<ImageBatch> {
        self.validate()?;
        let a = log_gamma(self.alpha, rng);
        let b = log_gamma(self.alpha, rng);
        let difference = b - a;
        let lambda = if difference.is_nan() {
            // The limit for alpha approaching zero places equal mass at 0/1.
            if rng.next_f64() < 0.5 { 0.0 } else { 1.0 }
        } else {
            1.0 / (1.0 + difference.exp())
        };
        let dims = batch.images.dims();
        if dims.len() != 4 || dims[0] == 0 {
            return Err(invalid_shape(
                "batch mixing requires nonempty rank-4 images",
            ));
        }
        let (h, w) = match batch.axis_order {
            ImageAxisOrder::Hwc => (dims[1], dims[2]),
            ImageAxisOrder::Chw => (dims[2], dims[3]),
        };
        let rect = if self.cutmix {
            if h == 0 || w == 0 {
                return Err(invalid_shape("CutMix requires nonempty spatial dimensions"));
            }
            let x = rng.gen_range_usize(0..w)?;
            let y = rng.gen_range_usize(0..h)?;
            let r = 0.5 * (1.0 - lambda).sqrt();
            let dx = (w as f64 * r) as usize;
            let dy = (h as f64 * r) as usize;
            Some([
                x.saturating_sub(dx),
                y.saturating_sub(dy),
                x.saturating_add(dx).min(w),
                y.saturating_add(dy).min(h),
            ])
        } else {
            None
        };
        self.apply_resolved(batch, lambda, rect)
    }
    pub fn apply_resolved(
        &self,
        batch: ImageBatch,
        lambda: f64,
        rect: Option<[usize; 4]>,
    ) -> RivetResult<ImageBatch> {
        self.validate()?;
        let dims = batch.images.dims();
        if dims.len() != 4 || dims[0] == 0 || !lambda.is_finite() || !(0.0..=1.0).contains(&lambda)
        {
            return Err(invalid_shape("invalid batch shape or mixing coefficient"));
        }
        let n = dims[0];
        let (h, w, c) = match batch.axis_order {
            ImageAxisOrder::Hwc => (dims[1], dims[2], dims[3]),
            ImageAxisOrder::Chw => (dims[2], dims[3], dims[1]),
        };
        if h == 0 || w == 0 || c == 0 {
            return Err(invalid_shape("batch mixing requires nonempty images"));
        }
        let mut lam = lambda as f32;
        let rectangle = if self.cutmix {
            let [x1, y1, x2, y2] =
                rect.ok_or_else(|| invalid_argument("CutMix requires a resolved rectangle"))?;
            if x1 > x2 || y1 > y2 || x2 > w || y2 > h {
                return Err(invalid_argument("CutMix rectangle is outside image"));
            }
            lam = (1.0 - ((x2 - x1) * (y2 - y1)) as f64 / (w * h) as f64) as f32;
            [x1, y1, x2, y2]
        } else {
            [0; 4]
        };
        let labels = soft_labels(&batch.labels, n, self.num_classes)?;
        let classes = labels.dims()[1];
        let lv = labels.to_vec::<f32>()?;
        let mixed_labels =
            Tensor::from_exact_writer::<f32, _, _>([n, classes], &Device::Cpu, |out| {
                for i in 0..n {
                    let prev = if i == 0 { n - 1 } else { i - 1 };
                    for k in 0..classes {
                        out.write_next(
                            lv[prev * classes + k] * (1.0 - lam) + lv[i * classes + k] * lam,
                        )?;
                    }
                }
                Ok(())
            })?;
        if !self.cutmix && batch.images.dtype() != DType::F32 {
            return Err(invalid_argument(
                "MixUp requires float32 images; call to_dtype(scale=True) first",
            ));
        }
        let images = batch.images.with_cpu_storage(|storage, layout| {
            let stride = layout.stride();
            let start = layout.start_offset();
            macro_rules! mix {
                ($values:expr,$ty:ty,$blend:expr) => {{
                    let values = $values;
                    Tensor::from_exact_writer::<$ty, _, _>(dims, &Device::Cpu, |out| {
                        for i in 0..n {
                            let prev = if i == 0 { n - 1 } else { i - 1 };
                            match batch.axis_order {
                                ImageAxisOrder::Hwc => {
                                    for y in 0..h {
                                        for x in 0..w {
                                            let replace = self.cutmix
                                                && x >= rectangle[0]
                                                && x < rectangle[2]
                                                && y >= rectangle[1]
                                                && y < rectangle[3];
                                            let base = start
                                                + i * stride[0]
                                                + y * stride[1]
                                                + x * stride[2];
                                            let other = start
                                                + prev * stride[0]
                                                + y * stride[1]
                                                + x * stride[2];
                                            for ch in 0..c {
                                                let a = *values
                                                    .get(base + ch * stride[3])
                                                    .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                                                let b = *values
                                                    .get(other + ch * stride[3])
                                                    .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                                                out.write_next(if self.cutmix {
                                                    if replace { b } else { a }
                                                } else {
                                                    ($blend)(a, b, lam)
                                                })?;
                                            }
                                        }
                                    }
                                }
                                ImageAxisOrder::Chw => {
                                    for ch in 0..c {
                                        for y in 0..h {
                                            let base = start
                                                + i * stride[0]
                                                + ch * stride[1]
                                                + y * stride[2];
                                            let other = start
                                                + prev * stride[0]
                                                + ch * stride[1]
                                                + y * stride[2];
                                            for x in 0..w {
                                                let replace = self.cutmix
                                                    && x >= rectangle[0]
                                                    && x < rectangle[2]
                                                    && y >= rectangle[1]
                                                    && y < rectangle[3];
                                                let a = *values
                                                    .get(base + x * stride[3])
                                                    .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                                                let b = *values
                                                    .get(other + x * stride[3])
                                                    .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                                                out.write_next(if self.cutmix {
                                                    if replace { b } else { a }
                                                } else {
                                                    ($blend)(a, b, lam)
                                                })?;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Ok(())
                    })
                }};
            }
            match storage {
                CpuStorageRef::F32(values) => {
                    mix!(values, f32, |a: f32, b: f32, l: f32| b * (1.0 - l) + a * l)
                }
                CpuStorageRef::U8(values) if self.cutmix => {
                    mix!(values, u8, |a: u8, _b: u8, _l: f32| a)
                }
                _ => Err(rivet_core::Error::UnsupportedDType {
                    dtype: batch.images.dtype(),
                    op: "batch mixing",
                }),
            }
        })?;
        Ok(ImageBatch {
            images,
            labels: mixed_labels,
            axis_order: batch.axis_order,
        })
    }
}
fn soft_labels(labels: &Tensor, n: usize, num_classes: Option<usize>) -> RivetResult<Tensor> {
    match labels.dims() {
        [size] if *size == n && labels.dtype() == DType::I64 => {
            let classes = num_classes
                .ok_or_else(|| invalid_argument("num_classes is required for index labels"))?;
            let values = labels.to_vec::<i64>()?;
            if values.iter().any(|&v| v < 0 || v as usize >= classes) {
                return Err(invalid_argument("class label outside num_classes"));
            }
            Ok(Tensor::from_exact_writer::<f32, _, _>(
                [n, classes],
                &Device::Cpu,
                |out| {
                    for value in values {
                        for k in 0..classes {
                            out.write_next(if value as usize == k { 1.0 } else { 0.0 })?;
                        }
                    }
                    Ok(())
                },
            )?)
        }
        [size, classes] if *size == n && *classes > 0 && labels.dtype() == DType::F32 => {
            if num_classes.is_some_and(|c| c != *classes)
                || labels.to_vec::<f32>()?.iter().any(|x| !x.is_finite())
            {
                return Err(invalid_argument("invalid soft labels or class count"));
            }
            Ok(labels.clone())
        }
        _ => Err(invalid_shape(
            "labels must be int64 [B] or float32 [B,num_classes]",
        )),
    }
}
fn log_gamma(alpha: f64, rng: &mut RandomStream) -> f64 {
    if alpha < 1.0 {
        return log_gamma(alpha + 1.0, rng) + rng.next_f64().max(f64::MIN_POSITIVE).ln() / alpha;
    }
    let d = alpha - 1.0 / 3.0;
    let c = (9.0 * d).sqrt().recip();
    loop {
        let z = (-2.0 * rng.next_f64().max(f64::MIN_POSITIVE).ln()).sqrt()
            * (std::f64::consts::TAU * rng.next_f64()).cos();
        let v = 1.0 + c * z;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * z.powi(4) || u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            return d.ln() + v.ln();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn input(dtype: DType) -> ImageBatch {
        let x = Tensor::from_vec(
            vec![0f32, 1., 2., 3., 10., 11., 12., 13.],
            [2, 2, 2, 1],
            &Device::Cpu,
        )
        .unwrap();
        ImageBatch {
            images: x.to_dtype(dtype).unwrap(),
            labels: Tensor::from_vec(vec![0i64, 1], [2], &Device::Cpu).unwrap(),
            axis_order: ImageAxisOrder::Hwc,
        }
    }
    #[test]
    fn mixup_resolved_pairs_previous_and_emits_soft_labels() {
        let cfg = BatchMixConfig {
            cutmix: false,
            alpha: 1.,
            num_classes: Some(2),
        };
        let out = cfg.apply_resolved(input(DType::F32), 0.25, None).unwrap();
        assert_eq!(
            out.images.to_vec::<f32>().unwrap(),
            [7.5, 8.5, 9.5, 10.5, 2.5, 3.5, 4.5, 5.5]
        );
        assert_eq!(
            out.labels.to_vec::<f32>().unwrap(),
            [0.25, 0.75, 0.75, 0.25]
        );
    }
    #[test]
    fn cutmix_uses_actual_rectangle_area_and_preserves_uint8() {
        let cfg = BatchMixConfig {
            cutmix: true,
            alpha: 1.,
            num_classes: Some(2),
        };
        let out = cfg
            .apply_resolved(input(DType::U8), 0.99, Some([0, 0, 1, 2]))
            .unwrap();
        assert_eq!(
            out.images.to_vec::<u8>().unwrap(),
            [10, 1, 12, 3, 0, 11, 2, 13]
        );
        assert_eq!(out.labels.to_vec::<f32>().unwrap(), [0.5, 0.5, 0.5, 0.5]);
        assert!(
            cfg.apply_resolved(input(DType::U8), 0.5, Some([0, 0, 3, 2]))
                .is_err()
        );
    }
    #[test]
    fn beta_distribution_moments_and_extreme_shapes() {
        let mut rng = RandomStream::from_seed(7);
        let mut values = Vec::new();
        for _ in 0..10000 {
            let a = log_gamma(0.7, &mut rng);
            let b = log_gamma(0.7, &mut rng);
            values.push(1. / (1. + (b - a).exp()));
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        assert!((mean - 0.5).abs() < 0.01);
        assert!((variance - 1. / (4. * 2.4)).abs() < 0.005);
        for alpha in [1e-300, 1e300] {
            let cfg = BatchMixConfig {
                cutmix: false,
                alpha,
                num_classes: Some(2),
            };
            let result = cfg.apply(input(DType::F32), &mut rng).unwrap();
            assert!(
                result
                    .labels
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .all(|v| v.is_finite())
            );
        }
    }
}
