use super::{
    BoxFormat, StructuredSample,
    overlay::{self, OverlayElement},
};
use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::ImageAxisOrder;
use rivet_core::{CpuStorageRef, DType, Tensor};
use rivet_data::random::RandomStream;

#[derive(Clone, Debug)]
pub struct CopyPasteDonor {
    pub sample: StructuredSample,
    pub semantic_mask: Option<Tensor>,
}

#[derive(Clone, Debug)]
pub struct CopyAndPasteConfig {
    pub donors: Vec<CopyPasteDonor>,
    pub blend_sigma_range: Option<(f64, f64)>,
    pub scale_range: (f64, f64),
    pub min_visibility_after_paste: f64,
    pub min_paste_area: usize,
}

impl CopyAndPasteConfig {
    pub fn validate(&self) -> RivetResult<()> {
        let (lo, hi) = self.scale_range;
        if self.donors.len() > 64
            || !(0.0..=1.0).contains(&self.min_visibility_after_paste)
            || lo <= 0.0
            || lo > hi
            || !hi.is_finite()
            || self.min_paste_area == 0
            || self
                .blend_sigma_range
                .is_some_and(|(lo, hi)| !(lo >= 0.0 && lo <= hi && hi <= 32.0))
        {
            return Err(invalid_argument("invalid CopyAndPaste parameters"));
        }
        Ok(())
    }
}

#[derive(Clone)]
struct Instance {
    mask: Vec<u8>,
    label: Option<i64>,
    points: Option<Vec<f32>>,
}

fn hwc(s: &StructuredSample) -> RivetResult<Tensor> {
    match s.axis_order {
        ImageAxisOrder::Hwc => Ok(s.image.clone()),
        ImageAxisOrder::Chw => Ok(s.image.permute(&[1, 2, 0])?),
    }
}

fn instances(s: &StructuredSample, h: usize, w: usize) -> RivetResult<Vec<Instance>> {
    let boxes = s.xyxy()?.map(|t| t.to_vec::<f32>()).transpose()?;
    let labels = s.labels.as_ref().map(Tensor::to_vec::<i64>).transpose()?;
    let points = s
        .keypoints
        .as_ref()
        .map(Tensor::to_vec::<f32>)
        .transpose()?;
    let k = s.keypoints.as_ref().map_or(0, |t| t.dims()[1]) * 2;
    let instance_masks = s.masks.as_ref().filter(|m| m.rank() == 3);
    let masks = instance_masks
        .map(|m| m.to_dtype(DType::I64)?.to_vec::<i64>())
        .transpose()?;
    let n = instance_masks
        .or(s.boxes.as_ref())
        .map_or(0, |t| t.dims()[0]);
    if n == 0
        && (labels.as_ref().is_some_and(|x| !x.is_empty())
            || s.keypoints.as_ref().is_some_and(|x| x.dims()[0] != 0))
    {
        return Err(invalid_argument(
            "CopyAndPaste annotations need instance masks or boxes",
        ));
    }
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let mask = if let Some(masks) = &masks {
            masks[i * h * w..(i + 1) * h * w]
                .iter()
                .map(|&v| u8::from(v != 0))
                .collect()
        } else {
            let b = &boxes.as_ref().expect("box-only instance")[i * 4..i * 4 + 4];
            let (l, t, r, b) = (
                (b[0].floor() as usize).min(w),
                (b[1].floor() as usize).min(h),
                (b[2].ceil() as usize).min(w),
                (b[3].ceil() as usize).min(h),
            );
            let mut mask = vec![0; h * w];
            if l < r {
                for y in t..b {
                    mask[y * w + l..y * w + r].fill(1);
                }
            }
            mask
        };
        result.push(Instance {
            mask,
            label: labels.as_ref().map(|l| l[i]),
            points: points.as_ref().map(|p| p[i * k..(i + 1) * k].to_vec()),
        });
    }
    Ok(result)
}

fn bounds(mask: &[u8], h: usize, w: usize) -> Option<[usize; 4]> {
    let mut b = [w, h, 0, 0];
    for (i, &v) in mask.iter().enumerate() {
        if v != 0 {
            let (x, y) = (i % w, i / w);
            b = [b[0].min(x), b[1].min(y), b[2].max(x + 1), b[3].max(y + 1)];
        }
    }
    (b[0] < b[2] && b[1] < b[3]).then_some(b)
}

impl StructuredSample {
    pub fn copy_and_paste(
        mut self,
        config: &CopyAndPasteConfig,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        config.validate()?;
        if config.donors.is_empty() {
            return Ok(self);
        }
        let (h, w) = self.size()?;
        let base = hwc(&self)?.contiguous()?;
        let channels = base.dims()[2];
        let mask_dtype = self.masks.as_ref().map_or(DType::U8, Tensor::dtype);
        let mut semantic = semantic_values(self.masks.as_ref().filter(|m| m.rank() == 2), h, w)?;
        let mut blended = config
            .blend_sigma_range
            .map(|_| image_values(&base))
            .transpose()?;
        let mut all = instances(&self, h, w)?;
        let mut areas: Vec<_> = all
            .iter()
            .map(|a| a.mask.iter().filter(|&&v| v != 0).count())
            .collect();
        let mut layers = Vec::new();
        let mut pixels = vec![None; h * w];
        let mut pasted = false;
        let (lo, hi) = config.scale_range;
        for donor in &config.donors {
            let sample = &donor.sample;
            sample.validate()?;
            let (dh, dw) = sample.size()?;
            let donor_semantic = semantic_values(
                donor
                    .semantic_mask
                    .as_ref()
                    .or(sample.masks.as_ref().filter(|m| m.rank() == 2)),
                dh,
                dw,
            )?;
            let image = hwc(sample)?;
            let donor_values = config
                .blend_sigma_range
                .map(|_| image_values(&image))
                .transpose()?;

            if image.dtype() != base.dtype() || image.dims()[2] != channels {
                return Err(invalid_argument(
                    "CopyAndPaste donors must match image dtype and channel count",
                ));
            }
            let layer = layers.len();
            layers.push(OverlayElement {
                image,
                mask: None,
                offset: None,
                bbox: None,
                mask_id: None,
            });
            for instance in instances(sample, dh, dw)? {
                let Some([mut l, mut t, mut r, mut b]) = bounds(&instance.mask, dh, dw) else {
                    continue;
                };
                if let Some(points) = &instance.points {
                    for p in points.chunks_exact(2) {
                        if p[0] >= 0.0 && p[1] >= 0.0 && p[0] < dw as f32 && p[1] < dh as f32 {
                            l = l.min(p[0].floor() as usize);
                            t = t.min(p[1].floor() as usize);
                            r = r.max(p[0].floor() as usize + 1);
                            b = b.max(p[1].floor() as usize + 1);
                        }
                    }
                }
                let (sw, sh) = (r - l, b - t);
                let fit = (w as f64 / sw as f64).min(h as f64 / sh as f64).min(1.0);
                let scale = fit * (lo + rng.next_f64() * (hi - lo)).min(1.0);
                let (pw, ph) = (
                    ((sw as f64 * scale).round() as usize).clamp(1, w),
                    ((sh as f64 * scale).round() as usize).clamp(1, h),
                );
                if pw * ph < config.min_paste_area {
                    continue;
                }
                let (x, y) = (
                    rng.gen_range_usize(0..w - pw + 1)?,
                    rng.gen_range_usize(0..h - ph + 1)?,
                );
                let mut mask = vec![0; h * w];
                let mut area = 0;
                let mut alpha = config.blend_sigma_range.map(|_| vec![0.0; pw * ph]);
                for dy in 0..ph {
                    let sy = t + dy * sh / ph;
                    for dx in 0..pw {
                        let sx = l + dx * sw / pw;
                        let src = sy * dw + sx;
                        if instance.mask[src] != 0 {
                            let p = (y + dy) * w + x + dx;
                            mask[p] = 1;
                            pixels[p] = Some((layer, src));
                            if let Some(alpha) = &mut alpha {
                                alpha[dy * pw + dx] = 1.0;
                            }
                            if let (Some(target), Some(values)) = (&mut semantic, &donor_semantic) {
                                target[p] = values[src];
                            }
                            area += 1;
                        }
                    }
                }
                if area == 0 {
                    continue;
                }
                if let (Some(blended), Some(values), Some(mut alpha), Some((lo, hi))) =
                    (&mut blended, &donor_values, alpha, config.blend_sigma_range)
                {
                    blur_alpha(&mut alpha, pw, ph, lo + rng.next_f64() * (hi - lo));
                    for dy in 0..ph {
                        let sy = t + dy * sh / ph;
                        for dx in 0..pw {
                            let sx = l + dx * sw / pw;
                            let weight = alpha[dy * pw + dx];
                            let dst = ((y + dy) * w + x + dx) * channels;
                            let src = (sy * dw + sx) * channels;
                            for c in 0..channels {
                                blended[dst + c] += weight * (values[src + c] - blended[dst + c]);
                            }
                        }
                    }
                }
                for existing in &mut all {
                    for (a, &b) in existing.mask.iter_mut().zip(&mask) {
                        if b != 0 {
                            *a = 0;
                        }
                    }
                }
                let points = instance.points.map(|mut p| {
                    for point in p.chunks_exact_mut(2) {
                        point[0] = x as f32 + (point[0] - l as f32) * pw as f32 / sw as f32;
                        point[1] = y as f32 + (point[1] - t as f32) * ph as f32 / sh as f32;
                    }
                    p
                });
                all.push(Instance {
                    mask,
                    label: instance.label,
                    points,
                });
                areas.push(area);
                pasted = true;
            }
        }
        if !pasted {
            return Ok(self);
        }
        let image = if let Some(values) = blended {
            match base.dtype() {
                DType::U8 => Tensor::from_exact_iter(
                    values.into_iter().map(|v| v.round() as u8),
                    base.dims(),
                    base.device(),
                )?,
                DType::F32 => Tensor::from_exact_iter(
                    values.into_iter().map(|v| v as f32),
                    base.dims(),
                    base.device(),
                )?,
                _ => unreachable!(),
            }
        } else {
            base.with_cpu_storage(|storage, layout| {
                let start = layout.start_offset();
                // The base is contiguous. Only the selected samples are read.
                match storage {
                    CpuStorageRef::U8(v) => Ok(overlay::compose(
                        &base,
                        &v[start..start + h * w * channels],
                        &layers,
                        &pixels,
                        channels,
                    )),
                    CpuStorageRef::F32(v) => Ok(overlay::compose(
                        &base,
                        &v[start..start + h * w * channels],
                        &layers,
                        &pixels,
                        channels,
                    )),
                    _ => unreachable!(),
                }
            })??
        };
        self.image = match self.axis_order {
            ImageAxisOrder::Hwc => image,
            ImageAxisOrder::Chw => image.permute(&[2, 0, 1])?,
        };

        let labels_present = self.labels.is_some() || all.iter().any(|a| a.label.is_some());
        let k = self.keypoints.as_ref().map(|t| t.dims()[1]).or_else(|| {
            all.iter()
                .find_map(|a| a.points.as_ref().map(|p| p.len() / 2))
        });
        let mut masks = Vec::new();
        let mut boxes = Vec::new();
        let mut labels = Vec::new();
        let mut points = Vec::new();
        for (instance, before) in all.into_iter().zip(areas) {
            let Some(b) = bounds(&instance.mask, h, w) else {
                continue;
            };
            let visible = instance.mask.iter().filter(|&&v| v != 0).count();
            if visible as f64 / (before.max(1) as f64) < config.min_visibility_after_paste {
                continue;
            }
            boxes.extend(b.map(|v| v as f32));
            masks.extend(instance.mask);
            if labels_present {
                labels.push(instance.label.ok_or_else(|| {
                    invalid_argument("CopyAndPaste instances must have consistent labels")
                })?);
            }
            if let Some(k) = k {
                let p = instance
                    .points
                    .filter(|p| p.len() == 2 * k)
                    .ok_or_else(|| {
                        invalid_argument("CopyAndPaste instances must have consistent keypoints")
                    })?;
                points.extend(p);
            }
        }
        let n = boxes.len() / 4;
        let xyxy = Tensor::from_vec(boxes, [n, 4], base.device())?;
        self.boxes = Some(super::annotations::convert_boxes(
            &xyxy,
            BoxFormat::Xyxy,
            self.box_format,
        )?);
        self.masks = Some(if let Some(values) = semantic {
            if mask_dtype == DType::U8 && values.iter().any(|&v| u8::try_from(v).is_err()) {
                return Err(invalid_argument(
                    "CopyAndPaste semantic IDs do not fit the target uint8 mask",
                ));
            }
            Tensor::from_vec(values, [h, w], base.device())?.to_dtype(mask_dtype)?
        } else {
            Tensor::from_vec(masks, [n, h, w], base.device())?.to_dtype(mask_dtype)?
        });
        self.labels = if labels_present {
            Some(Tensor::from_vec(labels, [n], base.device())?)
        } else {
            None
        };
        self.keypoints = if let Some(k) = k {
            Some(Tensor::from_vec(points, [n, k, 2], base.device())?)
        } else {
            None
        };
        Ok(self)
    }
}

fn semantic_values(mask: Option<&Tensor>, h: usize, w: usize) -> RivetResult<Option<Vec<i64>>> {
    let Some(mask) = mask else {
        return Ok(None);
    };
    if mask.dims() != [h, w] || !matches!(mask.dtype(), DType::U8 | DType::I64) {
        return Err(invalid_argument(
            "CopyAndPaste semantic mask must match donor HW and use uint8 or int64",
        ));
    }
    Ok(Some(mask.to_dtype(DType::I64)?.to_vec::<i64>()?))
}

fn image_values(image: &Tensor) -> RivetResult<Vec<f64>> {
    Ok(match image.dtype() {
        DType::U8 => image.to_vec::<u8>()?.into_iter().map(f64::from).collect(),
        DType::F32 => image.to_vec::<f32>()?.into_iter().map(f64::from).collect(),
        _ => {
            return Err(invalid_argument(
                "CopyAndPaste requires uint8 or float32 pixels",
            ));
        }
    })
}

fn blur_alpha(alpha: &mut [f64], w: usize, h: usize, sigma: f64) {
    if sigma == 0.0 {
        return;
    }
    let radius = (3.0 * sigma).ceil() as isize;
    let mut kernel: Vec<_> = (-radius..=radius)
        .map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp())
        .collect();
    let sum = kernel.iter().sum::<f64>();
    for v in &mut kernel {
        *v /= sum;
    }
    let mut temporary = vec![0.0; alpha.len()];
    let samples = |p: usize, n: usize| {
        kernel.iter().enumerate().filter_map(move |(j, &v)| {
            p.checked_add_signed(j as isize - radius)
                .filter(|&i| i < n)
                .map(|i| (i, v))
        })
    };
    for y in 0..h {
        for x in 0..w {
            temporary[y * w + x] = samples(x, w).map(|(i, v)| alpha[y * w + i] * v).sum();
        }
    }
    for y in 0..h {
        for x in 0..w {
            alpha[y * w + x] = samples(y, h).map(|(i, v)| temporary[i * w + x] * v).sum();
        }
    }
}
