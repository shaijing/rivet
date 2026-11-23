use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::ImageAxisOrder;
use rivet_core::{DType, Device, Tensor};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxFormat {
    Xyxy,
    Xywh,
    Cxcywh,
}
impl std::str::FromStr for BoxFormat {
    type Err = crate::errors::VisionError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "xyxy" => Ok(Self::Xyxy),
            "xywh" => Ok(Self::Xywh),
            "cxcywh" => Ok(Self::Cxcywh),
            _ => Err(invalid_argument("box_format must be xyxy, xywh, or cxcywh")),
        }
    }
}
impl BoxFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Xyxy => "xyxy",
            Self::Xywh => "xywh",
            Self::Cxcywh => "cxcywh",
        }
    }
}
#[derive(Clone, Debug)]
pub struct StructuredSample {
    pub image: Tensor,
    pub axis_order: ImageAxisOrder,
    pub boxes: Option<Tensor>,
    pub box_format: BoxFormat,
    pub labels: Option<Tensor>,
    /// Instance masks [N,H,W] or semantic map [H,W], nearest geometry, fill 0.
    pub masks: Option<Tensor>,
    /// Keypoint coordinates [N,K,2]. Coordinates may remain outside the canvas.
    pub keypoints: Option<Tensor>,
}
impl StructuredSample {
    pub fn size(&self) -> RivetResult<(usize, usize)> {
        let d = self.image.dims();
        if d.len() != 3 {
            return Err(invalid_shape("structured image must have rank 3"));
        }
        Ok(match self.axis_order {
            ImageAxisOrder::Hwc => (d[0], d[1]),
            ImageAxisOrder::Chw => (d[1], d[2]),
        })
    }
    pub fn validate(&self) -> RivetResult<()> {
        let (h, w) = self.size()?;
        let c = match self.axis_order {
            ImageAxisOrder::Hwc => self.image.dims()[2],
            ImageAxisOrder::Chw => self.image.dims()[0],
        };
        if h == 0
            || w == 0
            || c == 0
            || h > u32::MAX as usize
            || w > u32::MAX as usize
            || !matches!(self.image.dtype(), DType::U8 | DType::F32)
        {
            return Err(invalid_argument(
                "structured image requires nonempty uint8/float32 pixels",
            ));
        }
        let mut instances = None;
        let mut check = |tensor: &Tensor| -> RivetResult<()> {
            let n = tensor
                .dims()
                .first()
                .copied()
                .ok_or_else(|| invalid_shape("instance field requires a leading dimension"))?;
            if instances.is_some_and(|count| count != n) {
                return Err(invalid_shape(
                    "all target fields must have equal instance counts",
                ));
            }
            instances = Some(n);
            Ok(())
        };
        if let Some(boxes) = &self.boxes {
            if boxes.rank() != 2
                || boxes.dims()[1] != 4
                || boxes.dtype() != DType::F32
                || boxes.to_vec::<f32>()?.iter().any(|v| !v.is_finite())
            {
                return Err(invalid_argument("boxes require finite float32 [N,4]"));
            }
            check(boxes)?;
        }
        if let Some(labels) = &self.labels {
            if labels.rank() != 1 || labels.dtype() != DType::I64 {
                return Err(invalid_argument("instance labels require int64 [N]"));
            }
            check(labels)?;
        }
        if let Some(masks) = &self.masks {
            let semantic = masks.rank() == 2 && masks.dims() == [h, w];
            let instance = masks.rank() == 3 && masks.dims()[1..] == [h, w];
            if !(semantic || instance) || !matches!(masks.dtype(), DType::U8 | DType::I64) {
                return Err(invalid_shape(
                    "masks require uint8/int64 [H,W] or [N,H,W] matching image",
                ));
            }
            if instance {
                check(masks)?;
            }
        }
        if let Some(points) = &self.keypoints {
            if points.rank() != 3
                || points.dims()[2] != 2
                || points.dtype() != DType::F32
                || points.to_vec::<f32>()?.iter().any(|v| !v.is_finite())
            {
                return Err(invalid_argument("keypoints require finite float32 [N,K,2]"));
            }
            check(points)?;
        }
        Ok(())
    }
    pub fn xyxy(&self) -> RivetResult<Option<Tensor>> {
        self.boxes
            .as_ref()
            .map(|b| convert_boxes(b, self.box_format, BoxFormat::Xyxy))
            .transpose()
    }
    pub fn convert_boxes(mut self, format: BoxFormat) -> RivetResult<Self> {
        if let Some(b) = &self.boxes {
            self.boxes = Some(convert_boxes(b, self.box_format, format)?);
        }
        self.box_format = format;
        Ok(self)
    }
    pub fn clamp_boxes(mut self) -> RivetResult<Self> {
        let (h, w) = self.size()?;
        if let Some(b) = self.xyxy()? {
            let values = b.to_vec::<f32>()?;
            let mut out = values;
            for b in out.chunks_exact_mut(4) {
                b[0] = b[0].clamp(0.0, w as f32);
                b[2] = b[2].clamp(0.0, w as f32);
                b[1] = b[1].clamp(0.0, h as f32);
                b[3] = b[3].clamp(0.0, h as f32);
            }
            let boxes = Tensor::from_vec(out, b.dims(), &Device::Cpu)?;
            self.boxes = Some(convert_boxes(&boxes, BoxFormat::Xyxy, self.box_format)?);
        }
        Ok(self)
    }
    /// Clamp keypoints to valid pixel centers on the current image canvas.
    pub fn clamp_keypoints(mut self) -> RivetResult<Self> {
        if let Some(points) = &self.keypoints {
            let (h, w) = self.size()?;
            if h == 0 || w == 0 {
                return Err(invalid_shape(
                    "keypoint clamping requires a nonempty canvas",
                ));
            }
            let mut values = points.to_vec::<f32>()?;
            for point in values.chunks_exact_mut(2) {
                point[0] = point[0].clamp(0.0, (w - 1) as f32);
                point[1] = point[1].clamp(0.0, (h - 1) as f32);
            }
            self.keypoints = Some(Tensor::from_vec(values, points.dims(), points.device())?);
        }
        Ok(self)
    }

    /// Remove instances with any keypoint outside the current image canvas.
    pub fn sanitize_keypoints(self) -> RivetResult<Self> {
        let (h, w) = self.size()?;
        let points = self
            .keypoints
            .as_ref()
            .ok_or_else(|| invalid_argument("sanitize_keypoints requires keypoints"))?;
        let &[instances, count, 2] = points.dims() else {
            return Err(invalid_shape("keypoints require shape [N,K,2]"));
        };
        if instances == 0 || count == 0 {
            return Ok(self);
        }
        let values = points.to_vec::<f32>()?;
        let keep: Vec<usize> = values
            .chunks_exact(count * 2)
            .enumerate()
            .filter_map(|(i, group)| {
                group
                    .chunks_exact(2)
                    .all(|point| {
                        (0.0..w as f32).contains(&point[0]) && (0.0..h as f32).contains(&point[1])
                    })
                    .then_some(i)
            })
            .collect();
        self.retain_instances(&keep)
    }

    pub fn sanitize(self, min_size: f32, min_area: f32) -> RivetResult<Self> {
        if !min_size.is_finite() || min_size < 1.0 || !min_area.is_finite() || min_area < 1.0 {
            return Err(invalid_argument(
                "sanitize min_size and min_area must be finite and >=1",
            ));
        }
        let (h, w) = self.size()?;
        let boxes = self
            .xyxy()?
            .ok_or_else(|| invalid_argument("sanitize requires bounding boxes"))?;
        let values = boxes.to_vec::<f32>()?;
        let keep: Vec<usize> = values
            .chunks_exact(4)
            .enumerate()
            .filter_map(|(i, b)| {
                let bw = b[2] - b[0];
                let bh = b[3] - b[1];
                (bw >= min_size
                    && bh >= min_size
                    && bw * bh >= min_area
                    && b[0] >= 0.0
                    && b[1] >= 0.0
                    && b[2] <= w as f32
                    && b[3] <= h as f32)
                    .then_some(i)
            })
            .collect();
        self.retain_instances(&keep)
    }

    fn retain_instances(mut self, keep: &[usize]) -> RivetResult<Self> {
        self.boxes = self.boxes.map(|t| select_instances(&t, keep)).transpose()?;
        self.labels = self
            .labels
            .map(|t| select_instances(&t, keep))
            .transpose()?;
        self.masks = self
            .masks
            .map(|t| {
                if t.rank() == 2 {
                    Ok(t)
                } else {
                    select_instances(&t, keep)
                }
            })
            .transpose()?;
        self.keypoints = self
            .keypoints
            .map(|t| select_instances(&t, keep))
            .transpose()?;
        Ok(self)
    }
    pub fn random_iou_crop(
        self,
        cfg: &RandomIoUCropConfig,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        cfg.validate()?;
        let (h, w) = self.size()?;
        let boxes = self
            .xyxy()?
            .ok_or_else(|| invalid_argument("RandomIoUCrop requires boxes"))?;
        let values = boxes.to_vec::<f32>()?;
        if values.is_empty() {
            return Ok(self);
        }
        // Bound option retries to make impossible configurations terminate.
        for _ in 0..cfg.max_attempts {
            let threshold = cfg.options[rng.gen_range_usize(0..cfg.options.len())?];
            if threshold >= 1.0 {
                return Ok(self);
            }
            for _ in 0..cfg.trials {
                let cw = (w as f64
                    * (cfg.min_scale + rng.next_f64() * (cfg.max_scale - cfg.min_scale)))
                    as usize;
                let ch = (h as f64
                    * (cfg.min_scale + rng.next_f64() * (cfg.max_scale - cfg.min_scale)))
                    as usize;
                if cw == 0 || ch == 0 || cw > w || ch > h {
                    continue;
                }
                let ratio = cw as f64 / ch as f64;
                if ratio < cfg.min_aspect || ratio > cfg.max_aspect {
                    continue;
                }
                let x = ((w - cw) as f64 * rng.next_f64()) as usize;
                let y = ((h - ch) as f64 * rng.next_f64()) as usize;
                let mut keep = Vec::new();
                let mut max_iou = 0.0f64;
                for b in values.chunks_exact(4) {
                    let cx = (b[0] + b[2]) * 0.5;
                    let cy = (b[1] + b[3]) * 0.5;
                    let inside = cx > x as f32
                        && cx < (x + cw) as f32
                        && cy > y as f32
                        && cy < (y + ch) as f32;
                    keep.push(inside);
                    if inside {
                        let iw = (b[2].min((x + cw) as f32) - b[0].max(x as f32)).max(0.0);
                        let ih = (b[3].min((y + ch) as f32) - b[1].max(y as f32)).max(0.0);
                        let inter = (iw * ih) as f64;
                        let area = ((b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)) as f64;
                        max_iou = max_iou.max(inter / (area + (cw * ch) as f64 - inter));
                    }
                }
                if !keep.iter().any(|v| *v) || max_iou < threshold {
                    continue;
                }
                let mut out =
                    super::geometry::crop(self, x as u32, y as u32, cw as u32, ch as u32)?;
                let xy = out.xyxy()?.unwrap();
                let mut bv = xy.to_vec::<f32>()?;
                for (b, valid) in bv.chunks_exact_mut(4).zip(keep) {
                    if !valid {
                        b.fill(0.0);
                    }
                }
                let xy = Tensor::from_vec(bv, xy.dims(), &Device::Cpu)?;
                out.boxes = Some(convert_boxes(&xy, BoxFormat::Xyxy, out.box_format)?);
                return Ok(out);
            }
        }
        Ok(self)
    }
}
pub(super) fn convert_boxes(boxes: &Tensor, from: BoxFormat, to: BoxFormat) -> RivetResult<Tensor> {
    if from == to {
        return Ok(boxes.clone());
    }
    let values = boxes.to_vec::<f32>()?;
    let mut out = Vec::with_capacity(values.len());
    for b in values.chunks_exact(4) {
        let xy = match from {
            BoxFormat::Xyxy => [b[0], b[1], b[2], b[3]],
            BoxFormat::Xywh => [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            BoxFormat::Cxcywh => [
                b[0] - b[2] * 0.5,
                b[1] - b[3] * 0.5,
                b[0] + b[2] * 0.5,
                b[1] + b[3] * 0.5,
            ],
        };
        out.extend(match to {
            BoxFormat::Xyxy => xy,
            BoxFormat::Xywh => [xy[0], xy[1], xy[2] - xy[0], xy[3] - xy[1]],
            BoxFormat::Cxcywh => [
                (xy[0] + xy[2]) * 0.5,
                (xy[1] + xy[3]) * 0.5,
                xy[2] - xy[0],
                xy[3] - xy[1],
            ],
        });
    }
    Ok(Tensor::from_vec(out, boxes.dims(), &Device::Cpu)?)
}
fn select_instances(input: &Tensor, keep: &[usize]) -> RivetResult<Tensor> {
    let mut dims = input.dims().to_vec();
    dims[0] = keep.len();
    let item = input.dims()[1..].iter().product::<usize>();
    input
        .with_cpu_storage(|storage, layout| {
            macro_rules! select {
                ($v:expr,$ty:ty) => {{
                    let v = $v;
                    Tensor::from_exact_writer::<$ty, _, _>(dims.clone(), &Device::Cpu, |out| {
                        for &i in keep {
                            let view = input.narrow(0, i, 1)?;
                            let layout = view.layout();
                            if layout.is_contiguous() {
                                out.extend_from_slice(
                                    &v[layout.start_offset()..layout.start_offset() + item],
                                )?;
                            } else {
                                for offset in layout.strided_index() {
                                    out.write_next(
                                        *v.get(offset)
                                            .ok_or(rivet_core::Error::StorageOutOfBounds)?,
                                    )?;
                                }
                            }
                        }
                        Ok(())
                    })
                }};
            }
            let _ = layout;
            match storage {
                rivet_core::CpuStorageRef::U8(v) => select!(v, u8),
                rivet_core::CpuStorageRef::I64(v) => select!(v, i64),
                rivet_core::CpuStorageRef::F32(v) => select!(v, f32),
                _ => unreachable!(),
            }
        })
        .map_err(Into::into)
}
#[derive(Clone, Debug)]
pub struct RandomIoUCropConfig {
    pub min_scale: f64,
    pub max_scale: f64,
    pub min_aspect: f64,
    pub max_aspect: f64,
    pub options: Vec<f64>,
    pub trials: usize,
    pub max_attempts: usize,
}
impl Default for RandomIoUCropConfig {
    fn default() -> Self {
        Self {
            min_scale: 0.3,
            max_scale: 1.0,
            min_aspect: 0.5,
            max_aspect: 2.0,
            options: vec![0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            trials: 40,
            max_attempts: 100,
        }
    }
}
impl RandomIoUCropConfig {
    pub fn validate(&self) -> RivetResult<()> {
        if !self.min_scale.is_finite()
            || !self.max_scale.is_finite()
            || self.min_scale <= 0.0
            || self.max_scale < self.min_scale
            || self.max_scale > 1.0
            || !self.min_aspect.is_finite()
            || !self.max_aspect.is_finite()
            || self.min_aspect <= 0.0
            || self.max_aspect < self.min_aspect
            || self.trials == 0
            || self.max_attempts == 0
            || self.options.is_empty()
            || self.options.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(invalid_argument("invalid RandomIoUCrop configuration"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::TransformSequence;
    use crate::structured::StructuredPipeline;
    fn sample() -> StructuredSample {
        let s = StructuredSample {
            image: Tensor::from_vec(vec![0u8; 4 * 5 * 3], [4, 5, 3], &Device::Cpu).unwrap(),
            axis_order: ImageAxisOrder::Hwc,
            boxes: Some(
                Tensor::from_vec(
                    vec![-1f32, 0., 2., 3., 4., 3., 5., 4.],
                    [2, 4],
                    &Device::Cpu,
                )
                .unwrap(),
            ),
            box_format: BoxFormat::Xyxy,
            labels: Some(Tensor::from_vec(vec![7i64, 9], [2], &Device::Cpu).unwrap()),
            masks: Some(Tensor::from_vec(vec![1i64; 40], [2, 4, 5], &Device::Cpu).unwrap()),
            keypoints: Some(
                Tensor::from_vec(vec![0f32, 1., 4., 3.], [2, 1, 2], &Device::Cpu).unwrap(),
            ),
        };
        s.validate().unwrap();
        s
    }
    #[test]
    fn sanitize_filters_aligned_fields_with_strided_masks() {
        let s = sample();
        let s = s.sanitize(1., 1.).unwrap();
        assert_eq!(s.labels.unwrap().to_vec::<i64>().unwrap(), [9]);
        assert_eq!(s.masks.unwrap().dims(), [1, 4, 5]);
        assert_eq!(s.keypoints.unwrap().to_vec::<f32>().unwrap(), [4., 3.]);
    }
    #[test]
    fn crop_clamp_and_empty_instance_geometry() {
        let s = sample().clamp_boxes().unwrap();
        let p = StructuredPipeline::new(vec![s.clone()])
            .unwrap()
            .compose(TransformSequence::new().crop(0, 0, 3, 3));
        let s = p.apply(s, 0).unwrap().sanitize(1., 1.).unwrap();
        assert_eq!(s.labels.as_ref().unwrap().to_vec::<i64>().unwrap(), [7]);
        let s = s.sanitize(20., 1.).unwrap();
        let s = StructuredPipeline::new(vec![])
            .unwrap()
            .compose(TransformSequence::new().horizontal_flip())
            .apply(s, 0)
            .unwrap();
        assert_eq!(s.boxes.unwrap().dims(), [0, 4]);
        assert_eq!(s.masks.unwrap().dims(), [0, 3, 3]);
    }
    #[test]
    fn impossible_iou_configuration_is_bounded() {
        let s = sample().clamp_boxes().unwrap();
        let config = RandomIoUCropConfig {
            options: vec![0.99],
            min_scale: 0.2,
            max_scale: 0.2,
            trials: 2,
            max_attempts: 2,
            ..Default::default()
        };
        let mut rng = RandomStream::from_seed(3);
        let out = s.clone().random_iou_crop(&config, &mut rng).unwrap();
        assert!(out.image.same_storage(&s.image));
    }
}
