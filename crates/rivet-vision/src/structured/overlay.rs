use super::StructuredSample;
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::ImageAxisOrder;
use rivet_core::{CpuStorageRef, DType, Tensor, WithDType};
use rivet_data::random::RandomStream;

#[derive(Clone, Debug)]
pub struct OverlayElement {
    pub image: Tensor,
    pub mask: Option<Tensor>,
    /// Selected placement in destination (x, y) pixels.
    pub offset: Option<(i64, i64)>,
    /// Optional normalized destination XYXY box, resizing the overlay.
    pub bbox: Option<[f64; 4]>,
    pub mask_id: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct OverlayElementsConfig {
    pub elements: Vec<OverlayElement>,
    pub min_visibility: f64,
}

impl OverlayElementsConfig {
    pub fn validate(&self) -> RivetResult<()> {
        if self.elements.len() > 64 || !(0.0..=1.0).contains(&self.min_visibility) {
            return Err(invalid_argument("invalid OverlayElements configuration"));
        }
        for e in &self.elements {
            let &[h, w, c] = e.image.dims() else {
                return Err(invalid_shape("overlay image must be HWC"));
            };
            if h == 0
                || w == 0
                || c == 0
                || !matches!(e.image.dtype(), DType::U8 | DType::F32)
                || e.mask
                    .as_ref()
                    .is_some_and(|m| m.dims() != [h, w] || m.dtype() != DType::U8)
                || (e.offset.is_some() && e.bbox.is_some())
                || e.bbox.is_some_and(|[l, t, r, b]| {
                    !(l >= 0.0 && l < r && r <= 1.0 && t >= 0.0 && t < b && b <= 1.0)
                })
            {
                return Err(invalid_argument("invalid overlay image, mask or placement"));
            }
        }
        Ok(())
    }
}

impl StructuredSample {
    pub fn overlay_elements(
        mut self,
        config: &OverlayElementsConfig,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        config.validate()?;
        if config.elements.is_empty() {
            return Ok(self);
        }
        let (h, w) = self.size()?;
        let base = match self.axis_order {
            ImageAxisOrder::Hwc => self.image.clone(),
            ImageAxisOrder::Chw => self.image.permute(&[1, 2, 0])?,
        };
        let c = base.dims()[2];
        let mut pixels = vec![None; h * w];
        let mut mask_ids = config
            .elements
            .iter()
            .any(|e| e.mask_id.is_some())
            .then(|| vec![None; h * w]);
        if mask_ids.is_some() && self.masks.as_ref().is_some_and(|m| m.rank() != 2) {
            return Err(invalid_argument(
                "OverlayElements mask_id requires a semantic mask",
            ));
        }
        for (index, e) in config.elements.iter().enumerate() {
            let &[eh, ew, ec] = e.image.dims() else {
                unreachable!()
            };
            if ec != c || e.image.dtype() != base.dtype() {
                return Err(invalid_argument(
                    "overlay dtype and channels must match the image",
                ));
            }
            let mask = e.mask.as_ref().map(Tensor::to_vec::<u8>).transpose()?;
            let (x, y, ow, oh) = if let Some([l, t, r, b]) = e.bbox {
                let (x, y) = (
                    (l * w as f64).floor() as usize,
                    (t * h as f64).floor() as usize,
                );
                let (right, bottom) = (
                    (r * w as f64).ceil() as usize,
                    (b * h as f64).ceil() as usize,
                );
                (x as i64, y as i64, right - x, bottom - y)
            } else {
                let (x, y) = if let Some(offset) = e.offset {
                    offset
                } else {
                    (
                        rng.gen_range_usize(0..w.saturating_sub(ew) + 1)? as i64,
                        rng.gen_range_usize(0..h.saturating_sub(eh) + 1)? as i64,
                    )
                };
                (x, y, ew, eh)
            };
            let clip = |start: i64, len: usize, end: usize| {
                let start = i128::from(start);
                start.clamp(0, end as i128) as usize
                    ..(start + len as i128).clamp(0, end as i128) as usize
            };
            for dy in clip(y, oh, h) {
                let sy =
                    (((dy as i128 - i128::from(y)) as u128) * eh as u128 / oh as u128) as usize;
                for dx in clip(x, ow, w) {
                    let sx =
                        (((dx as i128 - i128::from(x)) as u128) * ew as u128 / ow as u128) as usize;
                    let src = sy * ew + sx;
                    if mask.as_ref().is_none_or(|m| m[src] != 0) {
                        pixels[dy * w + dx] = Some((index, src));
                        if let (Some(ids), Some(id)) = (&mut mask_ids, e.mask_id) {
                            ids[dy * w + dx] = Some(id);
                        }
                    }
                }
            }
        }
        if pixels.iter().all(Option::is_none) {
            return Ok(self);
        }
        let base = base.contiguous()?;
        let output = base.with_cpu_storage(|storage, layout| match storage {
            CpuStorageRef::U8(v) => Ok(compose(
                &base,
                crate::transforms::contiguous_values(v, layout)?,
                &config.elements,
                &pixels,
                c,
            )),
            CpuStorageRef::F32(v) => Ok(compose(
                &base,
                crate::transforms::contiguous_values(v, layout)?,
                &config.elements,
                &pixels,
                c,
            )),
            _ => unreachable!("validated overlay dtype"),
        })??;
        self.image = match self.axis_order {
            ImageAxisOrder::Hwc => output,
            ImageAxisOrder::Chw => output.permute(&[2, 0, 1])?,
        };
        if let Some(ids) = mask_ids {
            if ids.iter().any(Option::is_some) {
                let mask = if let Some(mask) = self.masks.take() {
                    mask
                } else {
                    Tensor::zeros([h, w], DType::I64, base.device())?
                };
                let mask = mask.contiguous()?;
                self.masks = Some(
                    mask.with_cpu_storage(|storage, layout| {
                        let start = layout.start_offset();
                        match storage {
                            CpuStorageRef::I64(v) => Tensor::from_exact_iter(
                                v[start..start + h * w]
                                    .iter()
                                    .zip(&ids)
                                    .map(|(&v, id)| id.unwrap_or(v)),
                                [h, w],
                                mask.device(),
                            ),
                            CpuStorageRef::U8(v) => {
                                Tensor::from_exact_writer([h, w], mask.device(), |out| {
                                    for (&v, id) in v[start..start + h * w].iter().zip(&ids) {
                                        let v = u8::try_from(id.unwrap_or(i64::from(v)))
                                            .map_err(|_| rivet_core::Error::StorageOutOfBounds)?;
                                        out.write_next(v)?;
                                    }
                                    Ok(())
                                })
                            }
                            _ => unreachable!(),
                        }
                    })
                    .map_err(|e| {
                        invalid_argument(format!(
                            "invalid OverlayElements mask_id for mask dtype: {e}"
                        ))
                    })?,
                );
            }
        }
        let instance_fill = self
            .masks
            .as_ref()
            .filter(|mask| mask.rank() == 3)
            .map(|_| 0);
        super::geometry::apply_occlusion_with_visibility(
            self,
            h,
            w,
            instance_fill,
            config.min_visibility,
            |y, x| pixels[y * w + x].is_some(),
        )
    }
}

fn compose<T: WithDType>(
    base: &Tensor,
    original: &[T],
    elements: &[OverlayElement],
    pixels: &[Option<(usize, usize)>],
    c: usize,
) -> RivetResult<Tensor> {
    let overlays = elements
        .iter()
        .map(|e| e.image.to_vec::<T>())
        .collect::<rivet_core::Result<Vec<_>>>()?;
    Ok(Tensor::from_exact_writer(
        base.dims(),
        base.device(),
        |out| {
            for (i, location) in pixels.iter().enumerate() {
                let (data, offset) = match *location {
                    Some((layer, p)) => (overlays[layer].as_slice(), p * c),
                    None => (original, i * c),
                };
                out.extend_from_slice(&data[offset..offset + c])?;
            }
            Ok(())
        },
    )?)
}
