use super::{BoxFormat, StructuredSample, geometry};
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::ImageAxisOrder;
use crate::transforms::{InterpolationMode, PadConfig};
use rivet_core::{DType, Tensor};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug)]
pub enum MosaicFit {
    Cover,
    Contain,
}

#[derive(Clone, Debug)]
pub struct MosaicConfig {
    pub donors: Vec<StructuredSample>,
    pub grid_yx: (u32, u32),
    pub cell_shape: (u32, u32),
    pub target_size: Option<(u32, u32)>,
    pub fit_mode: MosaicFit,
    pub center_range: (f64, f64),
    pub interpolation: InterpolationMode,
    pub fill: f32,
    pub fill_mask: i64,
}

impl MosaicConfig {
    pub fn canvas(&self) -> RivetResult<(u32, u32)> {
        let (rows, cols) = self.grid_yx;
        let (h, w) = self.cell_shape;
        match (rows.checked_mul(h), cols.checked_mul(w)) {
            (Some(h), Some(w)) if h > 0 && w > 0 => Ok((h, w)),
            _ => Err(invalid_shape(
                "Mosaic canvas must have positive axes that fit u32",
            )),
        }
    }
    pub fn validate(&self) -> RivetResult<()> {
        let (h, w) = self.canvas()?;
        let (lo, hi) = self.center_range;
        if !matches!(self.grid_yx.0.checked_mul(self.grid_yx.1), Some(1..=64))
            || self
                .target_size
                .is_some_and(|(th, tw)| th == 0 || tw == 0 || th > h || tw > w)
            || !(lo >= 0.0 && lo <= hi && hi <= 1.0)
            || !self.fill.is_finite()
        {
            return Err(invalid_argument("invalid Mosaic geometry or fill"));
        }
        Ok(())
    }

    fn fit(&self, mut s: StructuredSample) -> RivetResult<StructuredSample> {
        let (h, w) = s.size()?;
        let (ch, cw) = self.cell_shape;
        let (ch64, cw64) = (f64::from(ch), f64::from(cw));
        let (sx, sy) = (cw64 / w as f64, ch64 / h as f64);
        let scale = match self.fit_mode {
            MosaicFit::Cover => sx.max(sy),
            MosaicFit::Contain => sx.min(sy),
        };
        let round = |n: usize| -> RivetResult<u32> {
            let v = (n as f64 * scale).round().max(1.0);
            if v > f64::from(u32::MAX) {
                return Err(invalid_shape("Mosaic fitted axis exceeds u32"));
            }
            Ok(v as u32)
        };
        let (rw, rh) = (round(w)?, round(h)?);
        if (rw as usize, rh as usize) != (w, h) {
            s = geometry::resize(s, rw, rh, self.interpolation)?;
        }
        match self.fit_mode {
            MosaicFit::Cover if rw != cw || rh != ch => {
                let (x, y) = ((rw - cw) as usize / 2, (rh - ch) as usize / 2);
                let boxes = s.xyxy()?.map(|b| b.to_vec::<f32>()).transpose()?;
                geometry::crop_visible_instances(
                    s,
                    boxes.as_deref(),
                    [x, y, x + cw as usize, y + ch as usize],
                )
            }
            MosaicFit::Contain if rw != cw || rh != ch => {
                let (x, y) = ((cw - rw) / 2, (ch - rh) / 2);
                geometry::pad_with_mask_fill(
                    s,
                    &PadConfig::with_sides(x, y, cw - rw - x, ch - rh - y).with_fill(self.fill),
                    self.fill_mask,
                )
            }
            _ => Ok(s),
        }
    }
}

fn join_grid(tiles: &[&Tensor], cols: usize, y_axis: usize, x_axis: usize) -> RivetResult<Tensor> {
    let rows = tiles
        .chunks_exact(cols)
        .map(|r| {
            if r.len() == 1 {
                Ok(r[0].clone())
            } else {
                Tensor::cat(r, x_axis)
            }
        })
        .collect::<rivet_core::Result<Vec<_>>>()?;
    if rows.len() == 1 {
        Ok(rows.into_iter().next().expect("one row"))
    } else {
        Ok(Tensor::cat(&rows.iter().collect::<Vec<_>>(), y_axis)?)
    }
}

impl StructuredSample {
    pub fn mosaic(self, config: &MosaicConfig, rng: &mut RandomStream) -> RivetResult<Self> {
        config.validate()?;
        let (rows, cols) = (config.grid_yx.0 as usize, config.grid_yx.1 as usize);
        let (ch, cw) = (config.cell_shape.0 as usize, config.cell_shape.1 as usize);
        let (h, w) = (rows * ch, cols * cw);
        let mut indices: Vec<_> = (0..config.donors.len()).collect();
        let used = (rows * cols - 1).min(indices.len());
        for i in 0..used {
            let j = rng.gen_range_usize(i..indices.len())?;
            indices.swap(i, j);
        }
        let axis = self.axis_order;
        let channels = if axis == ImageAxisOrder::Hwc {
            self.image.dims()[2]
        } else {
            self.image.dims()[0]
        };
        let dtype = self.image.dtype();
        let box_format = self.box_format;
        let mut tiles = Vec::with_capacity(rows * cols);
        for i in 0..rows * cols {
            let item = if i > 0 && i <= used {
                config.donors[indices[i - 1]].clone()
            } else {
                self.clone()
            };
            let c = if item.axis_order == ImageAxisOrder::Hwc {
                item.image.dims()[2]
            } else {
                item.image.dims()[0]
            };
            if c != channels || item.image.dtype() != dtype || item.axis_order != axis {
                return Err(invalid_argument(
                    "Mosaic samples must share image dtype, layout and channel count",
                ));
            }
            tiles.push(config.fit(item)?);
        }
        let (ya, xa) = if axis == ImageAxisOrder::Hwc {
            (0, 1)
        } else {
            (1, 2)
        };
        let mut output = StructuredSample {
            image: join_grid(
                &tiles.iter().map(|s| &s.image).collect::<Vec<_>>(),
                cols,
                ya,
                xa,
            )?,
            box_format,
            axis_order: axis,
            boxes: None,
            labels: None,
            masks: None,
            keypoints: None,
        };
        let semantic = tiles
            .iter()
            .filter_map(|s| s.masks.as_ref())
            .filter(|m| m.rank() == 2)
            .max_by_key(|m| m.dtype() == DType::I64);
        let instance = tiles
            .iter()
            .filter_map(|s| s.masks.as_ref())
            .filter(|m| m.rank() == 3)
            .max_by_key(|m| m.dtype() == DType::I64);
        if semantic.is_some() && instance.is_some() {
            return Err(invalid_argument(
                "Mosaic cannot mix semantic and instance masks",
            ));
        }
        let has_boxes = tiles.iter().any(|s| s.boxes.is_some());
        let has_labels = tiles.iter().any(|s| s.labels.is_some());
        let point_count = tiles
            .iter()
            .find_map(|s| s.keypoints.as_ref().map(|t| t.dims()[1]));
        let mut boxes = Vec::new();
        let mut labels = Vec::new();
        let mut points = Vec::new();
        let mut masks = Vec::new();
        for (i, tile) in tiles.iter().enumerate() {
            let (x, y) = (i % cols * cw, i / cols * ch);
            let instance_count = tile
                .boxes
                .as_ref()
                .or(tile.labels.as_ref())
                .or(tile.keypoints.as_ref())
                .or(tile.masks.as_ref().filter(|m| m.rank() == 3))
                .map_or(0, |t| t.dims()[0]);
            if instance_count > 0
                && ((has_boxes && tile.boxes.is_none())
                    || (has_labels && tile.labels.is_none())
                    || (point_count.is_some() && tile.keypoints.is_none())
                    || (instance.is_some() && tile.masks.is_none()))
            {
                return Err(invalid_argument(
                    "Mosaic instance fields must match across tiles with instances",
                ));
            }
            if let Some(b) = tile.xyxy()? {
                for rect in b.to_vec::<f32>()?.chunks_exact(4) {
                    boxes.extend([
                        rect[0] + x as f32,
                        rect[1] + y as f32,
                        rect[2] + x as f32,
                        rect[3] + y as f32,
                    ]);
                }
            }
            if let Some(l) = &tile.labels {
                labels.extend(l.to_vec::<i64>()?);
            }
            if let Some(k) = &tile.keypoints {
                if Some(k.dims()[1]) != point_count {
                    return Err(invalid_argument(
                        "Mosaic samples must share keypoint count per instance",
                    ));
                }
                for p in k.to_vec::<f32>()?.chunks_exact(2) {
                    points.extend([p[0] + x as f32, p[1] + y as f32]);
                }
            }
            if let Some(template) = semantic {
                let mask = tile
                    .masks
                    .clone()
                    .map(|m| m.to_dtype(template.dtype()))
                    .transpose()?;
                let mask = if let Some(mask) = mask {
                    mask
                } else {
                    if template.dtype() == DType::I64 {
                        Tensor::full(config.fill_mask, [ch, cw], template.device())?
                    } else {
                        let fill = u8::try_from(config.fill_mask).map_err(|_| {
                            invalid_argument("Mosaic uint8 mask fill is outside [0, 255]")
                        })?;
                        Tensor::full(fill, [ch, cw], template.device())?
                    }
                };
                masks.push(mask);
            } else if let Some(template) = instance
                && let Some(mask) = &tile.masks
            {
                let mask = mask.to_dtype(template.dtype())?;
                masks.push(geometry::masks_map_fill(&mask, h, w, 0, |yy, xx| {
                    Some((yy.checked_sub(y)?, xx.checked_sub(x)?))
                })?);
            }
        }
        if has_boxes {
            let b = Tensor::from_slice(&boxes, [boxes.len() / 4, 4], output.image.device())?;
            output.boxes = Some(super::annotations::convert_boxes(
                &b,
                BoxFormat::Xyxy,
                box_format,
            )?);
        }
        if has_labels {
            let n = labels.len();
            output.labels = Some(Tensor::from_vec(labels, [n], output.image.device())?);
        }
        if let Some(k) = point_count {
            let n = tiles
                .iter()
                .filter_map(|s| s.keypoints.as_ref())
                .map(|p| p.dims()[0])
                .sum::<usize>();
            output.keypoints = Some(Tensor::from_vec(points, [n, k, 2], output.image.device())?);
        }
        if semantic.is_some() {
            output.masks = Some(join_grid(&masks.iter().collect::<Vec<_>>(), cols, 0, 1)?);
        } else if instance.is_some() {
            output.masks = Some(Tensor::cat(&masks.iter().collect::<Vec<_>>(), 0)?);
        }
        let (th, tw) = config.target_size.unwrap_or((h as u32, w as u32));
        if (th as usize, tw as usize) == (h, w) {
            return Ok(output);
        }
        let (lo, hi) = config.center_range;
        let mut start =
            |extra: usize| ((lo + rng.next_f64() * (hi - lo)) * extra as f64).round() as usize;
        let (x, y) = (start(w - tw as usize), start(h - th as usize));
        geometry::crop_visible_instances(
            output,
            has_boxes.then_some(boxes.as_slice()),
            [x, y, x + tw as usize, y + th as usize],
        )
    }
}
