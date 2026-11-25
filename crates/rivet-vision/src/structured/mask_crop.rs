use super::StructuredSample;
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use rivet_core::{CpuStorageRef, Tensor};
use rivet_data::random::RandomStream;

#[derive(Clone, Debug, PartialEq)]
pub struct CropNonEmptyMaskConfig {
    pub width: u32,
    pub height: u32,
    pub ignore_values: Vec<i64>,
    pub ignore_channels: Vec<usize>,
}

impl CropNonEmptyMaskConfig {
    pub fn validate(&self) -> RivetResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(invalid_argument(
                "CropNonEmptyMaskIfExists requires positive dimensions",
            ));
        }
        Ok(())
    }
}

impl StructuredSample {
    pub fn crop_non_empty_mask_if_exists(
        self,
        config: &CropNonEmptyMaskConfig,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        config.validate()?;
        let (height, width) = self.size()?;
        let (crop_w, crop_h) = (config.width as usize, config.height as usize);
        if crop_w > width || crop_h > height {
            return Err(invalid_shape(
                "CropNonEmptyMaskIfExists exceeds image dimensions",
            ));
        }
        let point = match &self.masks {
            Some(mask) => mask_point(mask, config, rng)?,
            None => None,
        };
        let mut origin = |point: Option<usize>, crop: usize, limit: usize| {
            let (lo, hi) = point.map_or((0, limit - crop), |p| {
                ((p + 1).saturating_sub(crop), p.min(limit - crop))
            });
            rng.gen_range_usize(lo..hi + 1)
        };
        let x = origin(point.map(|(x, _)| x), crop_w, width)?;
        let y = origin(point.map(|(_, y)| y), crop_h, height)?;
        let mut sample = self;
        if let Some(boxes) = sample.xyxy()? {
            let values = boxes.to_vec::<f32>()?;
            let keep: Vec<_> = values
                .chunks_exact(4)
                .enumerate()
                .filter_map(|(i, b)| {
                    (b[0].max(x as f32) < b[2].min((x + crop_w) as f32)
                        && b[1].max(y as f32) < b[3].min((y + crop_h) as f32))
                    .then_some(i)
                })
                .collect();
            if keep.len() != values.len() / 4 {
                sample = sample.retain_instances(&keep)?;
            }
        }
        super::geometry::crop(sample, x as u32, y as u32, config.width, config.height)
    }
}

fn mask_point(
    mask: &Tensor,
    config: &CropNonEmptyMaskConfig,
    rng: &mut RandomStream,
) -> RivetResult<Option<(usize, usize)>> {
    let mask = if mask.rank() == 2 {
        mask.unsqueeze(0)?
    } else {
        mask.clone()
    };
    let [planes, height, width] = *mask.dims() else {
        return Err(invalid_shape(
            "CropNonEmptyMaskIfExists expects a semantic or instance mask",
        ));
    };
    let chosen: Vec<_> = (0..planes)
        .filter(|i| !config.ignore_channels.contains(i))
        .collect();
    if chosen.is_empty() {
        return Ok(None);
    }
    let point = mask.with_cpu_storage(|storage, layout| {
        let [sn, sy, sx] = *layout.stride() else {
            unreachable!()
        };
        let start = layout.start_offset();
        macro_rules! sample {
            ($active:expr) => {{
                let active = $active;
                let eligible_points = || {
                    (0..height)
                        .flat_map(|y| (0..width).map(move |x| (x, y)))
                        .filter(|&(x, y)| {
                            let offset = start + y * sy + x * sx;
                            chosen.iter().any(|&i| active(offset + i * sn))
                        })
                };
                let count = eligible_points().count();
                if count == 0 {
                    None
                } else {
                    let index = rng
                        .gen_range_usize(0..count)
                        .expect("positive eligible count");
                    eligible_points().nth(index)
                }
            }};
        }
        let eligible = |v| v != 0 && !config.ignore_values.contains(&v);
        Ok(match storage {
            CpuStorageRef::U8(values) => sample!(|i| eligible(i64::from(values[i]))),
            CpuStorageRef::I64(values) => sample!(|i| eligible(values[i])),
            _ => unreachable!("validated masks"),
        })
    })?;
    Ok(point)
}
