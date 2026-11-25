use super::StructuredSample;
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use rivet_core::{CpuStorageRef, Tensor};
use rivet_data::random::RandomStream;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskDropoutConfig {
    pub max_objects_range: (u32, u32),
    pub fill: f32,
    pub fill_mask: i64,
}

impl MaskDropoutConfig {
    pub fn validate(&self) -> RivetResult<()> {
        let (lo, hi) = self.max_objects_range;
        if lo > hi || !self.fill.is_finite() {
            return Err(invalid_argument("invalid MaskDropout configuration"));
        }
        Ok(())
    }
}

impl StructuredSample {
    pub fn mask_dropout(
        mut self,
        config: &MaskDropoutConfig,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        config.validate()?;
        let Some(mask) = &self.masks else {
            return Ok(self);
        };
        let (h, w) = self.size()?;
        let instance = mask.rank() == 3;
        let (mut objects, pixels) = object_pixels(mask, h, w)?;
        let (lo, hi) = config.max_objects_range;
        let count =
            (lo as usize + rng.gen_range_usize(0..(hi - lo) as usize + 1)?).min(objects.len());
        if count == 0 {
            return Ok(self);
        }
        for i in 0..count {
            let j = rng.gen_range_usize(i..objects.len())?;
            objects.swap(i, j);
        }
        let selected: BTreeSet<_> = objects[..count].iter().copied().collect();
        let mut covered = vec![false; h * w];
        for (index, ids) in pixels.chunks_exact(h * w).enumerate() {
            if instance && !selected.contains(&(index as i64)) {
                continue;
            }
            for (covered, &id) in covered.iter_mut().zip(ids) {
                *covered |= id != 0 && (instance || selected.contains(&id));
            }
        }
        if instance {
            let keep: Vec<_> = (0..mask.dims()[0])
                .filter(|&i| !selected.contains(&(i as i64)))
                .collect();
            self = self.retain_instances(&keep)?;
        }
        let dropped = |y: usize, x: usize| covered[y * w + x];
        self.image = crate::transforms::replace_spatial_pixels(
            &self.image,
            self.axis_order,
            h,
            w,
            [config.fill; 2],
            |y, x| dropped(y, x).then_some(false),
        )?;
        super::geometry::apply_occlusion(self, h, w, Some(config.fill_mask), dropped)
    }
}

pub(super) fn object_pixels(
    mask: &Tensor,
    h: usize,
    w: usize,
) -> RivetResult<(Vec<i64>, Vec<i64>)> {
    let pixels = mask
        .with_cpu_storage(|storage, layout| {
            let mut pixels = Vec::with_capacity(layout.checked_elem_count()?);
            let start = layout.start_offset();
            let (n, sn, sy, sx) = match (mask.dims(), layout.stride()) {
                ([_, _], [sy, sx]) => (1, 0, *sy, *sx),
                ([n, _, _], [sn, sy, sx]) => (*n, *sn, *sy, *sx),
                _ => return Err(rivet_core::Error::StorageOutOfBounds),
            };
            macro_rules! read {
                ($values:expr) => {
                    for i in 0..n {
                        for y in 0..h {
                            let row = start + i * sn + y * sy;
                            for x in 0..w {
                                pixels.push(i64::from($values[row + x * sx]));
                            }
                        }
                    }
                };
            }
            match storage {
                CpuStorageRef::U8(values) => {
                    read!(values);
                }
                CpuStorageRef::I64(values) => {
                    read!(values);
                }
                _ => return Err(rivet_core::Error::StorageOutOfBounds),
            }
            Ok(pixels)
        })
        .map_err(|e| invalid_shape(format!("invalid MaskDropout mask: {e}")))?;
    let ids = if mask.rank() == 2 {
        pixels
            .iter()
            .copied()
            .filter(|&v| v != 0)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    } else {
        pixels
            .chunks_exact(h * w)
            .enumerate()
            .filter_map(|(i, plane)| plane.iter().any(|&v| v != 0).then_some(i as i64))
            .collect()
    };
    Ok((ids, pixels))
}
