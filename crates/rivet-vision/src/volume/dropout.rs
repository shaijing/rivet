use super::VolumeSample;
use super::crop_pad::crop_offset;
use crate::errors::{RivetResult, invalid_argument};
use rivet_core::{CpuStorageRef, DType, Tensor};

#[derive(Clone, Debug)]
pub struct CoarseDropout3DConfig {
    pub num_holes_range: [usize; 2],
    pub hole_ranges: [[f64; 2]; 3],
    pub fill: Vec<f32>,
    pub fill_mask: Option<i64>,
    pub remove_invisible: bool,
}

#[derive(Clone, Copy)]
struct VolumeHole {
    start: [usize; 3],
    end: [usize; 3],
}

impl VolumeSample {
    pub fn coarse_dropout(
        mut self,
        config: &CoarseDropout3DConfig,
        rng: &mut rivet_data::random::RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        let shape = self.spatial_shape()?;
        let channels = self.volume.dims().get(3).copied().unwrap_or(1);
        let [min, max] = config.num_holes_range;
        if min > max
            || max == usize::MAX
            || config
                .hole_ranges
                .iter()
                .any(|&[lo, hi]| !(0.0..=1.0).contains(&lo) || !(lo..=1.0).contains(&hi))
        {
            return Err(invalid_argument(
                "invalid CoarseDropout3D hole count or fractional range",
            ));
        }
        if (config.fill.len() != 1 && config.fill.len() != channels)
            || config.fill.iter().any(|v| !v.is_finite())
        {
            return Err(invalid_argument(
                "CoarseDropout3D fill requires a finite scalar or one value per channel",
            ));
        }
        if self.mask.as_ref().is_some_and(|m| m.dtype() == DType::U8)
            && config.fill_mask.is_some_and(|v| u8::try_from(v).is_err())
        {
            return Err(invalid_argument(
                "CoarseDropout3D uint8 mask fill must be in [0, 255]",
            ));
        }
        let count = if min == max {
            min
        } else {
            rng.gen_range_usize(min..max + 1)?
        };
        let mut holes = Vec::new();
        holes.try_reserve(count).map_err(|_| {
            invalid_argument("CoarseDropout3D hole count exceeds allocation limits")
        })?;
        for _ in 0..count {
            let size: [usize; 3] = std::array::from_fn(|i| {
                let [lo, hi] = config.hole_ranges[i];
                ((shape[i] as f64 * (lo + (hi - lo) * rng.next_f64())).ceil() as usize)
                    .min(shape[i])
            });
            if size.contains(&0) {
                continue;
            }
            let mut start = [0; 3];
            for i in 0..3 {
                start[i] = crop_offset(shape[i] - size[i], Some(rng))?;
            }
            holes.push(VolumeHole {
                start,
                end: std::array::from_fn(|i| start[i] + size[i]),
            });
        }
        if holes.is_empty() {
            return Ok(self);
        }
        self.volume = erase_volume(&self.volume, shape, &holes, &config.fill, None)?;
        if let (Some(mask), Some(fill)) = (&self.mask, config.fill_mask) {
            self.mask = Some(erase_volume(mask, shape, &holes, &[], Some(fill))?);
        }
        if let Some(points) = &self.keypoints
            && config.remove_invisible
        {
            let mut coords = points.to_vec::<f32>()?;
            let mut keep = 0;
            for first in (0..coords.len()).step_by(3) {
                let p = [coords[first + 2], coords[first + 1], coords[first]];
                if !holes.iter().any(|hole| {
                    (0..3).all(|i| {
                        (hole.start[i] as f64..hole.end[i] as f64).contains(&f64::from(p[i]))
                    })
                }) {
                    coords.copy_within(first..first + 3, keep);
                    keep += 3;
                }
            }
            coords.truncate(keep);
            self.keypoints = Some(Tensor::from_vec(coords, [keep / 3, 3], points.device())?);
        }
        Ok(self)
    }
}

fn erase_volume(
    input: &Tensor,
    [depth, height, width]: [usize; 3],
    holes: &[VolumeHole],
    fill: &[f32],
    mask_fill: Option<i64>,
) -> RivetResult<Tensor> {
    let input = input.contiguous()?;
    let channels = input.dims().get(3).copied().unwrap_or(1);
    let mut intervals = Vec::with_capacity(holes.len());
    Ok(input.with_cpu_storage(|storage, layout| {
        macro_rules! erase {
            ($pixels:expr, $value:expr) => {{
                let fill: Vec<_> = (0..channels).map($value).collect();
                Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                    for z in 0..depth {
                        for y in 0..height {
                            intervals.clear();
                            intervals.extend(holes.iter().filter_map(|h| {
                                ((h.start[0]..h.end[0]).contains(&z)
                                    && (h.start[1]..h.end[1]).contains(&y))
                                .then_some((h.start[2], h.end[2]))
                            }));
                            intervals.sort_unstable();
                            let row = layout.start_offset() + (z * height + y) * width * channels;
                            let mut x = 0;
                            for &(start, end) in &intervals {
                                if start > x {
                                    out.extend_from_slice(
                                        &$pixels[row + x * channels..row + start * channels],
                                    )?;
                                }
                                for _ in x.max(start)..end {
                                    out.extend_from_slice(&fill)?;
                                }
                                x = x.max(end);
                            }
                            out.extend_from_slice(
                                &$pixels[row + x * channels..row + width * channels],
                            )?;
                        }
                    }
                    Ok(())
                })
            }};
        }
        let channel_fill = |c| fill[if fill.len() == 1 { 0 } else { c }];
        match storage {
            CpuStorageRef::U8(v) => erase!(v, |c| mask_fill
                .map_or_else(|| channel_fill(c).round() as u8, |v| v as u8)),
            CpuStorageRef::F32(v) => erase!(v, channel_fill),
            CpuStorageRef::I64(v) => erase!(v, |_| mask_fill.expect("validated mask fill")),
            _ => unreachable!("validated volume dropout dtype"),
        }
    })?)
}
