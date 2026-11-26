use super::{Resize3DConfig, VolumeSample};
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use rivet_core::{CpuStorageRef, DType, Tensor, WithDType};

#[derive(Clone, Debug, PartialEq)]
pub struct Pad3DConfig {
    /// Front/back, top/bottom, left/right.
    pub padding: [usize; 6],
    pub fill: Vec<f32>,
    pub fill_mask: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PadIfNeeded3DConfig {
    pub min_zyx: Option<[usize; 3]>,
    pub pad_divisor_zyx: Option<[usize; 3]>,
    pub random_position: bool,
    pub fill: Vec<f32>,
    pub fill_mask: i64,
}

impl VolumeSample {
    pub fn pad_if_needed(
        self,
        config: &PadIfNeeded3DConfig,
        rng: &mut rivet_data::random::RandomStream,
    ) -> RivetResult<Self> {
        let shape = self.spatial_shape()?;
        if config.min_zyx.is_none() && config.pad_divisor_zyx.is_none()
            || config.min_zyx.is_some_and(|s| s.contains(&0))
            || config.pad_divisor_zyx.is_some_and(|s| s.contains(&0))
        {
            return Err(invalid_argument(
                "PadIfNeeded3D requires positive minimum axes or divisors",
            ));
        }
        let mins = config.min_zyx.unwrap_or([1; 3]);
        let divisors = config.pad_divisor_zyx.unwrap_or([1; 3]);
        let mut padding = [0; 6];
        for i in 0..3 {
            let size = shape[i].max(mins[i]);
            let output = size
                .div_ceil(divisors[i])
                .checked_mul(divisors[i])
                .ok_or_else(|| invalid_shape("PadIfNeeded3D axis overflow"))?;
            let extra = output - shape[i];
            let before = if config.random_position && extra > 0 {
                let end = extra
                    .checked_add(1)
                    .ok_or_else(|| invalid_shape("PadIfNeeded3D axis overflow"))?;
                rng.gen_range_usize(0..end)?
            } else {
                extra / 2
            };
            padding[i * 2] = before;
            padding[i * 2 + 1] = extra - before;
        }
        self.pad(&Pad3DConfig {
            padding,
            fill: config.fill.clone(),
            fill_mask: config.fill_mask,
        })
    }

    pub fn pad(mut self, config: &Pad3DConfig) -> RivetResult<Self> {
        self.validate()?;
        let old = self.spatial_shape()?;
        let channels = self.volume.dims().get(3).copied().unwrap_or(1);
        if (config.fill.len() != 1 && config.fill.len() != channels)
            || config.fill.iter().any(|v| !v.is_finite())
        {
            return Err(invalid_argument(
                "Pad3D fill requires a finite scalar or one value per channel",
            ));
        }
        if config.padding == [0; 6] {
            return Ok(self);
        }
        let before = [config.padding[0], config.padding[2], config.padding[4]];
        let after = [config.padding[1], config.padding[3], config.padding[5]];
        let mut new = [0; 3];
        for i in 0..3 {
            new[i] = old[i]
                .checked_add(before[i])
                .and_then(|n| n.checked_add(after[i]))
                .ok_or_else(|| invalid_shape("Pad3D axis overflow"))?;
        }
        self.volume = pad_tensor(&self.volume, old, new, before, &config.fill, None)?;
        if let Some(mask) = &self.mask {
            if mask.dtype() == DType::U8 && u8::try_from(config.fill_mask).is_err() {
                return Err(invalid_argument(
                    "Pad3D uint8 mask fill must be in [0, 255]",
                ));
            }
            self.mask = Some(pad_tensor(
                mask,
                old,
                new,
                before,
                &[],
                Some(config.fill_mask),
            )?);
        }
        if let Some(points) = &self.keypoints {
            let delta = [before[2] as f64, before[1] as f64, before[0] as f64];
            let coords = points.to_vec::<f32>()?;
            self.keypoints = Some(Tensor::from_exact_iter(
                coords
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (f64::from(v) + delta[i % 3]) as f32),
                points.dims(),
                points.device(),
            )?);
        }
        Ok(self)
    }
}

fn pad_tensor(
    input: &Tensor,
    old: [usize; 3],
    new: [usize; 3],
    before: [usize; 3],
    fill: &[f32],
    mask_fill: Option<i64>,
) -> RivetResult<Tensor> {
    let channels = input.dims().get(3).copied().unwrap_or(1);
    let mut shape = new.to_vec();
    if input.rank() == 4 {
        shape.push(channels);
    }
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| {
        let start = layout.start_offset();
        let fill_index = |c| if fill.len() == 1 { 0 } else { c };
        macro_rules! run {
            ($values:expr,$value:expr) => {{
                let value = $value;
                let fill: Vec<_> = (0..channels).map(value).collect();
                write_pad(
                    shape.as_slice(),
                    input.device(),
                    &$values[start..],
                    &fill,
                    old,
                    new,
                    before,
                )
            }};
        }
        match storage {
            CpuStorageRef::U8(v) => run!(v, |c| mask_fill
                .map_or_else(|| fill[fill_index(c)].round() as u8, |v| v as u8)),
            CpuStorageRef::F32(v) => run!(v, |c| fill[fill_index(c)]),
            CpuStorageRef::I64(v) => run!(v, |_| mask_fill.expect("validated int64 mask")),
            _ => unreachable!("validated padding dtype"),
        }
    })?)
}

fn write_pad<T: WithDType>(
    shape: &[usize],
    device: &rivet_core::Device,
    pixels: &[T],
    fill: &[T],
    [d, h, w]: [usize; 3],
    [nd, nh, nw]: [usize; 3],
    [z0, y0, x0]: [usize; 3],
) -> rivet_core::Result<Tensor> {
    let c = fill.len();
    let source_row = w * c;
    Tensor::from_exact_writer(shape, device, |out| {
        for z in 0..nd {
            for y in 0..nh {
                if (z0..z0 + d).contains(&z) && (y0..y0 + h).contains(&y) {
                    for _ in 0..x0 {
                        out.extend_from_slice(fill)?;
                    }
                    let first = ((z - z0) * h + y - y0) * source_row;
                    out.extend_from_slice(&pixels[first..first + source_row])?;
                    for _ in x0 + w..nw {
                        out.extend_from_slice(fill)?;
                    }
                } else {
                    for _ in 0..nw {
                        out.extend_from_slice(fill)?;
                    }
                }
            }
        }
        Ok(())
    })
}

#[derive(Clone, Debug)]
pub struct Crop3DConfig {
    pub size: [usize; 3],
    pub pad_if_needed: bool,
    pub fill: Vec<f32>,
    pub fill_mask: i64,
}

impl VolumeSample {
    /// Crop around the center, or use the supplied stream for random placement.
    /// Keypoints are translated and retained, including points outside the crop.
    pub fn crop(
        mut self,
        config: &Crop3DConfig,
        mut rng: Option<&mut rivet_data::random::RandomStream>,
    ) -> RivetResult<Self> {
        self.validate()?;
        if config.size.contains(&0) {
            return Err(invalid_shape("Crop3D axes must be positive"));
        }
        let shape = self.spatial_shape()?;
        let mut padding = [0; 6];
        let mut start = [0; 3];
        for axis in 0..3 {
            let input = shape[axis];
            let output = config.size[axis];
            if input < output {
                if !config.pad_if_needed {
                    return Err(invalid_shape("Crop3D size exceeds input volume"));
                }
                let extra = output - input;
                let before = crop_offset(extra, rng.as_deref_mut())?;
                padding[axis * 2] = before;
                padding[axis * 2 + 1] = extra - before;
            } else {
                start[axis] = crop_offset(input - output, rng.as_deref_mut())?;
            }
        }
        let unpadded_size = std::array::from_fn(|i| shape[i].min(config.size[i]));
        self = self.crop_window(start, unpadded_size)?;
        if padding != [0; 6] {
            self = self.pad(&Pad3DConfig {
                padding,
                fill: config.fill.clone(),
                fill_mask: config.fill_mask,
            })?;
        }
        Ok(self)
    }

    fn crop_window(mut self, start: [usize; 3], size: [usize; 3]) -> RivetResult<Self> {
        let crop = |mut t: Tensor| -> rivet_core::Result<Tensor> {
            for axis in 0..3 {
                t = t.narrow(axis, start[axis], size[axis])?;
            }
            Ok(t)
        };
        self.volume = crop(self.volume)?;
        self.mask = self.mask.map(crop).transpose()?;
        if let Some(points) = &self.keypoints
            && start != [0; 3]
        {
            let coords = points.to_vec::<f32>()?;
            self.keypoints = Some(Tensor::from_exact_iter(
                coords
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (f64::from(v) - start[2 - i % 3] as f64) as f32),
                points.dims(),
                points.device(),
            )?);
        }
        Ok(self)
    }
}

pub(super) fn crop_offset(
    extra: usize,
    rng: Option<&mut rivet_data::random::RandomStream>,
) -> RivetResult<usize> {
    match rng {
        Some(rng) if extra > 0 => Ok(rng.gen_range_usize(0..extra + 1)?),
        _ => Ok(extra / 2),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RandomResizedCrop3DConfig {
    pub resize: Resize3DConfig,
    pub scale: [f64; 2],
    pub ratio: Option<f64>,
    pub uniform_scale: bool,
}

impl VolumeSample {
    pub fn random_resized_crop(
        self,
        config: RandomResizedCrop3DConfig,
        rng: &mut rivet_data::random::RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        let source = self.spatial_shape()?;
        let [low, high] = config.scale;
        if !(0.0..=1.0).contains(&low)
            || !(low..=1.0).contains(&high)
            || config.ratio.is_some_and(|v| !v.is_finite() || v < 1.0)
            || config.resize.size.contains(&0)
        {
            return Err(invalid_argument(
                "invalid RandomResizedCrop3D size, scale or ratio",
            ));
        }
        let (size, random) = resized_crop_shape(source, config, rng);
        let mut start = [0; 3];
        for axis in 0..3 {
            start[axis] = crop_offset(source[axis] - size[axis], random.then_some(&mut *rng))?;
        }
        self.crop_window(start, size)?.resize(config.resize)
    }
}

fn resized_crop_shape(
    source: [usize; 3],
    config: RandomResizedCrop3DConfig,
    rng: &mut rivet_data::random::RandomStream,
) -> ([usize; 3], bool) {
    let bounds = source.map(|v| v as f64);
    let source_volume = bounds.iter().product::<f64>();
    let [requested_low, requested_high] = config.scale.map(|s| s * source_volume);
    let longest = match config.ratio {
        Some(ratio) => {
            let shortest = bounds.into_iter().fold(f64::INFINITY, f64::min);
            bounds.map(|v| v.min(shortest * ratio))
        }
        None => {
            let output = config.resize.size.map(|v| v as f64);
            let scale = (0..3)
                .map(|i| bounds[i] / output[i])
                .fold(f64::INFINITY, f64::min);
            output.map(|v| v * scale)
        }
    };
    let round = |sides: [f64; 3]| {
        let mut size = std::array::from_fn(|i| (sides[i].round() as usize).clamp(1, source[i]));
        if let Some(ratio) = config.ratio {
            let shortest = size.into_iter().min().unwrap();
            let largest = (shortest as f64 * ratio) as usize;
            size = size.map(|v| v.min(largest));
        }
        size
    };
    let max_volume = longest.iter().product::<f64>();
    if requested_low > max_volume {
        return (round(longest), false);
    }
    let high = requested_high.min(max_volume);
    let sampled_volume = requested_low + (high - requested_low) * rng.next_f64();
    let adjust = |edges: [f64; 3], volume: f64| {
        let factor = (volume / edges.iter().product::<f64>()).cbrt();
        edges.map(|v| v * factor)
    };
    if let Some(ratio) = config.ratio {
        let spread = ratio.ln();
        let attempts = if ratio == 1.0 { 0 } else { 32 };
        for _ in 0..attempts {
            let logs = bounds.map(|_| rng.next_f64() * spread);
            let edges = logs.map(f64::exp);
            let volume = if config.uniform_scale {
                sampled_volume
            } else {
                requested_low + (requested_high - requested_low) * rng.next_f64()
            };
            let proposed = adjust(edges, volume);
            if (0..3).all(|i| proposed[i] <= bounds[i]) {
                return (round(proposed), true);
            }
        }
    }
    // A scaled maximum-volume box always fits. This also covers tight
    // ranges where random log-shape proposals rarely fit.
    (round(adjust(longest, sampled_volume)), true)
}
