//! Decoded volumes use DHW or DHWC. Masks are DHW class/instance IDs,
//! and keypoints are stored as [N, 3] in X, Y, Z pixel-center coordinates.
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use rivet_core::{CpuStorageRef, DType, Tensor, WithDType};

#[derive(Clone, Debug)]
pub struct VolumeSample {
    pub volume: Tensor,
    pub mask: Option<Tensor>,
    pub keypoints: Option<Tensor>,
    pub label: i64,
}

impl VolumeSample {
    pub fn spatial_shape(&self) -> RivetResult<[usize; 3]> {
        match self.volume.dims() {
            &[d, h, w] if d > 0 && h > 0 && w > 0 => Ok([d, h, w]),
            &[d, h, w, c] if d > 0 && h > 0 && w > 0 && c > 0 => Ok([d, h, w]),
            _ => Err(invalid_shape("volume must be nonempty DHW or DHWC")),
        }
    }

    pub fn validate(&self) -> RivetResult<()> {
        let size = self.spatial_shape()?;
        if !matches!(self.volume.dtype(), DType::U8 | DType::F32) {
            return Err(invalid_argument("volume requires uint8 or float32 pixels"));
        }
        if self.mask.as_ref().is_some_and(|mask| {
            mask.dims() != size || !matches!(mask.dtype(), DType::U8 | DType::I64)
        }) {
            return Err(invalid_shape(
                "volume mask requires matching DHW uint8 or int64 data",
            ));
        }
        if let Some(points) = &self.keypoints {
            if !matches!(points.dims(), [_, 3])
                || points.dtype() != DType::F32
                || points.to_vec::<f32>()?.iter().any(|v| !v.is_finite())
            {
                return Err(invalid_argument(
                    "volume keypoints require finite float32 [N,3] coordinates",
                ));
            }
        }
        Ok(())
    }

    pub fn resize(mut self, config: Resize3DConfig) -> RivetResult<Self> {
        self.validate()?;
        let old = self.spatial_shape()?;
        let new = config.size;
        if new.contains(&0) {
            return Err(invalid_shape("Resize3D axes must be positive"));
        }
        if new == old {
            return Ok(self);
        }
        self.volume = resample(&self.volume, old, new, config.interpolation)?;
        if let Some(mask) = &self.mask {
            self.mask = Some(resample(mask, old, new, VolumeInterpolation::Nearest)?);
        }
        if let Some(points) = &self.keypoints {
            let scale = [
                new[2] as f64 / old[2] as f64,
                new[1] as f64 / old[1] as f64,
                new[0] as f64 / old[0] as f64,
            ];
            let coords = points.to_vec::<f32>()?;
            self.keypoints = Some(Tensor::from_exact_iter(
                coords
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| ((f64::from(v) + 0.5) * scale[i % 3] - 0.5) as f32),
                points.dims(),
                points.device(),
            )?);
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolumeInterpolation {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resize3DConfig {
    pub size: [usize; 3],
    pub interpolation: VolumeInterpolation,
}

#[derive(Clone, Copy)]
struct AxisSample {
    nearest: usize,
    low: usize,
    high: usize,
    fraction: f64,
}

fn axis_samples(old: usize, new: usize) -> Vec<AxisSample> {
    let scale = old as f64 / new as f64;
    (0..new)
        .map(|i| {
            let p = (i as f64 + 0.5) * scale;
            let nearest = (p as usize).min(old - 1);
            let coordinate = (p - 0.5).clamp(0.0, (old - 1) as f64);
            let low = coordinate as usize;
            AxisSample {
                nearest,
                low,
                high: (low + 1).min(old - 1),
                fraction: coordinate - low as f64,
            }
        })
        .collect()
}

fn resample(
    input: &Tensor,
    old: [usize; 3],
    new: [usize; 3],
    mode: VolumeInterpolation,
) -> RivetResult<Tensor> {
    let [zd, yd, xd] = std::array::from_fn(|i| axis_samples(old[i], new[i]));
    let axes = [zd.as_slice(), yd.as_slice(), xd.as_slice()];
    let channels = input.dims().get(3).copied().unwrap_or(1);
    let mut shape = new.to_vec();
    if input.rank() == 4 {
        shape.push(channels);
    }
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| {
        let start = layout.start_offset();
        let read = |z: usize, y: usize, x: usize, c: usize| {
            start + ((z * old[1] + y) * old[2] + x) * channels + c
        };
        macro_rules! run {
            ($values:expr,$finish:expr) => {{
                let finish = $finish;
                write(
                    &input,
                    &shape,
                    axes,
                    channels,
                    mode,
                    |z, y, x, c| $values[read(z, y, x, c)],
                    finish,
                )
            }};
        }
        match storage {
            CpuStorageRef::U8(v) => run!(v, |p: f64| p.round() as u8),
            CpuStorageRef::F32(v) => run!(v, |p: f64| p as f32),
            CpuStorageRef::I64(v) => run!(v, |p: f64| p as i64),
            _ => unreachable!("validated volume dtype"),
        }
    })?)
}

fn write<T: WithDType + IntoVolumeFloat>(
    input: &Tensor,
    shape: &[usize],
    [zs, ys, xs]: [&[AxisSample]; 3],
    channels: usize,
    mode: VolumeInterpolation,
    read: impl Fn(usize, usize, usize, usize) -> T,
    finish: impl Fn(f64) -> T,
) -> rivet_core::Result<Tensor> {
    Tensor::from_exact_writer(shape, input.device(), |out| {
        if mode == VolumeInterpolation::Nearest {
            for z in zs {
                for y in ys {
                    for x in xs {
                        for c in 0..channels {
                            out.write_next(read(z.nearest, y.nearest, x.nearest, c))?;
                        }
                    }
                }
            }
        } else {
            for z in zs {
                for y in ys {
                    for x in xs {
                        for c in 0..channels {
                            let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
                            let along_x = |z, y| {
                                lerp(
                                    read(z, y, x.low, c).as_f64(),
                                    read(z, y, x.high, c).as_f64(),
                                    x.fraction,
                                )
                            };
                            let along_y =
                                |z| lerp(along_x(z, y.low), along_x(z, y.high), y.fraction);
                            out.write_next(finish(lerp(
                                along_y(z.low),
                                along_y(z.high),
                                z.fraction,
                            )))?;
                        }
                    }
                }
            }
        }
        Ok(())
    })
}

trait IntoVolumeFloat {
    fn as_f64(self) -> f64;
}
impl IntoVolumeFloat for u8 {
    fn as_f64(self) -> f64 {
        f64::from(self)
    }
}
impl IntoVolumeFloat for f32 {
    fn as_f64(self) -> f64 {
        f64::from(self)
    }
}
impl IntoVolumeFloat for i64 {
    fn as_f64(self) -> f64 {
        self as f64
    }
}

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

fn crop_offset(
    extra: usize,
    rng: Option<&mut rivet_data::random::RandomStream>,
) -> RivetResult<usize> {
    match rng {
        Some(rng) if extra > 0 => Ok(rng.gen_range_usize(0..extra + 1)?),
        _ => Ok(extra / 2),
    }
}

impl VolumeSample {
    /// Reorient output DHW axes by permuting input axes and then reflecting
    /// selected output axes. The channel axis and label are preserved.
    pub fn reorient(mut self, order: [usize; 3], reflected_axes: &[usize]) -> RivetResult<Self> {
        self.validate()?;
        if volume_axis_mask(&order)? != 0b111 {
            return Err(invalid_argument("3D axis order must be a permutation"));
        }
        let reflected = volume_axis_mask(reflected_axes)?;
        let shape = self.spatial_shape()?;
        let reorder = |t: Tensor| -> rivet_core::Result<Tensor> {
            let t = if order == [0, 1, 2] {
                t
            } else if t.rank() == 4 {
                t.permute(&[order[0], order[1], order[2], 3])?
            } else {
                t.permute(&order)?
            };
            if reflected_axes.is_empty() {
                Ok(t)
            } else {
                t.flip(reflected_axes)
            }
        };
        self.volume = reorder(self.volume)?;
        self.mask = self.mask.map(reorder).transpose()?;
        if let Some(points) = &self.keypoints
            && (order != [0, 1, 2] || reflected != 0)
        {
            let coords = points.to_vec::<f32>()?;
            self.keypoints = Some(Tensor::from_exact_iter(
                (0..coords.len()).map(|i| {
                    let axis = 2 - i % 3;
                    let original = order[axis];
                    let value = coords[i / 3 * 3 + 2 - original];
                    if reflected & (1 << axis) == 0 {
                        value
                    } else {
                        ((shape[original] - 1) as f64 - f64::from(value)) as f32
                    }
                }),
                points.dims(),
                points.device(),
            )?);
        }
        Ok(self)
    }

    pub fn rotate90(self, [a, b]: [usize; 2], turns: u8) -> RivetResult<Self> {
        if a >= b || b >= 3 || turns >= 4 {
            return Err(invalid_argument(
                "expected ascending DHW axes and 0 to 3 quarter turns",
            ));
        }
        let mut order = [0, 1, 2];
        if turns % 2 == 1 {
            order.swap(a, b);
        }
        match turns {
            0 => self.reorient(order, &[]),
            1 => self.reorient(order, &[a]),
            2 => self.reorient(order, &[a, b]),
            _ => self.reorient(order, &[b]),
        }
    }

    /// Apply one of the 48 signed spatial permutations, with identity at 0.
    pub fn cubic_symmetry(self, element: usize) -> RivetResult<Self> {
        const PERMUTATIONS: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let order = PERMUTATIONS
            .get(element / 8)
            .copied()
            .ok_or_else(|| invalid_argument("CubicSymmetry element must be in [0, 47]"))?;
        let reflected = (0..3)
            .filter(|axis| element & (1 << axis) != 0)
            .collect::<Vec<_>>();
        self.reorient(order, &reflected)
    }
}

pub fn volume_axis_mask(axes: &[usize]) -> RivetResult<u8> {
    let mut mask = 0;
    for &axis in axes {
        if axis >= 3 || mask & (1 << axis) != 0 {
            return Err(invalid_argument("expected distinct DHW spatial axes"));
        }
        mask |= 1 << axis;
    }
    Ok(mask)
}

impl VolumeSample {
    pub fn grid_shuffle(
        mut self,
        grid: [usize; 3],
        rng: &mut rivet_data::random::RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        let shape = self.spatial_shape()?;
        if (0..3).any(|i| grid[i] == 0 || grid[i] > shape[i]) {
            return Err(invalid_shape("GridShuffle3D grid must fit the DHW volume"));
        }
        let bounds: [Vec<usize>; 3] = std::array::from_fn(|i| {
            let count = grid[i];
            let (width, extra) = (shape[i] / count, shape[i] % count);
            (0..=count).map(|j| j * width + j.min(extra)).collect()
        });
        let mut origins = Vec::new();
        let mut groups = [const { Vec::new() }; 8];
        for z in 0..grid[0] {
            for y in 0..grid[1] {
                for x in 0..grid[2] {
                    let cell = [z, y, x];
                    let class = (0..3).fold(0, |bits, i| {
                        bits | (usize::from(cell[i] < shape[i] % grid[i]) << i)
                    });
                    groups[class].push(origins.len());
                    origins.push(std::array::from_fn(|i| bounds[i][cell[i]]));
                }
            }
        }
        let mut sources: Vec<_> = (0..origins.len()).collect();
        for group in &groups {
            for i in (1..group.len()).rev() {
                let j = rng.gen_range_usize(0..i + 1)?;
                sources.swap(group[i], group[j]);
            }
        }
        if sources.iter().copied().eq(0..origins.len()) {
            return Ok(self);
        }
        let shuffle = |t: &Tensor| shuffle_volume(t, &bounds, grid, &origins, &sources);
        self.volume = shuffle(&self.volume)?;
        self.mask = self.mask.as_ref().map(shuffle).transpose()?;
        if let Some(points) = &self.keypoints {
            let mut destination = vec![0; sources.len()];
            for (dest, &source) in sources.iter().enumerate() {
                destination[source] = dest;
            }
            let mut coords = points.to_vec::<f32>()?;
            for point in coords.chunks_exact_mut(3) {
                let zyx = [point[2], point[1], point[0]];
                if (0..3).all(|i| zyx[i] >= 0.0 && f64::from(zyx[i]) < shape[i] as f64) {
                    let [z, y, x]: [usize; 3] = std::array::from_fn(|i| {
                        bounds[i].partition_point(|&bound| bound <= zyx[i] as usize) - 1
                    });
                    let source = (z * grid[1] + y) * grid[2] + x;
                    let dest = destination[source];
                    for i in 0..3 {
                        point[2 - i] = (f64::from(zyx[i]) + origins[dest][i] as f64
                            - origins[source][i] as f64)
                            as f32;
                    }
                }
            }
            self.keypoints = Some(Tensor::from_vec(coords, points.dims(), points.device())?);
        }
        Ok(self)
    }
}

fn shuffle_volume(
    tensor: &Tensor,
    bounds: &[Vec<usize>; 3],
    grid: [usize; 3],
    origins: &[[usize; 3]],
    sources: &[usize],
) -> RivetResult<Tensor> {
    let input = tensor.contiguous()?;
    let channels = input.dims().get(3).copied().unwrap_or(1);
    let height = input.dims()[1];
    let width = input.dims()[2];
    Ok(input.with_cpu_storage(|storage, layout| {
        macro_rules! copy_rows {
            ($values:expr) => {
                Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                    for z in 0..grid[0] {
                        for dz in 0..bounds[0][z + 1] - bounds[0][z] {
                            for y in 0..grid[1] {
                                for dy in 0..bounds[1][y + 1] - bounds[1][y] {
                                    let first_cell = (z * grid[1] + y) * grid[2];
                                    for x in 0..grid[2] {
                                        let [sz, sy, sx] = origins[sources[first_cell + x]];
                                        let first = layout.start_offset()
                                            + (((sz + dz) * height + sy + dy) * width + sx)
                                                * channels;
                                        let len = (bounds[2][x + 1] - bounds[2][x]) * channels;
                                        out.extend_from_slice(&$values[first..first + len])?;
                                    }
                                }
                            }
                        }
                    }
                    Ok(())
                })
            };
        }
        match storage {
            CpuStorageRef::U8(v) => copy_rows!(v),
            CpuStorageRef::F32(v) => copy_rows!(v),
            CpuStorageRef::I64(v) => copy_rows!(v),
            _ => unreachable!("validated volume grid dtype"),
        }
    })?)
}

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

#[derive(Clone, Debug)]
pub struct Anisotropy3DConfig {
    pub axes: Vec<usize>,
    pub num_axes_range: [usize; 2],
    pub downscale_factor_range: [f64; 2],
    pub antialias: bool,
}

impl VolumeSample {
    /// Simulate reduced acquisition resolution while preserving the label grid.
    pub fn anisotropy(
        mut self,
        config: &Anisotropy3DConfig,
        rng: &mut rivet_data::random::RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        volume_axis_mask(&config.axes)?;
        let [min, max] = config.num_axes_range;
        let [low, high] = config.downscale_factor_range;
        if min == 0
            || min > max
            || max > config.axes.len()
            || !low.is_finite()
            || !high.is_finite()
            || low <= 1.0
            || high < low
        {
            return Err(invalid_argument(
                "invalid Anisotropy3D axis count or downscale factor range",
            ));
        }
        let shape = self.spatial_shape()?;
        let count = rng.gen_range_usize(min..max + 1)?;
        let mut reduced = shape;
        let mut axes = config.axes.clone();
        for index in 0..count {
            let selected = rng.gen_range_usize(index..axes.len())?;
            axes.swap(index, selected);
            let axis = axes[index];
            let factor = low + (high - low) * rng.next_f64();
            reduced[axis] = ((shape[axis] as f64 / factor).round() as usize).max(1);
        }
        if reduced != shape {
            let down = if config.antialias {
                area_downsample(&self.volume, shape, reduced)?
            } else {
                resample(&self.volume, shape, reduced, VolumeInterpolation::Linear)?
            };
            self.volume = resample(&down, reduced, shape, VolumeInterpolation::Linear)?;
        }
        Ok(self)
    }
}

fn area_downsample(input: &Tensor, old: [usize; 3], new: [usize; 3]) -> RivetResult<Tensor> {
    let weights: [Vec<Vec<(usize, f64)>>; 3] = std::array::from_fn(|axis| {
        let ratio = old[axis] as f64 / new[axis] as f64;
        (0..new[axis])
            .map(|i| {
                let left = i as f64 * ratio;
                let right = (i + 1) as f64 * ratio;
                (left.floor() as usize..(right.ceil() as usize).min(old[axis]))
                    .map(|j| (j, (right.min((j + 1) as f64) - left.max(j as f64)) / ratio))
                    .collect()
            })
            .collect()
    });
    let mut dims = input.dims().to_vec();
    dims[..3].copy_from_slice(&new);
    let channels = dims.get(3).copied().unwrap_or(1);
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| {
        macro_rules! sample {
            ($data:expr, $cast:expr) => {
                Tensor::from_exact_writer(dims.as_slice(), input.device(), |out| {
                    for zs in &weights[0] {
                        for ys in &weights[1] {
                            for xs in &weights[2] {
                                for c in 0..channels {
                                    let mut sum = 0.0;
                                    for &(z, wz) in zs {
                                        for &(y, wy) in ys {
                                            let row = layout.start_offset()
                                                + (z * old[1] + y) * old[2] * channels;
                                            let wyz = wy * wz;
                                            for &(x, wx) in xs {
                                                sum += f64::from($data[row + x * channels + c])
                                                    * wx
                                                    * wyz;
                                            }
                                        }
                                    }
                                    out.write_next($cast(sum))?;
                                }
                            }
                        }
                    }
                    Ok(())
                })
            };
        }
        match storage {
            CpuStorageRef::U8(v) => sample!(v, |v: f64| v.round() as u8),
            CpuStorageRef::F32(v) => sample!(v, |v: f64| v as f32),
            _ => unreachable!("validated volume dtype"),
        }
    })?)
}

mod warp;
pub use warp::{Affine3DConfig, VolumeBorder};
mod elastic;
pub use elastic::ElasticTransform3DConfig;

impl VolumeSample {
    pub fn rician_noise(
        mut self,
        std_range: [f64; 2],
        per_channel: bool,
        rng: &mut crate::RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        let channels = self.volume.dims().get(3).copied().unwrap_or(1);
        self.volume = crate::transforms::noise::rician_noise(
            &self.volume,
            channels,
            std_range,
            per_channel,
            rng,
        )?;
        Ok(self)
    }
}

mod bias;
pub use bias::BiasFieldConfig;

mod spectral;
pub use spectral::GibbsRingingConfig;
