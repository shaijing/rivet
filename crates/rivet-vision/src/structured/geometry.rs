use super::annotations::{BoxFormat, StructuredSample, convert_boxes};
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use crate::transforms::spatial::remap::{SpatialRemap, remap_image};
use crate::transforms::{
    AffineConfig, CropConfig, FlipConfig, InterpolationMode, LayoutConfig, PadConfig, PaddingMode,
    ResizeConfig,
};
use rivet_core::{CpuStorageRef, Device, Tensor};

pub(super) type Matrix = [[f64; 3]; 3];
pub(super) fn image(sample: &StructuredSample) -> ImageSample {
    ImageSample::Decoded(DecodedSample {
        image: sample.image.clone(),
        label: 0,
    })
}
fn point(m: Matrix, x: f64, y: f64) -> [f32; 2] {
    let d = m[2][0] * x + m[2][1] * y + m[2][2];
    [
        ((m[0][0] * x + m[0][1] * y + m[0][2]) / d) as f32,
        ((m[1][0] * x + m[1][1] * y + m[1][2]) / d) as f32,
    ]
}
pub(super) fn inverse(m: Matrix) -> RivetResult<Matrix> {
    let [[a, b, c], [d, e, f], [g, h, i]] = m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if !det.is_finite() || det.abs() < 1e-14 {
        return Err(invalid_argument("singular geometric transform"));
    }
    Ok([
        [
            (e * i - f * h) / det,
            (c * h - b * i) / det,
            (b * f - c * e) / det,
        ],
        [
            (f * g - d * i) / det,
            (a * i - c * g) / det,
            (c * d - a * f) / det,
        ],
        [
            (d * h - e * g) / det,
            (b * g - a * h) / det,
            (a * e - b * d) / det,
        ],
    ])
}
pub(super) fn affine_matrix(m: [f64; 6]) -> Matrix {
    [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [0., 0., 1.]]
}
pub(super) fn targets(
    mut s: StructuredSample,
    m: Matrix,
    k: Matrix,
    clamp: bool,
) -> RivetResult<StructuredSample> {
    if let Some(boxes) = s.xyxy()? {
        let values = boxes.to_vec::<f32>()?;
        let mut out = Vec::with_capacity(values.len());
        for b in values.chunks_exact(4) {
            let p = [
                point(m, b[0] as f64, b[1] as f64),
                point(m, b[2] as f64, b[1] as f64),
                point(m, b[2] as f64, b[3] as f64),
                point(m, b[0] as f64, b[3] as f64),
            ];
            if p.iter().flatten().any(|v| !v.is_finite()) {
                return Err(invalid_argument(
                    "box maps to a nonfinite projective coordinate",
                ));
            }
            let x1 = p.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min);
            let y1 = p.iter().map(|v| v[1]).fold(f32::INFINITY, f32::min);
            let x2 = p.iter().map(|v| v[0]).fold(f32::NEG_INFINITY, f32::max);
            let y2 = p.iter().map(|v| v[1]).fold(f32::NEG_INFINITY, f32::max);
            out.extend([x1, y1, x2, y2]);
        }
        let b = Tensor::from_vec(out, boxes.dims(), &Device::Cpu)?;
        s.boxes = Some(convert_boxes(&b, BoxFormat::Xyxy, s.box_format)?);
    }
    if let Some(points) = &s.keypoints {
        let values = points.to_vec::<f32>()?;
        let out = values
            .chunks_exact(2)
            .flat_map(|p| point(k, p[0] as f64, p[1] as f64))
            .collect::<Vec<_>>();
        if out.iter().any(|v| !v.is_finite()) {
            return Err(invalid_argument(
                "keypoint maps to a nonfinite projective coordinate",
            ));
        }
        s.keypoints = Some(Tensor::from_vec(out, points.dims(), &Device::Cpu)?);
    }
    if clamp { s.clamp_boxes() } else { Ok(s) }
}
/// Map instance masks using precomputed integer coordinates and typed writers.
fn masks_map<F>(input: &Tensor, oh: usize, ow: usize, mapping: F) -> RivetResult<Tensor>
where
    F: FnMut(usize, usize) -> Option<(usize, usize)>,
{
    masks_map_fill(input, oh, ow, 0, mapping)
}

pub(super) fn masks_map_fill<F>(
    input: &Tensor,
    oh: usize,
    ow: usize,
    fill: i64,
    mut mapping: F,
) -> RivetResult<Tensor>
where
    F: FnMut(usize, usize) -> Option<(usize, usize)>,
{
    if input.rank() == 2 {
        return Ok(masks_map_fill(&input.unsqueeze(0)?, oh, ow, fill, mapping)?.squeeze(0)?);
    }
    let dims = input.dims();
    let n = dims[0];
    if n == 0 {
        return Ok(
            Tensor::from_exact_iter(std::iter::empty::<u8>(), [0, oh, ow], &Device::Cpu)?
                .to_dtype(input.dtype())?,
        );
    }
    let ih = dims[1];
    let iw = dims[2];
    let mut coords = Vec::with_capacity(
        oh.checked_mul(ow)
            .ok_or_else(|| invalid_shape("mask output size overflow"))?,
    );
    for y in 0..oh {
        for x in 0..ow {
            coords.push(mapping(y, x).filter(|&(sy, sx)| sy < ih && sx < iw));
        }
    }
    input
        .with_cpu_storage(|storage, layout| {
            let st = layout.stride();
            let start = layout.start_offset();
            macro_rules! map {
                ($values:expr,$t:ty) => {{
                    let values = $values;
                    Tensor::from_exact_writer::<$t, _, _>([n, oh, ow], &Device::Cpu, |out| {
                        for instance in 0..n {
                            let base = start + instance * st[0];
                            for coord in &coords {
                                let v = match coord {
                                    Some((y, x)) => *values
                                        .get(base + y * st[1] + x * st[2])
                                        .ok_or(rivet_core::Error::StorageOutOfBounds)?,
                                    None => fill as $t,
                                };
                                out.write_next(v)?;
                            }
                        }
                        Ok(())
                    })
                }};
            }
            match storage {
                CpuStorageRef::U8(values) => map!(values, u8),
                CpuStorageRef::I64(values) => map!(values, i64),
                _ => unreachable!("validated masks"),
            }
        })
        .map_err(Into::into)
}
pub(super) fn crop_visible_instances(
    mut sample: StructuredSample,
    boxes: Option<&[f32]>,
    [x0, y0, x1, y1]: [usize; 4],
) -> RivetResult<StructuredSample> {
    if let Some(boxes) = boxes {
        let keep: Vec<_> = boxes
            .chunks_exact(4)
            .enumerate()
            .filter_map(|(i, b)| {
                (f64::from(b[0]).max(x0 as f64) < f64::from(b[2]).min(x1 as f64)
                    && f64::from(b[1]).max(y0 as f64) < f64::from(b[3]).min(y1 as f64))
                .then_some(i)
            })
            .collect();
        if keep.len() != boxes.len() / 4 {
            sample = sample.retain_instances(&keep)?;
        }
    }
    crop(
        sample,
        x0 as u32,
        y0 as u32,
        (x1 - x0) as u32,
        (y1 - y0) as u32,
    )
}

pub(super) fn crop(
    mut s: StructuredSample,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> RivetResult<StructuredSample> {
    if w == 0 || h == 0 {
        return Err(invalid_shape("crop dimensions must be positive"));
    }
    s.image = CropConfig::new(x, y, w, h)
        .apply(image(&s), s.axis_order)?
        .into_decoded()?
        .image;
    if let Some(m) = &s.masks {
        s.masks = Some(masks_map(m, h as usize, w as usize, |iy, ix| {
            Some((iy + y as usize, ix + x as usize))
        })?);
    }
    let m = [[1., 0., -(x as f64)], [0., 1., -(y as f64)], [0., 0., 1.]];
    targets(s, m, m, true)
}
pub(super) fn pad(s: StructuredSample, p: &PadConfig) -> RivetResult<StructuredSample> {
    pad_with_mask_fill(s, p, 0)
}

pub(super) fn pad_with_mask_fill(
    mut s: StructuredSample,
    p: &PadConfig,
    mask_fill: i64,
) -> RivetResult<StructuredSample> {
    if s.masks
        .as_ref()
        .is_some_and(|mask| mask.dtype() == rivet_core::DType::U8)
        && u8::try_from(mask_fill).is_err()
    {
        return Err(invalid_argument(
            "LetterBox mask_fill must be in [0, 255] for uint8 masks",
        ));
    }
    let (h, w) = s.size()?;
    s.image = p.apply(image(&s), s.axis_order)?.into_decoded()?.image;
    if let Some(mask) = &s.masks {
        let (oh, ow) = s.size()?;
        let coord = |value: isize, len: usize| -> Option<usize> {
            if (0..len as isize).contains(&value) {
                return Some(value as usize);
            }
            match p.mode {
                PaddingMode::Constant => None,
                PaddingMode::Edge => Some(value.clamp(0, len as isize - 1) as usize),
                PaddingMode::Reflect => {
                    if len == 1 {
                        return Some(0);
                    }
                    let period = 2 * (len as isize - 1);
                    let r = value.rem_euclid(period);
                    Some(if r < len as isize { r } else { period - r } as usize)
                }
                PaddingMode::Symmetric => {
                    let period = 2 * len as isize;
                    let r = value.rem_euclid(period);
                    Some(if r < len as isize { r } else { period - r - 1 } as usize)
                }
            }
        };
        s.masks = Some(masks_map_fill(mask, oh, ow, mask_fill, |y, x| {
            Some((
                coord(y as isize - p.top as isize, h)?,
                coord(x as isize - p.left as isize, w)?,
            ))
        })?);
    }
    let m = [
        [1., 0., p.left as f64],
        [0., 1., p.top as f64],
        [0., 0., 1.],
    ];
    targets(s, m, m, false)
}
pub(super) fn resize(
    mut s: StructuredSample,
    w: u32,
    h: u32,
    interpolation: InterpolationMode,
) -> RivetResult<StructuredSample> {
    let (ih, iw) = s.size()?;
    let mut input = image(&s);
    if s.axis_order == ImageAxisOrder::Chw {
        input = LayoutConfig::new(ImageAxisOrder::Hwc).apply(input, s.axis_order)?;
    }
    input = ResizeConfig::with_interpolation(w, h, interpolation).apply(input)?;
    if s.axis_order == ImageAxisOrder::Chw {
        input = LayoutConfig::new(ImageAxisOrder::Chw).apply(input, ImageAxisOrder::Hwc)?;
    }
    s.image = input.into_decoded()?.image;
    if let Some(mask) = &s.masks {
        s.masks = Some(masks_map(mask, h as usize, w as usize, |y, x| {
            Some((y * ih / h as usize, x * iw / w as usize))
        })?);
    }
    let m = [
        [w as f64 / iw as f64, 0., 0.],
        [0., h as f64 / ih as f64, 0.],
        [0., 0., 1.],
    ];
    targets(s, m, m, false)
}
pub(super) fn flip(mut s: StructuredSample, horizontal: bool) -> RivetResult<StructuredSample> {
    let (h, w) = s.size()?;
    let cfg = if horizontal {
        FlipConfig::horizontal()
    } else {
        FlipConfig::vertical()
    };
    s.image = if s.image.dtype() == rivet_core::DType::F32 {
        let axis = match (s.axis_order, horizontal) {
            (ImageAxisOrder::Hwc, true) => 1,
            (ImageAxisOrder::Hwc, false) => 0,
            (ImageAxisOrder::Chw, true) => 2,
            (ImageAxisOrder::Chw, false) => 1,
        };
        s.image.flip(&[axis])?
    } else {
        cfg.apply_with_axis_order(image(&s), s.axis_order)?
            .into_decoded()?
            .image
    };
    if let Some(mask) = &s.masks {
        s.masks = Some(masks_map(mask, h, w, |y, x| {
            Some(if horizontal {
                (y, w - 1 - x)
            } else {
                (h - 1 - y, x)
            })
        })?);
    }
    let (m, k) = if horizontal {
        (
            [[-1., 0., w as f64], [0., 1., 0.], [0., 0., 1.]],
            [[-1., 0., w as f64 - 1.], [0., 1., 0.], [0., 0., 1.]],
        )
    } else {
        (
            [[1., 0., 0.], [0., -1., h as f64], [0., 0., 1.]],
            [[1., 0., 0.], [0., -1., h as f64 - 1.], [0., 0., 1.]],
        )
    };
    targets(s, m, k, false)
}
pub(super) fn warp_targets(
    mut s: StructuredSample,
    m_inv: Matrix,
    projective: bool,
) -> RivetResult<StructuredSample> {
    let (h, w) = s.size()?;
    if let Some(mask) = &s.masks {
        let md = mask.dims();
        let ih = md[md.len() - 2];
        let iw = md[md.len() - 1];
        let denominator = [0.5 * iw as f32, 0.5 * ih as f32];
        let mut scaled = [[0f32; 3]; 2];
        for axis in 0..2 {
            for j in 0..2 {
                scaled[axis][j] = m_inv[axis][j] as f32 / denominator[axis];
            }
            let offset = if projective {
                m_inv[axis][2]
            } else {
                m_inv[axis][2]
                    + m_inv[axis][0] * (w as f64 * 0.5)
                    + m_inv[axis][1] * (h as f64 * 0.5)
                    - if axis == 0 {
                        iw as f64 * 0.5
                    } else {
                        ih as f64 * 0.5
                    }
            };
            scaled[axis][2] = offset as f32 / denominator[axis];
        }
        s.masks = Some(masks_map(mask, h, w, |y, x| {
            // Mirror torchvision's float32 normalized grid construction and
            // grid_sample(align_corners=false), including rounding at ties.
            let bx = if projective {
                x as f32 + 0.5
            } else {
                x as f32 + 0.5 - w as f32 * 0.5
            };
            let by = if projective {
                y as f32 + 0.5
            } else {
                y as f32 + 0.5 - h as f32 * 0.5
            };
            let d = if projective {
                bx * m_inv[2][0] as f32 + by * m_inv[2][1] as f32 + m_inv[2][2] as f32
            } else {
                1.0
            };
            let mut source = [0f32; 2];
            for axis in 0..2 {
                let norm = bx * scaled[axis][0] + by * scaled[axis][1] + scaled[axis][2];
                let norm = if projective { norm / d - 1.0 } else { norm };
                let extent = if axis == 0 { iw as f32 } else { ih as f32 };
                source[axis] = (((norm + 1.0) * extent - 1.0) * 0.5).round_ties_even();
            }
            let [sx, sy] = source;
            if sx < 0.0 || sy < 0.0 || !sx.is_finite() || !sy.is_finite() {
                None
            } else {
                Some((sy as usize, sx as usize))
            }
        })?);
    }
    let m = inverse(m_inv)?;
    targets(s, m, m, true)
}

pub(super) fn affine_inverse(c: &AffineConfig, w: usize, h: usize) -> Matrix {
    let a = c.angle.to_radians();
    let sx = c.shear[0].to_radians();
    let sy = c.shear[1].to_radians();
    let aa = (a - sy).cos() / sy.cos();
    let b = -(a - sy).cos() * sx.tan() / sy.cos() - a.sin();
    let cc = (a - sy).sin() / sy.cos();
    let d = -(a - sy).sin() * sx.tan() / sy.cos() + a.cos();
    let m = [d / c.scale, -b / c.scale, -cc / c.scale, aa / c.scale];
    let [cx, cy] = c.center.unwrap_or([w as f64 * 0.5, h as f64 * 0.5]);
    affine_matrix([
        m[0],
        m[1],
        m[0] * (-cx - c.translate[0]) + m[1] * (-cy - c.translate[1]) + cx,
        m[2],
        m[3],
        m[2] * (-cx - c.translate[0]) + m[3] * (-cy - c.translate[1]) + cy,
    ])
}
pub(super) fn validate_occlusion_fill(s: &StructuredSample, fill: Option<i64>) -> RivetResult<()> {
    if fill.is_some_and(|v| u8::try_from(v).is_err())
        && s.masks
            .as_ref()
            .is_some_and(|mask| mask.dtype() == rivet_core::DType::U8)
    {
        return Err(invalid_argument(
            "occlusion mask fill must be in [0, 255] for uint8 masks",
        ));
    }
    Ok(())
}

pub(super) fn apply_occlusion(
    s: StructuredSample,
    h: usize,
    w: usize,
    fill: Option<i64>,
    covered: impl Fn(usize, usize) -> bool,
) -> RivetResult<StructuredSample> {
    apply_occlusion_with_visibility(s, h, w, fill, 0.0, covered)
}

pub(super) fn apply_occlusion_with_visibility(
    mut s: StructuredSample,
    h: usize,
    w: usize,
    fill: Option<i64>,
    min_visibility: f64,
    covered: impl Fn(usize, usize) -> bool,
) -> RivetResult<StructuredSample> {
    validate_occlusion_fill(&s, fill)?;
    if let (Some(mask), Some(fill)) = (&s.masks, fill) {
        s.masks = Some(masks_map_fill(mask, h, w, fill, |y, x| {
            (!covered(y, x)).then_some((y, x))
        })?);
    }
    if let Some(boxes) = s.xyxy()? {
        let boxes = boxes.to_vec::<f32>()?;
        let keep: Vec<_> = boxes
            .chunks_exact(4)
            .enumerate()
            .filter_map(|(i, b)| {
                let xs = (b[0].floor() as usize).min(w)..(b[2].ceil() as usize).min(w);
                let ys = (b[1].floor() as usize).min(h)..(b[3].ceil() as usize).min(h);
                if xs.is_empty() || ys.is_empty() {
                    return Some(i);
                }
                let survives = if min_visibility == 0.0 {
                    ys.into_iter().any(|y| xs.clone().any(|x| !covered(y, x)))
                } else {
                    let area = xs.len() as f64 * ys.len() as f64;
                    let visible = ys
                        .flat_map(|y| xs.clone().map(move |x| (y, x)))
                        .filter(|&(y, x)| !covered(y, x))
                        .count();
                    visible > 0 && visible as f64 / area >= min_visibility
                };
                survives.then_some(i)
            })
            .collect();
        if keep.len() != boxes.len() / 4 {
            s = s.retain_instances(&keep)?;
        }
    }
    Ok(s)
}

pub(super) fn transpose(mut s: StructuredSample) -> RivetResult<StructuredSample> {
    let order = match s.axis_order {
        ImageAxisOrder::Hwc => [1, 0, 2],
        ImageAxisOrder::Chw => [0, 2, 1],
    };
    s.image = s.image.permute(&order)?;
    if let Some(mask) = &s.masks {
        s.masks = Some(if mask.rank() == 2 {
            mask.permute(&[1, 0])?
        } else {
            mask.permute(&[0, 2, 1])?
        });
    }
    let swap = [[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]];
    targets(s, swap, swap, false)
}

pub(super) fn rotate_quarters(s: StructuredSample, turns: usize) -> RivetResult<StructuredSample> {
    match turns {
        0 => Ok(s),
        1 => flip(transpose(s)?, false),
        2 => flip(flip(s, true)?, false),
        _ => flip(transpose(s)?, true),
    }
}

pub(super) fn remap_structured(
    mut sample: StructuredSample,
    map: &impl SpatialRemap,
    interpolation: InterpolationMode,
    fill: u8,
    fill_mask: i64,
) -> RivetResult<StructuredSample> {
    let (h, w) = sample.size()?;
    if sample.image.dtype() != rivet_core::DType::U8 {
        return Err(invalid_argument(
            "structured spatial remap requires uint8 input",
        ));
    }
    if map.is_identity() {
        return Ok(sample);
    }
    validate_occlusion_fill(&sample, Some(fill_mask))?;
    let backward: Vec<_> = (0..h)
        .flat_map(|y| (0..w).map(move |x| map.source([x as f64, y as f64])))
        .collect();
    sample.image = remap_image(
        image(&sample),
        sample.axis_order,
        (h, w),
        interpolation,
        fill,
        |x, y| backward[y * w + x],
    )?
    .into_decoded()?
    .image;
    remap_targets(sample, map, fill_mask, false, |x, y| backward[y * w + x])
}

pub(super) fn remap_targets(
    mut sample: StructuredSample,
    map: &impl SpatialRemap,
    fill_mask: i64,
    tensor_nearest: bool,
    source: impl Fn(usize, usize) -> [f64; 2],
) -> RivetResult<StructuredSample> {
    let (h, w) = sample.size()?;
    if let Some(mask) = &sample.masks {
        sample.masks = Some(masks_map_fill(mask, h, w, fill_mask, |y, x| {
            let [sx, sy] = source(x, y);
            let round = |v: f64| {
                if tensor_nearest {
                    f64::from((v as f32).round_ties_even())
                } else {
                    (v + 0.5).floor()
                }
            };
            let (sx, sy) = (round(sx), round(sy));
            (sx >= 0.0 && sy >= 0.0).then_some((sy as usize, sx as usize))
        })?);
    }
    if let Some(boxes) = sample.xyxy()? {
        let boxes = boxes.to_vec::<f32>()?;
        let mut keep = Vec::new();
        let mut mapped = Vec::new();
        for (i, b) in boxes.chunks_exact(4).enumerate() {
            let [l, t, r, b] = [b[0], b[1], b[2], b[3]].map(f64::from);
            let (l, t, r, b) = (l.max(0.0), t.max(0.0), r.min(w as f64), b.min(h as f64));
            let mut bounds = [w, h, 0, 0];
            for p in 0..h * w {
                let (x, y) = (p % w, p / w);
                let [sx, sy] = source(x, y);
                let (sx, sy) = (sx + 0.5, sy + 0.5);
                if sx >= l && sx < r && sy >= t && sy < b {
                    let (x, y) = (p % w, p / w);
                    bounds = [
                        bounds[0].min(x),
                        bounds[1].min(y),
                        bounds[2].max(x + 1),
                        bounds[3].max(y + 1),
                    ];
                }
            }
            if bounds[0] < bounds[2] && bounds[1] < bounds[3] {
                keep.push(i);
                mapped.extend(bounds.map(|v| v as f32));
            }
        }
        let b = Tensor::from_vec(mapped, [keep.len(), 4], &Device::Cpu)?;
        if keep.len() != boxes.len() / 4 {
            sample = sample.retain_instances(&keep)?;
        }
        sample.boxes = Some(convert_boxes(&b, BoxFormat::Xyxy, sample.box_format)?);
    }
    if let Some(points) = &sample.keypoints {
        let values = points.to_vec::<f32>()?;
        let out: Vec<_> = values
            .chunks_exact(2)
            .flat_map(|p| {
                map.destination([f64::from(p[0]), f64::from(p[1])])
                    .map(|v| v as f32)
            })
            .collect();
        if out.iter().any(|v| !v.is_finite()) {
            return Err(invalid_argument(
                "spatial remap produced a nonfinite keypoint",
            ));
        }
        sample.keypoints = Some(Tensor::from_vec(out, points.dims(), &Device::Cpu)?);
    }
    Ok(sample)
}
