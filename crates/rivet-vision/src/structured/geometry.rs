use super::annotations::{BoxFormat, StructuredSample, convert_boxes};
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::pipeline::op::{ImageOp, PipelineImageState, SampleContext};
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageSample};
use crate::transforms::*;
use rivet_core::{CpuStorageRef, Device, Tensor};
use rivet_data::random::{OpKey, RandomStream};

type Matrix = [[f64; 3]; 3];
fn image(sample: &StructuredSample) -> ImageSample {
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
fn inverse(m: Matrix) -> RivetResult<Matrix> {
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
fn affine_matrix(m: [f64; 6]) -> Matrix {
    [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [0., 0., 1.]]
}
fn targets(
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

fn masks_map_fill<F>(
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
fn pad(s: StructuredSample, p: &PadConfig) -> RivetResult<StructuredSample> {
    pad_with_mask_fill(s, p, 0)
}

fn pad_with_mask_fill(
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
fn flip(mut s: StructuredSample, horizontal: bool) -> RivetResult<StructuredSample> {
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
fn warp_targets(
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

fn affine_inverse(c: &AffineConfig, w: usize, h: usize) -> Matrix {
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
fn apply_sequence(
    mut sample: StructuredSample,
    ops: &[ImageOp],
    ctx: &SampleContext,
    parent: OpKey,
) -> RivetResult<StructuredSample> {
    let mut counts = std::collections::HashMap::<&str, u32>::new();
    for op in ops {
        let count = counts.entry(op.name()).or_default();
        let child = parent.derive(OpKey::from_parts(op.name(), *count));
        *count += 1;
        let mut rng = ctx.stream(child);
        sample = apply(sample, op, &mut rng, ctx, child)?;
    }
    Ok(sample)
}
pub(super) fn apply(
    mut s: StructuredSample,
    op: &ImageOp,
    rng: &mut RandomStream,
    ctx: &SampleContext,
    key: OpKey,
) -> RivetResult<StructuredSample> {
    op.validate()?;
    let (h, w) = s.size()?;
    match op {
        ImageOp::Resize(c) => {
            let (ow, oh) = c.output_dimensions(w as u32, h as u32)?;
            resize(s, ow, oh, c.interpolation)
        }
        ImageOp::Crop(c) => crop(s, c.x, c.y, c.width, c.height),
        ImageOp::CenterCrop(c) => {
            let pw = (c.width as usize).saturating_sub(w) as u32;
            let ph = (c.height as usize).saturating_sub(h) as u32;
            if pw > 0 || ph > 0 {
                s = pad(
                    s,
                    &PadConfig::with_sides(pw / 2, ph / 2, pw - pw / 2, ph - ph / 2),
                )?;
            }
            let (h, w) = s.size()?;
            let round = |d: u32| d / 2 + u32::from(d % 4 == 3);
            crop(
                s,
                round(w as u32 - c.width),
                round(h as u32 - c.height),
                c.width,
                c.height,
            )
        }
        ImageOp::Pad(c) => pad(s, c),
        ImageOp::Flip(c) => flip(s, c.direction == FlipDirection::Horizontal),
        ImageOp::RandomHorizontalFlip(c) => {
            if rng.gen_bool(c.probability)? {
                flip(s, true)
            } else {
                Ok(s)
            }
        }
        ImageOp::RandomCrop(c) => {
            let [l, t, r, b] = c.padding_sides.unwrap_or([c.padding; 4]);
            let mut p = PadConfig::with_sides(l, t, r, b).with_mode(c.padding_mode);
            p = if c.fill.len() == 1 {
                p.with_fill(c.fill[0])
            } else {
                p.with_fill_channels([c.fill[0], c.fill[1], c.fill[2]])
            };
            if l | t | r | b != 0 {
                s = pad(s, &p)?;
            }
            let (h, w) = s.size()?;
            if c.pad_if_needed {
                if w < c.width as usize {
                    s = pad(
                        s,
                        &PadConfig {
                            left: c.width - w as u32,
                            right: c.width - w as u32,
                            top: 0,
                            bottom: 0,
                            ..p
                        },
                    )?;
                }
                if h < c.height as usize {
                    s = pad(
                        s,
                        &PadConfig {
                            top: c.height - h as u32,
                            bottom: c.height - h as u32,
                            left: 0,
                            right: 0,
                            ..p
                        },
                    )?;
                }
            }
            let (h, w) = s.size()?;
            if w < c.width as usize || h < c.height as usize {
                return Err(invalid_shape("random_crop exceeds padded canvas"));
            }
            let y = rng.gen_range_usize(0..h - c.height as usize + 1)?;
            let x = rng.gen_range_usize(0..w - c.width as usize + 1)?;
            crop(s, x as u32, y as u32, c.width, c.height)
        }
        ImageOp::RandomResizedCrop(c) => {
            let p = c.resolve(w as u32, h as u32, rng);
            resize(
                crop(s, p.x, p.y, p.width, p.height)?,
                c.width,
                c.height,
                c.interpolation,
            )
        }
        ImageOp::Affine(c) => {
            let m = affine_inverse(c, w, h);
            s.image = c.apply(image(&s), s.axis_order)?.into_decoded()?.image;
            warp_targets(s, m, false)
        }
        ImageOp::RandomAffine(c) => {
            let dr = c.degree_range.unwrap_or([-c.degrees, c.degrees]);
            let mut uniform = |a: f32, b: f32| a + (b - a) * rng.next_f32();
            let angle = uniform(dr[0], dr[1]) as f64;
            let tx = (uniform(-c.translate[0], c.translate[0]) as f64 * w as f64).round_ties_even();
            let ty = (uniform(-c.translate[1], c.translate[1]) as f64 * h as f64).round_ties_even();
            let scale = uniform(c.scale[0], c.scale[1]) as f64;
            let shear = c
                .shear_ranges
                .unwrap_or([[-c.shear[0], c.shear[0]], [-c.shear[1], c.shear[1]]]);
            let sh = [
                uniform(shear[0][0], shear[0][1]) as f64,
                uniform(shear[1][0], shear[1][1]) as f64,
            ];
            let mut cfg =
                AffineConfig::new(angle, [tx, ty], scale, sh).with_options(c.interpolation, c.fill);
            cfg.fill_channels = c.fill_channels;
            apply(s, &ImageOp::Affine(cfg), rng, ctx, key)
        }
        ImageOp::ArbitraryRotate(c) => {
            let (m, _, _) = crate::transforms::rotation_inverse(w, h, c.angle, c.expand, c.center);
            s.image = c.apply(image(&s), s.axis_order)?.into_decoded()?.image;
            warp_targets(s, affine_matrix(m), false)
        }
        ImageOp::RandomRotation(c) => {
            let mut cfg = ArbitraryRotateConfig::new(
                c.degrees[0] + (c.degrees[1] - c.degrees[0]) * rng.next_f32(),
            )
            .with_options(c.expand, c.interpolation, c.fill);
            cfg.center = c.center;
            cfg.fill_channels = c.fill_channels;
            apply(s, &ImageOp::ArbitraryRotate(cfg), rng, ctx, key)
        }
        ImageOp::Rotate(c) => {
            let angle = match c.angle {
                RotationAngle::Deg90 => 90.,
                RotationAngle::Deg180 => 180.,
                RotationAngle::Deg270 => 270.,
            };
            let cfg =
                ArbitraryRotateConfig::new(angle).with_options(true, InterpolationMode::Nearest, 0);
            apply(s, &ImageOp::ArbitraryRotate(cfg), rng, ctx, key)
        }
        ImageOp::Perspective(c) => {
            let m = crate::transforms::homography(&c.end_points, &c.start_points)
                .ok_or_else(|| invalid_argument("singular perspective"))?;
            s.image = c.apply(image(&s), s.axis_order)?.into_decoded()?.image;
            warp_targets(s, m, true)
        }
        ImageOp::RandomPerspective(c) => {
            if !rng.gen_bool(c.probability)? || w < 2 || h < 2 || c.distortion_scale == 0.0 {
                return Ok(s);
            }
            let mx = (w - 1) as f32;
            let my = (h - 1) as f32;
            let hx = (c.distortion_scale * (w / 2) as f32) as usize;
            let hy = (c.distortion_scale * (h / 2) as f32) as usize;
            let mut draw = |v| rng.gen_range_usize(0..v + 1).map(|x| x as f32);
            let start = [
                Point2::new(0., 0.),
                Point2::new(mx, 0.),
                Point2::new(mx, my),
                Point2::new(0., my),
            ];
            let end = [
                Point2::new(draw(hx)?, draw(hy)?),
                Point2::new(mx - hx as f32 + draw(hx)?, draw(hy)?),
                Point2::new(mx - hx as f32 + draw(hx)?, my - hy as f32 + draw(hy)?),
                Point2::new(draw(hx)?, my - hy as f32 + draw(hy)?),
            ];
            let mut cfg = PerspectiveConfig::new(start, end).with_options(c.interpolation, c.fill);
            cfg.fill_channels = c.fill_channels;
            apply(s, &ImageOp::Perspective(cfg), rng, ctx, key)
        }
        ImageOp::RandomApply { probability, ops } => {
            if rng.gen_bool(*probability)? {
                s = apply_sequence(s, ops, ctx, key)?;
            }
            Ok(s)
        }
        ImageOp::RandomChoice { choices, weights } => {
            let index = if let Some(weights) = weights {
                let total: f64 = weights.iter().sum();
                let r = rng.next_f64() * total;
                let mut cumulative = 0.;
                let mut selected = weights.len() - 1;
                for (i, v) in weights.iter().enumerate() {
                    cumulative += v;
                    if r < cumulative {
                        selected = i;
                        break;
                    }
                }
                selected
            } else {
                rng.gen_range_usize(0..choices.len())?
            };
            apply_sequence(
                s,
                &choices[index],
                ctx,
                key.derive(OpKey::from_parts("branch", index as u32)),
            )
        }
        ImageOp::RandomOrder { ops } => {
            let mut order: Vec<_> = (0..ops.len()).collect();
            for i in (1..order.len()).rev() {
                let j = rng.gen_range_usize(0..i + 1)?;
                order.swap(i, j);
            }
            for i in order {
                let child = key.derive(OpKey::from_parts(ops[i].name(), i as u32));
                let mut child_rng = ctx.stream(child);
                s = apply(s, &ops[i], &mut child_rng, ctx, child)?;
            }
            Ok(s)
        }
        ImageOp::V2(V2Config::XyMasking(config)) => {
            let mask = config.sample(h, w, rng)?;
            s.image = config.apply(&s.image, s.axis_order, &mask)?;
            apply_occlusion(s, h, w, config.fill_mask, |y, x| mask.covers(y, x))
        }
        ImageOp::V2(V2Config::PixelDropout(config)) => {
            let fill = config.mask_drop_value;
            validate_occlusion_fill(&s, fill)?;
            let mut dropped = Vec::new();
            let record =
                !config.per_channel && (s.boxes.is_some() || (fill.is_some() && s.masks.is_some()));
            s.image = config.apply(&s.image, s.axis_order, rng, record.then_some(&mut dropped))?;
            if record && config.dropout_prob > 0.0 {
                apply_occlusion(s, h, w, fill, |y, x| dropped[y * w + x])
            } else {
                Ok(s)
            }
        }
        ImageOp::V2(V2Config::RandomScale {
            range,
            y,
            interpolation,
        }) => {
            let (w, h) = crate::transforms::sample_scale(w, h, *range, *y, rng)?;
            resize(s, w, h, *interpolation)
        }
        ImageOp::V2(V2Config::LongestMaxSize(config)) => {
            let (w, h) = config.sample(w, h, rng)?;
            resize(s, w, h, config.interpolation)
        }
        ImageOp::V2(V2Config::PadIfNeeded(config)) => {
            let padding = config.sample(h, w, rng)?;
            pad_with_mask_fill(s, &padding, config.fill_mask)
        }
        ImageOp::V2(V2Config::LetterBox(config)) => {
            let (h, w) = s.size()?;
            let (scale, padding) = config.resolve(w, h)?;
            let resized = resize(s, scale.width, scale.height, scale.interpolation)?;
            pad_with_mask_fill(resized, &padding, config.mask_fill)
        }
        ImageOp::V2(V2Config::RandomResize {
            min_size,
            max_size,
            interpolation,
        }) => {
            let size = rng.gen_range_usize(*min_size as usize..*max_size as usize)? as u32;
            apply(
                s,
                &ImageOp::Resize(ResizeConfig::short_edge(size, None, *interpolation)),
                rng,
                ctx,
                key,
            )
        }
        ImageOp::V2(V2Config::RandomShortestSize {
            min_sizes,
            max_size,
            interpolation,
        }) => {
            let short = min_sizes[rng.gen_range_usize(0..min_sizes.len())?] as f64;
            let ratio = (short / h.min(w) as f64)
                .min(max_size.map_or(f64::INFINITY, |v| v as f64 / h.max(w) as f64));
            resize(
                s,
                (w as f64 * ratio) as u32,
                (h as f64 * ratio) as u32,
                *interpolation,
            )
        }
        ImageOp::V2(V2Config::ScaleJitter {
            target_width,
            target_height,
            scale_range: (a, b),
            interpolation,
        }) => {
            let r = (*target_width as f64 / w as f64).min(*target_height as f64 / h as f64)
                * (a + rng.next_f64() * (b - a));
            resize(
                s,
                (w as f64 * r) as u32,
                (h as f64 * r) as u32,
                *interpolation,
            )
        }
        ImageOp::V2(V2Config::RandomZoomOut {
            fill,
            side_range: (a, b),
            probability,
        }) => {
            if !rng.gen_bool(*probability)? {
                return Ok(s);
            }
            let r = a + rng.next_f64() * (b - a);
            let ow = w as f64 * r;
            let oh = h as f64 * r;
            if ow > u32::MAX as f64 || oh > u32::MAX as f64 {
                return Err(invalid_shape("zoom-out canvas too large"));
            }
            let dx = ow as u32 - w as u32;
            let dy = oh as u32 - h as u32;
            let l = (dx as f64 * rng.next_f64()) as u32;
            let t = (dy as f64 * rng.next_f64()) as u32;
            let mut cfg = PadConfig::with_sides(l, t, dx - l, dy - t);
            cfg = if fill.len() == 1 {
                cfg.with_fill(fill[0])
            } else {
                cfg.with_fill_channels([fill[0], fill[1], fill[2]])
            };
            pad(s, &cfg)
        }
        ImageOp::Decode(_)
        | ImageOp::FiveCrop(_)
        | ImageOp::TenCrop(_)
        | ImageOp::ElasticTransform(_)
        | ImageOp::AutoAugment(_)
        | ImageOp::RandAugment(_)
        | ImageOp::TrivialAugmentWide(_)
        | ImageOp::AugMix(_) => Err(invalid_argument(format!(
            "{} is not supported for structured samples",
            op.name()
        ))),
        _ => {
            let state = PipelineImageState::Decoded {
                dtype: s.image.dtype(),
                axis_order: s.axis_order,
            };
            s.image = if op.execution_kind() == crate::pipeline::op::ExecutionKind::Batch {
                op.apply_batch(s.image.unsqueeze(0)?, s.axis_order)?
                    .squeeze(0)?
            } else {
                op.apply_sample_with_key(image(&s), ctx, state, Some(key))?
                    .into_decoded()?
                    .image
            };
            if let ImageOp::Layout(c) = op {
                s.axis_order = c.axis_order;
            }
            Ok(s)
        }
    }
}

fn validate_occlusion_fill(s: &StructuredSample, fill: Option<i64>) -> RivetResult<()> {
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

fn apply_occlusion(
    mut s: StructuredSample,
    h: usize,
    w: usize,
    fill: Option<i64>,
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
                (xs.is_empty()
                    || ys.is_empty()
                    || ys.into_iter().any(|y| xs.clone().any(|x| !covered(y, x))))
                .then_some(i)
            })
            .collect();
        if keep.len() != boxes.len() / 4 {
            s = s.retain_instances(&keep)?;
        }
    }
    Ok(s)
}
