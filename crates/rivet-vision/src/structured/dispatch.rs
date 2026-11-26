use super::annotations::{BoxFormat, StructuredSample, convert_boxes};
use super::geometry::{
    affine_inverse, affine_matrix, apply_occlusion, crop, crop_visible_instances, flip, image,
    inverse, masks_map_fill, pad, pad_with_mask_fill, remap_structured, remap_targets, resize,
    rotate_quarters, targets, transpose, validate_occlusion_fill, warp_targets,
};
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::pipeline::op::{ImageOp, PipelineImageState, SampleContext};
use crate::sample::image::ImageAxisOrder;
use crate::transforms::{
    AffineConfig, ArbitraryRotateConfig, FlipDirection, InterpolationMode, PadConfig,
    PerspectiveConfig, Point2, ResizeConfig, RotationAngle, V2Config,
};
use rivet_core::{Device, Tensor};
use rivet_data::random::{OpKey, RandomStream};

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
        ImageOp::Callback(op) => {
            if let Some(seed) = op.sample(rng)? {
                s.image = op.apply("image", &s.image, seed)?;
                for (name, slot) in [
                    ("mask", &mut s.masks),
                    ("bboxes", &mut s.boxes),
                    ("keypoints", &mut s.keypoints),
                    ("labels", &mut s.labels),
                ] {
                    if let Some(value) = slot {
                        *value = op.apply(name, value, seed)?;
                    }
                }
                s.validate()?;
            }
            Ok(s)
        }
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
        ImageOp::V2(V2Config::Morphological(config)) => {
            s.image = config.apply(&s.image, s.axis_order)?;
            if let Some(mask) = &s.masks {
                s.masks = Some(config.apply(mask, ImageAxisOrder::Chw)?);
            }
            Ok(s)
        }
        ImageOp::ElasticTransform(config) => {
            use crate::transforms::spatial::remap::SpatialRemap;
            if s.image.dtype() != rivet_core::DType::U8 {
                return Err(invalid_argument("elastic transform requires uint8 input"));
            }
            let map = config.sample(h, w, rng)?;
            if map.is_identity() {
                return Ok(s);
            }
            s.image = config
                .apply_sampled(image(&s).into_decoded()?, s.axis_order, &map)?
                .into_decoded()?
                .image;
            remap_targets(s, &map, 0, true, |x, y| map.source([x as f64, y as f64]))
        }
        ImageOp::V2(V2Config::WaterRefraction(config)) => {
            let map = config.sample(h, w, rng)?;
            remap_structured(s, &map, config.interpolation, config.fill, config.fill_mask)
        }
        ImageOp::V2(V2Config::RandomGridShuffle(config)) => {
            let map = config.sample(h, w, rng)?;
            if map.identity {
                return Ok(s);
            }
            s.image = map.apply(&s.image, s.axis_order)?;
            if let Some(mask) = &s.masks {
                s.masks = Some(map.apply(mask, ImageAxisOrder::Chw)?);
            }
            if let Some(boxes) = s.xyxy()? {
                let values = boxes.to_vec::<f32>()?;
                let mut keep = Vec::new();
                let mut mapped = Vec::new();
                for (i, b) in values.chunks_exact(4).enumerate() {
                    if let Some(b) = map.box_bounds(b) {
                        keep.push(i);
                        mapped.extend(b);
                    }
                }
                if keep.len() != values.len() / 4 {
                    s = s.retain_instances(&keep)?;
                }
                let b = Tensor::from_vec(mapped, [keep.len(), 4], &Device::Cpu)?;
                s.boxes = Some(convert_boxes(&b, BoxFormat::Xyxy, s.box_format)?);
            }
            if let Some(points) = &s.keypoints {
                let mapped: Vec<_> = points
                    .to_vec::<f32>()?
                    .chunks_exact(2)
                    .flat_map(|p| map.keypoint([p[0], p[1]]))
                    .collect();
                s.keypoints = Some(Tensor::from_vec(mapped, points.dims(), &Device::Cpu)?);
            }
            Ok(s)
        }
        ImageOp::V2(V2Config::PixelSpread(config)) => {
            let map = config.sample(h, w, rng)?;
            remap_structured(
                s,
                &map,
                InterpolationMode::Nearest,
                config.fill,
                config.fill_mask,
            )
        }
        ImageOp::V2(V2Config::ThinPlateSpline(config)) => {
            let map = config.sample(h, w, rng);
            remap_structured(s, &map, config.interpolation, config.fill, config.fill_mask)
        }
        ImageOp::V2(V2Config::GridElasticDeform(config)) => {
            let map = config.sample(h, w, rng);
            remap_structured(s, &map, config.interpolation, config.fill, config.fill_mask)
        }
        ImageOp::V2(V2Config::PiecewiseAffine(config)) => {
            let map = config.sample(h, w, rng)?;
            remap_structured(s, &map, config.interpolation, config.fill, config.fill_mask)
        }
        ImageOp::V2(V2Config::GridDistortion(config)) => {
            let map = config.sample(h, w, rng);
            remap_structured(s, &map, config.interpolation, config.fill, config.fill_mask)
        }
        ImageOp::V2(V2Config::OpticalDistortion(config)) => {
            let map = config.sample(h, w, rng);
            remap_structured(s, &map, config.interpolation, config.fill, config.fill_mask)
        }
        ImageOp::V2(V2Config::SafeRotate(config)) => {
            let matrix = config.sample(w, h, rng);
            if s.image.dtype() != rivet_core::DType::U8 {
                return Err(invalid_argument("SafeRotate requires uint8 input"));
            }
            if matrix == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
                return Ok(s);
            }
            validate_occlusion_fill(&s, Some(config.fill_mask))?;
            s.image = config
                .apply(image(&s), s.axis_order, matrix)?
                .into_decoded()?
                .image;
            if let Some(mask) = &s.masks {
                let map =
                    crate::transforms::pillow_affine_map(matrix, w, h, InterpolationMode::Nearest);
                s.masks = Some(masks_map_fill(mask, h, w, config.fill_mask, |y, x| {
                    let [sx, sy] = map(x, y);
                    (sx >= 0.0 && sy >= 0.0).then_some((sy as usize, sx as usize))
                })?);
            }
            let m = inverse(affine_matrix(matrix))?;
            let mut keypoints = m;
            for row in keypoints.iter_mut().take(2) {
                row[2] += (row[0] + row[1]) * 0.5 - 0.5;
            }
            targets(s, m, keypoints, true)
        }
        ImageOp::V2(V2Config::SquareSymmetry(element)) => {
            let (reflect, turns) =
                crate::transforms::SquareSymmetry::sample(*element, rng)?.components();
            if reflect {
                s = transpose(s)?;
            }
            rotate_quarters(s, turns)
        }
        ImageOp::V2(V2Config::RandomRotate90) => rotate_quarters(s, rng.gen_range_usize(0..4)?),
        ImageOp::V2(V2Config::Transpose) => transpose(s),
        ImageOp::V2(V2Config::GridMask(config)) => {
            let mask = config.sample(h, w, rng)?;
            s.image = config.apply(&s.image, s.axis_order, mask)?;
            apply_occlusion(s, h, w, config.fill_mask, |y, x| mask.covers(y, x))
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
        ImageOp::V2(V2Config::RandomCropFromBorders { fractions }) => {
            let [x, y, cw, ch] = crate::transforms::sample_border_crop(h, w, *fractions, rng)?;
            if (x, y, cw as usize, ch as usize) == (0, 0, w, h) {
                return Ok(s);
            }
            let boxes = s.xyxy()?.map(|b| b.to_vec::<f32>()).transpose()?;
            let (x, y, cw, ch) = (x as usize, y as usize, cw as usize, ch as usize);
            crop_visible_instances(s, boxes.as_deref(), [x, y, x + cw, y + ch])
        }
        ImageOp::V2(V2Config::RandomSizedCrop(config)) => {
            let [x, y, cw, ch] = config.sample(h, w, rng)?;
            let boxes = s.xyxy()?.map(|b| b.to_vec::<f32>()).transpose()?;
            let (x, y, cw, ch) = (x as usize, y as usize, cw as usize, ch as usize);
            let cropped = crop_visible_instances(s, boxes.as_deref(), [x, y, x + cw, y + ch])?;
            resize(cropped, config.width, config.height, config.interpolation)
        }
        ImageOp::V2(V2Config::CropAndPad(config)) => {
            let ([x, y, cw, ch], padding) = config.sample(h, w, rng)?;
            if (x, y, cw as usize, ch as usize) != (0, 0, w, h) {
                let boxes = s.xyxy()?.map(|b| b.to_vec::<f32>()).transpose()?;
                let (x, y, cw, ch) = (x as usize, y as usize, cw as usize, ch as usize);
                s = crop_visible_instances(s, boxes.as_deref(), [x, y, x + cw, y + ch])?;
            }
            if padding.left != 0 || padding.right != 0 || padding.top != 0 || padding.bottom != 0 {
                s = pad_with_mask_fill(s, &padding, config.fill_mask)?;
            }
            if config.keep_size && s.size()? != (h, w) {
                s = resize(s, w as u32, h as u32, config.interpolation)?;
            }
            Ok(s)
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
