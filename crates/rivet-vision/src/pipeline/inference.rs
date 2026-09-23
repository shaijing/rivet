//! Vision-specific value-property inference and validation.

use rivet_core::DType as CoreDType;
use rivet_plan::{
    AxisOrder, Contiguity, DataType, DeviceCut, LogicalNode, NodeKind, OperatorProperties,
    OperatorStage, PropertyInference, Representation, Residency, ShapeDim, ValueGranularity,
    ValueProperties, ValueShape,
};

use super::logical::FusionGroupPayload;
use super::op::{BatchConfig, ImageOp, IndexOp, PipelineImageState, SourceOp};
use crate::errors::{RivetResult, invalid_pipeline};
use crate::sample::image::ImageAxisOrder;

pub struct VisionPropertyInference {
    num_workers: usize,
}

impl VisionPropertyInference {
    pub fn new(num_workers: usize) -> Self {
        Self { num_workers }
    }
}

impl Default for VisionPropertyInference {
    fn default() -> Self {
        Self::new(0)
    }
}

impl PropertyInference for VisionPropertyInference {
    fn infer_node(
        &self,
        node: &LogicalNode,
        inputs: &[ValueProperties],
    ) -> Result<ValueProperties, String> {
        let result = match node.kind() {
            NodeKind::Source => node
                .payload_as::<SourceOp>()
                .ok_or_else(|| "source node is missing its vision SourceOp payload".to_owned())
                .map(|source| properties_from_source(source))?,
            NodeKind::Index => {
                let _ = node
                    .payload_as::<IndexOp>()
                    .ok_or_else(|| "index node is missing its vision IndexOp payload".to_owned())?;
                single_input(inputs, node.kind())?.clone()
            }
            NodeKind::Op => {
                let input = single_input(inputs, node.kind())?;
                // Property inference describes operator semantics independently
                // of the backend that will execute them. Capability validation
                // decides whether an op is legal after a DeviceCut; shape and
                // dtype inference still need to propagate through that region.
                let mut semantic_input = input.clone();
                semantic_input.residency = Some(Residency::Host);
                if let Some(op) = node.payload_as::<ImageOp>() {
                    let mut output =
                        infer_image_op_with_workers(op, &semantic_input, self.num_workers)
                            .map_err(|error| error.to_string())?;
                    output.residency = input.residency.clone();
                    output
                } else if let Some(group) = node.payload_as::<FusionGroupPayload>() {
                    let mut properties = semantic_input;
                    for op in &group.ops {
                        properties = infer_image_op_with_workers(op, &properties, self.num_workers)
                            .map_err(|error| error.to_string())?;
                    }
                    // Normalize + HWC-to-CHW has a dedicated contiguous output
                    // kernel even though the unfused layout is a strided view.
                    if group.name == "NormalizeToChw" {
                        properties.contiguity = Some(Contiguity::Contiguous);
                    }
                    properties.residency = input.residency.clone();
                    properties
                } else {
                    return Err(
                        "op node is missing its vision ImageOp or FusionGroup payload".to_owned(),
                    );
                }
            }
            NodeKind::Batch => {
                let batch = node
                    .payload_as::<BatchConfig>()
                    .ok_or_else(|| "batch node is missing its BatchConfig payload".to_owned())?;
                infer_batch(batch, single_input(inputs, node.kind())?)
                    .map_err(|error| error.to_string())?
            }
            NodeKind::DeviceCut => {
                let cut = node
                    .payload_as::<DeviceCut>()
                    .ok_or_else(|| "device cut node is missing its DeviceCut payload".to_owned())?;
                let mut properties = single_input(inputs, node.kind())?.clone();
                properties.residency = Some(Residency::Device(cut.target.class.clone()));
                properties
            }
            NodeKind::Cache | NodeKind::Sink => single_input(inputs, node.kind())?.clone(),
        };
        Ok(result)
    }
}

fn single_input(inputs: &[ValueProperties], kind: NodeKind) -> Result<&ValueProperties, String> {
    match inputs {
        [input] => Ok(input),
        [] => Err(format!("{kind:?} node requires one input")),
        _ => Err(format!(
            "{kind:?} node requires one input, got {}",
            inputs.len()
        )),
    }
}

pub(crate) fn properties_from_state(state: PipelineImageState) -> ValueProperties {
    match state {
        PipelineImageState::Encoded => ValueProperties {
            representation: Some(Representation::EncodedImage),
            shape: Some(ValueShape(vec![ShapeDim::Dynamic])),
            contiguity: Some(Contiguity::Contiguous),
            residency: Some(Residency::Host),
            granularity: Some(ValueGranularity::Sample),
            mutability: Some(rivet_plan::Mutability::Immutable),
            operator: Some(OperatorProperties {
                stage: OperatorStage::Source,
                sample_stage_has_work: false,
            }),
            ..ValueProperties::default()
        },
        PipelineImageState::Decoded { dtype, axis_order } => image_properties(
            core_dtype(dtype),
            axis_order,
            dynamic_image_shape(axis_order),
            Contiguity::Contiguous,
        ),
    }
}

fn properties_from_source(source: &SourceOp) -> ValueProperties {
    properties_from_state(source.state())
}

pub(crate) fn state_from_properties(
    properties: &ValueProperties,
) -> RivetResult<PipelineImageState> {
    match properties.representation.as_ref() {
        Some(Representation::EncodedImage) => Ok(PipelineImageState::Encoded),
        Some(Representation::Image) => {
            let dtype = properties
                .dtype
                .as_ref()
                .ok_or_else(|| invalid_pipeline("image properties are missing dtype"))?;
            let axis_order = image_axis(properties.axis_order.as_ref())
                .ok_or_else(|| invalid_pipeline("image properties are missing axis order"))?;
            Ok(PipelineImageState::Decoded {
                dtype: core_dtype_from_property(dtype.clone())?,
                axis_order,
            })
        }
        _ => Err(invalid_pipeline(
            "value properties do not describe an image",
        )),
    }
}

pub(crate) fn infer_image_op(
    op: &ImageOp,
    input: &ValueProperties,
) -> RivetResult<ValueProperties> {
    infer_image_op_with_workers(op, input, 0)
}

fn infer_image_op_with_workers(
    op: &ImageOp,
    input: &ValueProperties,
    num_workers: usize,
) -> RivetResult<ValueProperties> {
    op.validate()?;
    match input.residency.as_ref() {
        Some(Residency::Host | Residency::Unknown) | None => {}
        Some(Residency::Device(_)) => {
            return Err(invalid_pipeline(format!(
                "{} requires host-resident input in the CPU vision planner",
                op.name()
            )));
        }
    }
    match input.granularity {
        Some(ValueGranularity::Sample) => {}
        Some(granularity) => {
            return Err(invalid_pipeline(format!(
                "{} requires sample granularity, current granularity is {granularity}",
                op.name()
            )));
        }
        None => {
            return Err(invalid_pipeline(format!(
                "{} requires known sample granularity",
                op.name()
            )));
        }
    }
    let incoming_operator = input.operator.unwrap_or(OperatorProperties {
        stage: OperatorStage::Source,
        sample_stage_has_work: false,
    });
    if incoming_operator.stage == OperatorStage::Batch
        && op.execution_kind() == super::op::ExecutionKind::Sample
    {
        return Err(invalid_pipeline(format!(
            "{} cannot follow the batch stage; move sample operations before normalize/layout (sample ops require uint8 HWC input)",
            op.name()
        )));
    }
    let input_state = state_from_properties(input)?;
    if matches!(input_state, PipelineImageState::Decoded { .. })
        && !matches!(op, ImageOp::Decode(_))
    {
        validate_rank(input, op.name())?;
    }

    use PipelineImageState::{Decoded, Encoded};
    let mut output = input.clone();
    match op {
        ImageOp::Decode(_) => match input_state {
            Encoded => {
                output = image_properties(
                    DataType::U8,
                    ImageAxisOrder::Hwc,
                    dynamic_image_shape(ImageAxisOrder::Hwc),
                    Contiguity::Contiguous,
                );
            }
            Decoded { .. } => {
                return Err(invalid_pipeline(
                    "Decode requires an encoded image, current state is decoded",
                ));
            }
        },
        ImageOp::Resize(config) => {
            require_u8_hwc(input_state, "Resize")?;
            set_spatial_shape(
                &mut output,
                ShapeDim::Known(config.height as usize),
                ShapeDim::Known(config.width as usize),
            );
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::Crop(config) => {
            require_u8_decoded(input_state, "Crop")?;
            validate_rank(input, "crop")?;
            validate_crop_bounds(
                input,
                config.x,
                config.y,
                config.width,
                config.height,
                "crop",
            )?;
            set_spatial_shape(
                &mut output,
                ShapeDim::Known(config.height as usize),
                ShapeDim::Known(config.width as usize),
            );
            output.contiguity = Some(Contiguity::Strided);
        }
        ImageOp::CenterCrop(config) => {
            require_u8_decoded(input_state, "CenterCrop")?;
            validate_rank(input, "center_crop")?;
            validate_center_crop_bounds(input, config.width, config.height)?;
            set_spatial_shape(
                &mut output,
                ShapeDim::Known(config.height as usize),
                ShapeDim::Known(config.width as usize),
            );
            output.contiguity = Some(Contiguity::Strided);
        }
        ImageOp::Pad(config) => {
            require_u8_or_f32_decoded(input_state, "Pad")?;
            validate_rank(input, "pad")?;
            if let Some(shape) = &mut output.shape {
                match image_axis(input.axis_order.as_ref()) {
                    Some(ImageAxisOrder::Hwc) => {
                        shape.0[0] = add_padding(shape.0[0], config.top, config.bottom);
                        shape.0[1] = add_padding(shape.0[1], config.left, config.right);
                    }
                    Some(ImageAxisOrder::Chw) => {
                        shape.0[1] = add_padding(shape.0[1], config.top, config.bottom);
                        shape.0[2] = add_padding(shape.0[2], config.left, config.right);
                    }
                    None => {}
                }
            }
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::Flip(_) => {
            require_u8_decoded(input_state, "Flip")?;
            validate_rank(input, "flip")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::RandomCrop(config) => {
            require_u8_hwc(input_state, "RandomCrop")?;
            validate_random_crop_fit(input, config.width, config.height, config.padding)?;
            output.shape = Some(ValueShape(vec![
                ShapeDim::Known(config.height as usize),
                ShapeDim::Known(config.width as usize),
                channel_dim(input),
            ]));
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::RandomResizedCrop(config) => {
            require_u8_hwc(input_state, "RandomResizedCrop")?;
            output.shape = Some(ValueShape(vec![
                ShapeDim::Known(config.height as usize),
                ShapeDim::Known(config.width as usize),
                channel_dim(input),
            ]));
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::RandomHorizontalFlip(_) => {
            require_u8_decoded(input_state, "RandomHorizontalFlip")?;
            output.contiguity = Some(Contiguity::Unknown);
        }
        ImageOp::Brightness(_) => {
            require_u8_hwc(input_state, "Brightness")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::Contrast(_) => {
            require_u8_hwc(input_state, "Contrast")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::Hue(_) => {
            require_u8_hwc(input_state, "Hue")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::ColorJitter(_) => {
            require_u8_hwc(input_state, "ColorJitter")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::Invert(_) => {
            require_u8_decoded(input_state, "Invert")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::Posterize(_) => {
            require_u8_decoded(input_state, "Posterize")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::Solarize(_) => {
            require_u8_decoded(input_state, "Solarize")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::Autocontrast(_) => {
            require_u8_decoded(input_state, "Autocontrast")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::Equalize(_) => {
            require_u8_decoded(input_state, "Equalize")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::Sharpness(_) => {
            require_u8_decoded(input_state, "Sharpness")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::ArbitraryRotate(_) => {
            require_u8_decoded(input_state, "ArbitraryRotate")?;
            make_spatial_dynamic(&mut output);
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::RandomAffine(_) => {
            require_u8_decoded(input_state, "RandomAffine")?;
            make_spatial_dynamic(&mut output);
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::Perspective(_) => {
            require_u8_decoded(input_state, "Perspective")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::RandomPerspective(_) => {
            require_u8_decoded(input_state, "RandomPerspective")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::ElasticTransform(_) => {
            require_u8_decoded(input_state, "ElasticTransform")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::RandomApply {
            probability: _,
            ops,
        } => {
            let branch = infer_sequence(ops, input)?;
            output = merge_optional_shape(input, &branch);
            output.contiguity = merge_contiguity(input.contiguity, branch.contiguity);
        }
        ImageOp::RandomChoice { choices } => {
            let mut choices_iter = choices.iter();
            let first_ops = choices_iter.next().ok_or_else(|| {
                crate::errors::invalid_argument("random_choice requires at least one choice")
            })?;
            let first = infer_sequence(first_ops, input)?;
            output = choices_iter.try_fold(first, |acc, ops| {
                let branch = infer_sequence(ops, input)?;
                if acc.dtype != branch.dtype || acc.axis_order != branch.axis_order {
                    return Err(invalid_pipeline(
                        "random_choice choices must produce the same image state",
                    ));
                }
                let mut merged = merge_optional_shape(&acc, &branch);
                merged.contiguity = merge_contiguity(acc.contiguity, branch.contiguity);
                Ok(merged)
            })?;
        }
        ImageOp::RandomOrder { ops } => {
            let mut merged = input.clone();
            for nested in ops {
                let branch = infer_image_op(nested, input)?;
                if branch.dtype != input.dtype || branch.axis_order != input.axis_order {
                    return Err(invalid_pipeline(
                        "RandomOrder nested transforms must preserve image state",
                    ));
                }
                merged = merge_optional_shape(&merged, &branch);
            }
            output = merged;
        }
        ImageOp::GaussianBlur(_) => {
            require_u8_hwc(input_state, "GaussianBlur")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::Grayscale(config) => {
            require_u8_decoded(input_state, "Grayscale")?;
            set_channel_dim(
                &mut output,
                ShapeDim::Known(config.num_output_channels as usize),
            );
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::RandomGrayscale(config) => {
            require_u8_decoded(input_state, "RandomGrayscale")?;
            let current = channel_dim(input);
            let gray = ShapeDim::Known(config.num_output_channels as usize);
            set_channel_dim(
                &mut output,
                if current == gray {
                    gray
                } else {
                    ShapeDim::Dynamic
                },
            );
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::RandomErasing(_) => {
            require_u8_or_f32_decoded(input_state, "RandomErasing")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::ConvertImageDtype(config) => match input_state {
            Encoded => {
                return Err(invalid_pipeline(
                    "ConvertImageDtype requires a decoded image, current state is encoded",
                ));
            }
            Decoded { axis_order, .. } => {
                output.dtype = Some(core_dtype(config.dtype));
                output.axis_order = Some(axis_property(axis_order));
                output.contiguity = Some(Contiguity::Contiguous);
            }
        },
        ImageOp::Rotate(config) => {
            require_u8_hwc(input_state, "Rotate")?;
            if matches!(
                config.angle,
                crate::transforms::RotationAngle::Deg90 | crate::transforms::RotationAngle::Deg270
            ) {
                swap_spatial_shape(&mut output);
            }
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::Normalize(_) => match input_state {
            Encoded => {
                return Err(invalid_pipeline(
                    "Normalize requires a decoded image, current state is encoded",
                ));
            }
            Decoded { axis_order, .. } => {
                output.dtype = Some(DataType::F32);
                output.axis_order = Some(axis_property(axis_order));
                output.contiguity = Some(Contiguity::Contiguous);
            }
        },
        ImageOp::Layout(config) => match input_state {
            Encoded => {
                return Err(invalid_pipeline(
                    "Layout requires a decoded image, current state is encoded",
                ));
            }
            Decoded { .. } => {
                let input_axis = image_axis(input.axis_order.as_ref()).unwrap();
                output.axis_order = Some(axis_property(config.axis_order));
                if input_axis != config.axis_order {
                    reorder_image_shape(&mut output, input_axis, config.axis_order);
                    output.contiguity = Some(Contiguity::Strided);
                }
            }
        },
    }

    output.representation = Some(Representation::Image);
    output.granularity = input.granularity;
    output.residency = input.residency.clone().or(Some(Residency::Host));
    output.mutability = Some(rivet_plan::Mutability::Immutable);
    let noop_layout = matches!(op, ImageOp::Layout(config)
        if image_axis(input.axis_order.as_ref()) == Some(config.axis_order));
    let sample_normalize = matches!(op, ImageOp::Normalize(_))
        && num_workers > 0
        && incoming_operator.sample_stage_has_work
        && incoming_operator.stage != OperatorStage::Batch;
    let is_batch = op.execution_kind() == super::op::ExecutionKind::Batch && !sample_normalize;
    output.operator = Some(if noop_layout {
        incoming_operator
    } else {
        OperatorProperties {
            stage: if is_batch {
                OperatorStage::Batch
            } else {
                OperatorStage::Sample
            },
            sample_stage_has_work: incoming_operator.sample_stage_has_work
                || (op.execution_kind() == super::op::ExecutionKind::Sample && !noop_layout)
                || sample_normalize,
        }
    });
    Ok(output)
}

fn infer_sequence(ops: &[ImageOp], input: &ValueProperties) -> RivetResult<ValueProperties> {
    let mut properties = input.clone();
    for op in ops {
        if op.execution_kind() != super::op::ExecutionKind::Sample {
            return Err(invalid_pipeline(format!(
                "{} nested transforms must be sample-stage operations",
                op.name()
            )));
        }
        properties = infer_image_op(op, &properties)?;
    }
    Ok(properties)
}

fn infer_batch(config: &BatchConfig, input: &ValueProperties) -> RivetResult<ValueProperties> {
    config.validate()?;
    if input.granularity != Some(ValueGranularity::Sample) {
        return Err(invalid_pipeline(format!(
            "batch requires sample granularity, current granularity is {}",
            input
                .granularity
                .map(|granularity| granularity.to_string())
                .unwrap_or_else(|| "unknown".to_owned())
        )));
    }
    if input.representation != Some(Representation::Image) {
        return Err(invalid_pipeline(
            "pipeline must decode images before batching",
        ));
    }
    validate_rank(input, "batch")?;
    let mut output = input.clone();
    let input_shape = input
        .shape
        .as_ref()
        .ok_or_else(|| invalid_pipeline("batch input shape is unknown"))?;
    let batch_dim = if config.drop_last {
        ShapeDim::Known(config.size)
    } else {
        ShapeDim::Dynamic
    };
    output.shape = Some(ValueShape(
        std::iter::once(batch_dim)
            .chain(input_shape.0.iter().copied())
            .collect(),
    ));
    output.axis_order = match image_axis(input.axis_order.as_ref()) {
        Some(ImageAxisOrder::Hwc) => Some(AxisOrder::Nhwc),
        Some(ImageAxisOrder::Chw) => Some(AxisOrder::Nchw),
        None => None,
    };
    output.granularity = Some(ValueGranularity::Batch);
    output.contiguity = Some(Contiguity::Contiguous);
    output.operator = Some(OperatorProperties {
        stage: OperatorStage::Batch,
        sample_stage_has_work: input
            .operator
            .is_some_and(|operator| operator.sample_stage_has_work),
    });
    Ok(output)
}

fn require_u8_decoded(input: PipelineImageState, name: &str) -> RivetResult<()> {
    match input {
        PipelineImageState::Encoded => Err(invalid_pipeline(format!(
            "{name} requires a decoded image, current state is encoded"
        ))),
        PipelineImageState::Decoded {
            dtype: CoreDType::U8,
            ..
        } => Ok(()),
        PipelineImageState::Decoded { dtype, axis_order } => Err(invalid_pipeline(format!(
            "{name} requires uint8 input, current state is {:?} {}",
            dtype,
            axis_order.as_str()
        ))),
    }
}

fn require_u8_or_f32_decoded(input: PipelineImageState, name: &str) -> RivetResult<()> {
    match input {
        PipelineImageState::Encoded => Err(invalid_pipeline(format!(
            "{name} requires a decoded image, current state is encoded"
        ))),
        PipelineImageState::Decoded {
            dtype: CoreDType::U8 | CoreDType::F32,
            ..
        } => Ok(()),
        PipelineImageState::Decoded { dtype, axis_order } => Err(invalid_pipeline(format!(
            "{name} requires uint8 or float32 input, current state is {:?} {}",
            dtype,
            axis_order.as_str()
        ))),
    }
}

fn require_u8_hwc(input: PipelineImageState, name: &str) -> RivetResult<()> {
    match input {
        PipelineImageState::Encoded => Err(invalid_pipeline(format!(
            "{name} requires a decoded image, current state is encoded"
        ))),
        PipelineImageState::Decoded {
            dtype: CoreDType::U8,
            axis_order: ImageAxisOrder::Hwc,
        } => Ok(()),
        PipelineImageState::Decoded { dtype, axis_order } => Err(invalid_pipeline(format!(
            "{name} requires uint8 HWC input, current state is {:?} {}",
            dtype,
            axis_order.as_str()
        ))),
    }
}

fn validate_rank(properties: &ValueProperties, op: &str) -> RivetResult<()> {
    match &properties.shape {
        Some(shape) if shape.rank() != 3 => Err(crate::errors::invalid_shape(format!(
            "{op} requires a rank-3 image, got rank {}",
            shape.rank()
        ))),
        Some(_) => Ok(()),
        None => Err(invalid_pipeline(format!(
            "{op} requires a known image rank"
        ))),
    }
}

fn validate_crop_bounds(
    input: &ValueProperties,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    op: &str,
) -> RivetResult<()> {
    let (height_dim, width_dim) = spatial_dims(input)?;
    if let (ShapeDim::Known(image_height), ShapeDim::Known(image_width)) = (height_dim, width_dim) {
        if (x as usize)
            .checked_add(width as usize)
            .is_none_or(|end| end > image_width)
            || (y as usize)
                .checked_add(height as usize)
                .is_none_or(|end| end > image_height)
        {
            return Err(crate::errors::invalid_shape(format!(
                "{op} rectangle ({x}, {y}, {width}, {height}) exceeds image shape {image_width}x{image_height}"
            )));
        }
    }
    Ok(())
}

fn validate_center_crop_bounds(
    input: &ValueProperties,
    width: u32,
    height: u32,
) -> RivetResult<()> {
    let (height_dim, width_dim) = spatial_dims(input)?;
    if let (ShapeDim::Known(image_height), ShapeDim::Known(image_width)) = (height_dim, width_dim) {
        if width as usize > image_width || height as usize > image_height {
            return Err(crate::errors::invalid_shape(format!(
                "center_crop size {width}x{height} exceeds image shape {image_width}x{image_height}"
            )));
        }
    }
    Ok(())
}

fn validate_random_crop_fit(
    input: &ValueProperties,
    width: u32,
    height: u32,
    padding: u32,
) -> RivetResult<()> {
    let (image_height, image_width) = spatial_dims(input)?;
    let (ShapeDim::Known(image_height), ShapeDim::Known(image_width)) = (image_height, image_width)
    else {
        return Ok(());
    };
    let pad = u64::from(padding);
    let padded_width = u64::try_from(image_width)
        .ok()
        .and_then(|width| width.checked_add(2 * pad));
    let padded_height = u64::try_from(image_height)
        .ok()
        .and_then(|height| height.checked_add(2 * pad));
    let (Some(padded_width), Some(padded_height)) = (padded_width, padded_height) else {
        return Err(crate::errors::invalid_shape(format!(
            "random_crop padding {padding} makes image {image_width}x{image_height} too large"
        )));
    };
    let (Ok(padded_width), Ok(padded_height)) =
        (u32::try_from(padded_width), u32::try_from(padded_height))
    else {
        return Err(crate::errors::invalid_shape(format!(
            "random_crop padding {padding} makes image {image_width}x{image_height} too large"
        )));
    };
    if width > padded_width || height > padded_height {
        return Err(crate::errors::invalid_shape(format!(
            "random_crop size {width}x{height} exceeds padded image {padded_width}x{padded_height}"
        )));
    }
    Ok(())
}

fn spatial_dims(input: &ValueProperties) -> RivetResult<(ShapeDim, ShapeDim)> {
    let shape = input
        .shape
        .as_ref()
        .ok_or_else(|| invalid_pipeline("image shape is unknown"))?;
    match image_axis(input.axis_order.as_ref()) {
        Some(ImageAxisOrder::Hwc) => Ok((shape.0[0], shape.0[1])),
        Some(ImageAxisOrder::Chw) => Ok((shape.0[1], shape.0[2])),
        None => Err(invalid_pipeline("image axis order is unknown")),
    }
}

fn image_properties(
    dtype: DataType,
    axis_order: ImageAxisOrder,
    shape: ValueShape,
    contiguity: Contiguity,
) -> ValueProperties {
    ValueProperties {
        representation: Some(Representation::Image),
        dtype: Some(dtype),
        shape: Some(shape),
        axis_order: Some(axis_property(axis_order)),
        contiguity: Some(contiguity),
        residency: Some(Residency::Host),
        granularity: Some(ValueGranularity::Sample),
        mutability: Some(rivet_plan::Mutability::Immutable),
        operator: Some(OperatorProperties {
            stage: OperatorStage::Source,
            sample_stage_has_work: false,
        }),
    }
}

fn dynamic_image_shape(axis_order: ImageAxisOrder) -> ValueShape {
    match axis_order {
        ImageAxisOrder::Hwc => ValueShape(vec![ShapeDim::Dynamic; 3]),
        ImageAxisOrder::Chw => ValueShape(vec![ShapeDim::Dynamic; 3]),
    }
}

fn core_dtype(dtype: CoreDType) -> DataType {
    match dtype {
        CoreDType::U8 => DataType::U8,
        CoreDType::I16 => DataType::I16,
        CoreDType::U32 => DataType::U32,
        CoreDType::I32 => DataType::I32,
        CoreDType::I64 => DataType::I64,
        CoreDType::BF16 => DataType::BF16,
        CoreDType::F16 => DataType::F16,
        CoreDType::F32 => DataType::F32,
        CoreDType::F64 => DataType::F64,
    }
}

fn core_dtype_from_property(dtype: DataType) -> RivetResult<CoreDType> {
    match dtype {
        DataType::U8 => Ok(CoreDType::U8),
        DataType::I16 => Ok(CoreDType::I16),
        DataType::U32 => Ok(CoreDType::U32),
        DataType::I32 => Ok(CoreDType::I32),
        DataType::I64 => Ok(CoreDType::I64),
        DataType::BF16 => Ok(CoreDType::BF16),
        DataType::F16 => Ok(CoreDType::F16),
        DataType::F32 => Ok(CoreDType::F32),
        DataType::F64 => Ok(CoreDType::F64),
        DataType::I8 | DataType::U16 | DataType::U64 | DataType::Bool | DataType::Other(_) => Err(
            invalid_pipeline("image properties use an unsupported dtype"),
        ),
    }
}

fn axis_property(axis: ImageAxisOrder) -> AxisOrder {
    match axis {
        ImageAxisOrder::Hwc => AxisOrder::Hwc,
        ImageAxisOrder::Chw => AxisOrder::Chw,
    }
}

fn image_axis(axis: Option<&AxisOrder>) -> Option<ImageAxisOrder> {
    match axis? {
        AxisOrder::Hwc | AxisOrder::Nhwc => Some(ImageAxisOrder::Hwc),
        AxisOrder::Chw | AxisOrder::Nchw => Some(ImageAxisOrder::Chw),
        AxisOrder::Other(_) => None,
    }
}

fn dynamic_dim() -> ShapeDim {
    ShapeDim::Dynamic
}

fn channel_dim(properties: &ValueProperties) -> ShapeDim {
    let Some(shape) = &properties.shape else {
        return dynamic_dim();
    };
    match image_axis(properties.axis_order.as_ref()) {
        Some(ImageAxisOrder::Hwc) => shape.0.get(2).copied().unwrap_or(ShapeDim::Dynamic),
        Some(ImageAxisOrder::Chw) => shape.0.first().copied().unwrap_or(ShapeDim::Dynamic),
        None => dynamic_dim(),
    }
}

fn set_spatial_shape(properties: &mut ValueProperties, height: ShapeDim, width: ShapeDim) {
    if let Some(shape) = &mut properties.shape {
        match image_axis(properties.axis_order.as_ref()) {
            Some(ImageAxisOrder::Hwc) if shape.rank() == 3 => {
                shape.0[0] = height;
                shape.0[1] = width;
            }
            Some(ImageAxisOrder::Chw) if shape.rank() == 3 => {
                shape.0[1] = height;
                shape.0[2] = width;
            }
            _ => {}
        }
    }
}

fn set_channel_dim(properties: &mut ValueProperties, channel: ShapeDim) {
    if let Some(shape) = &mut properties.shape {
        match image_axis(properties.axis_order.as_ref()) {
            Some(ImageAxisOrder::Hwc) if shape.rank() == 3 => shape.0[2] = channel,
            Some(ImageAxisOrder::Chw) if shape.rank() == 3 => shape.0[0] = channel,
            _ => {}
        }
    }
}

fn make_spatial_dynamic(properties: &mut ValueProperties) {
    set_spatial_shape(properties, ShapeDim::Dynamic, ShapeDim::Dynamic);
}

fn swap_spatial_shape(properties: &mut ValueProperties) {
    if let Some(shape) = &mut properties.shape {
        if shape.rank() == 3 {
            match image_axis(properties.axis_order.as_ref()) {
                Some(ImageAxisOrder::Hwc) => shape.0.swap(0, 1),
                Some(ImageAxisOrder::Chw) => shape.0.swap(1, 2),
                None => {}
            }
        }
    }
}

fn reorder_image_shape(properties: &mut ValueProperties, from: ImageAxisOrder, to: ImageAxisOrder) {
    let Some(shape) = &mut properties.shape else {
        return;
    };
    if shape.rank() != 3 {
        return;
    }
    shape.0 = match (from, to) {
        (ImageAxisOrder::Hwc, ImageAxisOrder::Chw) => vec![shape.0[2], shape.0[0], shape.0[1]],
        (ImageAxisOrder::Chw, ImageAxisOrder::Hwc) => vec![shape.0[1], shape.0[2], shape.0[0]],
        _ => return,
    };
}

fn add_padding(dim: ShapeDim, before: u32, after: u32) -> ShapeDim {
    match dim {
        ShapeDim::Known(value) => value
            .checked_add(before as usize)
            .and_then(|value| value.checked_add(after as usize))
            .map(ShapeDim::Known)
            .unwrap_or(ShapeDim::Dynamic),
        ShapeDim::Dynamic => ShapeDim::Dynamic,
    }
}

fn merge_optional_shape(a: &ValueProperties, b: &ValueProperties) -> ValueProperties {
    let mut result = a.clone();
    result.shape = match (&a.shape, &b.shape) {
        (Some(a), Some(b)) if a.rank() == b.rank() => Some(ValueShape(
            a.0.iter()
                .zip(&b.0)
                .map(|(&left, &right)| {
                    if left == right {
                        left
                    } else {
                        ShapeDim::Dynamic
                    }
                })
                .collect(),
        )),
        _ => None,
    };
    result.contiguity = merge_contiguity(a.contiguity, b.contiguity);
    result
}

fn merge_contiguity(a: Option<Contiguity>, b: Option<Contiguity>) -> Option<Contiguity> {
    if a == b { a } else { Some(Contiguity::Unknown) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rivet_plan::{DeviceClass, Mutability};

    fn decoded_properties(dtype: DataType) -> ValueProperties {
        image_properties(
            dtype,
            ImageAxisOrder::Hwc,
            ValueShape(vec![
                ShapeDim::Known(6),
                ShapeDim::Known(8),
                ShapeDim::Known(3),
            ]),
            Contiguity::Contiguous,
        )
    }

    #[test]
    fn inference_rejects_dtype_residency_and_granularity_conflicts() {
        let float = decoded_properties(DataType::F32);
        let error = infer_image_op(&ImageOp::resize(4, 4), &float).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Resize requires uint8 HWC input")
        );

        let mut device = decoded_properties(DataType::U8);
        device.residency = Some(Residency::Device(DeviceClass::Cuda));
        let error = infer_image_op(&ImageOp::resize(4, 4), &device).unwrap_err();
        assert!(error.to_string().contains("host-resident input"));

        let mut batch = decoded_properties(DataType::U8);
        batch.granularity = Some(ValueGranularity::Batch);
        let error = infer_image_op(&ImageOp::horizontal_flip(), &batch).unwrap_err();
        assert!(error.to_string().contains("sample granularity"));
    }

    #[test]
    fn layout_axis_order_and_contiguity_are_independent_properties() {
        let input = decoded_properties(DataType::U8);
        let output = infer_image_op(&ImageOp::hwc_to_chw(), &input).unwrap();
        assert_eq!(output.axis_order, Some(AxisOrder::Chw));
        assert_eq!(output.contiguity, Some(Contiguity::Strided));
        assert_eq!(output.mutability, Some(Mutability::Immutable));
        assert_eq!(
            output.shape.as_ref().unwrap().dims(),
            [ShapeDim::Known(3), ShapeDim::Known(6), ShapeDim::Known(8)]
        );
    }
}
