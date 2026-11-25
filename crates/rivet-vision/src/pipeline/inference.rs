//! Vision-specific value-property inference and validation.

use rivet_core::DType as CoreDType;
use rivet_plan::{
    AxisOrder, Contiguity, DataType, DeviceCut, DomainPropertiesValue, LogicalNode, NodeKind,
    OperatorProperties, OperatorStage, PropertyInference, Representation, Residency, ShapeDim,
    ValueGranularity, ValueProperties, ValueShape,
};

use super::logical::FusionGroupPayload;
use super::op::{BatchConfig, ImageOp, IndexOp, PipelineImageState, SourceOp};
use crate::errors::{RivetResult, invalid_pipeline};
use crate::sample::image::ImageAxisOrder;

/// Logical image semantics are independent of executor worker configuration.
pub struct VisionPropertyInference;

#[derive(Clone, Debug, PartialEq, Eq)]
enum VisionRepresentation {
    EncodedImage,
    Image,
}

/// Vision-only facts are carried through the generic plan as an opaque domain
/// property, while common dtype/shape/residency facts remain directly visible
/// to generic planning passes.
#[derive(Clone, Debug, PartialEq, Eq)]
struct VisionProperties {
    representation: VisionRepresentation,
    axis_order: Option<ImageAxisOrder>,
    views_per_source: usize,
}

impl VisionPropertyInference {
    pub fn new(_num_workers: usize) -> Self {
        Self
    }
}

impl Default for VisionPropertyInference {
    fn default() -> Self {
        Self::new(0)
    }
}

impl PropertyInference for VisionPropertyInference {
    fn domain_id(&self) -> Option<rivet_plan::DomainId> {
        Some(crate::VISION_DOMAIN_ID)
    }

    fn infer_node(
        &self,
        node: &LogicalNode,
        inputs: &[ValueProperties],
    ) -> Result<ValueProperties, String> {
        if node.kind() == NodeKind::Source && !inputs.is_empty() {
            return Err("vision source must not have inputs".to_owned());
        }
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
                if let Some(concat) = node.payload_as::<super::ImageConcat>() {
                    return rivet_plan::DomainOp::infer_properties(concat, inputs);
                }
                let input = single_input(inputs, node.kind())?;
                // Property inference describes operator semantics independently
                // of the backend that will execute them. Capability validation
                // decides whether an op is legal after a DeviceCut; shape and
                // dtype inference still need to propagate through that region.
                let mut semantic_input = input.clone();
                semantic_input.residency = Some(Residency::Host);
                if let Some(op) = node.payload_as::<ImageOp>() {
                    let mut output =
                        infer_image_op(op, &semantic_input).map_err(|error| error.to_string())?;
                    output.residency = input.residency.clone();
                    output
                } else if let Some(group) = node.payload_as::<FusionGroupPayload>() {
                    let mut properties = semantic_input;
                    for op in &group.ops {
                        properties =
                            infer_image_op(op, &properties).map_err(|error| error.to_string())?;
                    }
                    // Fusion expresses value semantics. The physical selector
                    // can choose a contiguous writer or a deferred layout view.
                    if group.name == "NormalizeToChw" {
                        properties.contiguity = Some(Contiguity::Unknown);
                    }
                    properties.residency = input.residency.clone();
                    properties
                } else if let Some(op) = node.payload().and_then(|payload| payload.as_domain_op()) {
                    op.infer_properties(inputs)?
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
    let properties = match state {
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
    };
    attach_vision_properties(properties)
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
    if input.granularity == Some(ValueGranularity::Batch) {
        if op.execution_kind() != super::op::ExecutionKind::Batch {
            return Err(invalid_pipeline(format!(
                "{} requires sample granularity",
                op.name()
            )));
        }
        let mut sample_input = input.clone();
        let batch_dim = sample_input
            .shape
            .as_mut()
            .filter(|shape| shape.rank() == 4)
            .ok_or_else(|| invalid_pipeline("batch image kernel requires rank four"))?
            .0
            .remove(0);
        sample_input.axis_order = match input.axis_order {
            Some(AxisOrder::Nhwc) => Some(AxisOrder::Hwc),
            Some(AxisOrder::Nchw) => Some(AxisOrder::Chw),
            _ => return Err(invalid_pipeline("batch image kernel requires NHWC or NCHW")),
        };
        sample_input.granularity = Some(ValueGranularity::Sample);
        let mut output = infer_image_op(op, &sample_input)?;
        if let Some(shape) = &mut output.shape {
            shape.0.insert(0, batch_dim);
        }
        output.axis_order = match output.axis_order {
            Some(AxisOrder::Hwc) => Some(AxisOrder::Nhwc),
            Some(AxisOrder::Chw) => Some(AxisOrder::Nchw),
            other => other,
        };
        output.granularity = Some(ValueGranularity::Batch);
        if let Some(operator) = &mut output.operator {
            operator.stage = OperatorStage::Batch;
        }
        return Ok(output);
    }
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
    let input_state = state_from_properties(input)?;
    if matches!(input_state, PipelineImageState::Decoded { .. })
        && !matches!(op, ImageOp::Decode(_))
    {
        validate_rank(input, op.name())?;
    }

    use PipelineImageState::{Decoded, Encoded};
    let mut output = input.clone();
    match op {
        ImageOp::V2(config) => {
            use crate::transforms::V2Config;
            require_u8_or_f32_decoded(input_state, op.name())?;
            match config {
                V2Config::SquareSymmetry(element) => {
                    if let Some(element) = element {
                        if element.swaps_axes() {
                            swap_spatial_shape(&mut output);
                        }
                    } else {
                        let (h, w) = spatial_dims(input)?;
                        if h != w {
                            make_spatial_dynamic(&mut output);
                        }
                    }
                    output.contiguity = Some(Contiguity::Unknown);
                }
                V2Config::ThinPlateSpline(_)
                | V2Config::GridElasticDeform(_)
                | V2Config::PiecewiseAffine(_)
                | V2Config::GridDistortion(_)
                | V2Config::OpticalDistortion(_) => {
                    require_u8_decoded(input_state, op.name())?;
                }
                V2Config::Superpixels(_) => {
                    if !matches!(
                        input_state,
                        Decoded {
                            axis_order: ImageAxisOrder::Hwc,
                            ..
                        }
                    ) || matches!(channel_dim(input),ShapeDim::Known(c) if c != 1 && c != 3)
                    {
                        return Err(invalid_pipeline(
                            "Superpixels requires HWC grayscale or RGB",
                        ));
                    }
                }
                V2Config::Morphological(_) => {
                    output.contiguity = Some(Contiguity::Unknown);
                }
                V2Config::SafeRotate(config) => {
                    require_u8_decoded(input_state, op.name())?;
                    if config.fill_channels.is_some()
                        && matches!(channel_dim(input), ShapeDim::Known(c) if c != 3)
                    {
                        return Err(invalid_pipeline("SafeRotate per-channel fill requires RGB"));
                    }
                }
                V2Config::RandomRotate90 => {
                    let (h, w) = spatial_dims(input)?;
                    if h != w {
                        make_spatial_dynamic(&mut output);
                    }
                    output.contiguity = Some(Contiguity::Unknown);
                }
                V2Config::Transpose => {
                    swap_spatial_shape(&mut output);
                    output.contiguity = Some(Contiguity::Unknown);
                }
                V2Config::RandomSizedCrop(config) => {
                    require_u8_hwc(input_state, op.name())?;
                    if matches!(channel_dim(input), ShapeDim::Known(c) if c != 3) {
                        return Err(invalid_pipeline("RandomSizedCrop requires RGB input"));
                    }
                    set_spatial_shape(
                        &mut output,
                        ShapeDim::Known(config.height as usize),
                        ShapeDim::Known(config.width as usize),
                    );
                    output.contiguity = Some(Contiguity::Contiguous);
                }
                V2Config::RandomResize { .. }
                | V2Config::RandomShortestSize { .. }
                | V2Config::RandomScale { .. }
                | V2Config::ScaleJitter { .. } => {
                    require_u8_hwc(input_state, op.name())?;
                    make_spatial_dynamic(&mut output);
                }
                V2Config::RandomZoomOut {
                    probability,
                    side_range,
                    fill,
                } => {
                    if fill.len() == 3 && matches!(channel_dim(input), ShapeDim::Known(c) if c != 3)
                    {
                        return Err(invalid_pipeline(
                            "per-channel zoom out fill requires three channels",
                        ));
                    }
                    if *probability != 0.0 && *side_range != (1.0, 1.0) {
                        make_spatial_dynamic(&mut output);
                    }
                }
                V2Config::RandomPhotometricDistort { .. } => {
                    require_u8_hwc(input_state, op.name())?;
                }
                V2Config::PadIfNeeded(config) => {
                    if config.fill_channels.is_some()
                        && matches!(channel_dim(input), ShapeDim::Known(c) if c != 3)
                    {
                        return Err(invalid_pipeline(
                            "PadIfNeeded per-channel fill requires RGB",
                        ));
                    }
                    let (h, w) = spatial_dims(input)?;
                    let size = |dim, min, divisor| -> RivetResult<ShapeDim> {
                        match dim {
                            ShapeDim::Known(v) => Ok(ShapeDim::Known(
                                crate::transforms::PadIfNeededConfig::output_axis(v, min, divisor)?,
                            )),
                            other => Ok(other),
                        }
                    };
                    set_spatial_shape(
                        &mut output,
                        size(h, config.min_height, config.pad_height_divisor)?,
                        size(w, config.min_width, config.pad_width_divisor)?,
                    );
                    output.contiguity = Some(Contiguity::Unknown);
                }
                V2Config::LetterBox(config) => {
                    require_u8_hwc(input_state, op.name())?;
                    if matches!(channel_dim(input), ShapeDim::Known(c) if c != 3) {
                        return Err(invalid_pipeline("LetterBox requires three channels"));
                    }
                    set_spatial_shape(
                        &mut output,
                        ShapeDim::Known(config.height as usize),
                        ShapeDim::Known(config.width as usize),
                    );
                }
                V2Config::RandomCropFromBorders { fractions } => {
                    if fractions.iter().any(|&v| v > 0.0) {
                        make_spatial_dynamic(&mut output);
                    }
                    output.contiguity = Some(Contiguity::Unknown);
                }
                V2Config::Blur { .. }
                | V2Config::AdvancedBlur(_)
                | V2Config::StochasticConvolution { .. }
                | V2Config::ZoomBlur { .. }
                | V2Config::GlassBlur { .. }
                | V2Config::RandomShadow(_)
                | V2Config::PlasmaShadow { .. }
                | V2Config::PlasmaBrightnessContrast { .. }
                | V2Config::Illumination { .. }
                | V2Config::RandomBrightnessContrast { .. }
                | V2Config::Halftone { .. }
                | V2Config::Dithering(_)
                | V2Config::RingingOvershoot { .. }
                | V2Config::Enhance { .. }
                | V2Config::Emboss { .. }
                | V2Config::ExposureMatching { .. }
                | V2Config::RandomGamma { .. }
                | V2Config::FilmGrain { .. }
                | V2Config::MultiplicativeNoise { .. }
                | V2Config::AdditiveNoise(_) => {
                    if !matches!(
                        input_state,
                        PipelineImageState::Decoded {
                            dtype: CoreDType::U8 | CoreDType::F32,
                            axis_order: ImageAxisOrder::Hwc,
                        }
                    ) || matches!(channel_dim(input), ShapeDim::Known(0))
                    {
                        return Err(invalid_pipeline(format!(
                            "{} requires uint8 or float32 HWC input with at least one channel",
                            op.name()
                        )));
                    }
                }
                V2Config::PlanckianJitter { .. }
                | V2Config::RgbShift(_)
                | V2Config::IsoNoise { .. }
                | V2Config::RandomSunFlare(..)
                | V2Config::LensFlare(..)
                | V2Config::RandomRain(..)
                | V2Config::RandomFog(..)
                | V2Config::AtmosphericFog(..)
                | V2Config::Spatter(..)
                | V2Config::RandomGravel(..)
                | V2Config::RandomSnow(..) => {
                    if !matches!(
                        input_state,
                        PipelineImageState::Decoded {
                            dtype: CoreDType::U8 | CoreDType::F32,
                            axis_order: ImageAxisOrder::Hwc,
                        }
                    ) || matches!(channel_dim(input), ShapeDim::Known(c) if c != 3)
                    {
                        return Err(invalid_pipeline(format!(
                            "{} requires uint8 or float32 HWC RGB input",
                            op.name()
                        )));
                    }
                }
                V2Config::Colorize(_) => {
                    if !matches!(
                        input_state,
                        PipelineImageState::Decoded {
                            dtype: CoreDType::U8 | CoreDType::F32,
                            axis_order: ImageAxisOrder::Hwc,
                        }
                    ) || matches!(channel_dim(input), ShapeDim::Known(c) if c != 1)
                    {
                        return Err(invalid_pipeline(
                            "Colorize requires uint8 or float32 HWC grayscale input",
                        ));
                    }
                    set_channel_dim(&mut output, ShapeDim::Known(3));
                }
                V2Config::HueSaturationValue(_)
                | V2Config::ToSepia { .. }
                | V2Config::FancyPca { .. } => {
                    if !matches!(
                        input_state,
                        PipelineImageState::Decoded {
                            dtype: CoreDType::U8 | CoreDType::F32,
                            axis_order: ImageAxisOrder::Hwc,
                        }
                    ) || matches!(channel_dim(input), ShapeDim::Known(c) if c != 1 && c != 3)
                    {
                        return Err(invalid_pipeline(format!(
                            "{} requires uint8 or float32 HWC grayscale or RGB input",
                            op.name()
                        )));
                    }
                }
                V2Config::Vignetting { .. } => {
                    if !matches!(
                        input_state,
                        PipelineImageState::Decoded {
                            dtype: CoreDType::U8 | CoreDType::F32,
                            axis_order: ImageAxisOrder::Hwc,
                        }
                    ) || matches!(channel_dim(input), ShapeDim::Known(0))
                    {
                        return Err(invalid_pipeline(
                            "Vignetting requires uint8 or float32 HWC input with at least one channel",
                        ));
                    }
                }
                V2Config::Clahe { .. } | V2Config::MedianBlur => {
                    require_u8_hwc(input_state, op.name())?;
                    if matches!(channel_dim(input), ShapeDim::Known(c) if c != 1 && c != 3) {
                        return Err(invalid_pipeline(format!(
                            "{} requires one or three channels",
                            op.name()
                        )));
                    }
                }
                V2Config::Downscale { .. }
                | V2Config::RandomToneCurve { .. }
                | V2Config::ChromaticAberration { .. }
                | V2Config::Defocus { .. }
                | V2Config::MotionBlur { .. }
                | V2Config::UnsharpMask { .. } => {
                    require_u8_hwc(input_state, op.name())?;
                    if matches!(channel_dim(input), ShapeDim::Known(c) if c != 3) {
                        return Err(invalid_pipeline(format!(
                            "{} requires three channels",
                            op.name()
                        )));
                    }
                }
                V2Config::ChannelDropout(config) => {
                    if !matches!(
                        input_state,
                        Decoded {
                            dtype: CoreDType::U8 | CoreDType::F32,
                            ..
                        }
                    ) {
                        return Err(invalid_pipeline(
                            "ChannelDropout requires uint8 or float32 input",
                        ));
                    }
                    if let ShapeDim::Known(channels) = channel_dim(input)
                        && (channels == 0
                            || config
                                .channels_to_drop
                                .as_ref()
                                .is_some_and(|indices| indices.iter().any(|&i| i >= channels)))
                    {
                        return Err(invalid_pipeline(
                            "ChannelDropout channel index is out of bounds",
                        ));
                    }
                }
                V2Config::SaltAndPepper { .. }
                | V2Config::GridMask(_)
                | V2Config::XyMasking(_)
                | V2Config::PixelDropout(_)
                | V2Config::GridDropout { .. }
                | V2Config::CoarseDropout { .. } => {
                    if !matches!(
                        input_state,
                        Decoded {
                            dtype: CoreDType::U8 | CoreDType::F32,
                            ..
                        }
                    ) {
                        return Err(invalid_pipeline(format!(
                            "{} requires uint8 or float32 input",
                            op.name()
                        )));
                    }
                }
                V2Config::LongestMaxSize(config) => {
                    require_u8_hwc(input_state, op.name())?;
                    if matches!(channel_dim(input), ShapeDim::Known(c) if c != 3) {
                        return Err(invalid_pipeline("LongestMaxSize requires RGB input"));
                    }
                    if let (ShapeDim::Known(h), ShapeDim::Known(w)) = spatial_dims(input)? {
                        if config.random() {
                            make_spatial_dynamic(&mut output);
                        } else {
                            let (w, h) = config.dimensions(w, h, 0)?;
                            set_spatial_shape(
                                &mut output,
                                ShapeDim::Known(h as usize),
                                ShapeDim::Known(w as usize),
                            );
                        }
                    } else {
                        make_spatial_dynamic(&mut output);
                    }
                }
                V2Config::CropAndPad(config) => {
                    if config.keep_size
                        && (!matches!(
                            input_state,
                            Decoded {
                                dtype: CoreDType::U8,
                                ..
                            }
                        ) || matches!(channel_dim(input), ShapeDim::Known(c) if c != 3))
                    {
                        return Err(invalid_pipeline(
                            "CropAndPad keep_size requires uint8 RGB input",
                        ));
                    }
                    if config.fill.len() == 3
                        && matches!(channel_dim(input), ShapeDim::Known(c) if c != 3)
                    {
                        return Err(invalid_pipeline(
                            "CropAndPad per-channel fill requires RGB input",
                        ));
                    }
                    if !config.keep_size {
                        make_spatial_dynamic(&mut output);
                    }
                    output.contiguity = Some(Contiguity::Unknown);
                }
                V2Config::Jpeg { .. } => {
                    if !matches!(
                        input_state,
                        Decoded {
                            dtype: CoreDType::U8,
                            ..
                        }
                    ) {
                        return Err(invalid_pipeline("JPEG requires uint8 input"));
                    }
                    if matches!(channel_dim(input), ShapeDim::Known(c) if c != 1 && c != 3) {
                        return Err(invalid_pipeline("JPEG requires one or three channels"));
                    }
                }
                V2Config::Rgb => {
                    if matches!(channel_dim(input), ShapeDim::Known(c) if !(1..=4).contains(&c)) {
                        return Err(invalid_pipeline("RGB requires one to four channels"));
                    }
                    set_channel_dim(&mut output, ShapeDim::Known(3));
                }
                V2Config::PermuteChannels(order) => {
                    if let ShapeDim::Known(c) = channel_dim(input) {
                        if order.len() != c
                            || order.iter().any(|&i| i >= c)
                            || (0..c).any(|i| !order.contains(&i))
                        {
                            return Err(invalid_pipeline(
                                "channel order must be a permutation of the input channels",
                            ));
                        }
                    }
                }
                V2Config::ToDtype { dtype, .. } => {
                    output.dtype = Some(core_dtype(*dtype));
                }
                V2Config::ShotNoise { .. } => {
                    if !matches!(
                        input_state,
                        Decoded {
                            dtype: CoreDType::F32,
                            ..
                        }
                    ) {
                        return Err(invalid_pipeline("ShotNoise requires float32 input"));
                    }
                }
                V2Config::GaussianNoise { .. } | V2Config::RandomChannelPermutation { .. } => {}
            }
            output.contiguity = Some(Contiguity::Unknown);
        }
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
            if config.short_edge.is_some() {
                let (ih, iw) = spatial_dims(input)?;
                if let (ShapeDim::Known(ih), ShapeDim::Known(iw)) = (ih, iw) {
                    let iw = u32::try_from(iw).map_err(|_| {
                        crate::errors::invalid_shape("resize input width exceeds u32")
                    })?;
                    let ih = u32::try_from(ih).map_err(|_| {
                        crate::errors::invalid_shape("resize input height exceeds u32")
                    })?;
                    let (ow, oh) = config.output_dimensions(iw, ih)?;
                    set_spatial_shape(
                        &mut output,
                        ShapeDim::Known(oh as usize),
                        ShapeDim::Known(ow as usize),
                    );
                } else {
                    set_spatial_shape(&mut output, ShapeDim::Dynamic, ShapeDim::Dynamic);
                }
            } else {
                set_spatial_shape(
                    &mut output,
                    ShapeDim::Known(config.height as usize),
                    ShapeDim::Known(config.width as usize),
                );
            }
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::FiveCrop(_) | ImageOp::TenCrop(_) => {
            require_u8_or_f32_decoded(input_state, op.name())?;
            let (width, height) = match op {
                ImageOp::FiveCrop(config) => (config.width, config.height),
                ImageOp::TenCrop(config) => (config.width, config.height),
                _ => unreachable!(),
            };
            let (ih, iw) = spatial_dims(input)?;
            if matches!(ih, ShapeDim::Known(ih) if ih < height as usize)
                || matches!(iw, ShapeDim::Known(iw) if iw < width as usize)
            {
                return Err(invalid_pipeline(
                    "multi-view crop size exceeds image dimensions",
                ));
            }
            set_spatial_shape(
                &mut output,
                ShapeDim::Known(height as usize),
                ShapeDim::Known(width as usize),
            );
            output.contiguity = Some(Contiguity::Strided);
        }
        ImageOp::Crop(config) => {
            require_u8_decoded(input_state, "Crop")?;
            validate_rank(input, "crop")?;
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
            validate_random_crop_fit(input, config)?;
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
        ImageOp::Saturation(_) | ImageOp::Gamma(_) | ImageOp::RandomGaussianBlur(_) => {
            require_u8_hwc(input_state, op.name())?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::Affine(_) => {
            require_u8_decoded(input_state, op.name())?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::RandomRotation(config) => {
            require_u8_decoded(input_state, op.name())?;
            if config.expand {
                make_spatial_dynamic(&mut output);
            }
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::Erase(_) => {
            require_u8_or_f32_decoded(input_state, op.name())?;
            validate_rank(input, "erase")?;
            output.contiguity = Some(Contiguity::Contiguous);
        }
        ImageOp::AutoAugment(_)
        | ImageOp::RandAugment(_)
        | ImageOp::TrivialAugmentWide(_)
        | ImageOp::AugMix(_) => {
            require_u8_hwc(input_state, op.name())?;
            output.contiguity = Some(Contiguity::Unknown);
        }
        ImageOp::LinearTransformation(_) => {
            require_u8_or_f32_decoded(input_state, op.name())?;
            if input.dtype != Some(rivet_plan::DataType::F32) {
                return Err(crate::errors::invalid_argument(
                    "LinearTransformation requires float32 input",
                ));
            }
            validate_rank(input, "linear transformation")?;
            output.contiguity = Some(Contiguity::Contiguous);
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
        ImageOp::RandomChoice { choices, weights } => {
            super::op::choice_cumulative_weights(choices.len(), weights.as_deref())?;
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
                output.dtype = Some(rivet_plan::DataType::F32);
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
    output.operator = Some(if noop_layout {
        incoming_operator
    } else {
        OperatorProperties {
            stage: OperatorStage::Sample,
            sample_stage_has_work: incoming_operator.sample_stage_has_work
                || op.execution_kind() == super::op::ExecutionKind::Sample,
        }
    });
    let views = views_per_source(input)
        .checked_mul(op.output_multiplier())
        .ok_or_else(|| invalid_pipeline("multi-view count overflows usize"))?;
    Ok(attach_vision_properties_with_views(output, views))
}

fn infer_sequence(ops: &[ImageOp], input: &ValueProperties) -> RivetResult<ValueProperties> {
    let mut properties = input.clone();
    for op in ops {
        if op.output_multiplier() != 1 {
            return Err(invalid_pipeline(
                "multi-view crops cannot be nested in random_apply, random_choice or random_order; use compose instead",
            ));
        }
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
        ShapeDim::Known(
            config
                .size
                .checked_mul(views_per_source(input))
                .ok_or_else(|| invalid_pipeline("multi-view batch size overflows usize"))?,
        )
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
    Ok(attach_vision_properties(output))
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

fn validate_random_crop_fit(
    input: &ValueProperties,
    config: &crate::transforms::RandomCropConfig,
) -> RivetResult<()> {
    if config.pad_if_needed {
        return Ok(());
    }
    let (image_height, image_width) = spatial_dims(input)?;
    let (ShapeDim::Known(image_height), ShapeDim::Known(image_width)) = (image_height, image_width)
    else {
        return Ok(());
    };
    let [left, top, right, bottom] = config.padding_sides.unwrap_or([config.padding; 4]);
    let padded_width = u64::try_from(image_width)
        .ok()
        .and_then(|width| width.checked_add(u64::from(left) + u64::from(right)));
    let padded_height = u64::try_from(image_height)
        .ok()
        .and_then(|height| height.checked_add(u64::from(top) + u64::from(bottom)));
    let (Some(padded_width), Some(padded_height)) = (padded_width, padded_height) else {
        return Err(crate::errors::invalid_shape(format!(
            "random_crop padding makes image {image_width}x{image_height} too large"
        )));
    };
    let (Ok(padded_width), Ok(padded_height)) =
        (u32::try_from(padded_width), u32::try_from(padded_height))
    else {
        return Err(crate::errors::invalid_shape(format!(
            "random_crop padding makes image {image_width}x{image_height} too large"
        )));
    };
    if config.width > padded_width || config.height > padded_height {
        return Err(crate::errors::invalid_shape(format!(
            "random_crop size {}x{} exceeds padded image {padded_width}x{padded_height}",
            config.width, config.height
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
        ..ValueProperties::default()
    }
}

pub(crate) fn views_per_source(properties: &ValueProperties) -> usize {
    properties
        .domain_as::<DomainPropertiesValue<VisionProperties>>()
        .map_or(1, |domain| domain.value().views_per_source)
}

fn attach_vision_properties(properties: ValueProperties) -> ValueProperties {
    let views = views_per_source(&properties);
    attach_vision_properties_with_views(properties, views)
}

fn attach_vision_properties_with_views(
    mut properties: ValueProperties,
    views_per_source: usize,
) -> ValueProperties {
    let representation = match properties.representation {
        Some(Representation::EncodedImage) => VisionRepresentation::EncodedImage,
        _ => VisionRepresentation::Image,
    };
    let axis_order = image_axis(properties.axis_order.as_ref());
    properties.domain = Some(std::sync::Arc::new(DomainPropertiesValue::new(
        crate::VISION_DOMAIN_ID,
        VisionProperties {
            representation,
            axis_order,
            views_per_source,
        },
    )));
    properties
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
    fn multiview_batch_shape_and_factor_survive_representation_changes() {
        let input = decoded_properties(DataType::U8);
        let cropped = infer_image_op(&ImageOp::five_crop(2, 2), &input).unwrap();
        let normalized =
            infer_image_op(&ImageOp::normalize(vec![0.0; 3], vec![1.0; 3]), &cropped).unwrap();
        let chw = infer_image_op(&ImageOp::hwc_to_chw(), &normalized).unwrap();
        let output = infer_batch(&BatchConfig::new(4, true), &chw).unwrap();
        assert_eq!(views_per_source(&output), 5);
        assert_eq!(
            output.shape.unwrap().dims(),
            [
                ShapeDim::Known(20),
                ShapeDim::Known(3),
                ShapeDim::Known(2),
                ShapeDim::Known(2)
            ]
        );
        assert_eq!(output.dtype, Some(DataType::F32));
    }

    #[test]
    fn concat_rejects_different_multiview_cardinalities() {
        let input = decoded_properties(DataType::U8);
        let cropped = infer_image_op(&ImageOp::five_crop(2, 2), &input).unwrap();
        let single = infer_image_op(&ImageOp::crop(0, 0, 2, 2), &input).unwrap();
        let err = rivet_plan::DomainOp::infer_properties(
            &super::super::ImageConcat { axis: 2 },
            &[cropped, single],
        )
        .unwrap_err();
        assert!(err.contains("matching view counts"));
    }

    #[test]
    fn multiview_batch_extent_overflow_is_rejected() {
        let cropped =
            infer_image_op(&ImageOp::five_crop(2, 2), &decoded_properties(DataType::U8)).unwrap();
        assert!(infer_batch(&BatchConfig::new(usize::MAX, true), &cropped).is_err());
    }

    #[test]
    fn short_edge_resize_propagates_known_invalid_shapes() {
        let op =
            ImageOp::resize_short_edge(1, Some(2), crate::transforms::InterpolationMode::Bilinear);
        for (height, width) in [(0, 8), (1, u32::MAX as usize), (8, u32::MAX as usize + 1)] {
            let mut input = decoded_properties(DataType::U8);
            input.shape = Some(ValueShape(vec![
                ShapeDim::Known(height),
                ShapeDim::Known(width),
                ShapeDim::Known(3),
            ]));
            assert!(infer_image_op(&op, &input).is_err());
        }
    }

    #[test]
    fn short_edge_resize_inference_matches_runtime_dimensions() {
        let input = image_properties(
            DataType::U8,
            ImageAxisOrder::Hwc,
            ValueShape(vec![
                ShapeDim::Known(4),
                ShapeDim::Known(3),
                ShapeDim::Known(3),
            ]),
            Contiguity::Contiguous,
        );
        let op =
            ImageOp::resize_short_edge(8, Some(9), crate::transforms::InterpolationMode::Bilinear);
        let inferred = infer_image_op(&op, &input).unwrap();
        assert_eq!(
            inferred.shape.unwrap().dims(),
            [ShapeDim::Known(9), ShapeDim::Known(7), ShapeDim::Known(3)]
        );
        assert_eq!(
            crate::transforms::ResizeConfig::short_edge(
                8,
                Some(9),
                crate::transforms::InterpolationMode::Bilinear
            )
            .output_dimensions(3, 4)
            .unwrap(),
            (7, 9)
        );
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
        device.residency = Some(Residency::Device(DeviceClass::Metal));
        let error = infer_image_op(&ImageOp::resize(4, 4), &device).unwrap_err();
        assert!(error.to_string().contains("host-resident input"));

        let mut batch = decoded_properties(DataType::U8);
        batch.granularity = Some(ValueGranularity::Batch);
        let error = infer_image_op(&ImageOp::horizontal_flip(), &batch).unwrap_err();
        assert!(error.to_string().contains("sample granularity"));
    }

    #[test]
    fn logical_properties_do_not_depend_on_worker_count() {
        let input = infer_image_op(
            &ImageOp::crop(1, 1, 4, 4),
            &decoded_properties(DataType::U8),
        )
        .unwrap();
        let node = LogicalNode::new(
            NodeKind::Op,
            [],
            Some(std::sync::Arc::new(ImageOp::normalize(
                vec![0.5; 3],
                vec![0.5; 3],
            ))),
        );
        let inline = VisionPropertyInference::new(0)
            .infer_node(&node, &[input.clone()])
            .unwrap();
        let threaded = VisionPropertyInference::new(4)
            .infer_node(&node, &[input])
            .unwrap();
        assert_eq!(inline.dtype, threaded.dtype);
        assert_eq!(inline.shape, threaded.shape);
        assert_eq!(inline.axis_order, threaded.axis_order);
        assert_eq!(inline.contiguity, threaded.contiguity);
        assert_eq!(inline.operator, threaded.operator);
        assert_eq!(inline.operator.unwrap().stage, OperatorStage::Sample);
    }

    #[test]
    fn dtype_identity_is_not_a_logical_scheduling_barrier() {
        let input = decoded_properties(DataType::U8);
        let converted =
            infer_image_op(&ImageOp::convert_image_dtype(CoreDType::U8), &input).unwrap();
        let cropped = infer_image_op(&ImageOp::crop(1, 1, 4, 4), &converted).unwrap();
        assert_eq!(cropped.dtype, Some(DataType::U8));
        assert_eq!(
            cropped.shape.unwrap().dims(),
            [ShapeDim::Known(4), ShapeDim::Known(4), ShapeDim::Known(3)]
        );
        let normalized =
            infer_image_op(&ImageOp::normalize(vec![0.5; 3], vec![0.5; 3]), &input).unwrap();
        assert!(infer_image_op(&ImageOp::crop(1, 1, 4, 4), &normalized).is_err());
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
    #[test]
    fn v2_inference_tracks_dtype_channels_and_random_spatial_extents() {
        use crate::transforms::{InterpolationMode, V2Config};
        let input = decoded_properties(DataType::U8);
        let output = infer_image_op(
            &ImageOp::V2(V2Config::ToDtype {
                dtype: CoreDType::F32,
                scale: false,
            }),
            &input,
        )
        .unwrap();
        assert_eq!(output.dtype, Some(DataType::F32));
        assert_eq!(output.shape, input.shape);
        let random = infer_image_op(
            &ImageOp::V2(V2Config::RandomResize {
                min_size: 3,
                max_size: 8,
                interpolation: InterpolationMode::Bilinear,
            }),
            &input,
        )
        .unwrap();
        assert_eq!(
            random.shape.unwrap().dims(),
            [ShapeDim::Dynamic, ShapeDim::Dynamic, ShapeDim::Known(3)]
        );
        let unchanged = infer_image_op(
            &ImageOp::V2(V2Config::RandomZoomOut {
                fill: vec![0.0],
                side_range: (1.0, 4.0),
                probability: 0.0,
            }),
            &input,
        )
        .unwrap();
        assert_eq!(unchanged.shape, input.shape);
        assert!(
            infer_image_op(
                &ImageOp::V2(V2Config::PermuteChannels(vec![0, 0, 1])),
                &input
            )
            .is_err()
        );
        let float = decoded_properties(DataType::F32);
        assert!(
            infer_image_op(&ImageOp::V2(V2Config::Jpeg { quality: (50, 80) }), &float).is_err()
        );
        assert!(
            infer_image_op(
                &ImageOp::V2(V2Config::RandomResize {
                    min_size: 3,
                    max_size: 8,
                    interpolation: InterpolationMode::Bilinear
                }),
                &float
            )
            .is_err()
        );
        let mut gray = input.clone();
        set_channel_dim(&mut gray, ShapeDim::Known(1));
        let rgb = infer_image_op(&ImageOp::V2(V2Config::Rgb), &gray).unwrap();
        assert_eq!(channel_dim(&rgb), ShapeDim::Known(3));
    }
}
