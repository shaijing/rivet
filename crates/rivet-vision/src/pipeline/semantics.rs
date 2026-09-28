//! Conservative semantic facts used by specific rewrite rules.
//!
//! These facts do not authorize arbitrary operator swaps. In particular,
//! sample independence does not prove that Crop and Normalize commute.

use rivet_plan::{LogicalNode, NodeKind, ValueGranularity, ValueProperties};

use super::op::ImageOp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Randomness {
    None,
    /// Random streams depend on source sample identity and a stable operator ID.
    SourceSample,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OperatorSemantics {
    pub one_to_one: bool,
    pub preserves_order: bool,
    pub sample_independent: bool,
    pub has_side_effects: bool,
    pub randomness: Randomness,
    /// Selection pushdown uses selected-sample error semantics: errors in
    /// samples excluded by Take/Skip need not be evaluated.
    pub may_error: bool,
}

impl Default for OperatorSemantics {
    fn default() -> Self {
        // Unknown operators must opt in through a specific rule, rather than
        // gaining permission from their NodeKind or inferred output shape.
        Self {
            one_to_one: false,
            preserves_order: false,
            sample_independent: false,
            has_side_effects: true,
            randomness: Randomness::Unknown,
            may_error: true,
        }
    }
}

pub(crate) fn image_semantics(op: &ImageOp) -> OperatorSemantics {
    // Exhaustively list built-ins. Adding a new variant requires reviewing its
    // effects rather than silently treating every image payload as pure.
    match op {
        ImageOp::Decode(_)
        | ImageOp::Resize(_)
        | ImageOp::Crop(_)
        | ImageOp::CenterCrop(_)
        | ImageOp::Pad(_)
        | ImageOp::Flip(_)
        | ImageOp::RandomCrop(_)
        | ImageOp::RandomResizedCrop(_)
        | ImageOp::RandomHorizontalFlip(_)
        | ImageOp::Brightness(_)
        | ImageOp::Contrast(_)
        | ImageOp::Hue(_)
        | ImageOp::ColorJitter(_)
        | ImageOp::Invert(_)
        | ImageOp::Posterize(_)
        | ImageOp::Solarize(_)
        | ImageOp::Autocontrast(_)
        | ImageOp::Equalize(_)
        | ImageOp::Sharpness(_)
        | ImageOp::ArbitraryRotate(_)
        | ImageOp::RandomAffine(_)
        | ImageOp::Perspective(_)
        | ImageOp::RandomPerspective(_)
        | ImageOp::ElasticTransform(_)
        | ImageOp::RandomApply { .. }
        | ImageOp::RandomChoice { .. }
        | ImageOp::RandomOrder { .. }
        | ImageOp::GaussianBlur(_)
        | ImageOp::Grayscale(_)
        | ImageOp::RandomGrayscale(_)
        | ImageOp::RandomErasing(_)
        | ImageOp::ConvertImageDtype(_)
        | ImageOp::Rotate(_)
        | ImageOp::Normalize(_)
        | ImageOp::Layout(_) => OperatorSemantics {
            one_to_one: true,
            preserves_order: true,
            sample_independent: true,
            has_side_effects: false,
            randomness: if op.random_key_kind().is_some() {
                Randomness::SourceSample
            } else {
                Randomness::None
            },
            may_error: true,
        },
    }
}

/// Conditions for moving an index selection across exactly this image node.
/// Sharing and graph arity are checked separately by the graph rewrite.
pub(crate) fn allows_index_pushdown(
    node: &LogicalNode,
    input: Option<&ValueProperties>,
) -> Result<OperatorSemantics, &'static str> {
    if node.kind() != NodeKind::Op {
        return Err("node is a source/index/batch/cache/device/sink barrier");
    }
    let Some(op) = node.payload_as::<ImageOp>() else {
        return Err("unknown or multi-input operation has no approved selection rewrite");
    };
    let semantics = image_semantics(op);
    if !semantics.one_to_one || !semantics.preserves_order || !semantics.sample_independent {
        return Err("operation does not preserve independent one-to-one source samples");
    }
    if semantics.has_side_effects || semantics.randomness == Randomness::Unknown {
        return Err("operation has unknown state or side effects");
    }
    if input.and_then(|properties| properties.granularity) != Some(ValueGranularity::Sample) {
        return Err("operation input does not have sample granularity");
    }
    if semantics.randomness == Randomness::SourceSample && node.semantic_identity().is_none() {
        return Err("random operation has no stable semantic identity");
    }
    Ok(semantics)
}
