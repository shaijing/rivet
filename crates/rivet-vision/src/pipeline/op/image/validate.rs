use super::{ExecutionKind, ImageOp, PipelineImageState};
use crate::errors::{RivetResult, invalid_argument, invalid_pipeline};
use crate::pipeline::op::choice_cumulative_weights;
use crate::pipeline::planning::inference;

impl ImageOp {
    pub fn transition(&self, input: PipelineImageState) -> RivetResult<PipelineImageState> {
        let properties = inference::properties_from_state(input);
        let output = inference::infer_image_op(self, &properties)?;
        inference::state_from_properties(&output)
    }

    /// Validate the op's own configuration (not its position in the
    /// pipeline; that is `transition`'s job).
    pub fn validate(&self) -> RivetResult<()> {
        match self {
            Self::Callback(op) => op.validate(),
            Self::V2(op) => op.validate(),
            Self::LinearTransformation(op) => op.validate(),
            Self::AutoAugment(op) => op.validate(),
            Self::RandAugment(op) => op.validate(),
            Self::TrivialAugmentWide(op) => op.validate(),
            Self::AugMix(op) => op.validate(),

            Self::Saturation(op) => op.validate(),
            Self::Gamma(op) => op.validate(),
            Self::Affine(op) => op.validate(),
            Self::Erase(op) => op.validate(),
            Self::RandomRotation(op) => op.validate(),
            Self::RandomGaussianBlur(op) => op.validate(),

            Self::Decode(_) | Self::Flip(_) | Self::Layout(_) => Ok(()),
            Self::Resize(op) => op.validate(),
            Self::Crop(op) => {
                if op.width == 0 || op.height == 0 {
                    return Err(invalid_argument(
                        "crop width and height must be greater than 0",
                    ));
                }
                Ok(())
            }
            Self::CenterCrop(op) => {
                if op.width == 0 || op.height == 0 {
                    return Err(invalid_argument(
                        "center_crop width and height must be greater than 0",
                    ));
                }
                Ok(())
            }
            Self::FiveCrop(op) => op.validate(),
            Self::TenCrop(op) => op.validate(),
            Self::Pad(op) => op.validate(),
            Self::RandomCrop(op) => op.validate(),
            Self::RandomResizedCrop(op) => op.validate(),
            Self::RandomHorizontalFlip(op) => {
                if !(0.0..=1.0).contains(&op.probability) {
                    return Err(invalid_argument(
                        "random_horizontal_flip probability must be in [0.0, 1.0]",
                    ));
                }
                Ok(())
            }
            Self::Contrast(op) => op.validate(),
            Self::Hue(op) => op.validate(),
            Self::Brightness(op) => op.validate(),
            Self::ColorJitter(op) => op.validate(),
            Self::Invert(_) => Ok(()),
            Self::Posterize(op) => op.validate(),
            Self::Solarize(_) => Ok(()),
            Self::Autocontrast(_) | Self::Equalize(_) => Ok(()),
            Self::Sharpness(op) => op.validate(),
            Self::ArbitraryRotate(op) => op.validate(),
            Self::RandomAffine(op) => op.validate(),
            Self::Perspective(op) => op.validate(),
            Self::RandomPerspective(op) => op.validate(),
            Self::ElasticTransform(op) => op.validate(),
            Self::RandomApply { probability, ops } => {
                validate_probability(*probability, "random_apply")?;
                validate_nested_ops(ops, "random_apply")
            }
            Self::RandomChoice { choices, weights } => {
                if choices.is_empty() {
                    return Err(invalid_argument(
                        "random_choice requires at least one choice",
                    ));
                }
                choice_cumulative_weights(choices.len(), weights.as_deref())?;
                for choice in choices {
                    validate_nested_ops(choice, "random_choice")?;
                }
                Ok(())
            }
            Self::RandomOrder { ops } => validate_nested_ops(ops, "random_order"),
            Self::GaussianBlur(op) => op.validate(),
            Self::Grayscale(op) => op.validate(),
            Self::RandomGrayscale(op) => op.validate(),
            Self::RandomErasing(op) => op.validate(),
            Self::Rotate(_) => Ok(()),
            Self::ConvertImageDtype(op) => op.validate(),
            Self::Normalize(op) => op.validate(),
        }
    }
}

fn validate_probability(probability: f64, op_name: &str) -> RivetResult<()> {
    if probability.is_finite() && (0.0..=1.0).contains(&probability) {
        Ok(())
    } else {
        Err(invalid_argument(format!(
            "{op_name} probability must be finite and in [0.0, 1.0]"
        )))
    }
}

fn validate_nested_ops(ops: &[ImageOp], op_name: &str) -> RivetResult<()> {
    for op in ops {
        op.validate()?;
        if op.execution_kind() != ExecutionKind::Sample {
            return Err(invalid_pipeline(format!(
                "{op_name} nested transforms must be sample-stage operations; {} is batch-stage",
                op.name()
            )));
        }
    }
    Ok(())
}
