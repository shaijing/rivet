use rivet_plan::{
    Contiguity, DomainId, DomainOp, OpId, PlanPayload, Representation, ShapeDim, ValueProperties,
};
use std::any::Any;

/// Concatenate image branches along an image axis (excluding the batch axis).
/// Inputs must describe the same samples, dtype, layout and non-concat extents.
#[derive(Clone, Copy, Debug)]
pub struct ImageConcat {
    pub axis: usize,
}

impl PlanPayload for ImageConcat {
    fn domain(&self) -> &'static str {
        "vision"
    }
    fn name(&self) -> &'static str {
        "ImageConcat"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn domain_id(&self) -> Option<DomainId> {
        Some(crate::VISION_DOMAIN_ID)
    }
    fn op_id(&self) -> Option<OpId> {
        Some(OpId::new(10000))
    }
    fn as_domain_op(&self) -> Option<&dyn DomainOp> {
        Some(self)
    }
}

impl DomainOp for ImageConcat {
    fn infer_properties(&self, inputs: &[ValueProperties]) -> Result<ValueProperties, String> {
        if self.axis >= 3 || inputs.len() < 2 {
            return Err("ImageConcat requires at least two images and an axis in 0..3".to_owned());
        }
        let mut output = inputs[0].clone();
        if output.representation != Some(Representation::Image) {
            return Err("ImageConcat requires decoded images".to_owned());
        }
        let batch = output.granularity == Some(rivet_plan::ValueGranularity::Batch);
        let axis = self.axis + usize::from(batch);
        for input in &inputs[1..] {
            if input.representation != output.representation
                || input.dtype != output.dtype
                || input.axis_order != output.axis_order
                || input.granularity != output.granularity
                || input.residency != output.residency
                || input
                    .operator
                    .is_some_and(|op| op.stage == rivet_plan::OperatorStage::Batch)
                    != output
                        .operator
                        .is_some_and(|op| op.stage == rivet_plan::OperatorStage::Batch)
            {
                return Err("ImageConcat inputs must have matching representation, dtype, layout, stage, granularity and residency".to_owned());
            }
            match (&mut output.shape, &input.shape) {
                (Some(out), Some(other)) if out.rank() == other.rank() && axis < out.rank() => {
                    for (dim, (a, b)) in out.0.iter_mut().zip(&other.0).enumerate() {
                        if dim == axis {
                            *a = match (*a, *b) {
                                (ShapeDim::Known(a), ShapeDim::Known(b)) => ShapeDim::Known(
                                    a.checked_add(b).ok_or("ImageConcat extent overflow")?,
                                ),
                                _ => ShapeDim::Dynamic,
                            };
                        } else if matches!((*a, *b), (ShapeDim::Known(a), ShapeDim::Known(b)) if a != b)
                        {
                            return Err(
                                "ImageConcat non-concatenated dimensions must match".to_owned()
                            );
                        }
                    }
                }
                _ => return Err("ImageConcat inputs must have matching image ranks".to_owned()),
            }
        }
        output.contiguity = Some(Contiguity::Contiguous);
        if let Some(op) = &mut output.operator {
            op.sample_stage_has_work = true;
            if op.stage == rivet_plan::OperatorStage::Source {
                op.stage = rivet_plan::OperatorStage::Sample;
            }
        }
        Ok(output)
    }
}
