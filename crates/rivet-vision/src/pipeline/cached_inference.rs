//! Reuse validated vision properties until a domain rewrite invalidates them.
//!
//! This is intentionally domain-local: vision rewrites either clear their
//! annotations or update affected properties after proving downstream facts
//! unchanged. Generic optimizer plugins need not honor that convention.

use std::sync::Arc;

use rivet_plan::{LogicalPlan, OptimizerContext, OptimizerPass, PassResult, PropertyInference};

pub(crate) struct CachedInferProperties(pub(crate) Arc<dyn PropertyInference>);

impl OptimizerPass for CachedInferProperties {
    fn name(&self) -> &'static str {
        "property-inference-validation"
    }

    fn run(
        &self,
        plan: &mut LogicalPlan,
        context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        // Check the whole arena, not just annotation count: newly appended
        // nodes may otherwise accidentally match a stale annotation's count.
        if plan
            .nodes()
            .all(|(id, _)| context.annotations().get(id).is_some())
        {
            return Ok(PassResult::unchanged());
        }
        let properties = plan
            .infer_properties(self.0.as_ref())
            .map_err(|error| error.to_string())?;
        context.set_annotations(properties);
        Ok(PassResult::unchanged())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::cache::DecodedImageMemoryDataset;
    use crate::pipeline::builder::ImagePipeline;
    use crate::pipeline::inference::VisionPropertyInference;
    use crate::pipeline::op::ImageOp;
    use crate::sample::image::DecodedSample;
    use crate::source::ImageSource;
    use rivet_core::{DType, Device, Tensor};
    use rivet_plan::{AxisOrder, DataType, LogicalNode, NodeKind, ValueProperties};

    struct CountingVisionInference(Arc<AtomicUsize>);

    impl PropertyInference for CountingVisionInference {
        fn infer_node(
            &self,
            node: &LogicalNode,
            inputs: &[ValueProperties],
        ) -> Result<ValueProperties, String> {
            self.0.fetch_add(1, Ordering::Relaxed);
            VisionPropertyInference::default().infer_node(node, inputs)
        }
    }

    fn fixture() -> (LogicalPlan, CachedInferProperties, Arc<AtomicUsize>) {
        let source = ImageSource::from_decoded(Arc::new(DecodedImageMemoryDataset::new(vec![
            DecodedSample {
                image: Tensor::from_vec(vec![1u8, 2, 3], [1, 1, 3], &Device::Cpu).unwrap(),
                label: 7,
            },
        ])));
        let plan = ImagePipeline::from_source(source)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .to_logical_plan();
        let calls = Arc::new(AtomicUsize::new(0));
        let pass = CachedInferProperties(Arc::new(CountingVisionInference(calls.clone())));
        (plan, pass, calls)
    }

    #[test]
    fn unchanged_plan_reuses_properties_and_cached_placement() {
        let (mut plan, pass, calls) = fixture();
        let mut context = OptimizerContext::default();
        pass.run(&mut plan, &mut context).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), plan.nodes().len());
        context.set_placement(rivet_plan::PlacementPlan::default());
        pass.run(&mut plan, &mut context).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), plan.nodes().len());
        assert!(context.placement().is_some());
        assert_eq!(
            context
                .annotations()
                .get(plan.root().unwrap())
                .unwrap()
                .dtype,
            Some(DataType::F32)
        );
    }

    #[test]
    fn appended_node_forces_inference_without_explicit_invalidation() {
        let (mut plan, pass, calls) = fixture();
        let mut context = OptimizerContext::default();
        pass.run(&mut plan, &mut context).unwrap();
        let previous_calls = calls.load(Ordering::Relaxed);
        let layout = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [plan.root().unwrap()],
            Some(Arc::new(ImageOp::hwc_to_chw())),
        ));
        plan.set_root(layout).unwrap();
        pass.run(&mut plan, &mut context).unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            previous_calls + plan.nodes().len()
        );
        assert_eq!(
            context.annotations().get(layout).unwrap().axis_order,
            Some(AxisOrder::Chw)
        );
    }

    #[test]
    fn invalidated_same_size_rewrite_updates_dtype_and_layout() {
        let (mut plan, pass, calls) = fixture();
        let mut context = OptimizerContext::default();
        pass.run(&mut plan, &mut context).unwrap();
        let old_calls = calls.load(Ordering::Relaxed);
        let normalize = plan
            .nodes()
            .find(|(_, node)| matches!(node.payload_as::<ImageOp>(), Some(ImageOp::Normalize(_))))
            .unwrap()
            .0;
        let source = plan.node(normalize).unwrap().inputs().get(0).unwrap();
        plan.replace_node(
            normalize,
            LogicalNode::new(
                NodeKind::Op,
                [source],
                Some(Arc::new(ImageOp::hwc_to_chw())),
            ),
        )
        .unwrap();
        context.clear_annotations();
        pass.run(&mut plan, &mut context).unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            old_calls + plan.nodes().len()
        );
        let output = context.annotations().get(plan.root().unwrap()).unwrap();
        assert_eq!(output.dtype, Some(DataType::U8));
        assert_eq!(output.axis_order, Some(AxisOrder::Chw));
    }

    #[test]
    fn invalidated_rewrite_preserves_semantic_validation_errors() {
        let (mut plan, pass, calls) = fixture();
        let mut context = OptimizerContext::default();
        pass.run(&mut plan, &mut context).unwrap();
        let old_calls = calls.load(Ordering::Relaxed);
        let normalize = plan
            .nodes()
            .find(|(_, node)| matches!(node.payload_as::<ImageOp>(), Some(ImageOp::Normalize(_))))
            .unwrap()
            .0;
        let source = plan.node(normalize).unwrap().inputs().get(0).unwrap();
        plan.replace_node(
            normalize,
            LogicalNode::new(
                NodeKind::Op,
                [source],
                Some(Arc::new(ImageOp::convert_image_dtype(DType::I64))),
            ),
        )
        .unwrap();
        context.clear_annotations();
        let error = pass.run(&mut plan, &mut context).unwrap_err();
        assert!(
            error.contains("image dtype conversion supports uint8 and float32"),
            "{error}"
        );
        assert!(calls.load(Ordering::Relaxed) > old_calls);
    }
}
