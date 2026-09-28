//! Local, exact canonicalizations. No image operation is exchanged with another.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use rivet_plan::{
    AxisOrder, Diagnostic, LogicalNode, LogicalPlan, NodeId, NodeKind, OptimizerContext,
    OptimizerPass, PassResult, Representation, ValueGranularity,
};

use super::op::{ImageOp, IndexOp};
use crate::sample::image::ImageAxisOrder;

pub(crate) struct CanonicalizeSelectionsLayouts {
    pub(crate) selections: bool,
    pub(crate) layouts: bool,
}

impl OptimizerPass for CanonicalizeSelectionsLayouts {
    fn name(&self) -> &'static str {
        "canonicalize-selections-layouts"
    }

    fn run(
        &self,
        plan: &mut LogicalPlan,
        context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        // Layout uses inferred properties. Perform it before selection rewrites
        // invalidate them; neither rewrite changes the upstream image value.
        let mut changed = false;
        if self.layouts {
            changed |= simplify_layouts(plan, context)?;
        }
        if self.selections {
            changed |= simplify_selections(plan, context)?;
        }
        if changed {
            context.clear_annotations();
        }
        Ok(PassResult {
            changed,
            ..PassResult::default()
        })
    }
}

fn consumers(plan: &LogicalPlan) -> Result<HashMap<NodeId, usize>, String> {
    let mut result = HashMap::new();
    for id in plan
        .topological_order()
        .map_err(|error| error.to_string())?
    {
        for input in plan
            .node(id)
            .map_err(|error| error.to_string())?
            .inputs()
            .iter()
        {
            *result.entry(input).or_default() += 1;
        }
    }
    // Being the graph result is an additional observation of the value.
    *result
        .entry(plan.root().map_err(|error| error.to_string())?)
        .or_default() += 1;
    Ok(result)
}

fn selection(node: &LogicalNode) -> Option<&IndexOp> {
    if node.kind() != NodeKind::Index || node.inputs().len() != 1 {
        return None;
    }
    match node.payload_as::<IndexOp>()? {
        op @ (IndexOp::Skip { .. } | IndexOp::Take { .. }) => Some(op),
        IndexOp::Shuffle { .. } => None,
    }
}

fn equivalent_slice(left: &IndexOp, right: &IndexOp) -> bool {
    match (left, right) {
        (IndexOp::Skip { count: a }, IndexOp::Skip { count: b })
        | (IndexOp::Take { count: a }, IndexOp::Take { count: b }) => a == b,
        _ => false,
    }
}

fn simplify_selections(
    plan: &mut LogicalPlan,
    context: &mut OptimizerContext,
) -> Result<bool, String> {
    let uses = consumers(plan)?;
    let mut visited = HashSet::new();
    let mut changed = false;
    let order = plan
        .topological_order()
        .map_err(|error| error.to_string())?;
    for tail in order.into_iter().rev() {
        if visited.contains(&tail)
            || selection(plan.node(tail).map_err(|error| error.to_string())?).is_none()
        {
            continue;
        }
        let mut chain = vec![tail];
        let mut input = plan
            .node(tail)
            .map_err(|error| error.to_string())?
            .inputs()
            .get(0)
            .unwrap();
        while uses.get(&input).copied() == Some(1) {
            let node = plan.node(input).map_err(|error| error.to_string())?;
            if selection(node).is_none() {
                break;
            }
            chain.push(input);
            input = node.inputs().get(0).unwrap();
        }
        visited.extend(chain.iter().copied());
        if chain.len() < 2 {
            continue;
        }
        chain.reverse();
        // This interval represents every possible input length <= usize::MAX:
        // actual bounds are simply clamped by the sampler to its input length.
        let (mut start, mut end) = (0usize, usize::MAX);
        for id in &chain {
            selection(plan.node(*id).map_err(|error| error.to_string())?)
                .unwrap()
                .apply_range(&mut start, &mut end);
        }
        let mut canonical = Vec::new();
        if start != 0 {
            canonical.push(IndexOp::Skip { count: start });
        }
        if end != usize::MAX {
            canonical.push(IndexOp::Take { count: end - start });
        }
        if canonical.len() == chain.len()
            && chain
                .iter()
                .zip(&canonical)
                .all(|(id, op)| equivalent_slice(selection(plan.node(*id).unwrap()).unwrap(), op))
        {
            continue;
        }
        match canonical.as_slice() {
            [] => plan
                .redirect_uses(tail, input)
                .map_err(|error| error.to_string())?,
            [op] => plan
                .replace_node(
                    tail,
                    LogicalNode::new(NodeKind::Index, [input], Some(Arc::new(op.clone()))),
                )
                .map_err(|error| error.to_string())?,
            [skip, take] => {
                let head = chain[0];
                plan.replace_node(
                    head,
                    LogicalNode::new(NodeKind::Index, [input], Some(Arc::new(skip.clone()))),
                )
                .map_err(|error| error.to_string())?;
                plan.replace_node(
                    tail,
                    LogicalNode::new(NodeKind::Index, [head], Some(Arc::new(take.clone()))),
                )
                .map_err(|error| error.to_string())?;
            }
            _ => unreachable!(),
        }
        context.diagnostics.push(Diagnostic {
            code: "rewrite.canonical-selection",
            message: format!(
                "composed {} adjacent selections ending at %{} into {} exact slice operations",
                chain.len(),
                tail.index(),
                canonical.len()
            ),
        });
        changed = true;
    }
    Ok(changed)
}

fn simplify_layouts(
    plan: &mut LogicalPlan,
    context: &mut OptimizerContext,
) -> Result<bool, String> {
    let uses = consumers(plan)?;
    let mut changed = false;
    // Reverse traversal avoids revisiting the now-unreachable inner view.
    for tail in plan
        .topological_order()
        .map_err(|error| error.to_string())?
        .into_iter()
        .rev()
    {
        let outer = plan.node(tail).map_err(|error| error.to_string())?;
        let Some(ImageOp::Layout(final_layout)) = outer.payload_as::<ImageOp>() else {
            continue;
        };
        if outer.kind() != NodeKind::Op || outer.inputs().len() != 1 {
            continue;
        }
        let middle = outer.inputs().get(0).unwrap();
        if uses.get(&middle).copied() != Some(1) {
            continue;
        }
        let inner = plan.node(middle).map_err(|error| error.to_string())?;
        let Some(ImageOp::Layout(first_layout)) = inner.payload_as::<ImageOp>() else {
            continue;
        };
        if inner.kind() != NodeKind::Op
            || inner.inputs().len() != 1
            || first_layout.axis_order == final_layout.axis_order
        {
            continue;
        }
        let input = inner.inputs().get(0).unwrap();
        let Some(properties) = context.annotations().get(input) else {
            continue;
        };
        let expected_rank = match properties.granularity {
            Some(ValueGranularity::Sample) => 3,
            Some(ValueGranularity::Batch) => 4,
            _ => continue,
        };
        // Preserve rank errors too: unknown rank cannot prove that the two
        // runtime view permutations would have accepted their input.
        if properties.representation != Some(Representation::Image)
            || properties.shape.as_ref().map(|shape| shape.rank()) != Some(expected_rank)
        {
            continue;
        }
        let axis = match properties.axis_order {
            Some(AxisOrder::Hwc | AxisOrder::Nhwc) => ImageAxisOrder::Hwc,
            Some(AxisOrder::Chw | AxisOrder::Nchw) => ImageAxisOrder::Chw,
            _ => continue,
        };
        if axis != final_layout.axis_order {
            continue;
        }
        // Layout and BatchKernel::Layout only permute strides. Inverse views
        // restore the original shape, strides, storage and contiguity. This
        // rule never crosses Normalize (including materializing fusion).
        plan.redirect_uses(tail, input)
            .map_err(|error| error.to_string())?;
        context.diagnostics.push(Diagnostic {
            code: "rewrite.inverse-layout",
            message: format!(
                "removed inverse Layout views %{} and %{}; preserved input strides and contiguity",
                middle.index(),
                tail.index()
            ),
        });
        changed = true;
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::op::compile_sampler;
    use crate::sample::image::{DecodedSample, ImageSample};
    use crate::sampler::IndexSampler;
    use crate::transforms::LayoutConfig;
    use rivet_core::{Device, Tensor};
    use rivet_data::random::RandomContext;
    use rivet_plan::{Contiguity, ShapeDim, ValueProperties, ValueShape};

    fn index_plan(ops: &[IndexOp]) -> LogicalPlan {
        let mut plan = LogicalPlan::new();
        let mut previous = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        for op in ops {
            previous = plan.add_node(LogicalNode::new(
                NodeKind::Index,
                [previous],
                Some(Arc::new(op.clone())),
            ));
        }
        let sink = plan.add_node(LogicalNode::new(NodeKind::Sink, [previous], None));
        plan.set_root(sink).unwrap();
        plan
    }

    fn index_ops(plan: &LogicalPlan) -> Vec<IndexOp> {
        plan.topological_order()
            .unwrap()
            .into_iter()
            .filter_map(|id| plan.node(id).unwrap().payload_as::<IndexOp>().cloned())
            .collect()
    }

    fn selected(len: usize, ops: &[IndexOp]) -> Vec<usize> {
        IndexSampler::new(
            compile_sampler(len, ops, RandomContext::new(11)).unwrap(),
            0,
        )
        .next_indices(usize::MAX)
        .unwrap_or_default()
    }

    #[test]
    fn canonical_slices_preserve_exhaustive_small_ranges_and_overflow() {
        let counts = [0, 1, 3, 5, usize::MAX - 1, usize::MAX];
        for a in counts {
            for b in counts {
                for c in counts {
                    for mask in 0..8 {
                        let ops = [a, b, c]
                            .into_iter()
                            .enumerate()
                            .map(|(i, count)| {
                                if mask & (1 << i) == 0 {
                                    IndexOp::Take { count }
                                } else {
                                    IndexOp::Skip { count }
                                }
                            })
                            .collect::<Vec<_>>();
                        let mut plan = index_plan(&ops);
                        let pass = CanonicalizeSelectionsLayouts {
                            selections: true,
                            layouts: false,
                        };
                        let mut context = OptimizerContext::default();
                        pass.run(&mut plan, &mut context).unwrap();
                        let canonical = index_ops(&plan);
                        assert!(canonical.len() <= 2);
                        for len in [0, 1, 4, 10] {
                            assert_eq!(selected(len, &ops), selected(len, &canonical));
                        }
                        // Huge source lengths use interval arithmetic, avoiding allocation.
                        let before =
                            compile_sampler(usize::MAX, &ops, RandomContext::new(0)).unwrap();
                        let after =
                            compile_sampler(usize::MAX, &canonical, RandomContext::new(0)).unwrap();
                        match (before, after) {
                            (
                                crate::sampler::SamplerPlan::Sequential { start: a, end: b },
                                crate::sampler::SamplerPlan::Sequential { start: c, end: d },
                            ) => assert_eq!((a, b), (c, d)),
                            _ => panic!("slice-only samplers must remain sequential"),
                        }
                        assert!(!pass.run(&mut plan, &mut context).unwrap().changed);
                        plan.validate().unwrap();
                    }
                }
            }
        }
    }

    #[test]
    fn shuffle_is_a_slice_barrier() {
        let ops = [
            IndexOp::Take { count: 8 },
            IndexOp::Shuffle { seed: 11 },
            IndexOp::Skip { count: 2 },
            IndexOp::Take { count: 3 },
        ];
        let mut plan = index_plan(&ops);
        CanonicalizeSelectionsLayouts {
            selections: true,
            layouts: false,
        }
        .run(&mut plan, &mut OptimizerContext::default())
        .unwrap();
        assert_eq!(index_ops(&plan).len(), 4);
        assert_eq!(selected(20, &ops), selected(20, &index_ops(&plan)));
    }

    #[test]
    fn slices_do_not_cross_batch_device_or_image_nodes() {
        for kind in [NodeKind::Batch, NodeKind::DeviceCut, NodeKind::Op] {
            let mut plan = index_plan(&[IndexOp::Take { count: 8 }, IndexOp::Take { count: 4 }]);
            let ids = plan.topological_order().unwrap();
            let payload = if kind == NodeKind::DeviceCut {
                Some(
                    Arc::new(rivet_plan::DeviceCut::new(rivet_plan::DeviceTarget::cuda(
                        0,
                    ))) as Arc<dyn rivet_plan::PlanPayload>,
                )
            } else {
                None
            };
            let boundary = plan.add_node(LogicalNode::new(kind, [ids[1]], payload));
            let tail = plan.node(ids[2]).unwrap().clone().with_inputs([boundary]);
            plan.replace_node(ids[2], tail).unwrap();
            assert!(
                !CanonicalizeSelectionsLayouts {
                    selections: true,
                    layouts: false
                }
                .run(&mut plan, &mut OptimizerContext::default())
                .unwrap()
                .changed
            );
            assert_eq!(index_ops(&plan).len(), 2);
        }
    }

    #[test]
    fn shared_selection_prefix_is_preserved() {
        let mut plan = index_plan(&[
            IndexOp::Take { count: 7 },
            IndexOp::Take { count: 2 },
            IndexOp::Take { count: 1 },
        ]);
        let prefix = plan.nodes().nth(1).unwrap().0;
        let branch = plan.root().unwrap();
        let join = plan.add_node(LogicalNode::new(NodeKind::Sink, [branch, prefix], None));
        plan.set_root(join).unwrap();
        CanonicalizeSelectionsLayouts {
            selections: true,
            layouts: false,
        }
        .run(&mut plan, &mut OptimizerContext::default())
        .unwrap();
        assert!(matches!(
            plan.node(prefix).unwrap().payload_as::<IndexOp>(),
            Some(IndexOp::Take { count: 7 })
        ));
        assert_eq!(index_ops(&plan).len(), 2);
        plan.validate().unwrap();
    }

    fn layout_plan(
        axis: ImageAxisOrder,
        contiguous: Contiguity,
        rank: Option<usize>,
    ) -> (LogicalPlan, OptimizerContext, NodeId, NodeId, NodeId) {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let opposite = match axis {
            ImageAxisOrder::Hwc => ImageAxisOrder::Chw,
            ImageAxisOrder::Chw => ImageAxisOrder::Hwc,
        };
        let middle = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [source],
            Some(Arc::new(ImageOp::Layout(LayoutConfig::new(opposite)))),
        ));
        let tail = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [middle],
            Some(Arc::new(ImageOp::Layout(LayoutConfig::new(axis)))),
        ));
        let sink = plan.add_node(LogicalNode::new(NodeKind::Sink, [tail], None));
        plan.set_root(sink).unwrap();
        let mut context = OptimizerContext::default();
        context.annotations_mut().insert(
            source,
            ValueProperties {
                representation: Some(Representation::Image),
                shape: rank.map(|rank| ValueShape(vec![ShapeDim::Dynamic; rank])),
                granularity: Some(if rank == Some(4) {
                    ValueGranularity::Batch
                } else {
                    ValueGranularity::Sample
                }),
                axis_order: Some(match (axis, rank == Some(4)) {
                    (ImageAxisOrder::Hwc, false) => AxisOrder::Hwc,
                    (ImageAxisOrder::Chw, false) => AxisOrder::Chw,
                    (ImageAxisOrder::Hwc, true) => AxisOrder::Nhwc,
                    (ImageAxisOrder::Chw, true) => AxisOrder::Nchw,
                }),
                contiguity: Some(contiguous),
                ..ValueProperties::default()
            },
        );
        (plan, context, source, middle, tail)
    }

    #[test]
    fn inverse_layouts_restore_contiguous_and_strided_inputs() {
        for axis in [ImageAxisOrder::Hwc, ImageAxisOrder::Chw] {
            for contiguity in [Contiguity::Contiguous, Contiguity::Strided] {
                for rank in [3, 4] {
                    let (mut plan, mut context, source, _, _) =
                        layout_plan(axis, contiguity, Some(rank));
                    assert!(
                        CanonicalizeSelectionsLayouts {
                            selections: false,
                            layouts: true
                        }
                        .run(&mut plan, &mut context)
                        .unwrap()
                        .changed
                    );
                    assert_eq!(
                        plan.node(plan.root().unwrap()).unwrap().inputs().get(0),
                        Some(source)
                    );
                }
            }
        }
        let hwc = Tensor::from_vec((0u8..24).collect::<Vec<_>>(), [2, 4, 3], &Device::Cpu).unwrap();
        for input in [hwc.clone(), hwc.permute(&[2, 0, 1]).unwrap()] {
            let axis = if input.dims()[0] == 3 {
                ImageAxisOrder::Chw
            } else {
                ImageAxisOrder::Hwc
            };
            let opposite = if axis == ImageAxisOrder::Hwc {
                ImageAxisOrder::Chw
            } else {
                ImageAxisOrder::Hwc
            };
            let sample = ImageSample::Decoded(DecodedSample {
                image: input.clone(),
                label: 0,
            });
            let first = LayoutConfig::new(opposite).apply(sample, axis).unwrap();
            let output = LayoutConfig::new(axis)
                .apply(first, opposite)
                .unwrap()
                .into_decoded()
                .unwrap()
                .image;
            assert_eq!(
                output.to_vec::<u8>().unwrap(),
                input.to_vec::<u8>().unwrap()
            );
            assert_eq!(output.dims(), input.dims());
            assert_eq!(output.is_contiguous(), input.is_contiguous());
            assert!(output.same_storage(&input));
            let batch_input = input.unsqueeze(0).unwrap();
            let first = LayoutConfig::new(opposite)
                .apply_batch(batch_input.clone(), axis)
                .unwrap();
            let output = LayoutConfig::new(axis)
                .apply_batch(first, opposite)
                .unwrap();
            assert_eq!(output.is_contiguous(), batch_input.is_contiguous());
            assert!(output.same_storage(&batch_input));
            assert_eq!(
                output.to_vec::<u8>().unwrap(),
                batch_input.to_vec::<u8>().unwrap()
            );
        }
    }

    #[test]
    fn inverse_layouts_never_cross_materializing_normalize() {
        let (mut plan, mut context, _, middle, tail) =
            layout_plan(ImageAxisOrder::Hwc, Contiguity::Contiguous, Some(3));
        let normalize = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [middle],
            Some(Arc::new(ImageOp::normalize(vec![0.5; 3], vec![0.5; 3]))),
        ));
        let outer = plan.node(tail).unwrap().clone().with_inputs([normalize]);
        plan.replace_node(tail, outer).unwrap();
        assert!(
            !CanonicalizeSelectionsLayouts {
                selections: false,
                layouts: true
            }
            .run(&mut plan, &mut context)
            .unwrap()
            .changed
        );
    }

    #[test]
    fn inverse_layout_requires_known_valid_rank_and_unshared_intermediate() {
        for rank in [None, Some(2), Some(5)] {
            let (mut plan, mut context, _, _, _) =
                layout_plan(ImageAxisOrder::Hwc, Contiguity::Contiguous, rank);
            assert!(
                !CanonicalizeSelectionsLayouts {
                    selections: false,
                    layouts: true
                }
                .run(&mut plan, &mut context)
                .unwrap()
                .changed
            );
        }
        let (mut plan, mut context, _, middle, tail) =
            layout_plan(ImageAxisOrder::Hwc, Contiguity::Contiguous, Some(3));
        let join = plan.add_node(LogicalNode::new(NodeKind::Sink, [tail, middle], None));
        plan.set_root(join).unwrap();
        assert!(
            !CanonicalizeSelectionsLayouts {
                selections: false,
                layouts: true
            }
            .run(&mut plan, &mut context)
            .unwrap()
            .changed
        );
    }
}
