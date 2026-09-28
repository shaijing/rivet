//! Exact common-subplan elimination for explicitly supported pure image ops.
//!
//! This pass never changes operation order or evaluates mathematical
//! equivalence. A key identifies the same operation, bit-exact parameters and
//! the same ordered input nodes. Sources and random/effectful operations are
//! deliberately absent from the whitelist.

use std::collections::HashMap;

use rivet_plan::{
    Diagnostic, LogicalPlan, NodeId, NodeKind, OptimizerContext, OptimizerPass, PassResult,
};

use super::op::ImageOp;
use crate::sample::image::ImageAxisOrder;
use crate::transforms::{FlipDirection, InterpolationMode, PaddingMode};

pub(crate) struct Cse;

#[derive(PartialEq, Eq, Hash)]
struct Key {
    inputs: Vec<NodeId>,
    parameters: Vec<u64>,
}

impl OptimizerPass for Cse {
    fn name(&self) -> &'static str {
        "common-subplan-elimination"
    }

    fn run(
        &self,
        plan: &mut LogicalPlan,
        context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        let order = plan
            .topological_order()
            .map_err(|error| error.to_string())?;
        // A root-reachable graph with only unary nodes is a single chain;
        // identical siblings cannot exist. Keep its common compilation path
        // free of semantic-key and hash-table allocations.
        if order
            .iter()
            .all(|&id| plan.node(id).is_ok_and(|node| node.inputs().len() <= 1))
        {
            return Ok(PassResult::default());
        }
        let mut representatives = HashMap::<Key, NodeId>::new();
        let mut canonical = vec![None; plan.nodes().len()];
        let mut merged = 0;

        // Canonicalize inputs as each consumer is visited. Unlike repeatedly
        // redirecting every use, this visits each reachable edge only once.
        for id in order {
            let original = plan.node(id).map_err(|error| error.to_string())?;
            let inputs = original
                .inputs()
                .iter()
                .map(|input| canonical[input.index()].unwrap_or(input))
                .collect::<Vec<_>>();
            let rewired = original.inputs().iter().ne(inputs.iter().copied());
            let key = if original.kind() == NodeKind::Op && inputs.len() == 1 {
                original
                    .payload_as::<ImageOp>()
                    .and_then(exact_parameters)
                    .map(|parameters| Key {
                        inputs: inputs.clone(),
                        parameters,
                    })
            } else {
                None
            };
            let replacement = if let Some(key) = key {
                if let Some(&representative) = representatives.get(&key) {
                    merged += 1;
                    representative
                } else {
                    representatives.insert(key, id);
                    id
                }
            } else {
                id
            };
            canonical[id.index()] = Some(replacement);
            if rewired {
                let replacement_node = original.clone().with_inputs(inputs);
                plan.replace_node(id, replacement_node)
                    .map_err(|error| error.to_string())?;
            }
        }

        if merged == 0 {
            return Ok(PassResult::default());
        }
        let root = plan.root().map_err(|error| error.to_string())?;
        plan.set_root(canonical[root.index()].unwrap_or(root))
            .map_err(|error| error.to_string())?;
        plan.prune_unreachable()
            .map_err(|error| error.to_string())?;
        context.clear_annotations();
        Ok(PassResult {
            changed: true,
            diagnostics: vec![Diagnostic {
                code: "rewrite.common-subplan",
                message: format!(
                    "shared {merged} identical deterministic image operations with bit-exact parameters and identical ordered inputs"
                ),
            }],
            ..PassResult::default()
        })
    }
}

// No Debug formatting, approximate float comparison or random-operation
// classification is used as an equality proof. Adding an operation requires
// explicitly accounting for every parameter and its purity here.
fn exact_parameters(op: &ImageOp) -> Option<Vec<u64>> {
    let mut key = vec![u64::from(op.planning_op_id())];
    match op {
        ImageOp::Decode(_) => {}
        ImageOp::Resize(config) => key.extend([
            u64::from(config.width),
            u64::from(config.height),
            match config.interpolation {
                InterpolationMode::Nearest => 0,
                InterpolationMode::Bilinear => 1,
                InterpolationMode::Bicubic => 2,
                InterpolationMode::Lanczos3 => 3,
            },
        ]),
        ImageOp::Crop(config) => key.extend([
            u64::from(config.x),
            u64::from(config.y),
            u64::from(config.width),
            u64::from(config.height),
        ]),
        ImageOp::CenterCrop(config) => {
            key.extend([u64::from(config.width), u64::from(config.height)]);
        }
        ImageOp::Pad(config) => key.extend([
            u64::from(config.left),
            u64::from(config.right),
            u64::from(config.top),
            u64::from(config.bottom),
            u64::from(config.value.to_bits()),
            match config.mode {
                PaddingMode::Constant => 0,
                PaddingMode::Edge => 1,
                PaddingMode::Reflect => 2,
                PaddingMode::Symmetric => 3,
            },
        ]),
        ImageOp::Flip(config) => key.push(match config.direction {
            FlipDirection::Horizontal => 0,
            FlipDirection::Vertical => 1,
        }),
        ImageOp::Normalize(config) => {
            key.push(config.mean.len() as u64);
            key.extend(config.mean.iter().map(|value| u64::from(value.to_bits())));
            key.push(config.std.len() as u64);
            key.extend(config.std.iter().map(|value| u64::from(value.to_bits())));
        }
        ImageOp::Layout(config) => key.push(match config.axis_order {
            ImageAxisOrder::Hwc => 0,
            ImageAxisOrder::Chw => 1,
        }),
        ImageOp::ConvertImageDtype(config) => key.push(match config.dtype {
            rivet_core::DType::U8 => 0,
            rivet_core::DType::F32 => 1,
            _ => return None,
        }),
        _ => return None,
    }
    Some(key)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rivet_plan::LogicalNode;

    use super::*;
    use crate::transforms::{
        CropConfig, DecodeImageConfig, LayoutConfig, NormalizeConfig, RandomCropConfig,
        ResizeConfig,
    };

    fn add(plan: &mut LogicalPlan, input: NodeId, op: ImageOp) -> NodeId {
        plan.add_node(LogicalNode::new(NodeKind::Op, [input], Some(Arc::new(op))))
    }

    fn join(plan: &mut LogicalPlan, inputs: &[NodeId]) {
        let root = plan.add_node(LogicalNode::new(
            NodeKind::Sink,
            inputs.iter().copied(),
            None,
        ));
        plan.set_root(root).unwrap();
    }

    fn apply(plan: &mut LogicalPlan) -> PassResult {
        let result = Cse.run(plan, &mut OptimizerContext::default()).unwrap();
        plan.validate().unwrap();
        result
    }

    fn duplicate_pair(left: ImageOp, right: ImageOp) -> LogicalPlan {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let a = add(&mut plan, source, left);
        let b = add(&mut plan, source, right);
        join(&mut plan, &[a, b]);
        plan
    }

    #[test]
    fn shares_duplicate_decode_resize_and_preserves_ordered_edges() {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let a_decode = add(&mut plan, source, ImageOp::Decode(DecodeImageConfig));
        let b_decode = add(&mut plan, source, ImageOp::Decode(DecodeImageConfig));
        let a = add(
            &mut plan,
            a_decode,
            ImageOp::Resize(ResizeConfig::new(8, 9)),
        );
        let b = add(
            &mut plan,
            b_decode,
            ImageOp::Resize(ResizeConfig::new(8, 9)),
        );
        let crop = add(&mut plan, b, ImageOp::Crop(CropConfig::new(0, 0, 4, 4)));
        join(&mut plan, &[b, crop, a, a]);
        assert!(apply(&mut plan).changed);
        assert_eq!(plan.nodes().len(), 5);
        let edges = plan
            .node(plan.root().unwrap())
            .unwrap()
            .inputs()
            .iter()
            .collect::<Vec<_>>();
        assert_eq!(edges[0], edges[2]);
        assert_eq!(edges[2], edges[3]);
        assert_eq!(plan.node(edges[1]).unwrap().inputs().get(0), Some(edges[0]));
        assert!(!apply(&mut plan).changed);
    }

    #[test]
    fn parameter_differences_prevent_sharing() {
        for (left, right) in [
            (
                ImageOp::Resize(ResizeConfig::new(8, 9)),
                ImageOp::Resize(ResizeConfig::new(9, 8)),
            ),
            (
                ImageOp::Resize(ResizeConfig::new(8, 9)),
                ImageOp::Resize(ResizeConfig::with_interpolation(
                    8,
                    9,
                    InterpolationMode::Nearest,
                )),
            ),
            (
                ImageOp::Crop(CropConfig::new(0, 0, 4, 4)),
                ImageOp::Crop(CropConfig::new(1, 0, 4, 4)),
            ),
            (
                ImageOp::Layout(LayoutConfig::new(ImageAxisOrder::Hwc)),
                ImageOp::Layout(LayoutConfig::new(ImageAxisOrder::Chw)),
            ),
        ] {
            assert!(!apply(&mut duplicate_pair(left, right)).changed);
        }
    }

    #[test]
    fn normalization_uses_float_bits_and_vector_boundaries() {
        let norm = |mean, std| ImageOp::Normalize(NormalizeConfig::new(mean, std));
        assert!(
            !apply(&mut duplicate_pair(
                norm(vec![0.0], vec![1.0]),
                norm(vec![-0.0], vec![1.0])
            ))
            .changed
        );
        assert!(
            !apply(&mut duplicate_pair(
                norm(vec![1.0], vec![2.0, 3.0]),
                norm(vec![1.0, 2.0], vec![3.0])
            ))
            .changed
        );
        let nan = f32::from_bits(0x7fc00003);
        assert!(
            apply(&mut duplicate_pair(
                norm(vec![nan], vec![1.0]),
                norm(vec![nan], vec![1.0])
            ))
            .changed
        );
        assert!(
            !apply(&mut duplicate_pair(
                norm(vec![nan], vec![1.0]),
                norm(vec![f32::from_bits(0x7fc00004)], vec![1.0])
            ))
            .changed
        );
    }

    #[test]
    fn random_nodes_and_their_deterministic_successors_are_distinct() {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let random = ImageOp::RandomCrop(RandomCropConfig::new(8, 8, 4));
        let a = add(&mut plan, source, random.clone());
        let b = add(&mut plan, source, random);
        let a = add(&mut plan, a, ImageOp::Resize(ResizeConfig::new(4, 4)));
        let b = add(&mut plan, b, ImageOp::Resize(ResizeConfig::new(4, 4)));
        join(&mut plan, &[a, b]);
        assert!(!apply(&mut plan).changed);
    }

    #[test]
    fn compaction_preserves_random_operator_identity() {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let a = add(&mut plan, source, ImageOp::Decode(DecodeImageConfig));
        let b = add(&mut plan, source, ImageOp::Decode(DecodeImageConfig));
        let random = ImageOp::RandomCrop(RandomCropConfig::new(8, 8, 4));
        let a = plan.add_node(
            LogicalNode::new(NodeKind::Op, [a], Some(Arc::new(random.clone())))
                .with_semantic_identity(41),
        );
        let b = plan.add_node(
            LogicalNode::new(NodeKind::Op, [b], Some(Arc::new(random))).with_semantic_identity(42),
        );
        join(&mut plan, &[a, b]);
        assert!(apply(&mut plan).changed);
        let random_identities = plan
            .nodes()
            .filter_map(|(_, node)| {
                matches!(node.payload_as::<ImageOp>(), Some(ImageOp::RandomCrop(_)))
                    .then(|| node.semantic_identity().unwrap())
            })
            .collect::<Vec<_>>();
        assert_eq!(random_identities, vec![41, 42]);
        assert_eq!(plan.nodes().len(), 5);
    }

    #[test]
    fn distinct_sources_and_unknown_nodes_are_not_shared() {
        let mut plan = LogicalPlan::new();
        let a_source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let b_source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let a = add(&mut plan, a_source, ImageOp::Decode(DecodeImageConfig));
        let b = add(&mut plan, b_source, ImageOp::Decode(DecodeImageConfig));
        let x = plan.add_node(LogicalNode::new(NodeKind::Op, [a_source], None));
        let y = plan.add_node(LogicalNode::new(NodeKind::Op, [a_source], None));
        join(&mut plan, &[a, b, x, y]);
        assert!(!apply(&mut plan).changed);
        assert_eq!(plan.nodes().len(), 7);
    }

    #[test]
    fn serial_identical_operations_are_not_confused_with_siblings() {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let op = ImageOp::Normalize(NormalizeConfig::new(vec![0.5], vec![0.5]));
        let a = add(&mut plan, source, op.clone());
        let b = add(&mut plan, a, op);
        plan.set_root(b).unwrap();
        assert!(!apply(&mut plan).changed);
        assert_eq!(plan.root().unwrap(), b);
    }
}
