//! Execute optimized and unshared DAGs against the same encoded image source.

use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_buffer::Buffer;
use rivet_data::dataset::Dataset;
use rivet_plan::{LogicalNode, LogicalPlan, NodeId, NodeKind};

use super::op::{BatchConfig, ImageOp};
use super::{ImageConcat, ImageOptimizationOptions, ImagePipeline};
use crate::runtime::ImageDataLoader;
use crate::sample::image::{EncodedImageSample, ImageBatch};
use crate::transforms::{CropConfig, DecodeImageConfig, RandomCropConfig, ResizeConfig};

struct Images {
    bytes: Vec<Buffer>,
    reads: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
}

impl Dataset for Images {
    type Item = EncodedImageSample;

    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn get_many(&self, indices: &[usize]) -> rivet_data::DataResult<Vec<Self::Item>> {
        self.reads.fetch_add(indices.len(), Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(indices
            .iter()
            .map(|&index| EncodedImageSample {
                image: self.bytes[index].clone(),
                label: index as i64,
            })
            .collect())
    }
}

fn source(workers: usize) -> (ImagePipeline, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let reads = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let bytes = (0..8)
        .map(|sample| {
            let pixels = image::RgbImage::from_fn(6, 6, |x, y| {
                image::Rgb([
                    20 + sample * 9 + x as u8 * 3,
                    40 + y as u8 * 7,
                    100 + x as u8 * 3 + y as u8,
                ])
            });
            let mut png = Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(pixels)
                .write_to(&mut png, image::ImageFormat::Png)
                .unwrap();
            Buffer::from(png.into_inner())
        })
        .collect();
    let builder = ImagePipeline::new(Arc::new(Images {
        bytes,
        reads: reads.clone(),
        calls: calls.clone(),
    }))
    .workers(workers)
    .seed(91)
    .profiling(true);
    (builder, reads, calls)
}

fn op(plan: &mut LogicalPlan, input: NodeId, image: ImageOp) -> NodeId {
    plan.add_node(LogicalNode::new(
        NodeKind::Op,
        [input],
        Some(Arc::new(image)),
    ))
}

fn graph(builder: ImagePipeline, different: bool, random: bool) -> LogicalPlan {
    let mut plan = builder.to_logical_plan();
    let source = plan
        .nodes()
        .find(|(_, node)| node.kind() == NodeKind::Source)
        .unwrap()
        .0;
    let branches = [0, 1].map(|branch| {
        let decoded = op(&mut plan, source, ImageOp::Decode(DecodeImageConfig));
        let resized = op(
            &mut plan,
            decoded,
            ImageOp::Resize(ResizeConfig::new(
                8 + u32::from(different && branch == 1),
                8,
            )),
        );
        if random {
            plan.add_node(
                LogicalNode::new(
                    NodeKind::Op,
                    [resized],
                    Some(Arc::new(ImageOp::RandomCrop(RandomCropConfig::new(
                        4, 4, 2,
                    )))),
                )
                .with_semantic_identity(41 + branch),
            )
        } else {
            op(
                &mut plan,
                resized,
                ImageOp::Crop(CropConfig::new(
                    u32::from(different && branch == 1),
                    0,
                    4,
                    4,
                )),
            )
        }
    });
    // Reverse ports so the rewrite also has to preserve declared input order.
    let joined = plan.add_node(LogicalNode::new(
        NodeKind::Op,
        [branches[1], branches[0]],
        Some(Arc::new(ImageConcat { axis: 2 })),
    ));
    let batch = plan.add_node(LogicalNode::new(
        NodeKind::Batch,
        [joined],
        Some(Arc::new(BatchConfig::new(4, false))),
    ));
    let sink = plan.add_node(LogicalNode::new(NodeKind::Sink, [batch], None));
    plan.set_root(sink).unwrap();
    plan
}

fn compile(plan: LogicalPlan, cse: bool) -> ImageDataLoader {
    ImagePipeline::compile_logical_plan_with_options(
        plan,
        0,
        ImageOptimizationOptions {
            common_subplan_elimination: cse,
            ..ImageOptimizationOptions::default()
        },
    )
    .unwrap()
}

fn drain(loader: &mut ImageDataLoader) -> Vec<ImageBatch> {
    let mut batches = Vec::new();
    while let Some(batch) = loader.next_batch().unwrap() {
        batches.push(batch);
    }
    batches
}

fn assert_same(left: &[ImageBatch], right: &[ImageBatch]) {
    assert_eq!(left.len(), right.len());
    for (a, b) in left.iter().zip(right) {
        assert_eq!(a.images.dims(), b.images.dims());
        assert_eq!(
            a.images.to_vec::<u8>().unwrap(),
            b.images.to_vec::<u8>().unwrap()
        );
        assert_eq!(
            a.labels.to_vec::<i64>().unwrap(),
            b.labels.to_vec::<i64>().unwrap()
        );
    }
}

fn executions(loader: &ImageDataLoader) -> u64 {
    loader
        .profiler()
        .snapshot()
        .values()
        .map(|profile| profile.executions)
        .sum()
}

#[test]
fn duplicate_encoded_branches_share_actual_execution_and_source_reads() {
    for workers in [0, 3] {
        let (builder, reads, calls) = source(workers);
        let plan = graph(builder, false, false);
        let mut unshared = compile(plan.clone(), false);
        let reference = drain(&mut unshared);
        assert_eq!(reads.load(Ordering::Relaxed), 8);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        reads.store(0, Ordering::Relaxed);
        calls.store(0, Ordering::Relaxed);
        let mut shared = compile(plan, true);
        let actual = drain(&mut shared);
        assert_same(&actual, &reference);
        assert_eq!(reads.load(Ordering::Relaxed), 8);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        // Decode, Resize and Crop are each eliminated from one branch and
        // therefore each execute two fewer times (once per input morsel).
        assert_eq!(executions(&unshared) - executions(&shared), 6);
        assert_eq!(
            shared
                .physical_explain()
                .unwrap()
                .matches("Decode lane=")
                .count(),
            1
        );
        assert_eq!(
            unshared
                .physical_explain()
                .unwrap()
                .matches("Decode lane=")
                .count(),
            2
        );
    }
}

#[test]
fn differing_parameters_preserve_both_branch_values_and_port_order() {
    for workers in [0, 3] {
        let (builder, _, _) = source(workers);
        let plan = graph(builder, true, false);
        let mut unshared = compile(plan.clone(), false);
        let reference = drain(&mut unshared);
        let mut shared = compile(plan, true);
        let actual = drain(&mut shared);
        assert_same(&actual, &reference);
        // Only Decode is common; Resize and Crop parameter differences survive.
        assert_eq!(executions(&unshared) - executions(&shared), 2);
        assert!(actual.iter().any(|batch| {
            batch
                .images
                .to_vec::<u8>()
                .unwrap()
                .chunks_exact(6)
                .any(|pixel| pixel[..3] != pixel[3..])
        }));
    }
}

#[test]
fn equal_random_parameters_keep_independent_streams_after_common_prefix_sharing() {
    for workers in [0, 3] {
        let (builder, _, _) = source(workers);
        let plan = graph(builder, false, true);
        let mut unshared = compile(plan.clone(), false);
        let reference = drain(&mut unshared);
        let mut shared = compile(plan, true);
        let actual = drain(&mut shared);
        assert_same(&actual, &reference);
        // Decode and Resize share, but the two random operators still execute.
        assert_eq!(executions(&unshared) - executions(&shared), 4);
        assert!(actual.iter().any(|batch| {
            batch
                .images
                .to_vec::<u8>()
                .unwrap()
                .chunks_exact(6)
                .any(|pixel| pixel[..3] != pixel[3..])
        }));
    }
}

#[test]
fn cse_keeps_crop_normalize_order_and_invalid_reverse_order() {
    for workers in [0, 3] {
        let (builder, _, _) = source(workers);
        let plan = builder
            .decode_image()
            .crop(1, 1, 4, 4)
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .batch(4, false)
            .to_logical_plan();
        let mut reference = compile(plan.clone(), false);
        let mut actual = compile(plan, true);
        let reference = drain(&mut reference);
        let actual = drain(&mut actual);
        for (expected, got) in reference.iter().zip(&actual) {
            assert_eq!(
                expected.images.to_vec::<f32>().unwrap(),
                got.images.to_vec::<f32>().unwrap()
            );
            assert_eq!(
                expected.labels.to_vec::<i64>().unwrap(),
                got.labels.to_vec::<i64>().unwrap()
            );
        }
        let (builder, _, _) = source(workers);
        let reversed = builder
            .decode_image()
            .normalize(vec![0.5; 3], vec![0.5; 3])
            .crop(1, 1, 4, 4)
            .batch(4, false)
            .to_logical_plan();
        for cse in [false, true] {
            let error = ImagePipeline::compile_logical_plan_with_options(
                reversed.clone(),
                0,
                ImageOptimizationOptions {
                    common_subplan_elimination: cse,
                    ..ImageOptimizationOptions::default()
                },
            )
            .err()
            .unwrap()
            .to_string();
            assert!(error.to_ascii_lowercase().contains("crop"), "{error}");
        }
    }
}

#[test]
fn layout_fusion_retains_shared_normalize_without_duplicating_work() {
    use crate::sample::image::ImageAxisOrder;
    use crate::transforms::{LayoutConfig, NormalizeConfig};

    for workers in [0, 3] {
        let (builder, _, _) = source(workers);
        let mut plan = builder.to_logical_plan();
        let source = plan
            .nodes()
            .find(|(_, node)| node.kind() == NodeKind::Source)
            .unwrap()
            .0;
        let branches = [false, true].map(|second_normalize| {
            let decoded = op(&mut plan, source, ImageOp::Decode(DecodeImageConfig));
            let first = op(
                &mut plan,
                decoded,
                ImageOp::Normalize(NormalizeConfig::new(vec![0.5; 3], vec![0.5; 3])),
            );
            let output = if second_normalize {
                op(
                    &mut plan,
                    first,
                    ImageOp::Normalize(NormalizeConfig::new(vec![0.1; 3], vec![0.7; 3])),
                )
            } else {
                first
            };
            op(
                &mut plan,
                output,
                ImageOp::Layout(LayoutConfig::new(ImageAxisOrder::Chw)),
            )
        });
        let joined = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            branches,
            Some(Arc::new(ImageConcat { axis: 0 })),
        ));
        let batch = plan.add_node(LogicalNode::new(
            NodeKind::Batch,
            [joined],
            Some(Arc::new(BatchConfig::new(4, false))),
        ));
        let sink = plan.add_node(LogicalNode::new(NodeKind::Sink, [batch], None));
        plan.set_root(sink).unwrap();
        let mut optimized = plan.clone();
        let context = super::optimizer::optimize_vision_plan(&mut optimized, workers).unwrap();
        assert!(!optimized.nodes().any(|(_, node)| {
            node.payload_as::<super::logical::FusionGroupPayload>()
                .is_some()
        }));
        let normalize_nodes = optimized
            .nodes()
            .filter_map(|(id, node)| match node.payload_as::<ImageOp>() {
                Some(ImageOp::Normalize(config)) => Some((id, config)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(normalize_nodes.len(), 2);
        let shared = normalize_nodes
            .iter()
            .find(|(_, config)| config.mean == vec![0.5; 3])
            .unwrap()
            .0;
        assert_eq!(optimized.children(shared).unwrap().len(), 2);
        assert!(context.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "fusion.illegal"
                && diagnostic
                    .message
                    .contains("retains shared Normalize output")
        }));
        let mut reference = compile(plan.clone(), false);
        let mut actual = compile(plan, true);
        let expected = drain(&mut reference);
        let actual = drain(&mut actual);
        assert_eq!(expected.len(), actual.len());
        for (expected, actual) in expected.iter().zip(&actual) {
            assert_eq!(expected.images.dims(), actual.images.dims());
            assert_eq!(
                expected.images.to_vec::<f32>().unwrap(),
                actual.images.to_vec::<f32>().unwrap()
            );
            assert_eq!(
                expected.labels.to_vec::<i64>().unwrap(),
                actual.labels.to_vec::<i64>().unwrap()
            );
        }
    }
}
