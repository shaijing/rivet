use super::op::{BatchConfig, ImageOp, IndexOp};
use super::{ImageConcat, ImagePipeline};
use crate::sample::image::{EncodedImageSample, ImageBatch};
use arrow_buffer::Buffer;
use rivet_data::dataset::Dataset;
use rivet_plan::{LogicalNode, LogicalPlan, NodeKind};
use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Images {
    bytes: Vec<Buffer>,
    reads: Arc<AtomicUsize>,
}

impl Dataset for Images {
    type Item = EncodedImageSample;

    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn get_many(&self, indices: &[usize]) -> rivet_data::DataResult<Vec<Self::Item>> {
        self.reads.fetch_add(indices.len(), Ordering::Relaxed);
        Ok(indices
            .iter()
            .map(|&index| EncodedImageSample {
                image: self.bytes[index].clone(),
                label: index as i64,
            })
            .collect())
    }
}

fn pipeline(workers: usize) -> (ImagePipeline, Arc<AtomicUsize>) {
    let reads = Arc::new(AtomicUsize::new(0));
    let bytes = (0..8)
        .map(|sample| {
            let image = image::RgbImage::from_fn(6, 6, |x, y| {
                image::Rgb([
                    20 + sample * 9 + x as u8,
                    40 + y as u8 * 7,
                    100 + x as u8 * 3 + y as u8,
                ])
            });
            let mut output = Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(image)
                .write_to(&mut output, image::ImageFormat::Png)
                .unwrap();
            Buffer::from(output.into_inner())
        })
        .collect();
    (
        ImagePipeline::new(Arc::new(Images {
            bytes,
            reads: reads.clone(),
        }))
        .seed(91)
        .workers(workers),
        reads,
    )
}

fn names(plan: &LogicalPlan) -> Vec<&'static str> {
    plan.topological_order()
        .unwrap()
        .into_iter()
        .map(|id| {
            let node = plan.node(id).unwrap();
            if let Some(index) = node.payload_as::<IndexOp>() {
                match index {
                    IndexOp::Take { .. } => "Take",
                    IndexOp::Skip { .. } => "Skip",
                    IndexOp::Shuffle { .. } => "Shuffle",
                }
            } else if let Some(op) = node.payload_as::<ImageOp>() {
                op.name()
            } else {
                match node.kind() {
                    NodeKind::Source => "Source",
                    NodeKind::Batch => "Batch",
                    NodeKind::Sink => "Sink",
                    _ => "Other",
                }
            }
        })
        .collect()
}

fn first(mut loader: crate::runtime::ImageDataLoader) -> ImageBatch {
    let batch = loader.next_batch().unwrap().unwrap();
    assert!(loader.next_batch().unwrap().is_none());
    batch
}

#[test]
fn declared_index_positions_survive_ir_round_trip_before_real_rewrite() {
    let (builder, _) = pipeline(0);
    let mut plan = builder
        .decode_image()
        .random_crop(4, 4, 4)
        .take(4)
        .shuffle(11)
        .batch(4, false)
        .to_logical_plan();
    assert_eq!(
        names(&plan),
        [
            "Source",
            "Decode",
            "RandomCrop",
            "Take",
            "Shuffle",
            "Batch",
            "Sink"
        ]
    );
    let restored = ImagePipeline::from_logical_plan(&plan).unwrap();
    assert_eq!(names(&restored.to_logical_plan()), names(&plan));
    let random = plan
        .nodes()
        .find(|(_, node)| {
            node.payload_as::<ImageOp>()
                .is_some_and(|op| op.name() == "RandomCrop")
        })
        .unwrap()
        .1
        .semantic_identity();
    assert!(random.is_some());
    let context = super::optimizer::optimize_vision_plan(&mut plan, 0).unwrap();
    assert_eq!(
        names(&plan),
        [
            "Source",
            "Take",
            "Shuffle",
            "Decode",
            "RandomCrop",
            "Batch",
            "Sink"
        ]
    );
    assert!(
        context
            .diagnostics
            .iter()
            .any(|d| d.code == "rewrite.index-source-pushdown")
    );
    let moved_random = plan
        .nodes()
        .find(|(_, node)| {
            node.payload_as::<ImageOp>()
                .is_some_and(|op| op.name() == "RandomCrop")
        })
        .unwrap()
        .1
        .semantic_identity();
    assert_eq!(random, moved_random);
}

#[test]
fn pushdown_matches_full_transform_reference_and_only_reads_selected_samples() {
    for workers in [0, 3] {
        let (reference, _) = pipeline(workers);
        let reference = first(
            reference
                .decode_image()
                .random_crop(4, 4, 4)
                .batch(8, false)
                .compile_unoptimized_for_test(0)
                .unwrap(),
        );
        let reference_pixels = reference.images.to_vec::<u8>().unwrap();
        let (selected, reads) = pipeline(workers);
        let selected = first(
            selected
                .decode_image()
                .random_crop(4, 4, 4)
                .take(4)
                .shuffle(11)
                .batch(4, false)
                .compile()
                .unwrap(),
        );
        assert_eq!(reads.load(Ordering::Relaxed), 4);
        let pixels = selected.images.to_vec::<u8>().unwrap();
        let labels = selected.labels.to_vec::<i64>().unwrap();
        let mut sorted = labels.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, [0, 1, 2, 3]);
        for (position, label) in labels.iter().enumerate() {
            assert_eq!(
                &pixels[position * 48..(position + 1) * 48],
                &reference_pixels[*label as usize * 48..(*label as usize + 1) * 48]
            );
        }
    }
}

#[test]
fn padding_normalize_order_and_values_are_preserved() {
    for workers in [0, 3] {
        let (reference, _) = pipeline(workers);
        let reference = first(
            reference
                .decode_image()
                .random_crop(10, 10, 2)
                .normalize(vec![0.5; 3], vec![0.5; 3])
                .batch(8, false)
                .compile_unoptimized_for_test(0)
                .unwrap(),
        );
        let reference_pixels = reference.images.to_vec::<f32>().unwrap();
        let (selected, _) = pipeline(workers);
        let selected = first(
            selected
                .decode_image()
                .random_crop(10, 10, 2)
                .normalize(vec![0.5; 3], vec![0.5; 3])
                .take(4)
                .shuffle(11)
                .batch(4, false)
                .compile()
                .unwrap(),
        );
        let pixels = selected.images.to_vec::<f32>().unwrap();
        let labels = selected.labels.to_vec::<i64>().unwrap();
        for (position, label) in labels.iter().enumerate() {
            assert_eq!(&pixels[position * 300..position * 300 + 3], &[-1.0; 3]);
            assert_eq!(
                &pixels[position * 300..(position + 1) * 300],
                &reference_pixels[*label as usize * 300..(*label as usize + 1) * 300]
            );
        }
    }
}

#[test]
fn shuffle_before_take_selects_from_full_permutation() {
    for workers in [0, 3] {
        let (builder, _) = pipeline(workers);
        let full = first(
            builder
                .decode_image()
                .shuffle(11)
                .batch(8, false)
                .compile()
                .unwrap(),
        );
        let (builder, _) = pipeline(workers);
        let selected = first(
            builder
                .decode_image()
                .shuffle(11)
                .take(4)
                .batch(4, false)
                .compile()
                .unwrap(),
        );
        assert_eq!(
            selected.labels.to_vec::<i64>().unwrap(),
            full.labels.to_vec::<i64>().unwrap()[..4]
        );
        assert_eq!(
            selected.images.to_vec::<u8>().unwrap(),
            full.images.to_vec::<u8>().unwrap()[..4 * 108]
        );
    }
}

#[test]
fn branch_local_take_cannot_mutate_a_shared_transform() {
    let (builder, _) = pipeline(0);
    let mut plan = builder.decode_image().to_logical_plan();
    let decoded = plan
        .nodes()
        .find(|(_, node)| node.payload_as::<ImageOp>().is_some())
        .unwrap()
        .0;
    let shared = plan.add_node(LogicalNode::new(
        NodeKind::Op,
        [decoded],
        Some(Arc::new(ImageOp::invert())),
    ));
    let take = plan.add_node(LogicalNode::new(
        NodeKind::Index,
        [shared],
        Some(Arc::new(IndexOp::Take { count: 4 })),
    ));
    let join = plan.add_node(LogicalNode::new(
        NodeKind::Op,
        [take, shared],
        Some(Arc::new(ImageConcat { axis: 2 })),
    ));
    let batch = plan.add_node(LogicalNode::new(
        NodeKind::Batch,
        [join],
        Some(Arc::new(BatchConfig::new(4, false))),
    ));
    let sink = plan.add_node(LogicalNode::new(NodeKind::Sink, [batch], None));
    plan.set_root(sink).unwrap();
    let error = super::optimizer::optimize_vision_plan(&mut plan, 0)
        .err()
        .expect("branch-local take requires a separate execution capability");
    assert!(error.to_string().contains("index"), "{error}");
    let take = plan
        .nodes()
        .find(|(_, node)| node.kind() == NodeKind::Index)
        .unwrap()
        .1;
    assert_eq!(
        plan.node(take.inputs().get(0).unwrap())
            .unwrap()
            .payload_as::<ImageOp>()
            .unwrap()
            .name(),
        "Invert"
    );
    let error = ImagePipeline::compile_logical_plan(plan, 0)
        .err()
        .expect("branch-local take requires a separate execution capability");
    assert!(error.to_string().contains("index"), "{error}");
}

#[test]
fn index_after_explicit_batch_stays_at_batch_barrier() {
    let (builder, _) = pipeline(0);
    let plan = builder
        .decode_image()
        .batch(2, false)
        .take(2)
        .to_logical_plan();
    assert_eq!(names(&plan), ["Source", "Decode", "Batch", "Take", "Sink"]);
    let error = ImagePipeline::compile_logical_plan(plan, 0)
        .err()
        .expect("batch-level index execution is unsupported");
    assert!(error.to_string().contains("index"), "{error}");
}

#[test]
fn removing_an_earlier_random_op_does_not_renumber_surviving_streams() {
    for workers in [0, 3] {
        let (builder, _) = pipeline(workers);
        let original = builder
            .decode_image()
            .random_horizontal_flip(0.0)
            .random_horizontal_flip(0.5)
            .random_crop(4, 4, 1)
            .batch(8, false);
        let expected = first(original.clone().compile().unwrap());
        let mut edited = original.to_logical_plan();
        let removed = edited
            .topological_order()
            .unwrap()
            .into_iter()
            .find(|id| {
                edited
                    .node(*id)
                    .unwrap()
                    .payload_as::<ImageOp>()
                    .is_some_and(|op| op.name() == "RandomHorizontalFlip")
            })
            .unwrap();
        let input = edited.node(removed).unwrap().inputs().get(0).unwrap();
        edited.redirect_uses(removed, input).unwrap();
        edited.prune_unreachable().unwrap();
        let restored = ImagePipeline::from_logical_plan(&edited).unwrap();
        for actual in [
            first(ImagePipeline::compile_logical_plan(edited, 0).unwrap()),
            first(restored.compile().unwrap()),
        ] {
            assert_eq!(
                actual.images.to_vec::<u8>().unwrap(),
                expected.images.to_vec::<u8>().unwrap()
            );
            assert_eq!(
                actual.labels.to_vec::<i64>().unwrap(),
                expected.labels.to_vec::<i64>().unwrap()
            );
        }
    }
}

#[test]
fn skip_after_crop_preserves_source_sample_randomness() {
    for workers in [0, 3] {
        let (builder, _) = pipeline(workers);
        let expected = first(
            builder
                .decode_image()
                .random_crop(4, 4, 4)
                .batch(8, false)
                .compile_unoptimized_for_test(0)
                .unwrap(),
        );
        let (builder, reads) = pipeline(workers);
        let actual = first(
            builder
                .decode_image()
                .random_crop(4, 4, 4)
                .skip(2)
                .take(3)
                .batch(3, false)
                .compile()
                .unwrap(),
        );
        assert_eq!(
            actual.images.to_vec::<u8>().unwrap(),
            expected.images.to_vec::<u8>().unwrap()[2 * 48..5 * 48]
        );
        assert_eq!(actual.labels.to_vec::<i64>().unwrap(), [2, 3, 4]);
        assert_eq!(reads.load(Ordering::Relaxed), 3);
    }
}

#[test]
fn appending_a_random_op_after_pruning_allocates_an_unused_identity() {
    let (builder, _) = pipeline(0);
    let mut plan = builder
        .decode_image()
        .random_crop(6, 6, 0)
        .random_crop(4, 4, 1)
        .to_logical_plan();
    let random = plan
        .topological_order()
        .unwrap()
        .into_iter()
        .filter(|id| plan.node(*id).unwrap().semantic_identity().is_some())
        .collect::<Vec<_>>();
    let survivor_identity = plan.node(random[1]).unwrap().semantic_identity();
    let input = plan.node(random[0]).unwrap().inputs().get(0).unwrap();
    plan.redirect_uses(random[0], input).unwrap();
    plan.prune_unreachable().unwrap();
    let restored = ImagePipeline::from_logical_plan(&plan)
        .unwrap()
        .random_crop(3, 3, 1)
        .batch(8, false);
    let appended = restored.to_logical_plan();
    let identities = appended
        .topological_order()
        .unwrap()
        .into_iter()
        .filter_map(|id| appended.node(id).unwrap().semantic_identity())
        .collect::<Vec<_>>();
    assert_eq!(identities.len(), 2);
    assert_eq!(Some(identities[0]), survivor_identity);
    assert_ne!(identities[0], identities[1]);
    let batch = first(restored.compile().unwrap());
    assert_eq!(batch.images.dims(), [8, 3, 3, 3]);
}
