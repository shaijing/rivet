//! Execute optimized and unshared DAGs against the same encoded image source.

use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_buffer::Buffer;
use rivet_data::dataset::Dataset;
use rivet_plan::{LogicalNode, LogicalPlan, NodeId, NodeKind};

use crate::pipeline::op::{BatchConfig, ImageOp};
use crate::pipeline::{ImageConcat, ImageOptimizationOptions, ImagePipeline};
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
fn cse_preserves_identical_lambda_occurrences_and_random_keys() {
    use crate::pipeline::op::CallbackConfig;
    use std::sync::Mutex;

    let keys = Arc::new(Mutex::new(Vec::new()));
    let (builder, _, _) = source(3);
    let mut plan = graph(builder, false, false);
    let (join, original) = plan
        .nodes()
        .find(|(_, node)| node.payload_as::<ImageConcat>().is_some())
        .map(|(id, node)| (id, node.clone()))
        .unwrap();
    let callback = Arc::new({
        let keys = Arc::clone(&keys);
        move |_: &str, input: &rivet_core::Tensor, seed| {
            keys.lock().unwrap().push(seed);
            Ok(input.clone())
        }
    });
    let ports = original
        .inputs()
        .iter()
        .enumerate()
        .map(|(port, id)| {
            plan.add_node(
                LogicalNode::new(
                    NodeKind::Op,
                    [id],
                    Some(Arc::new(ImageOp::Callback(CallbackConfig {
                        probability: 1.0,
                        callback: callback.clone(),
                    }))),
                )
                .with_semantic_identity(181 + port as u64),
            )
        })
        .collect::<Vec<_>>();
    plan.replace_node(join, original.with_inputs(ports))
        .unwrap();
    let mut unshared = compile(plan.clone(), false);
    let mut shared = compile(plan, true);
    let reference = drain(&mut unshared);
    let mut unshared_keys = std::mem::take(&mut *keys.lock().unwrap());
    let actual = drain(&mut shared);
    let mut shared_keys = std::mem::take(&mut *keys.lock().unwrap());
    unshared_keys.sort_unstable();
    shared_keys.sort_unstable();
    assert_same(&actual, &reference);
    assert_eq!(shared_keys.len(), 16);
    assert_eq!(shared_keys, unshared_keys);
    shared_keys.dedup();
    assert_eq!(shared_keys.len(), 16);
}

#[test]
fn managed_dag_checks_decode_and_stack_before_their_kernel_calls() {
    // Decode allocates 432B. Resize releases that backing, so stack can
    // allocate 896B with the same reservation.
    fn resized_graph(builder: ImagePipeline) -> LogicalPlan {
        let mut plan = graph(builder, false, false);
        let crops = plan
            .nodes()
            .filter_map(|(id, node)| {
                matches!(node.payload_as::<ImageOp>(), Some(ImageOp::Crop(_)))
                    .then(|| (id, node.inputs().get(0).unwrap()))
            })
            .collect::<Vec<_>>();
        for (id, input) in crops {
            plan.replace_node(
                id,
                LogicalNode::new(
                    NodeKind::Op,
                    [input],
                    Some(Arc::new(ImageOp::Crop(CropConfig::new(0, 0, 6, 6)))),
                ),
            )
            .unwrap();
        }
        plan
    }

    for (budget, failing_op) in [
        (431, Some("Decode")),
        (895, Some("ImageStack")),
        (896, None),
    ] {
        let (builder, _, _) = source(3);
        let plan = resized_graph(builder.dag_managed_budget_bytes(budget));
        let mut loader = compile(plan, true);
        if let Some(failing_op) = failing_op {
            let error = loader.next_batch().unwrap_err().to_string();
            assert!(error.contains(failing_op), "{error}");
            assert!(error.contains("before allocation"), "{error}");
            let explain = loader.physical_explanation().unwrap();
            let node = explain
                .nodes
                .iter()
                .find(|node| node.operator.as_deref() == Some(failing_op))
                .unwrap();
            assert!(!loader.profiler().snapshot().contains_key(&node.id));
            assert!(loader.next_batch().unwrap().is_none());
        } else {
            let (builder, _, _) = source(3);
            let mut baseline = compile(resized_graph(builder), true);
            assert_same(&drain(&mut loader), &drain(&mut baseline));
        }
        let accounting = loader.managed_memory_snapshot().unwrap();
        assert_eq!(accounting.reserved_bytes, 0);
        assert_eq!(accounting.peak_reserved_bytes, budget);
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
        let context =
            crate::pipeline::planning::optimizer::optimize_vision_plan(&mut optimized, workers)
                .unwrap();
        assert!(!optimized.nodes().any(|(_, node)| {
            node.payload_as::<crate::pipeline::planning::logical::FusionGroupPayload>()
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

#[test]
#[ignore = "release DAG workload benchmark"]
fn dag_workload_matrix() {
    use std::time::{Duration, Instant};

    let rows = 512;
    let images = (0..rows)
        .map(|i| {
            let pixels = image::RgbImage::from_fn(32 + i % 17, 32 + i % 13, |x, y| {
                image::Rgb([(i + x) as u8, (i + y * 7) as u8, (x * y) as u8])
            });
            let mut bytes = Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(pixels)
                .write_to(&mut bytes, image::ImageFormat::Jpeg)
                .unwrap();
            Buffer::from(bytes.into_inner())
        })
        .collect::<Vec<_>>();

    let prefetch = std::env::var("RIVET_DAG_BENCH_PREFETCH")
        .map_or(6, |value| value.parse::<usize>().unwrap());
    let mut shared_output = None;
    for workload in ["shared", "unshared", "views", "managed"] {
        let mut reference = None;
        for profile in [false, true] {
            let source = Arc::new(Images {
                bytes: images.clone(),
                reads: Arc::new(AtomicUsize::new(0)),
                calls: Arc::new(AtomicUsize::new(0)),
            });
            let builder = ImagePipeline::new(Arc::clone(&source))
                .workers(4)
                .prefetch_batches(prefetch)
                .profiling(profile);
            let mut loader = match workload {
                "views" => builder
                    .decode_image()
                    .resize(12, 12)
                    .five_crop(8, 8)
                    .batch(4, false)
                    .compile()
                    .unwrap(),
                "managed" => compile(
                    graph(builder.dag_managed_budget_bytes(32 * 1024), false, false),
                    true,
                ),
                _ => compile(graph(builder, false, false), workload == "shared"),
            };
            let expected_views = if workload == "views" { 5 } else { 1 };
            let mut output = Vec::new();
            let mut latencies = Vec::new();
            let mut labels = Vec::new();
            loop {
                let start = Instant::now();
                let Some(batch) = loader.next_batch().unwrap() else {
                    break;
                };
                latencies.push(start.elapsed());
                labels.extend(batch.labels.to_vec::<i64>().unwrap());
                output.push(batch);
                if workload == "managed" {
                    std::thread::sleep(Duration::from_micros(25));
                }
            }
            assert_eq!(
                labels,
                (0..i64::from(rows))
                    .flat_map(|id| std::iter::repeat_n(id, expected_views))
                    .collect::<Vec<_>>()
            );
            assert_eq!(source.reads.load(Ordering::Relaxed), rows as usize);
            assert_eq!(source.calls.load(Ordering::Relaxed), rows as usize / 4);
            let first = latencies[0];
            latencies.sort_unstable();
            let pctl = |p: usize| latencies[(latencies.len() * p).div_ceil(100) - 1];
            let profiler = loader.profiler();
            let explanation = loader.physical_explanation().unwrap();
            let by_node = profiler.snapshot();
            if profile {
                let decodes = explanation
                    .nodes
                    .iter()
                    .filter(|node| node.operator.as_deref() == Some("Decode"));
                let decode_calls: u64 = decodes.map(|node| by_node[&node.id].executions).sum();
                let expected = if workload == "unshared" { 2 } else { 1 };
                assert_eq!(decode_calls, u64::from(rows) / 4 * expected);
                assert!(
                    profiler
                        .memory_snapshot()
                        .unwrap()
                        .peak_known_backing_bytes_per_invocation
                        > 0
                );
            }
            if let Some(memory) = loader.managed_memory_snapshot() {
                assert_eq!(memory.reserved_bytes, 0);
                assert!(memory.peak_reserved_bytes <= memory.limit_bytes);
            }
            match workload {
                "shared" if shared_output.is_none() => shared_output = Some(output.clone()),
                "shared" | "unshared" | "managed" => {
                    assert_same(&output, shared_output.as_deref().unwrap());
                }
                _ => {}
            }
            if let Some(baseline) = reference.as_deref() {
                assert_same(&output, baseline);
            } else {
                reference = Some(output);
            }
            let nodes = explanation
                .nodes
                .iter()
                .filter_map(|node| {
                    let profile = by_node.get(&node.id)?;
                    Some((
                        node.operator.as_deref(),
                        profile.executions,
                        profile.output_bytes,
                    ))
                })
                .collect::<Vec<_>>();
            println!(
                "{workload} profile={profile} first={first:?} p95={:?} p99={:?} nodes={nodes:?} backing={:?} reservation={:?} scheduler={:?}",
                pctl(95),
                pctl(99),
                profiler.memory_snapshot(),
                loader.managed_memory_snapshot(),
                profiler.dag_scheduler_snapshot()
            );
        }
    }
}
