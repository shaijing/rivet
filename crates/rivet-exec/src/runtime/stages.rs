use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crate::physical::{ExecutionLane, PhysNodeId, PhysicalProfiler, TransferKind};

use super::pipeline::{PhysicalPipelineAdapter, PipelineError};
use super::stage_queue::{BoundedStageQueue, StageMessage, StageQueueLimits};
use super::{RuntimeResult, WorkItem, WorkerPool, runtime_error};

enum SourcePayload<S, B> {
    Samples(Vec<(usize, S)>),
    Batch { indices: Vec<usize>, batch: B },
}

struct IndexedBatch<B> {
    indices: Vec<usize>,
    batch: B,
}

struct StageChannels<P: PhysicalPipelineAdapter> {
    requests: BoundedStageQueue<Vec<usize>>,
    source: BoundedStageQueue<StageResult<SourcePayload<P::Sample, P::Batch>, P::Error>>,
    decoded: BoundedStageQueue<StageResult<SourcePayload<P::Sample, P::Batch>, P::Error>>,
    transfer: BoundedStageQueue<StageResult<IndexedBatch<P::Batch>, P::Error>>,
    device: BoundedStageQueue<StageResult<IndexedBatch<P::Batch>, P::Error>>,
    sink: BoundedStageQueue<StageResult<P::Batch, P::Error>>,
}

#[derive(Clone, Copy)]
enum CpuStageOutput {
    Transfer,
    Device,
    Sink,
}

#[derive(Clone, Copy)]
struct StageTopology {
    decode: bool,
    transfer: bool,
    device: bool,
}

type StageResult<T, E> = Result<T, PipelineError<E>>;

impl<P: PhysicalPipelineAdapter> StageChannels<P> {
    fn cancel_source(&self) {
        self.requests.cancel();
        self.source.cancel();
    }

    fn cancel_cpu_inputs(&self) {
        self.cancel_source();
        self.decoded.cancel();
    }

    fn cancel_upstream(&self) {
        self.cancel_cpu_inputs();
        self.transfer.cancel();
        self.device.cancel();
    }

    fn cancel_all(&self) {
        self.cancel_upstream();
        self.sink.cancel();
    }
}

#[derive(Clone)]
pub(crate) struct StageNodes {
    pub source: PhysNodeId,
    pub decode: PhysNodeId,
    pub sample: Option<PhysNodeId>,
    pub batch: PhysNodeId,
    pub batch_kernel: PhysNodeId,
    pub sink: PhysNodeId,
    pub device_batch_kernel: bool,
    pub device_target: ExecutionLane,
    pub transfer_nodes: Vec<(PhysNodeId, TransferKind, ExecutionLane)>,
}

/// Persistent, bounded runtime stages connected by sequence-preserving queues.
pub(crate) struct PersistentStageGraph<P: PhysicalPipelineAdapter> {
    channels: Arc<StageChannels<P>>,
    handles: Vec<JoinHandle<()>>,
    limits: StageQueueLimits,
    topology: StageTopology,
}

impl<P: PhysicalPipelineAdapter> PersistentStageGraph<P> {
    pub(crate) fn new(
        adapter: Arc<P>,
        workers: usize,
        worker_capacity: usize,
        limits: StageQueueLimits,
        nodes: StageNodes,
        profiler: PhysicalProfiler,
    ) -> RuntimeResult<Self> {
        let limits = limits.validate()?;
        let topology = StageTopology {
            decode: nodes.decode != nodes.source,
            transfer: !nodes.transfer_nodes.is_empty(),
            device: nodes.device_batch_kernel,
        };
        let cpu_output = if topology.transfer {
            CpuStageOutput::Transfer
        } else if topology.device {
            CpuStageOutput::Device
        } else {
            CpuStageOutput::Sink
        };
        let channels = Arc::new(StageChannels {
            requests: BoundedStageQueue::new(limits)?,
            source: BoundedStageQueue::new(limits)?,
            decoded: BoundedStageQueue::new(limits)?,
            transfer: BoundedStageQueue::new(limits)?,
            device: BoundedStageQueue::new(limits)?,
            sink: BoundedStageQueue::new(limits)?,
        });
        if !topology.decode {
            channels.source.close();
        }
        if !topology.transfer {
            channels.transfer.close();
        }
        if !topology.device {
            channels.device.close();
        }
        let mut stages = Self {
            channels: Arc::clone(&channels),
            handles: Vec::with_capacity(5),
            limits,
            topology,
        };

        let source_adapter = Arc::clone(&adapter);
        let source_channels = Arc::clone(&channels);
        let source_profiler = profiler.clone();
        let source_nodes = nodes.clone();
        stages.spawn("rivet-source", move || {
            run_source_stage(
                source_adapter,
                source_channels,
                source_nodes,
                source_profiler,
                topology.decode,
            )
        })?;

        if topology.decode {
            let decode_adapter = Arc::clone(&adapter);
            let decode_channels = Arc::clone(&channels);
            let decode_profiler = profiler.clone();
            let decode_nodes = nodes.clone();
            stages.spawn("rivet-decode", move || {
                run_decode_stage(
                    decode_adapter,
                    decode_channels,
                    decode_nodes,
                    decode_profiler,
                )
            })?;
        }

        let worker_pool = if workers > 0 && !adapter.is_batch_native() {
            let worker_adapter = Arc::clone(&adapter);
            let worker_profiler = profiler.clone();
            let worker_node = nodes.sample.unwrap_or(nodes.decode);
            Some(WorkerPool::new(
                workers,
                worker_capacity.max(1),
                move |sample, index| {
                    let started = Instant::now();
                    let input_bytes = worker_adapter.sample_bytes(&sample);
                    let result = worker_adapter.process_sample(sample, index);
                    let output_bytes = result
                        .as_ref()
                        .map(|output| worker_adapter.output_bytes(output))
                        .unwrap_or(0);
                    worker_profiler.record(
                        worker_node,
                        started.elapsed(),
                        input_bytes,
                        output_bytes,
                    );
                    result
                },
                {
                    let worker_adapter = Arc::clone(&adapter);
                    move |worker, index| worker_adapter.worker_panic_error(worker, index)
                },
            )?)
        } else {
            None
        };
        let cpu_channels = Arc::clone(&channels);
        let cpu_adapter = Arc::clone(&adapter);
        let cpu_profiler = profiler.clone();
        let cpu_nodes = nodes.clone();
        stages.spawn("rivet-cpu-transform", move || {
            run_cpu_stage(
                cpu_adapter,
                cpu_channels,
                cpu_nodes,
                cpu_profiler,
                worker_pool,
                cpu_output,
            )
        })?;

        if topology.transfer {
            let transfer_channels = Arc::clone(&channels);
            let transfer_nodes = nodes.clone();
            let transfer_adapter = Arc::clone(&adapter);
            let transfer_profiler = profiler.clone();
            let transfer_device = topology.device;
            stages.spawn("rivet-transfer", move || {
                run_transfer_stage(
                    transfer_adapter,
                    transfer_channels,
                    transfer_nodes,
                    transfer_profiler,
                    transfer_device,
                )
            })?;
        }

        if topology.device {
            let device_channels = Arc::clone(&channels);
            let device_nodes = nodes;
            stages.spawn("rivet-device", move || {
                run_device_stage(adapter, device_channels, device_nodes, profiler)
            })?;
        }

        Ok(stages)
    }

    fn spawn(&mut self, name: &str, task: impl FnOnce() + Send + 'static) -> RuntimeResult<()> {
        let channels = Arc::clone(&self.channels);
        let handle = thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                if catch_unwind(AssertUnwindSafe(task)).is_err() {
                    // A stage panic must wake every blocked producer/consumer;
                    // the caller then receives a terminal graph error.
                    channels.cancel_all();
                }
            })
            .map_err(|error| runtime_error(format!("failed to spawn {name} stage: {error}")))?;
        self.handles.push(handle);
        Ok(())
    }

    pub(crate) fn submit(
        &self,
        sequence_id: u64,
        indices: Vec<usize>,
    ) -> RuntimeResult<std::time::Duration> {
        let bytes = indices.len().saturating_mul(std::mem::size_of::<usize>());
        self.channels.requests.send(sequence_id, indices, bytes)
    }

    pub(crate) fn close_requests(&self) {
        self.channels.requests.close();
    }

    pub(crate) fn cancel_upstream(&self) {
        self.channels.cancel_upstream();
    }

    pub(crate) fn recv_output(
        &self,
    ) -> RuntimeResult<(
        Option<StageMessage<StageResult<P::Batch, P::Error>>>,
        std::time::Duration,
    )> {
        self.channels.sink.recv()
    }

    pub(crate) fn queue_explain(&self) -> String {
        let mut rows = vec![("Sampler->Source", self.channels.requests.snapshot())];
        if self.topology.decode {
            rows.push(("Source->Decode", self.channels.source.snapshot()));
            rows.push(("Decode->CPU", self.channels.decoded.snapshot()));
        } else {
            rows.push(("Source->CPU", self.channels.decoded.snapshot()));
        }
        match (self.topology.transfer, self.topology.device) {
            (true, true) => {
                rows.push(("CPU->Transfer", self.channels.transfer.snapshot()));
                rows.push(("Transfer->Device", self.channels.device.snapshot()));
                rows.push(("Device->Sink", self.channels.sink.snapshot()));
            }
            (true, false) => {
                rows.push(("CPU->Transfer", self.channels.transfer.snapshot()));
                rows.push(("Transfer->Sink", self.channels.sink.snapshot()));
            }
            (false, true) => {
                rows.push(("CPU->Device", self.channels.device.snapshot()));
                rows.push(("Device->Sink", self.channels.sink.snapshot()));
            }
            (false, false) => rows.push(("CPU->Sink", self.channels.sink.snapshot())),
        }
        let mut output = String::from("Runtime stage queues:\n");
        for (name, stats) in rows {
            output.push_str(&format!(
                "  {name}: max_items={} max_bytes={} queued_items={} queued_bytes={} peak_items={} peak_bytes={} sends={} receives={} producer_wait={:?} consumer_wait={:?}\n",
                self.limits.max_items,
                self.limits.max_bytes,
                stats.queued_items,
                stats.queued_bytes,
                stats.peak_items,
                stats.peak_bytes,
                stats.sends,
                stats.receives,
                stats.producer_wait,
                stats.consumer_wait,
            ));
        }
        output
    }
}

impl<P: PhysicalPipelineAdapter> Drop for PersistentStageGraph<P> {
    fn drop(&mut self) {
        self.channels.cancel_all();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn run_source_stage<P: PhysicalPipelineAdapter>(
    adapter: Arc<P>,
    channels: Arc<StageChannels<P>>,
    nodes: StageNodes,
    profiler: PhysicalProfiler,
    decode_stage: bool,
) {
    loop {
        let (request, waited) = match channels.requests.recv() {
            Ok(result) => result,
            Err(_) => break,
        };
        profiler.record_wait(nodes.source, waited);
        let Some(request) = request else { break };
        let sequence_id = request.sequence_id;
        let indices = request.value;
        let input_bytes = indices.len().saturating_mul(std::mem::size_of::<usize>());
        let started = Instant::now();
        let fetched = catch_unwind(AssertUnwindSafe(|| {
            if adapter.is_batch_native() {
                adapter
                    .fetch_batch(&indices)
                    .map_err(PipelineError::Domain)?
                    .map(|batch| SourcePayload::Batch {
                        indices: indices.clone(),
                        batch,
                    })
                    .ok_or_else(|| {
                        PipelineError::Runtime(runtime_error(
                            "batch-native source capability disappeared",
                        ))
                    })
            } else {
                let samples = adapter
                    .fetch_samples(&indices)
                    .map_err(PipelineError::Domain)?;
                if samples.len() != indices.len() {
                    return Err(PipelineError::Runtime(runtime_error(format!(
                        "source returned {} samples for {} indices",
                        samples.len(),
                        indices.len()
                    ))));
                }
                Ok(SourcePayload::Samples(
                    indices.into_iter().zip(samples).collect(),
                ))
            }
        }))
        .unwrap_or_else(|_| {
            Err(PipelineError::Runtime(runtime_error(format!(
                "source stage panicked while fetching sequence {sequence_id}"
            ))))
        });
        let output_bytes = fetched
            .as_ref()
            .map(|payload| source_payload_bytes(payload, &*adapter))
            .unwrap_or(0);
        let output = if decode_stage {
            &channels.source
        } else {
            &channels.decoded
        };
        let fetched = if output_bytes > output.limits().max_bytes {
            Err(PipelineError::Runtime(runtime_error(format!(
                "source batch is {output_bytes} bytes, exceeding stage queue max_bytes {}",
                output.limits().max_bytes
            ))))
        } else {
            fetched
        };
        let output_bytes = if fetched.is_ok() { output_bytes } else { 0 };
        profiler.record(nodes.source, started.elapsed(), input_bytes, output_bytes);
        if fetched.is_err() {
            channels.requests.cancel();
        }
        let waited = match output.send(sequence_id, fetched, output_bytes) {
            Ok(waited) => waited,
            Err(_) => break,
        };
        profiler.record_wait(nodes.source, waited);
    }
    if decode_stage {
        channels.source.close();
    } else {
        channels.decoded.close();
    }
}

fn run_decode_stage<P: PhysicalPipelineAdapter>(
    adapter: Arc<P>,
    channels: Arc<StageChannels<P>>,
    nodes: StageNodes,
    profiler: PhysicalProfiler,
) {
    loop {
        let (message, waited) = match channels.source.recv() {
            Ok(result) => result,
            Err(_) => break,
        };
        profiler.record_wait(nodes.decode, waited);
        let Some(message) = message else { break };
        let sequence_id = message.sequence_id;
        let decoded = match message.value {
            Err(error) => Err(error),
            Ok(SourcePayload::Batch { indices, batch }) => {
                Ok(SourcePayload::Batch { indices, batch })
            }
            Ok(SourcePayload::Samples(samples)) => {
                let mut decoded = Vec::with_capacity(samples.len());
                let mut failure = None;
                for (index, sample) in samples {
                    let started = Instant::now();
                    let input_bytes = adapter.sample_bytes(&sample);
                    let result = match catch_unwind(AssertUnwindSafe(|| {
                        adapter.decode_sample(sample, index)
                    })) {
                        Ok(Ok(sample)) => Ok(sample),
                        Ok(Err(error)) => Err(PipelineError::Domain(error)),
                        Err(_) => Err(PipelineError::Runtime(runtime_error(format!(
                            "decode stage panicked at sample {index}"
                        )))),
                    };
                    let output_bytes = result
                        .as_ref()
                        .map(|sample| adapter.sample_bytes(sample))
                        .unwrap_or(0);
                    profiler.record(nodes.decode, started.elapsed(), input_bytes, output_bytes);
                    match result {
                        Ok(sample) => decoded.push((index, sample)),
                        Err(error) => {
                            failure = Some(error);
                            break;
                        }
                    }
                }
                match failure {
                    Some(error) => Err(error),
                    None => Ok(SourcePayload::Samples(decoded)),
                }
            }
        };
        let output_bytes = decoded
            .as_ref()
            .map(|payload| source_payload_bytes(payload, &*adapter))
            .unwrap_or(0);
        let decoded = if output_bytes > channels.decoded.limits().max_bytes {
            Err(PipelineError::Runtime(runtime_error(format!(
                "decoded batch is {output_bytes} bytes, exceeding stage queue max_bytes {}",
                channels.decoded.limits().max_bytes
            ))))
        } else {
            decoded
        };
        let output_bytes = if decoded.is_ok() { output_bytes } else { 0 };
        if decoded.is_err() {
            channels.cancel_source();
        }
        let waited = match channels.decoded.send(sequence_id, decoded, output_bytes) {
            Ok(waited) => waited,
            Err(_) => break,
        };
        profiler.record_wait(nodes.decode, waited);
    }
    channels.decoded.close();
}

fn run_cpu_stage<P: PhysicalPipelineAdapter>(
    adapter: Arc<P>,
    channels: Arc<StageChannels<P>>,
    nodes: StageNodes,
    profiler: PhysicalProfiler,
    worker_pool: Option<WorkerPool<P::Sample, P::Output, P::Error>>,
    output: CpuStageOutput,
) {
    loop {
        let (message, waited) = match channels.decoded.recv() {
            Ok(result) => result,
            Err(_) => break,
        };
        profiler.record_wait(nodes.sample.unwrap_or(nodes.batch), waited);
        let Some(message) = message else { break };
        let sequence_id = message.sequence_id;
        let result = match message.value {
            Err(error) => Err(error),
            Ok(SourcePayload::Batch { indices, batch }) => {
                execute_cpu_batch(&adapter, batch, indices, &nodes, &profiler)
            }
            Ok(SourcePayload::Samples(samples)) => process_batch_samples(
                &adapter,
                samples,
                sequence_id,
                worker_pool.as_ref(),
                &profiler,
                nodes.sample.unwrap_or(nodes.decode),
                nodes.batch,
            )
            .and_then(|(batch, indices)| {
                execute_cpu_batch(&adapter, batch, indices, &nodes, &profiler)
            }),
        };
        let waited = match output {
            CpuStageOutput::Sink => {
                let result = result.map(|indexed| indexed.batch);
                let output_bytes = result
                    .as_ref()
                    .map(|batch| adapter.batch_bytes(batch))
                    .unwrap_or(0);
                let result = if output_bytes > channels.sink.limits().max_bytes {
                    Err(PipelineError::Runtime(runtime_error(format!(
                        "CPU output batch is {output_bytes} bytes, exceeding stage queue max_bytes {}",
                        channels.sink.limits().max_bytes
                    ))))
                } else {
                    result
                };
                if result.is_err() {
                    channels.cancel_cpu_inputs();
                }
                let bytes = if result.is_ok() { output_bytes } else { 0 };
                channels.sink.send(sequence_id, result, bytes)
            }
            CpuStageOutput::Transfer | CpuStageOutput::Device => {
                let output_bytes = result
                    .as_ref()
                    .map(|indexed| {
                        adapter.batch_bytes(&indexed.batch).saturating_add(
                            indexed
                                .indices
                                .len()
                                .saturating_mul(std::mem::size_of::<usize>()),
                        )
                    })
                    .unwrap_or(0);
                let output_queue = match output {
                    CpuStageOutput::Transfer => &channels.transfer,
                    CpuStageOutput::Device => &channels.device,
                    CpuStageOutput::Sink => unreachable!(),
                };
                let result = if output_bytes > output_queue.limits().max_bytes {
                    Err(PipelineError::Runtime(runtime_error(format!(
                        "CPU output batch is {output_bytes} bytes, exceeding stage queue max_bytes {}",
                        output_queue.limits().max_bytes
                    ))))
                } else {
                    result
                };
                if result.is_err() {
                    channels.cancel_cpu_inputs();
                }
                let bytes = if result.is_ok() { output_bytes } else { 0 };
                output_queue.send(sequence_id, result, bytes)
            }
        };
        let waited = match waited {
            Ok(waited) => waited,
            Err(_) => break,
        };
        profiler.record_wait(
            if matches!(output, CpuStageOutput::Sink) {
                nodes.sink
            } else {
                nodes.batch
            },
            waited,
        );
    }
    match output {
        CpuStageOutput::Sink => channels.sink.close(),
        CpuStageOutput::Transfer => channels.transfer.close(),
        CpuStageOutput::Device => channels.device.close(),
    }
}

fn process_batch_samples<P: PhysicalPipelineAdapter>(
    adapter: &Arc<P>,
    samples: Vec<(usize, P::Sample)>,
    sequence_id: u64,
    worker_pool: Option<&WorkerPool<P::Sample, P::Output, P::Error>>,
    profiler: &PhysicalProfiler,
    sample_node: PhysNodeId,
    batch_node: PhysNodeId,
) -> StageResult<(P::Batch, Vec<usize>), P::Error> {
    let indices = samples.iter().map(|(index, _)| *index).collect::<Vec<_>>();
    let outputs = if let Some(pool) = worker_pool {
        let mut slots: Vec<Option<P::Output>> = std::iter::repeat_with(|| None)
            .take(samples.len())
            .collect();
        for (position, (sample_index, payload)) in samples.into_iter().enumerate() {
            pool.submit(WorkItem {
                batch_id: sequence_id,
                position,
                sample_index,
                payload,
            })
            .map_err(PipelineError::Runtime)?;
        }
        let mut first_error = None;
        for _ in 0..slots.len() {
            let result = pool.recv().map_err(PipelineError::Runtime)?;
            if result.batch_id != sequence_id {
                first_error.get_or_insert_with(|| {
                    PipelineError::Runtime(runtime_error(format!(
                        "sample worker returned batch {}, expected {sequence_id}",
                        result.batch_id
                    )))
                });
                continue;
            }
            let Some(slot) = slots.get_mut(result.position) else {
                first_error.get_or_insert_with(|| {
                    PipelineError::Runtime(runtime_error(format!(
                        "sample worker returned invalid position {}",
                        result.position
                    )))
                });
                continue;
            };
            if slot.is_some() {
                first_error.get_or_insert_with(|| {
                    PipelineError::Runtime(runtime_error(format!(
                        "sample worker duplicated position {}",
                        result.position
                    )))
                });
                continue;
            }
            match result.result {
                Ok(output) => *slot = Some(output),
                Err(error) => {
                    first_error.get_or_insert(PipelineError::Domain(error));
                }
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        slots
            .into_iter()
            .map(|slot| {
                slot.ok_or_else(|| {
                    PipelineError::Runtime(runtime_error("sample worker omitted a batch result"))
                })
            })
            .collect::<StageResult<Vec<_>, _>>()?
    } else {
        let mut outputs = Vec::with_capacity(samples.len());
        for (sample_index, sample) in samples {
            let started = Instant::now();
            let input_bytes = adapter.sample_bytes(&sample);
            let output_result = catch_unwind(AssertUnwindSafe(|| {
                adapter.process_sample(sample, sample_index)
            }));
            let output = match output_result {
                Ok(Ok(output)) => output,
                Ok(Err(error)) => return Err(PipelineError::Domain(error)),
                Err(_) => {
                    return Err(PipelineError::Runtime(runtime_error(format!(
                        "CPU transform stage panicked at sample {sample_index}"
                    ))));
                }
            };
            let output_bytes = adapter.output_bytes(&output);
            profiler.record(sample_node, started.elapsed(), input_bytes, output_bytes);
            outputs.push(output);
        }
        outputs
    };

    let capacity = outputs.len();
    let started = Instant::now();
    let batch = catch_unwind(AssertUnwindSafe(|| {
        let mut builder = adapter.batch_builder(capacity);
        for output in outputs {
            adapter
                .push_batch_sample(&mut builder, output)
                .map_err(PipelineError::Domain)?;
        }
        adapter.finish_batch(builder).map_err(PipelineError::Domain)
    }))
    .unwrap_or_else(|_| {
        Err(PipelineError::Runtime(runtime_error(
            "batch builder panicked while assembling a logical batch",
        )))
    })?;
    let output_bytes = adapter.batch_bytes(&batch);
    profiler.record(batch_node, started.elapsed(), 0, output_bytes);
    Ok((batch, indices))
}

fn execute_cpu_batch<P: PhysicalPipelineAdapter>(
    adapter: &Arc<P>,
    mut batch: P::Batch,
    indices: Vec<usize>,
    nodes: &StageNodes,
    profiler: &PhysicalProfiler,
) -> StageResult<IndexedBatch<P::Batch>, P::Error> {
    if !nodes.device_batch_kernel {
        let started = Instant::now();
        let input_bytes = adapter.batch_bytes(&batch);
        batch = match catch_unwind(AssertUnwindSafe(|| {
            adapter.apply_batch_with_indices(batch, &indices)
        })) {
            Ok(Ok(batch)) => batch,
            Ok(Err(error)) => return Err(PipelineError::Domain(error)),
            Err(_) => {
                return Err(PipelineError::Runtime(runtime_error(
                    "CPU batch transform stage panicked",
                )));
            }
        };
        profiler.record(
            nodes.batch_kernel,
            started.elapsed(),
            input_bytes,
            adapter.batch_bytes(&batch),
        );
    }
    Ok(IndexedBatch { indices, batch })
}

fn run_transfer_stage<P: PhysicalPipelineAdapter>(
    adapter: Arc<P>,
    channels: Arc<StageChannels<P>>,
    nodes: StageNodes,
    profiler: PhysicalProfiler,
    device_stage: bool,
) {
    loop {
        let (message, waited) = match channels.transfer.recv() {
            Ok(result) => result,
            Err(_) => break,
        };
        let profile_node = nodes
            .transfer_nodes
            .first()
            .map(|(node, _, _)| *node)
            .unwrap_or(nodes.batch);
        profiler.record_wait(profile_node, waited);
        let Some(message) = message else { break };
        let sequence_id = message.sequence_id;
        let result = match message.value {
            Err(error) => Err(error),
            Ok(IndexedBatch { batch, indices }) => {
                run_transfers(&adapter, batch, &nodes, &profiler)
                    .map(|batch| IndexedBatch { indices, batch })
            }
        };
        let waited = if device_stage {
            let output_bytes = result
                .as_ref()
                .map(|indexed| {
                    adapter.batch_bytes(&indexed.batch).saturating_add(
                        indexed
                            .indices
                            .len()
                            .saturating_mul(std::mem::size_of::<usize>()),
                    )
                })
                .unwrap_or(0);
            let result = if output_bytes > channels.device.limits().max_bytes {
                Err(PipelineError::Runtime(runtime_error(format!(
                    "device input batch is {output_bytes} bytes, exceeding stage queue max_bytes {}",
                    channels.device.limits().max_bytes
                ))))
            } else {
                result
            };
            if result.is_err() {
                channels.cancel_cpu_inputs();
            }
            let bytes = if result.is_ok() { output_bytes } else { 0 };
            channels.device.send(sequence_id, result, bytes)
        } else {
            let result = result.map(|indexed| indexed.batch);
            let output_bytes = result
                .as_ref()
                .map(|batch| adapter.batch_bytes(batch))
                .unwrap_or(0);
            let result = if output_bytes > channels.sink.limits().max_bytes {
                Err(PipelineError::Runtime(runtime_error(format!(
                    "sink batch is {output_bytes} bytes, exceeding stage queue max_bytes {}",
                    channels.sink.limits().max_bytes
                ))))
            } else {
                result
            };
            if result.is_err() {
                channels.cancel_cpu_inputs();
            }
            let bytes = if result.is_ok() { output_bytes } else { 0 };
            channels.sink.send(sequence_id, result, bytes)
        };
        let waited = match waited {
            Ok(waited) => waited,
            Err(_) => break,
        };
        profiler.record_wait(profile_node, waited);
    }
    if device_stage {
        channels.device.close();
    } else {
        channels.sink.close();
    }
}

fn run_device_stage<P: PhysicalPipelineAdapter>(
    adapter: Arc<P>,
    channels: Arc<StageChannels<P>>,
    nodes: StageNodes,
    profiler: PhysicalProfiler,
) {
    loop {
        let (message, waited) = match channels.device.recv() {
            Ok(result) => result,
            Err(_) => break,
        };
        profiler.record_wait(nodes.batch_kernel, waited);
        let Some(message) = message else { break };
        let sequence_id = message.sequence_id;
        let result = match message.value {
            Err(error) => Err(error),
            Ok(IndexedBatch { batch, indices }) if nodes.device_batch_kernel => {
                let started = Instant::now();
                let input_bytes = adapter.batch_bytes(&batch);
                let target = nodes
                    .transfer_nodes
                    .last()
                    .map(|(_, _, target)| *target)
                    .unwrap_or(nodes.device_target);
                catch_unwind(AssertUnwindSafe(|| {
                    adapter.apply_device_batch_with_indices(batch, target, &indices)
                }))
                .unwrap_or_else(|_| {
                    Err(PipelineError::Runtime(runtime_error(
                        "device transform stage panicked",
                    )))
                })
                .map(|batch| {
                    profiler.record(
                        nodes.batch_kernel,
                        started.elapsed(),
                        input_bytes,
                        adapter.batch_bytes(&batch),
                    );
                    batch
                })
            }
            Ok(IndexedBatch { batch, .. }) => Ok(batch),
        };
        let bytes = result
            .as_ref()
            .map(|batch| adapter.batch_bytes(batch))
            .unwrap_or(0);
        let result = if bytes > channels.sink.limits().max_bytes {
            Err(PipelineError::Runtime(runtime_error(format!(
                "sink batch is {bytes} bytes, exceeding stage queue max_bytes {}",
                channels.sink.limits().max_bytes
            ))))
        } else {
            result
        };
        let bytes = if result.is_ok() { bytes } else { 0 };
        if result.is_err() {
            channels.cancel_upstream();
        }
        let waited = match channels.sink.send(sequence_id, result, bytes) {
            Ok(waited) => waited,
            Err(_) => break,
        };
        profiler.record_wait(nodes.sink, waited);
    }
    channels.sink.close();
}

fn run_transfers<P: PhysicalPipelineAdapter>(
    adapter: &Arc<P>,
    mut batch: P::Batch,
    nodes: &StageNodes,
    profiler: &PhysicalProfiler,
) -> StageResult<P::Batch, P::Error> {
    for (node, kind, target) in nodes.transfer_nodes.iter().copied() {
        let started = Instant::now();
        let input_bytes = adapter.batch_bytes(&batch);
        batch = catch_unwind(AssertUnwindSafe(|| {
            adapter.transfer_batch(batch, kind, target)
        }))
        .unwrap_or_else(|_| {
            Err(PipelineError::Runtime(runtime_error(
                "transfer stage panicked",
            )))
        })?;
        profiler.record(
            node,
            started.elapsed(),
            input_bytes,
            adapter.batch_bytes(&batch),
        );
    }
    Ok(batch)
}

fn source_payload_bytes<P: PhysicalPipelineAdapter>(
    payload: &SourcePayload<P::Sample, P::Batch>,
    adapter: &P,
) -> usize {
    match payload {
        SourcePayload::Samples(samples) => samples.iter().fold(
            samples.len().saturating_mul(std::mem::size_of::<usize>()),
            |total, (_, sample)| total.saturating_add(adapter.sample_bytes(sample)),
        ),
        SourcePayload::Batch { indices, batch } => adapter
            .batch_bytes(batch)
            .saturating_add(indices.len().saturating_mul(std::mem::size_of::<usize>())),
    }
}
