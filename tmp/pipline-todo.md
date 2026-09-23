# Rivet Pipeline 改进：多阶段行动指南

> 设计原则、架构图和备选方案见 tmp/pipline.md。本文件只记录实施顺序、具体交付物和阶段验收，不重复解释设计背景。

## 执行规则

- 按 Phase 0 → Phase 11 顺序推进；前一阶段未达到验收条件，不开始依赖它的后续阶段。
- 每个 checkbox 对应可审查的代码、测试、benchmark 或文档证据；证据未完成前不勾选。
- 保留现有 ImagePipeline 用户 API。迁移阶段先做等价 lowering，再逐步替换执行实现。
- 每阶段使用目标 crate 的定向 check/test/bench；不从 workspace 根目录发起全量构建。
- logical plan 不持有 backend runtime 句柄；H2D、D2H、D2D 必须是显式 physical node。
- rivet-core 保持通用 tensor/backend 职责，不依赖 rivet-vision、rivet-text、rivet-audio 或 pipeline/runtime。
- 不支持的 CUDA op 不得在 kernel 或 Tensor API 内隐式做 D2H → CPU → H2D。
- 保留 HWC/NHWC 三通道 Normalize 和专用 NHWC → NCHW 写入路径；任何热点变化都要用对应 benchmark 验证。
- 性能判断使用 release benchmark 的多次结果；记录命令、环境、均值、范围和离散程度。

## 当前进度快照

以下是已经落地的结构基础，不代表相应阶段完成：

- [x] rivet-plan crate 已加入 workspace；Logical IR 和 optimizer 尚未实现。
- [x] rivet-exec crate 已加入 workspace；通用 worker/prefetch 和通用 cache 已从 rivet-data 移入，vision 专用 loader/scheduler 尚未迁移。
- [x] rivet-data 已有 Dataset、Source、Sampler 和基础 SourceCapabilities。
- [x] rivet-core 当前承载 Tensor、Storage、Layout、DType、Device、CPU 和 CUDA backend。
- [x] rivet-text crate 已建立空壳，并有 SST-5 ArrowText 读取示例；文本 semantic op 尚未实现。
- [ ] 尚无 rivet-plan IR → optimizer → placement → rivet-exec physical graph 的完整路径。

## Phase 0：冻结当前行为与建立基线

目标：留下可比较的旧实现证据，不先改变 pipeline/runtime 行为。

任务：

- [x] 保留现有 vision pipeline、runtime、batch、transform 正确性测试。
- [x] 保留既有参考值（均值 731303 img/s）；本 commit 重测五次均值 723429 img/s、范围 710259–736701 img/s、标准差 9020 img/s，详见 `docs/perf/baseline-023ed68.md`。
- [x] 用定向命令确认现有 vision 测试和 core/data 相关测试通过，并记录提交和运行环境：core 9/9、data 20/20、vision 130/130；commit `023ed68`。
- [x] 建立 CPU pipeline 的 throughput、首批 latency、稳态 batch latency、p50/p95、peak memory 基线，详见 `docs/perf/baseline-023ed68.md`。
- [x] 记录 trainer wait ratio、CPU/GPU utilization、H2D bytes、kernel launch、sync 和 allocation 指标；当前没有 trainer/GPU pipeline benchmark，无法提供 wait ratio、H2D、launch、sync，限制已记录；CPU 进程用时和 operator allocations 已采集。
- [x] 保存当前 ImagePipeline 编译摘要和关键 workload 输出，作为迁移前对照；当前没有独立的 Logical/Physical explain 输出。
- [x] 记录代表 workload：decode-only、CPU transforms、Normalize + Layout、decoded CIFAR-10；当前 benchmark 没有 CUDA sink 路径，限制已记录。
- [x] 记录 benchmark 的数据目录、batch、worker、warmup、epoch、release 配置和原始输出，详见 `docs/perf/baseline-023ed68.md`。

验收门槛：

- 可以在同一环境重复跑旧路径并得到带离散范围的基线。
- 新旧语义对照至少覆盖 shape、dtype、axis order、随机性和错误行为。

## Phase 1：完善 rivet-plan Logical IR 与兼容 lowering

目标：让现有 ImagePipeline 经过 LogicalPlan 后仍可执行旧 ExecutionPlan。

任务：

- [x] workspace 中已建立 rivet-plan crate。
- [x] 定义 NodeId、PlanArena、LogicalNode 和 LogicalPlan。
- [x] 定义 Source、Index、Op、Batch、Cache、Sink 等稳定 node kind 和输入边。
- [x] 使用 inline 容量为 2、可 spill 的小型输入边容器；按 Phase 1 采用 PlanPayload trait-object 擦除领域 payload，rivet-plan 不依赖 rivet-vision。
- [x] 支持 root 更新、parent/child traversal、节点替换和一致性校验。
- [x] 实现 logical explain，展示 node id、kind、inputs 和 root。
- [x] 将 ImagePipeline builder 输出 lowering 成 LogicalPlan，保留 builder API 和错误语义。
- [x] 将 LogicalPlan lowering 回现有 ExecutionPlan；旧执行路径仍作为迁移期 backend。
- [x] 为 logical plan 增加 explain 稳定性和 traversal 测试。
- [x] 对新旧 lowering 做输出和错误等价测试。

验收门槛：

- 现有 vision 行为不变；旧测试通过。
- logical plan 可稳定打印、snapshot 和定位非法输入。
- 该阶段不执行 Tensor 运算，不改变 scheduler 和设备行为。

## Phase 2：Property inference 与 validation

目标：用通用 ValueProperties 替换 pipeline compile 内不断扩展的 image state 推导。

任务：

- [x] 定义 representation、dtype、shape、layout、residency、granularity、mutability properties。
- [x] 定义 Known/Dynamic shape dimension，并以缺省 shape 表示未知 rank。
- [x] 定义 axis order 与 contiguity 两个独立 property。
- [x] 定义 Host/Device(DeviceClass)/Unknown residency；具体 device ordinal 保留给 physical planning。
- [x] 定义 Sample/Batch/Stream granularity，并通过独立 OperatorProperties 表示 sample/batch 执行 barrier。
- [x] 定义 PropertyInference hook 和带 node id 的 inference/validation error。
- [x] 将 vision op 的 state transition 语义集中到 property inference；compile 入口先推导校验，PipelineImageState 只作旧 ExecutionPlan 的兼容投影。
- [x] 为全部 ImageOp 变体实现 property inference，重点覆盖 Decode、Resize、Normalize、Layout。
- [x] 推导 Resize 固定 shape、Decode dynamic H/W/C、Batch 新增 N 轴。
- [x] 检查 dtype、rank/shape、axis order、contiguity、residency、granularity 的非法组合，并在已知 shape 时预检 crop bounds。
- [x] vision source 首版 residency 推导为 Host；已知 Device 输入在 CPU vision planner 中拒绝，不增加 CUDA placement。
- [x] 为所有 ImageOp 变体添加合法推导覆盖，并测试 dtype、shape、axis order、contiguity、residency 和 granularity 冲突。

验收门槛：

- 编译器不再依赖 PipelineImageState 决定合法性。
- 每个现有 vision op 有可解释的输入和输出属性。
- 非法 pipeline 在执行前失败，并指出节点及冲突属性。

## Phase 3：Optimizer、semantic rewrite、fusion 与 domain hooks

目标：把当前 compile 中的规则拆成可排序、可测试的 planning passes。

任务：

- [x] 定义 optimizer pass protocol，返回 changed 和诊断信息。
- [x] 支持 pass 读取/写入 property annotation、替换节点、更新 root。
- [x] 实现 canonicalization、validation/property inference、identity Layout/Convert/Crop 消除和 dead-node elimination。
- [x] 实现有上限的 fixed-point runner；无变化时提前结束，超限时报出诊断。
- [x] 固定顺序为 canonicalization → inference/validation → simplify → pushdown/rewrite → fusion discovery → placement。
- [x] placement 前完成语义 rewrite；placement 后如需改写必须通过 `PassResult::request_replan()` 重跑 canonicalization 到 placement。
- [x] 将 Normalize + Layout 迁移为正式 FusionGroup rule，并按 U8/HWC/CHW 和 worker stage 属性检查合法性。
- [x] 评估 Convert + Normalize、Convert + Normalize + Layout、Crop + Resize、Flip + Normalize 等融合；legality、reference benchmark 和限制记录在 `docs/perf/phase3-fusions.md`。
- [x] 实现 U8→F32 late dtype promotion、shape 已知时的 full-image crop 消除和 Skip/Take/Shuffle source sampler pushdown；Crop/Resize 重排因插值坐标语义不同而明确拒绝。
- [x] 每条 rewrite 明确 dtype、shape、layout、随机数和数值语义前置条件，并通过诊断说明是否合法或为什么拒绝。
- [x] 定义最小 type-erased logical payload 协议和 PlanRegistry；generic optimizer 不 downcast 具体 VisionOp。
- [x] domain crate 通过显式 plugin 注册 property inference、rewrite passes、fusion rule 和 physical candidate provider。
- [x] 禁止依赖 inventory、linkme 或全局 mutable registry；由应用 composition root 显式组装插件。
- [x] 为 plan snapshots、rewrite legality、fixed-point、placement restart、FusionGroup 应用及兼容 lowering 增加测试。

当前落地范围：canonicalization、属性推导和校验、identity rewrite、dead-node 清理、U8→F32 late promotion、已知 shape 下的 full-image crop 消除、source sampler index pushdown、FusionGroup 和 placement 重规划均接入 vision optimizer。Convert+Normalize 和 Convert+Normalize+Layout 的改写输出与旧 compiler 对照通过；Crop+Resize reorder 与 Flip+Normalize 未注册融合 kernel，planner 会报告语义条件并保持原执行顺序。基准结果见 `docs/perf/phase3-fusions.md`。

验收门槛：

- optimizer 输出与现有 compiler 等价，已注册规则可通过 explain 检查。
- generic planner 不依赖 rivet-vision、rivet-text 或 rivet-audio。

## Phase 4：Kernel capability、cost model 与 placement

目标：由 planner 按整段 pipeline 选择设备和 kernel，不再由 semantic op 固定执行位置。

任务：

- [x] 定义 KernelCapability、KernelRequirements、PlacementCandidate 和 KernelClass。
- [x] 定义显式 KernelCapabilities registry；只注册已有实现。
- [x] 记录输入/输出属性约束、device、granularity、fusion tags、alignment、contiguity、temporary bytes 和 in-place 能力。
- [x] 定义 CostEstimate：host/device bytes、transfer bytes、allocation、launch、sync 和 compute score。
- [x] 定义 MachineProfile：CPU threads、available devices 和传输带宽。
- [x] 实现静态 heuristic，不引入 autotuner 或 ML cost model。
- [ ] 从最终 Sink 约束向前/向后计算候选，优先形成较长的 same-device region。
- [x] 把 transfer、allocation、launch 和同步代价纳入区域选择，避免逐 op 贪心切换。
- [x] 第一版只允许一个主要 CPU → CUDA boundary；Decode 和普通 augmentation 默认 CPU。
- [x] CUDA sink 下仅将已有 GPU-friendly batch op 作为 CUDA 候选。
- [ ] 逐步移除 ImageOp::execution_kind() 对 placement 的主导作用。
- [x] 扩展 SourceCapabilities，明确 random/batch/zero-copy/parallel/async/read-device 等实际 source 能力。
- [x] 为 Lance、filesystem、memory source 声明准确 capability，未知能力保持保守值。
- [x] 输出 placement explain：候选、选择、cost、fusion、transfer boundary、shape/dtype/layout 和 parallelism。
- [x] 增加 placement snapshot、source strategy、CPU-only op 与 sink-device constraint 测试。

当前落地范围：vision 显式注册现有 CPU source/index/sample/batch/sink 路径；没有注册 CUDA vision kernel，因此 CUDA sink 请求会明确失败，避免选中未实现路径。planner 已生成 CPU placement 与成本说明，profile 会影响 CPU 并行策略。多设备 region 的全局动态规划和执行计划消费 placement 结果仍待后续阶段接入。

验收门槛：

- 相同 LogicalPlan 可根据 MachineProfile 产生不同 placement。
- 未注册的 kernel 不会被 planner 选择；CPU-only op 不会被放到 CUDA。
- placement 与 cost 选择都可通过 explain 解释。

## Phase 5：PhysicalGraph 与 rivet-exec 迁移

目标：由 rivet-exec lowering 和执行 physical graph，逐步接管当前 vision runtime。

任务：

- [x] workspace 中已建立 rivet-exec crate，并允许依赖 plan/core/data。
- [x] 通用 worker/prefetch 与通用 cache 已迁入 rivet-exec。
- [x] 定义 PhysNodeId、PhysicalGraph、PhysicalNode 和 Source/Kernel/Batch/Transfer/Cache/Sink node。
- [x] 定义 LogicalPlan → PhysicalGraph lowering contract；vision lowering 按属性标注构造物理节点，并将 batch kernel 放到 Batch barrier 之后。
- [x] 定义 PhysicalOperator 和 execution value protocol；首批值类型限定为 encoded samples、decoded samples、Tensor。
- [x] 定义 Morsel、sequence id、source index、bytes、shape/dtype、residency、granularity metadata。
- [x] 定义 IO、CPU、Transfer、Device execution lanes；logical IR 不携带 CUDA stream/context。
- [x] 将通用 ImageDataLoader 调度、sampler 消费、worker/prefetch 状态和 batch-builder 生命周期编排迁至 rivet-exec；ExecutionPlan 和图像 batch builder 保留在 vision adapter，提供领域语义回调。
- [x] 将现有单 worker-pool路径表示为 Sampler → Source → SampleStage → Batch → BatchStage → Sink；batch-native source 可跳过逐样本 stage。
- [x] 迁移期保留 rivet-vision 用户 API，由 vision adapter 提供 source、domain kernels 和 batch builder 实现。
- [x] physical graph 支持 last-use metadata 和按节点记录执行次数、耗时与输入/输出字节的 profiler instrumentation。
- [x] 增加 physical plan explain 稳定快照，以及 rivet-exec inline/worker 顺序测试和 vision loader 对直接 ExecutionPlan 结果的等价检查。

当前边界：首版 GraphExecutor 只执行线性 graph；vision CPU path 的通用采样、调度、预取和批次阶段由 rivet-exec 的 PhysicalPipelineExecutor 驱动，领域回调由 vision adapter 提供。Transfer/Device node 与对应异构执行仍由 Phase 6 接入。

验收门槛：

- ImagePipeline API 不变，CPU path 由 rivet-exec 执行。
- 领域 crate 不再拥有通用 scheduler、worker pool、buffer lifetime 或 profiler。

## Phase 6：显式 Transfer 与单 stream CUDA path

目标：先建立 CPU → CUDA 的正确异构执行闭环，再优化并发。

任务：

- [x] 定义 TransferKind::HostToDevice/DeviceToHost/DeviceToDevice 和显式 transfer node。
- [x] physical lowering 按 placement 插入 transfer；kernel 不得暗中调用 to_device。
- [x] transfer 参与 cost、lane、memory lifetime 和 profiler。
- [x] 第一版限制为单 CUDA device、单 default stream。
- [x] physical runtime 通过 Device lane 使用 stream；不把 stream/context 暴露到 LogicalPlan。
- [x] 明确 storage/view 在 transfer 前后的所有权和生命周期。
- [x] 未支持的 op 保留在 CPU region；任何 fallback 都通过显式 transfer node 表达。
- [x] 增加 transfer shape/dtype/residency、边界数量、错误路径和 stream ordering 测试。

验收门槛：

- [x] 生成 CPU → H2D → CUDA → Sink 的 physical plan，并可 explain。
- [x] 没有计划外 D2H/H2D，也没有依赖默认隐式同步的生命周期漏洞。

Phase 6 完成记录（2026-09-23）：

- `rivet-vision/cuda` 是 opt-in feature；`ImagePipeline::cuda_sink(ordinal)` 只请求目标 ordinal，LogicalPlan 不持有 CUDA runtime handle。当前所有 image op 仍留在 CPU；placement 选择 CUDA sink 并把 CPU→CUDA 边界作为单个 transfer boundary 计价，physical lowering 将其物化为 Transfer(H2D)。
- `PhysicalNode` 记录 transfer target、估算字节数和 last-use；pipeline runtime 对 transfer 单独计时并记录实际输入/输出字节。当前 vision path 只执行 H2D，而且要求 H2D 直接馈入 sink。
- `Tensor::to_device` 仅由显式 transfer callback 调用。images 与 labels 使用同一个 `Device` 的 default stream；返回前显式 synchronize，保证 pageable host batch 在异步 copy 完成前保持存活。Pinned staging 与 copy/compute overlap 留在 Phase 9。
- 验证：`cargo check -j 12 -p rivet-exec --lib`、`cargo check -j 12 -p rivet-vision --lib`、`cargo check -j 12 -p rivet-vision --lib --features cuda` 均通过；`cargo test -j 12 -p rivet-exec --lib`（16 项）、`cargo test -j 12 -p rivet-plan --lib`（9 项）、vision 默认 feature（153 项）和 CUDA feature（154 项）均通过。CUDA sink 测试在 RTX 4070 Ti 上实际运行，覆盖一个 H2D 边界、shape/dtype/residency/value、两个 H2D copy、stream 同步和物理 explain。
- 限制：vision runtime 当前不执行 D2H/D2D 节点，也不支持 transfer 后接 CUDA kernel；这些类型已在 physical IR 表达，后续随着相应执行 lane/capability 接入。

## Phase 7：CUDA batch-native kernel 与第一条完整 vision pipeline

目标：减少完整 tensor pass、CPU F32 临时量和 kernel launch，先打通 Normalize + Layout。

任务：

- [x] 实现 U8 NHWC batch → normalized F32 NCHW CUDA fused kernel。
- [x] 在同一 batch kernel 中实现 Flip 和 RandomResizedCrop/CropResize；目前融合路径要求固定 shape 的 batch-readable RGB HWC source、至多一个 RandomResizedCrop，并覆盖 nearest/bilinear/bicubic/lanczos3；其他来源继续走 CPU augment fallback。
- [x] semantic random parameter 用 seed + epoch + sample index + OpKey 生成，再上传给 backend kernel。
- [x] 同一 semantic random parameters 在 worker count、lane 和 backend 变化时保持一致；CPU/CUDA 对照覆盖乱序 sample indices 与 0/3 workers。
- [x] 验证每 batch 为 one launch，不对每个 sample 单独 launch。
- [x] physical planner 按 capability 选择 CPU 或 CUDA fused implementation；不满足融合条件的操作保留 CPU 执行。
- [x] 保留 CPU HWC/NHWC 三通道 normalize 和 NHWC → NCHW 专用 nested-loop/typed-writer path。
- [x] 直接从 CPU U8 HWC/NHWC 传输，避免先扩展为 CPU F32；输出写入最终 CUDA allocation。
- [x] CUDA 热路径按 RGB 通道和插值模式静态特化，没有 per-element dynamic dispatch、回调 writer 或图像中间 CPU Vec。
- [x] 实施 CPU/CUDA shape、dtype、layout、数值 tolerance 和随机性对照；覆盖四种 Resize 插值模式与前后翻转。
- [ ] 用 batch 32/64/128/256、224×224×3 kernel benchmarks 调优。
- [ ] 对 CIFAR-10 decoded benchmark 按 AGENTS.md 指定 release 命令重复测量，比较均值/范围/标准差（按用户要求，暂缓性能诊断）。

验收门槛：

- CUDA fused output 与 CPU reference 在规定 tolerance 内一致。
- CUDA path 的 launch、transfer、allocation 数可解释。
- CIFAR 和热点 benchmark 没有未解释的显著回退。

Phase 7 功能实现记录（2026-09-23）：

- `NormalizeToChw` 的 CUDA fusion 可以吸收相邻的 deterministic Flip、RandomHorizontalFlip 和一个 RandomResizedCrop；由 `RandomContext + sample index + OpKey` 在 host 端生成每个样本参数，worker 顺序与 sampler 排列不会改变参数。
- 新 fused RGB kernel 单 launch 完成可选 flip/crop-resize、U8 采样、归一化和 NHWC→NCHW；nearest、bilinear、bicubic、lanczos3 采用静态 kernel specialization。图像保持 U8 到显式 H2D，最终 F32 tensor 直接分配在 CUDA。
- 参数化 CPU/CUDA 对照验证了 output shape/dtype/layout/数值、shuffle 顺序、worker count 和 interpolation；Phase 7 功能测试通过。
- 性能 bench 和 CIFAR-10 decoded 重复测量尚未执行；用户要求先完善功能，Phase 7 的性能验收仍待后续完成。

## Phase 8：Persistent Stage Graph 与 bounded backpressure

目标：从一次 batch 调度升级为有界、可重叠的 persistent stages。

任务：

- [x] 建立 Sampler → Source → Decode → CPU Transform/Batch → Transfer → Device Transform → Sink stage graph。Sampler 仍由 `next_batch(&mut IndexSampler)` 驱动，并通过有界 request edge 向常驻 stage 提交请求。
- [x] 所有 stage edge 使用 bounded FIFO queue，并保留 sequence id；结果严格按 sampler 顺序交付。
- [x] queue 支持 max_items 和 max_bytes，按 index request、encoded/decoded sample、batch payload 计量；单个 payload 超限时返回错误。
- [x] 在内存压力下阻塞上游；prefetch window 和每条 edge 的 item/byte 容量均有硬上限。
- [x] 将旧 worker pool 接入 CPU Transform stage，按 batch sequence/position 回收结果并恢复原样本顺序。
- [x] Source、CPU Transform/Batch、Transfer、Device Transform、Sink 在不同 sequence 上由常驻 stage 并行推进。
- [x] 区分 inter-sample worker 数与 backend-managed intra-op parallelism，避免运行时额外创建 Rayon 池。
- [x] 覆盖 queue ordering/backpressure、sequence 顺序、错误传播、worker panic、取消、满 prefetch drop 和超 byte budget 错误。
- [x] profiler 记录 physical node action time、相邻 queue wait 与 queue peak/current item/byte 统计。

验收门槛：

- 每条 queue 的 item/byte 上限在 enqueue 前强制执行，因此 retained queue memory 有界；常规长时间吞吐/RSS 对比留待后续性能验收。
- 关闭 pipeline 或发生 source/decode/transform/transfer/device 错误时，剩余 worker 和 queue 能安全退出。

Phase 8 功能实现记录（2026-09-23）：

- `rivet-exec` 现有 persistent Source、Decode、CPU Transform/Batch、Transfer、Device Transform workers，通过六条 FIFO bounded queues 连接；caller-side sampler 只提交当前 `prefetch_batches + 1` 窗口内的请求。
- 每条 queue 默认最多保留 3 个 item（默认 prefetch 为 2）和 512 MiB payload，可由 vision builder 的 `.stage_queue_max_bytes(...)` 调整。超出单条 byte 上限的 item 会显式报错；生产者在队列达到容量时阻塞。
- CPU sample worker pool 保持每 batch 的 position 并在组 batch 前恢复顺序。物理 profiler 与 explain 输出 action/wait 时间、queue 当前和峰值占用，以及 inter-sample worker 配置。
- CUDA transfer 和 device transform 已拆成独立持久 stage；当前 pageable H2D adapter 会同步 copy 生命周期，实际 copy/compute overlap 与 pinned staging 仍属于 Phase 9。
- `cargo test -j 12 -p rivet-exec --lib` 通过 22 项；`cargo test -j 12 -p rivet-vision --features cuda --lib` 通过 158 项。未运行性能或 CIFAR benchmark。

## Phase 9：Pinned memory 与 copy/compute overlap

目标：确认 pinned memory 和独立 transfer lane 能提高端到端 throughput 后再引入。

任务：

- [ ] 定义 Pageable/Pinned host memory property 和 PinnedHostPool。
- [ ] 第一版用普通 CPU output 加显式 pinned staging buffer。
- [ ] 加入 CUDA transfer lane、compute lane 和 CUDA events。
- [ ] 正确建立 H2D(N+1) 与 compute(N) 的依赖。
- [ ] 对比 pageable/pinned H2D 不同大小和端到端 latency。
- [ ] 只有 benchmark 证明 decode 可直接写 pinned destination 有收益时才实施该路径。
- [ ] 不将所有 CpuStorage 默认改成 pinned memory。
- [ ] 验证 buffer 生命周期覆盖异步 transfer 完成时点。

验收门槛：

- H2D 和 compute overlap 有 profiler/event 证据。
- pinned pool 有界；无 use-after-free、无计划外同步。

## Phase 10：Memory planner、cache lifecycle 与 profiler

目标：基于 physical graph 的 liveness 管理 buffer，减少峰值内存和重复 allocation。

任务：

- [ ] 对 physical graph 计算 consumer count 和 last use。
- [ ] 按固定 shape、dtype、residency 和 alignment 定义 buffer reuse class。
- [ ] 管理 cache policy、key、lifetime、residency 和 cache node。
- [ ] 增加 inflight byte budget，backpressure 与 memory planner 使用同一预算。
- [ ] 评估 CpuAlignedPool、PinnedHostPool、CudaBufferPool、MetalBufferPool；仅实现基准证明必要的 pool。
- [ ] direct-output kernel 写最终 allocation；禁止重新引入中间 Vec 和多余 full tensor pass。
- [ ] profiler 记录 calls/items/input-output bytes/queue wait/action time/allocation/H2D-D2H-D2D/launch/sync。
- [ ] 输出 logical、optimized、physical explain。
- [ ] 对 fixed-shape batch 比较 allocation count、peak host/GPU memory 和 throughput。

验收门槛：

- liveness 结果覆盖多消费者、cache 分支、异步 transfer 和异常退出。
- 内存指标下降或保持不回退，且 buffer reuse 不破坏值和生命周期。

## Phase 11：Metal backend

目标：验证 Metal 仅作为新增 backend/physical implementation 接入，不改变 semantic plan。

任务：

- [ ] 在 rivet-core 增加 Metal storage/device/backend dispatch。
- [ ] 在 vision 注册 Metal NormalizeToNchw 等已实现 kernel capability 和 lowering。
- [ ] 在 rivet-exec 增加 Metal Device lane adapter。
- [ ] 加入 Metal device descriptor 到 physical planning context 的映射。
- [ ] 复用同一 LogicalPlan、properties、optimizer 和 transfer/memory protocols。
- [ ] 增加 CPU/CUDA/Metal semantic parity、layout、memory lifetime 与 explain 测试。
- [ ] 测量 Metal transfer、kernel 和 end-to-end performance。

验收门槛：

- 新增 Metal 不要求 rivet-plan 依赖或识别 Metal-specific domain op。
- logical/optimized plan 对 backend 保持相同；差异仅出现在 physical plan。

## 每阶段通用验收

每个阶段完成时更新对应 phase 的 checkbox，并在提交说明或阶段记录中附上：

- 目标 crate 的定向编译结果。
- 与改动对应的 correctness、determinism 或 plan snapshot 测试结果。
- 涉及性能路径时的 release benchmark 命令和多次结果。
- 新增/删除 API、当前限制和未完成迁移点。

不要为了勾选而运行不相关的全 workspace 测试或 benchmarks。

## 暂缓到对应阶段的工作

以下约束贯穿所有阶段；需要新增相应能力时，按前述 phase 的边界和验收门槛推进：

- 不一次性把全部 vision ops 写成 CUDA；先按 capability 和收益逐个接入。
- 不提前实现通用多 GPU/peer copy、multi-stream scheduler、ML cost model、autotuner 或全局自动注册系统。
- 不为了 text/audio 的未来需求提前构建任意 Any/trait-object 框架；由 domain crate 显式注册已实现的扩展。
- Logical IR 不持有 CUDA/Metal kernel、stream、context 或 allocation pointer。
- Tensor op/kernel 不得隐藏设备搬运；batch 内不能按每个 sample 单独 launch GPU kernel。
- 不为了统一 abstraction 移除已有的 CPU specialized image kernels。

## 最终里程碑

- [ ] M0：旧行为、测试和基线可重复。
- [ ] M1：ImagePipeline → LogicalPlan → legacy ExecutionPlan，行为等价。
- [ ] M2：properties、validation、rewrite 和 fusion 可 explain/snapshot。
- [ ] M3：capability、cost、placement 和 transfer boundary 可 explain。
- [ ] M4：PhysicalGraph/rivet-exec CPU path 保持旧行为。
- [ ] M5：CUDA fused NormalizeToNchw 通过 parity/launch-count 检查。
- [ ] M6：persistent stages 与 bounded backpressure 稳定。
- [ ] M7：pinned overlap、memory planner、profiler 有实测收益。
- [ ] M8：Metal 仅增加 backend implementation，不改变 logical architecture。
