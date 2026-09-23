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

- [ ] 定义 optimizer pass protocol，返回 changed 和诊断信息。
- [ ] 支持 pass 读取/写入 property annotation、替换节点、更新 root。
- [ ] 实现 canonicalization、validation/property inference、no-op elimination 和 dead-node elimination。
- [ ] 实现有上限的 fixed-point runner；无变化时提前结束，超限时报出诊断。
- [ ] 固定顺序为 canonicalization → inference/validation → simplify → pushdown/rewrite → fusion discovery → placement。
- [ ] placement 前完成语义 rewrite；placement 后如需改写必须显式触发重规划。
- [ ] 将 Normalize + Layout 现有融合迁移为正式 FusionGroup rule。
- [ ] 评估 Convert + Normalize、Convert + Normalize + Layout、Crop + Resize、Flip + Normalize 等融合，逐项记录 legality 和 benchmark 结果。
- [ ] 实现 late dtype promotion、合法的 crop/resize 提前和 Skip/Take/Shuffle source pushdown。
- [ ] 每条 rewrite 明确 dtype、shape、layout、随机数和数值语义前置条件。
- [ ] 定义最小 LogicalOp 扩展协议和 PlanRegistry；generic optimizer 不 downcast 具体 VisionOp。
- [ ] domain crate 通过显式 plugin 注册 property inference、rewrite、fusion 和 physical candidate。
- [ ] 禁止依赖 inventory、linkme 或全局 mutable registry；由应用 composition root 显式组装插件。
- [ ] 为 plan 前后 snapshots、rewrite legality、fixed-point 和 fusion discovery 增加测试。

验收门槛：

- optimizer 输出与现有 compiler 等价，已注册规则可通过 explain 检查。
- generic planner 不依赖 rivet-vision、rivet-text 或 rivet-audio。

## Phase 4：Kernel capability、cost model 与 placement

目标：由 planner 按整段 pipeline 选择设备和 kernel，不再由 semantic op 固定执行位置。

任务：

- [ ] 定义 KernelCapability、KernelRequirements、PlacementCandidate 和 KernelClass。
- [ ] 定义显式 KernelCapabilities registry；只注册已有实现。
- [ ] 记录输入/输出属性约束、device、granularity、fusion tags、alignment、contiguity、temporary bytes 和 in-place 能力。
- [ ] 定义 CostEstimate：host/device bytes、transfer bytes、allocation、launch、sync 和 compute score。
- [ ] 定义 MachineProfile：CPU threads、available devices 和传输带宽。
- [ ] 实现静态 heuristic，不引入 autotuner 或 ML cost model。
- [ ] 从最终 Sink 约束向前/向后计算候选，优先形成较长的 same-device region。
- [ ] 把 transfer、allocation、launch 和同步代价纳入区域选择，避免逐 op 贪心切换。
- [ ] 第一版只允许一个主要 CPU → CUDA boundary；Decode 和普通 augmentation 默认 CPU。
- [ ] CUDA sink 下仅将已有 GPU-friendly batch op 作为 CUDA 候选。
- [ ] 逐步移除 ImageOp::execution_kind() 对 placement 的主导作用。
- [ ] 扩展 SourceCapabilities，明确 random/batch/zero-copy/parallel/async/read-device 等实际 source 能力。
- [ ] 为 Lance、filesystem、memory source 声明准确 capability，未知能力保持保守值。
- [ ] 输出 placement explain：候选、选择、cost、fusion、transfer boundary、shape/dtype/layout 和 parallelism。
- [ ] 增加 placement snapshot、source strategy、CPU-only op 与 sink-device constraint 测试。

验收门槛：

- 相同 LogicalPlan 可根据 MachineProfile 产生不同 placement。
- 未注册的 kernel 不会被 planner 选择；CPU-only op 不会被放到 CUDA。
- placement 与 cost 选择都可通过 explain 解释。

## Phase 5：PhysicalGraph 与 rivet-exec 迁移

目标：由 rivet-exec lowering 和执行 physical graph，逐步接管当前 vision runtime。

任务：

- [x] workspace 中已建立 rivet-exec crate，并允许依赖 plan/core/data。
- [x] 通用 worker/prefetch 与通用 cache 已迁入 rivet-exec。
- [ ] 定义 PhysNodeId、PhysicalGraph、PhysicalNode 和 Source/Kernel/Batch/Transfer/Cache/Sink node。
- [ ] 定义 LogicalPlan → PhysicalGraph lowering contract。
- [ ] 定义 PhysicalOperator 和 execution value protocol；首批值类型限定为 encoded samples、decoded samples、Tensor。
- [ ] 定义 Morsel、sequence id、source index、bytes、shape/dtype、residency、granularity metadata。
- [ ] 定义 IO、CPU、Transfer、Device execution lanes；logical IR 不携带 CUDA stream/context。
- [ ] 将现有 ExecutionPlan、ImageDataLoader、scheduler、batch builder 逐步迁至 rivet-exec。
- [ ] 将现有单 worker-pool 路径表示为 Source → SampleStage → Batch → BatchStage → Sink。
- [ ] 迁移期保留 rivet-vision 用户 API，由 vision plugin/adapter 构造 domain ops 和 kernel implementations。
- [ ] physical graph 支持 last-use metadata 和 profiler instrumentation。
- [ ] 增加 physical plan explain/snapshot 和迁移前后 runtime 等价检查。

验收门槛：

- ImagePipeline API 不变，CPU path 由 rivet-exec 执行。
- 领域 crate 不再拥有通用 scheduler、worker pool、buffer lifetime 或 profiler。

## Phase 6：显式 Transfer 与单 stream CUDA path

目标：先建立 CPU → CUDA 的正确异构执行闭环，再优化并发。

任务：

- [ ] 定义 TransferKind::HostToDevice/DeviceToHost/DeviceToDevice 和显式 transfer node。
- [ ] physical lowering 按 placement 插入 transfer；kernel 不得暗中调用 to_device。
- [ ] transfer 参与 cost、lane、memory lifetime 和 profiler。
- [ ] 第一版限制为单 CUDA device、单 default stream。
- [ ] physical runtime 通过 Device lane 使用 stream；不把 stream/context 暴露到 LogicalPlan。
- [ ] 明确 storage/view 在 transfer 前后的所有权和生命周期。
- [ ] 未支持的 op 保留在 CPU region；任何 fallback 都通过显式 transfer node 表达。
- [ ] 增加 transfer shape/dtype/residency、边界数量、错误路径和 stream ordering 测试。

验收门槛：

- 生成 CPU → H2D → CUDA → Sink 的 physical plan，并可 explain。
- 没有计划外 D2H/H2D，也没有依赖默认隐式同步的生命周期漏洞。

## Phase 7：CUDA batch-native kernel 与第一条完整 vision pipeline

目标：减少完整 tensor pass、CPU F32 临时量和 kernel launch，先打通 Normalize + Layout。

任务：

- [ ] 实现 U8 NHWC batch → normalized F32 NCHW CUDA fused kernel。
- [ ] 再实现 batch Flip 和 CropResize kernel；先按收益/合法性排序。
- [ ] semantic random parameter 用 seed + epoch + sample index + OpKey 生成，再传给 backend kernel。
- [ ] 同一 semantic random parameters 在 worker count、lane 和 backend 变化时保持一致。
- [ ] 验证每 batch 为 one/few launches，不对每个 sample 单独 launch。
- [ ] physical planner 按 capability 选择 CPU 或 CUDA fused implementation。
- [ ] 保留 CPU HWC/NHWC 三通道 normalize 和 NHWC → NCHW 专用 nested-loop/typed-writer path。
- [ ] 直接从 CPU U8 HWC/NHWC 传输，避免先扩展为 CPU F32；输出写入最终对齐 allocation。
- [ ] 不用通用 per-element iterator、动态回调 writer 或中间 Vec 替代热路径。
- [ ] 实施 CPU/CUDA shape、dtype、layout、数值 tolerance 和随机性对照。
- [ ] 用 batch 32/64/128/256、224×224×3 kernel benchmarks 调优。
- [ ] 对 CIFAR-10 decoded benchmark 按 AGENTS.md 指定 release 命令重复测量，比较均值/范围/标准差。

验收门槛：

- CUDA fused output 与 CPU reference 在规定 tolerance 内一致。
- CUDA path 的 launch、transfer、allocation 数可解释。
- CIFAR 和热点 benchmark 没有未解释的显著回退。

## Phase 8：Persistent Stage Graph 与 bounded backpressure

目标：从一次 batch 调度升级为有界、可重叠的 persistent stages。

任务：

- [ ] 建立 Sampler → Source → Decode → CPU Transform → Batch → Transfer → Device Transform → Sink stages。
- [ ] 所有 stage edge 使用 bounded queue，并保留 sequence id。
- [ ] queue 支持 max_items 和 max_bytes，按 encoded/decoded/batch 数据大小计量。
- [ ] 在内存压力下阻塞上游；不得无限 prefetch。
- [ ] 将旧 worker pool 映射到 stage graph，先保持确定性和输出顺序。
- [ ] 支持 Source N+2、CPU Transform N+1、H2D N、GPU Transform N-1 的流水重叠。
- [ ] 区分 inter-sample 与 intra-op parallelism，避免 Rayon 与外层 workers 过度订阅。
- [ ] 测试 queue ordering、sequence reorder、错误传播、worker panic、取消和 backpressure。
- [ ] 测量 decode/transform/transfer/device stage wait 与 action time。

验收门槛：

- 在固定 item/byte budget 下吞吐稳定，内存不随运行时长无界增长。
- 关闭任一 stage 或发生错误时，剩余 worker 和 queue 能安全退出。

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
