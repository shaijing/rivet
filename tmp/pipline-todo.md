# Rivet Pipeline 架构改进指南

> 目标：将当前 `rivet-vision` 中逐步增长的 pipeline 编译与执行逻辑，重构为可复用于 `rivet-vision`、`rivet-text`、`rivet-audio` 等领域的通用计划层和执行层，同时为 CPU、CUDA、未来 Metal 提供统一的逻辑规划、设备 placement、物理 lowering、流式执行和内存优化能力。

---

# Phase 索引

- **Phase 0：目标结构与 crate 边界**：原章节 1, 2, 3
- **Phase 1：Logical IR、属性与领域扩展**：原章节 4, 5, 6, 7, 41, 46
- **Phase 2：语义改写、融合与优化器**：原章节 8, 9, 10, 42, 43
- **Phase 3：设备 Placement 与成本模型**：原章节 11, 12, 13, 14, 15, 34, 35, 36
- **Phase 4：Physical Graph 与执行 Runtime**：原章节 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 47, 48
- **Phase 5：Cache、内存规划与输出分配**：原章节 30, 31, 32
- **Phase 6：Backend 能力与设备支持**：原章节 16, 44, 45, 50
- **Phase 7：迁移与实施顺序**：原章节 40, 49, 55
- **Phase 8：Explain、性能与正确性验证**：原章节 37, 38, 39, 51, 52, 53
- **Phase 9：边界约束与目标架构收束**：原章节 54, 56, 58
- **Phase 10：当前 IR 编译器重构细则**：原章节 57

---

# Phase 0：目标结构与 crate 边界

# 1. 总体目标

Rivet 不应继续演化成：

```text
ImagePipeline
  -> Vec<ImageOp>
  -> compile_image_ops()
  -> sample_ops + batch_ops
  -> worker pool
```

而应逐步演化成：

```text
Domain DSL
  |
  v
rivet-plan
  Logical IR
  Property Inference
  Rewrite / Fusion
  Placement
  Optimized Logical IR
  |
  v
rivet-exec
  Physical Planner
  Physical Graph
  Scheduler
  Memory Planner
  Streaming Runtime
  |
  v
rivet-core
  Tensor / Storage / Device
  CPU / CUDA / future Metal
```

其中：

```text
rivet-vision
rivet-text
rivet-audio
```

负责领域语义；

```text
rivet-plan
```

负责“应该怎么执行”；

```text
rivet-exec
```

负责“什么时候、在哪个执行 lane 上执行”；

```text
rivet-core
```

负责 Tensor 和设备 backend 的底层执行能力。

核心原则：

> semantic op 不决定 backend；optimizer 决定计划；physical planner 决定具体 kernel 和设备；runtime 只执行已经决定好的 physical graph。

---

# 2. 推荐 crate 边界

最终建议：

```text
crates/
├── rivet-core
├── rivet-data
├── rivet-plan
├── rivet-exec
├── rivet-vision
├── rivet-text        # future
└── rivet-audio       # future
```

## 2.1 rivet-core

职责：

```text
Tensor
Storage
Layout
DType
Device
CPU backend
CUDA backend
Metal backend (future)
```

它不应该知道：

```text
Resize
Tokenize
MelSpectrogram
Dataset
Pipeline
Worker
```

典型结构：

```text
Tensor
  ├── Arc<Storage>
  ├── Layout
  ├── DType
  └── Device

Storage
  ├── Cpu(CpuStorage)
  ├── Cuda(CudaStorage)
  └── Metal(MetalStorage)     # future
```

---

## 2.2 rivet-data

职责：

```text
Dataset abstraction
Source
Sampler
Index stream
shuffle / take / skip
Lance / filesystem / memory source
```

不要让这里承担：

```text
CUDA image preprocessing
physical scheduling
optimizer
```

---

## 2.3 rivet-plan

职责：

```text
Logical IR
Arena IR
Property system
Validation
Semantic rewrite
Pushdown
Fusion planning
Placement
Cost model
Optimized logical plan
Explain logical plan
```

关键点：

> rivet-plan 不执行 Tensor 计算。

它只产生一个“已经确定设备 placement、数据属性和 fusion group 的优化计划”。

---

## 2.4 rivet-exec

职责：

```text
Logical -> Physical lowering
Physical graph
Physical kernels
Morsel
Execution lanes
Scheduler
Worker pools
CPU/GPU overlap
Transfer scheduling
Memory planner
Buffer pool
Cache lifecycle
Profiler
Explain physical plan
```

它允许依赖：

```text
rivet-plan
rivet-core
rivet-data
```

而领域 crate 可以向它注册 physical implementation。

---

## 2.5 rivet-vision / text / audio

领域 crate 负责：

```text
semantic ops
领域-specific properties
领域-specific legality rules
领域-specific rewrite / fusion candidates
backend kernel implementation
```

例如 vision：

```text
Decode
Resize
Crop
Normalize
ColorJitter
RandomResizedCrop
```

text：

```text
Tokenize
Truncate
Pad
Pack
Mask
```

audio：

```text
Decode
Resample
STFT
Spectrogram
MelFilter
```

但是它们不应该重复实现：

```text
Arena
placement search
transfer node
scheduler
buffer lifetime
worker runtime
profiling
```

---

# 3. 从当前 pipeline 到目标结构

当前：

```text
ImagePipeline
  source
  index_ops
  Vec<ImageOp>
  batch
  runtime
```

建议暂时保留用户 API。

也就是说第一阶段不需要重写 builder API。

用户仍然：

```rust
pipeline
    .decode_image()
    .resize(224, 224)
    .random_horizontal_flip(0.5)
    .normalize(mean, std)
    .hwc_to_chw()
    .batch(256)
```

但 compile 路径变为：

```text
ImagePipeline
   |
   v
Domain DSL lowering
   |
   v
rivet-plan::LogicalPlan
   |
   v
Arena IR
   |
   v
Optimizer
   |
   v
Optimized IR
   |
   v
rivet-exec::PhysicalPlanner
   |
   v
PhysicalGraph
```

这是最低风险迁移方式。

---

# Phase 1：Logical IR、属性与领域扩展

# 4. rivet-plan：Logical IR

不要让 logical IR 出现 backend-specific kernel。

错误：

```rust
enum IR {
    CpuResize,
    CudaResize,
    MetalResize,
}
```

正确：

```rust
enum PlanNodeKind {
    Source(SourceNode),
    Transform(OpId),
    Batch(BatchSpec),
    Cache(CacheSpec),
    Sink(SinkSpec),
}
```

或者 domain 为 op 提供一个 opaque semantic payload：

```rust
pub struct LogicalNode {
    pub kind: LogicalNodeKind,
    pub inputs: SmallVec<[NodeId; 2]>,
}

pub enum LogicalNodeKind {
    Source(SourcePlan),
    Op(PlanOp),
    Batch(BatchSpec),
    Cache(CacheSpec),
    Sink(SinkSpec),
}
```

其中：

```rust
pub struct PlanOp {
    pub domain: DomainId,
    pub op_id: OpId,
    pub payload: Arc<dyn Any + Send + Sync>,
}
```

不过不要过早做过度动态化。

第一版更简单：

```rust
pub enum DomainOp {
    Vision(VisionOp),
}
```

以后真正增加 text/audio 再决定是否做 registry / type erasure。

---

# 5. Arena IR

参考 Polars，不建议 optimizer 一直操作递归 `Arc` tree。

建议：

```rust
pub type NodeId = usize;

pub struct PlanArena {
    nodes: Vec<LogicalNode>,
}

pub struct LogicalPlan {
    pub root: NodeId,
    pub arena: PlanArena,
}
```

优势：

```text
节点替换简单
parent/child traversal 简单
CSE 简单
fusion 简单
属性缓存简单
placement annotation 简单
explain 简单
```

例如：

```text
Node 0 Source
Node 1 Decode(0)
Node 2 Resize(1)
Node 3 Normalize(2)
Node 4 Layout(3)
Node 5 Batch(4)
Node 6 Sink(5)
```

rewrite 后只需要改 node linkage。

---

# 6. Property System：替代 PipelineImageState

当前：

```rust
PipelineImageState::Encoded
PipelineImageState::Decoded {
    dtype,
    axis_order,
}
```

对于 CPU/CUDA/Metal 已经不够。

建议 rivet-plan 只定义通用 property container：

```rust
pub struct ValueProperties {
    pub representation: RepresentationId,
    pub dtype: Option<DType>,
    pub shape: ShapeProperty,
    pub layout: LayoutProperty,
    pub residency: Residency,
    pub granularity: Granularity,
    pub mutability: MutabilityProperty,
}
```

## 6.1 Shape

```rust
pub enum DimProperty {
    Known(usize),
    Dynamic,
}

pub struct ShapeProperty {
    pub dims: Vec<DimProperty>,
}
```

vision 例子：

```text
Decode:
  [H?, W?, 3]

Resize(224,224):
  [224,224,3]

Batch(256):
  [256,224,224,3]
```

---

## 6.2 Layout

```rust
pub struct LayoutProperty {
    pub axis_order: AxisOrderId,
    pub contiguous: Contiguity,
}

pub enum Contiguity {
    Contiguous,
    Strided,
    Unknown,
}
```

---

## 6.3 Residency

Logical property 不应该直接保存 `Arc<CudaDevice>`。

建议：

```rust
pub enum Residency {
    Host,
    Device(DeviceClass),
    Unknown,
}

pub enum DeviceClass {
    Cpu,
    Cuda,
    Metal,
}
```

具体 ordinal 属于 physical planning context：

```text
CUDA class -> CUDA:0
Metal class -> Metal:0
```

---

## 6.4 Granularity

```rust
pub enum Granularity {
    Sample,
    Batch,
    Stream,
}
```

注意：

> logical op 的语义粒度和 physical kernel 粒度不要强绑定。

例如 semantic `RandomCrop` 是 sample semantic，CUDA lowering 可以变成 batch kernel。

---

# 7. Domain Property Inference

`rivet-plan` 提供 framework：

```rust
pub trait PropertyInference {
    fn infer(
        &self,
        op: &PlanOp,
        inputs: &[ValueProperties],
    ) -> Result<ValueProperties>;
}
```

vision 实现：

```text
Decode:
  Encoded -> U8 HWC

Resize:
  U8 HWC -> U8 HWC

Normalize:
  U8/F32 -> F32

Layout:
  HWC -> CHW
```

text/audio 各自实现。

这里同时完成：

```text
state validation
shape inference
dtype checking
layout checking
```

---

# 41. Domain Op 注册体系：vision / text / audio 如何接入统一 IR

这里是整个多 domain 架构最重要的扩展点之一。

目标必须是：

```text
rivet-plan 不直接依赖：
    rivet-vision
    rivet-text
    rivet-audio
```

否则一旦 central IR 写成：

```rust
enum LogicalOp {
    VisionResize(...),
    VisionCrop(...),
    TextTokenize(...),
    AudioResample(...),
}
```

每增加一个 domain 都必须修改 `rivet-plan`，并且产生反向依赖。

正确方向应该是：

```text
rivet-vision ──┐
rivet-text   ──┼── implements / registers domain planning capabilities
rivet-audio  ──┘
                    |
                    v
                rivet-plan
          generic arena IR / optimizer
                    |
                    v
                rivet-exec
          generic physical runtime
```

也就是说：

```text
不是把所有 domain op 塞进中央 enum
```

而是：

```text
中央 IR 承载 opaque domain op
每个 domain crate 注册自己的：
    property inference
    semantic rewrite
    fusion
    physical candidates
    lowering
```

## 41.1 rivet-plan 的 LogicalOp 保持极小

建议：

```rust
pub type NodeId = u32;

pub struct PlanNode {
    pub op: LogicalOp,
    pub inputs: SmallVec<[NodeId; 2]>,
}

pub enum LogicalOp {
    Source(SourceOp),
    Batch(BatchOp),
    Cache(CacheOp),
    Sink(SinkOp),

    Domain(DomainOpNode),
}
```

其中真正的 domain 扩展点：

```rust
pub struct DomainOpNode {
    pub id: OpId,
    pub op: Arc<dyn DomainOp>,
}
```

这样 arena IR 中可以出现：

```text
Node 10: Domain(vision.resize)
Node 11: Domain(vision.normalize)
Node 20: Domain(text.tokenize)
Node 21: Domain(text.pack)
Node 30: Domain(audio.resample)
```

但 `rivet-plan` 本身不需要知道 Resize / Tokenize / Resample 的 Rust concrete type。

## 41.2 DomainOp trait

第一版建议只暴露 planning 真正需要的接口：

```rust
pub trait DomainOp:
    Send + Sync + std::fmt::Debug
{
    fn op_id(&self) -> OpId;

    fn name(&self) -> &'static str;

    fn infer_properties(
        &self,
        inputs: &[ValueProperties],
    ) -> PlanResult<ValueProperties>;

    fn as_any(&self) -> &dyn std::any::Any;
}
```

注意：

```text
DomainOp 只表达语义
```

它不应该包含：

```text
CudaStream
Metal command queue
worker count
physical kernel
execution thread
```

例如 vision：

```rust
pub struct ResizeOp {
    pub width: usize,
    pub height: usize,
    pub interpolation: InterpolationMode,
}

impl DomainOp for ResizeOp {
    fn op_id(&self) -> OpId {
        VISION_RESIZE
    }

    fn name(&self) -> &'static str {
        "Resize"
    }

    fn infer_properties(
        &self,
        inputs: &[ValueProperties],
    ) -> PlanResult<ValueProperties> {
        // validate decoded image
        // update spatial shape
        // preserve dtype / device as logical properties
        todo!()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
```

Text：

```rust
pub struct TokenizeOp {
    pub tokenizer: TokenizerId,
    pub max_length: Option<usize>,
}
```

Audio：

```rust
pub struct ResampleOp {
    pub target_sample_rate: u32,
}
```

三个 domain 都通过同一个 `DomainOp` 协议进入 IR。

## 41.3 不要把 ValueProperties 再做成中央 domain enum

同样不要写：

```rust
pub enum ValueProperties {
    Vision(VisionProperties),
    Text(TextProperties),
    Audio(AudioProperties),
}
```

否则增加 `rivet-video` 时又必须修改 `rivet-plan`。

推荐：

```rust
pub struct ValueProperties {
    pub common: CommonProperties,
    pub domain: Arc<dyn DomainProperties>,
}
```

公共属性：

```rust
pub struct CommonProperties {
    pub device: DevicePlacement,
    pub granularity: Granularity,
    pub dtype: Option<DType>,
    pub shape: ShapeProperties,
    pub contiguous: Contiguity,
}
```

Domain properties：

```rust
pub trait DomainProperties:
    Send + Sync + std::fmt::Debug
{
    fn domain_id(&self) -> DomainId;
    fn as_any(&self) -> &dyn Any;
}
```

Vision：

```rust
pub struct VisionProperties {
    pub representation: VisionRepresentation,
    pub axis_order: Option<ImageAxisOrder>,
    pub color_space: Option<ColorSpace>,
}
```

Text：

```rust
pub struct TextProperties {
    pub representation: TextRepresentation,
    pub length: LengthProperty,
}
```

Audio：

```rust
pub struct AudioProperties {
    pub representation: AudioRepresentation,
    pub sample_rate: Option<u32>,
    pub channels: Option<usize>,
}
```

这样 generic optimizer 能理解：

```text
device
dtype
shape
granularity
contiguity
```

而 domain optimizer 才理解：

```text
HWC / NCHW
TokenIds / RaggedTokens
Waveform / Spectrogram
```

## 41.4 OpId / DomainId

不要在 optimizer hot planning path 上大量比较字符串。

建议注册阶段分配稳定 ID：

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DomainId(u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpId(u32);
```

例如：

```text
vision -> DomainId(1)
text   -> DomainId(2)
audio  -> DomainId(3)

vision.resize    -> OpId(...)
vision.normalize -> OpId(...)
text.tokenize    -> OpId(...)
text.pack        -> OpId(...)
audio.resample   -> OpId(...)
```

Debug / explain 中仍显示：

```text
vision.resize
text.tokenize
audio.resample
```

但内部 lookup 使用 `OpId`。

## 41.5 PlanRegistry

真正需要注册的不是一个东西，而至少是三类 capability：

```rust
pub struct PlanRegistry {
    ops: OpRegistry,
    rewrites: RewriteRegistry,
    physical: PhysicalRegistry,
}
```

分别负责：

```text
OpRegistry
    op metadata / property hooks

RewriteRegistry
    semantic simplification / fusion / pushdown rules

PhysicalRegistry
    CPU / CUDA / Metal implementation candidates
```

## 41.6 Domain Plugin

建议每个 domain crate 提供显式 plugin：

```rust
pub trait PlanPlugin {
    fn domain(&self) -> &'static str;

    fn register(
        &self,
        registry: &mut PlanRegistry,
    );
}
```

Vision：

```rust
pub struct VisionPlugin;

impl PlanPlugin for VisionPlugin {
    fn domain(&self) -> &'static str {
        "vision"
    }

    fn register(&self, registry: &mut PlanRegistry) {
        register_vision_ops(registry);
        register_vision_rewrites(registry);
        register_vision_physical_impls(registry);
    }
}
```

Text：

```rust
pub struct TextPlugin;

impl PlanPlugin for TextPlugin {
    fn domain(&self) -> &'static str {
        "text"
    }

    fn register(&self, registry: &mut PlanRegistry) {
        register_text_ops(registry);
        register_text_rewrites(registry);
        register_text_physical_impls(registry);
    }
}
```

Audio 同理。

应用初始化：

```rust
let planner = PlannerBuilder::new()
    .plugin(VisionPlugin)
    .plugin(TextPlugin)
    .plugin(AudioPlugin)
    .build()?;
```

Python extension 也可以根据 feature 显式组装：

```text
vision feature enabled -> register VisionPlugin
text feature enabled   -> register TextPlugin
audio feature enabled  -> register AudioPlugin
```

## 41.7 不推荐全局自动注册

第一版不要依赖：

```text
inventory
linkme
lazy_static global registry
全局 mutable singleton
```

显式 registry 的优点：

```text
feature 可控
测试隔离简单
不同 planner 可使用不同能力集合
不会产生隐藏初始化顺序
Python extension 初始化更清楚
CUDA / Metal feature boundary 更清楚
```

## 41.8 Rewrite Rule 注册

Domain-specific optimizer rule 也注册进去，而不是写死进 `rivet-plan`。

通用接口：

```rust
pub trait RewriteRule: Send + Sync {
    fn name(&self) -> &'static str;

    fn apply(
        &self,
        node: NodeId,
        plan: &mut PlanArena,
        ctx: &OptimizerContext,
    ) -> PlanResult<RewriteResult>;
}
```

Vision：

```text
ConvertDtype
    + Normalize
    + Layout

=> FusedNormalizeToNchw
```

注册：

```rust
registry.rewrites.register(
    Box::new(FuseNormalizeLayout),
);
```

Text：

```text
Template
    + Tokenize
    + Truncate

=> TokenizeTemplate(max_length)
```

Audio：

```text
STFT
    + MelFilter

=> MelSpectrogram
```

Generic optimizer 则继续由 `rivet-plan` 自己提供：

```text
dead-node elimination
identity elimination
transfer elimination
transfer coalescing
common-subplan analysis
cache planning
device placement
batch placement
liveness metadata
```

原则：

```text
generic optimizer 不应该到处 downcast ResizeOp / TokenizeOp
```

如果 generic optimizer 大量：

```rust
op.as_any().downcast_ref::<ResizeOp>()
```

说明 domain boundary 已经被破坏。

## 41.9 Physical implementation 也通过 registry 注册

不要把 physical planner 写成：

```rust
match op {
    Resize => {
        if cuda { ... }
        else if metal { ... }
    }
    Tokenize => ...
}
```

否则每个 backend/domain 都会让 central planner 膨胀。

推荐：

```rust
pub trait PhysicalLowering: Send + Sync {
    fn op_id(&self) -> OpId;

    fn backend(&self) -> BackendKind;

    fn supports(
        &self,
        op: &dyn DomainOp,
        inputs: &[ValueProperties],
        ctx: &PlanningContext,
    ) -> bool;

    fn estimate_cost(
        &self,
        op: &dyn DomainOp,
        inputs: &[ValueProperties],
        ctx: &PlanningContext,
    ) -> CostEstimate;

    fn lower(
        &self,
        op: &dyn DomainOp,
        inputs: &[PhysNodeId],
        ctx: &mut LoweringContext,
    ) -> ExecResult<PhysNodeId>;
}
```

同一个 logical op 可以注册多个候选：

```text
vision.resize
    CpuResizeLowering
    CudaResizeLowering
    MetalResizeLowering

vision.normalize
    CpuNormalizeLowering
    CudaNormalizeLowering
    MetalNormalizeLowering

text.tokenize
    CpuTokenizerLowering

text.pad
    CpuPadLowering
    CudaPadLowering
    MetalPadLowering

audio.resample
    CpuResampleLowering
    CudaResampleLowering
    MetalResampleLowering
```

placement planner 的工作就是：

```text
1. 查询当前 OpId 的所有 physical candidates
2. 根据 input properties 检查 supports()
3. 计算 transfer cost + kernel cost + downstream target cost
4. 选择 candidate
5. 必要时插入 Transfer physical node
6. 调用 candidate.lower()
```

## 41.10 as_any 只允许 domain lowering 使用

例如：

```rust
impl PhysicalLowering for CudaResizeLowering {
    fn lower(...) -> Result<PhysNodeId> {
        let resize = op
            .as_any()
            .downcast_ref::<ResizeOp>()
            .ok_or(...)?;

        ...
    }
}
```

这是合理的，因为 `CudaResizeLowering` 本来就是 vision-specific implementation。

但：

```text
rivet-plan generic optimizer
rivet-exec generic scheduler
```

不应该知道 `ResizeOp` concrete type。

## 41.11 Planning 使用 trait object，runtime hot path 不受影响

`dyn DomainOp` / `dyn RewriteRule` / `dyn PhysicalLowering` 只发生在：

```text
plan construction
property inference
optimizer
placement
physical lowering
```

不发生在：

```text
每个 pixel
每个 token
每个 audio sample
```

所以这一层 virtual dispatch 成本完全可以接受。

Tensor hot kernel 仍保持 `Storage::{Cpu,Cuda,Metal}` enum/static typed dispatch。

## 41.12 rivet-exec 可以使用 Box<dyn PhysicalOperator>

pipeline physical operator 的 dispatch 粒度是：

```text
一个 morsel
一个 batch
一个 stage
```

不是 element-level。

所以可以非常通用：

```rust
pub trait PhysicalOperator: Send {
    fn name(&self) -> &'static str;

    fn process(
        &mut self,
        input: Morsel,
        ctx: &mut ExecutionContext,
    ) -> ExecResult<Morsel>;
}
```

具体实现：

```text
rivet-vision
    CpuResizeExec
    CudaNormalizeExec

rivet-text
    TokenizerExec
    PackExec

rivet-audio
    ResampleExec
    MelSpectrogramExec
```

`rivet-exec` 只看到：

```text
Box<dyn PhysicalOperator>
```

因此不需要依赖任何 domain crate。

## 41.13 完整 Vision 示例

DSL：

```rust
VisionPipeline::new(source)
    .decode()
    .resize(224, 224)
    .normalize(mean, std)
    .layout(ImageAxisOrder::Chw)
```

Logical Arena：

```text
Node 0 Source

Node 1 Domain(vision.decode)
    input = 0

Node 2 Domain(vision.resize)
    input = 1

Node 3 Domain(vision.normalize)
    input = 2

Node 4 Domain(vision.layout)
    input = 3
```

Domain optimizer：

```text
vision.normalize
+ vision.layout
=> vision.normalize_to_nchw
```

Physical candidates：

```text
vision.decode
    CPU

vision.resize
    CPU
    CUDA
    Metal

vision.normalize_to_nchw
    CPU batch
    CUDA batch
    Metal batch
```

Planner 最终可能选择：

```text
CpuDecode
    ↓
CpuResize
    ↓
Batch
    ↓
Transfer(CPU -> CUDA0)
    ↓
CudaNormalizeToNchw
```

然后 `rivet-exec` 只负责运行这张 physical graph。

## 41.14 Text 示例

Logical：

```text
Source
 ↓
text.template
 ↓
text.tokenize
 ↓
text.truncate
 ↓
text.pack
 ↓
Batch
```

Text optimizer：

```text
Template
+ Tokenize
+ Truncate

=> TokenizeTemplate(max_length)
```

Physical candidates：

```text
TokenizeTemplate
    CPU

Pack
    CPU stateful executor

Pad
    CPU / CUDA / Metal
```

最终：

```text
ArrowUtf8Source
    ↓
CpuTokenizeTemplate
    ↓
CpuPack
    ↓
TokenBudgetBatcher
    ↓
Transfer(CPU -> CUDA0)
    ↓
CudaPadAndMask
```

## 41.15 Audio 示例

Logical：

```text
Source
 ↓
audio.decode
 ↓
audio.resample
 ↓
audio.stft
 ↓
audio.mel
```

Audio optimizer：

```text
STFT + Mel
=> MelSpectrogram
```

Physical candidates：

```text
Decode
    CPU

Resample
    CPU / CUDA / Metal

MelSpectrogram
    CPU / CUDA / Metal
```

因此 Metal 后端加入时不需要改 logical IR。

## 41.16 依赖方向

必须守住：

```text
rivet-plan
    不依赖 rivet-vision/text/audio

rivet-exec
    不依赖 rivet-vision/text/audio

rivet-vision/text/audio
    依赖 rivet-plan 的 extension protocol
    依赖 rivet-exec 的 PhysicalOperator protocol
    依赖 rivet-core 的 Tensor / Device
```

理想关系：

```text
                    rivet-core
                        ^
                        |
                 +------+------+
                 |             |
             rivet-plan    rivet-data
                 ^             ^
                 |             |
       +---------+---------+   |
       |         |         |   |
    vision      text     audio  |
       |         |         |   |
       +---------+---------+   |
                 |             |
                 v             |
             rivet-exec <------+
```

实际 Cargo 依赖要避免 cycle，可以把 `PhysicalOperator` / `Morsel` 等稳定 protocol 放到 `rivet-exec` 的低层公共模块，domain crate 提供具体实现，而最终应用层负责 plugin 装配。

更严格的做法是：

```text
rivet-plan
    planning protocols

rivet-exec
    execution protocols + runtime

rivet-vision/text/audio
    domain plugin + domain physical operator implementations

rivet-python / top-level rivet crate
    composition root
    注册所有启用的 plugins
```

这样中央 crate 永远不知道具体 domain。

## 41.17 Composition Root

Plugin 注册最好集中在最高层：

```rust
pub fn default_planner() -> Planner {
    let mut builder = PlannerBuilder::new();

    #[cfg(feature = "vision")]
    builder = builder.plugin(rivet_vision::VisionPlugin);

    #[cfg(feature = "text")]
    builder = builder.plugin(rivet_text::TextPlugin);

    #[cfg(feature = "audio")]
    builder = builder.plugin(rivet_audio::AudioPlugin);

    builder.build()
}
```

以后：

```text
rivet-video
rivet-pointcloud
rivet-genomics
```

都只是增加 plugin，不修改 `rivet-plan::LogicalOp`。

## 41.18 最终规则

这一部分应该形成非常明确的架构约束：

```text
LogicalOp
    只保存通用 control/dataflow 节点
    + opaque DomainOp

DomainOp
    描述 domain semantic

DomainProperties
    描述 domain-specific state

RewriteRegistry
    注册 domain semantic optimization

PhysicalRegistry
    注册 CPU / CUDA / Metal candidates

Placement Planner
    在 candidates 中选择 physical implementation

rivet-exec
    只执行已经 lower 完成的 physical graph
```

最终效果：

```text
rivet-vision
rivet-text
rivet-audio
rivet-video (future)
       |
       v
  same rivet-plan IR
       |
       v
 same generic optimizer framework
       |
       v
 CPU / CUDA / Metal placement
       |
       v
 same rivet-exec runtime
```

这比中央大 enum 更适合作为 Rivet 长期的 domain extension model。

---

# 46. 推荐 rivet-plan 目录

```text
rivet-plan/src/
├── lib.rs
├── arena.rs
├── node.rs
├── plan.rs
├── property/
│   ├── mod.rs
│   ├── shape.rs
│   ├── layout.rs
│   └── device.rs
├── optimizer/
│   ├── mod.rs
│   ├── canonicalize.rs
│   ├── simplify.rs
│   ├── pushdown.rs
│   ├── fusion.rs
│   └── fixed_point.rs
├── placement/
│   ├── mod.rs
│   ├── candidate.rs
│   ├── cost.rs
│   └── linear_dp.rs
├── capability.rs
└── explain.rs
```

---

# Phase 2：语义改写、融合与优化器

# 8. Optimizer phase 拆分

当前 `compile.rs` 同时承担：

```text
validation
state transition
sample/batch placement
fusion
random key assignment
```

需要拆开。

建议固定 phase：

```text
1. DSL lowering
2. canonicalization
3. property inference
4. semantic rewrite
5. pushdown
6. fusion discovery
7. placement
8. transfer insertion
9. physical lowering
```

不要在一个大 match 里全部完成。

---

# 9. Semantic Rewrite

第一批高 ROI 规则：

## 9.1 Late dtype promotion

```text
Convert(U8 -> F32)
  -> Resize
```

如果语义允许：

```text
Resize(U8)
  -> Convert(F32)
```

减少大图 F32 memory traffic。

---

## 9.2 Crop/Resize 提前

```text
Normalize
 -> Crop
```

若数学语义等价则变成：

```text
Crop
 -> Normalize
```

原则：

> 尽可能先降低空间尺寸，再提升 dtype 或做逐像素昂贵处理。

---

## 9.3 Index pushdown

```text
Skip / Take / Shuffle
```

尽量下推到：

```text
Source
```

之前完成 selection，避免无用：

```text
IO
decode
H2D
transform
```

---

# 10. Fusion

Fusion 应该在 logical optimizer 中先发现候选，在 physical planner 中选择是否真正 lower 成 fused kernel。

不要 logical IR 直接变成 CUDA-specific fusion。

例如：

```text
ConvertDtype
 -> Normalize
 -> Layout
```

logical fusion group：

```text
FusionGroup {
    ops: [ConvertDtype, Normalize, Layout]
}
```

physical planner 可以选择：

```text
CPU:
  FusedNormalizeToChw

CUDA:
  CudaNormalizeToNchwBatch

Metal:
  MetalNormalizeToNchwBatch
```

---

## 10.1 推荐优先 fusion

```text
Convert + Normalize
Normalize + Layout
Convert + Normalize + Layout
Crop + Resize
Flip + Normalize
Crop + Resize + Normalize + Layout
```

不要为了 fusion 而 fusion。

重要指标：

```text
减少 full tensor pass
减少 allocation
减少 device launch
减少 temporary buffer
```

---

# 42. Optimizer Fixed Point

一些 rewrite 可以迭代到稳定：

```text
simplify
 -> pushdown
 -> fusion discovery
 -> simplify again
```

例如：

```text
Layout(HWC->HWC)
```

消除后可能使：

```text
Normalize + Layout
```

fusion 结构变化。

建议：

```rust
for _ in 0..MAX_PASSES {
    let changed = run_rewrite_passes(...)?;
    if !changed {
        break;
    }
}
```

但是 placement 最好在 semantic optimization 基本稳定后做。

---

# 43. 推荐优化顺序

```text
Canonicalization
  ↓
Validation / property inference
  ↓
No-op elimination
  ↓
Pushdown
  ↓
Semantic reorder
  ↓
Fusion discovery
  ↓
Placement
  ↓
Transfer insertion
  ↓
Physical lowering
  ↓
Memory planning
```

重要：

> 先语义优化，再 device placement。

否则 placement 后改写会导致大量重新规划。

---

# Phase 3：设备 Placement 与成本模型

# 11. Placement：rivet-plan 的核心

当前 `ExecutionKind::{Sample,Batch}` 不应继续作为主要 placement abstraction。

建议：

```rust
pub struct PlacementCandidate {
    pub device: DeviceClass,
    pub granularity: Granularity,
    pub kernel: KernelClass,
    pub requirements: KernelRequirements,
    pub cost: CostEstimate,
}
```

例如 Normalize：

```text
Candidate 1:
CPU + Sample

Candidate 2:
CPU + Batch

Candidate 3:
CUDA + Batch

Candidate 4:
Metal + Batch
```

optimizer/planner 根据整条 pipeline 选择。

---

# 12. Backend Capability Registry

不要在 optimizer 中写大量：

```rust
if cuda && op == Normalize { ... }
```

建议：

```rust
pub struct KernelCapabilities {
    implementations: Vec<KernelCapability>,
}

pub struct KernelCapability {
    pub op_kind: OpKind,
    pub device: DeviceClass,
    pub input_constraints: PropertyConstraints,
    pub output_constraints: PropertyConstraints,
    pub granularity: Granularity,
    pub fusion_tags: Vec<FusionTag>,
    pub cost_hint: KernelCostHint,
}
```

例如：

```text
NormalizeToNchw

CPU:
  U8 NHWC -> F32 NCHW
  batch

CUDA:
  U8 NHWC -> F32 NCHW
  batch

Metal:
  U8 NHWC -> F32 NCHW
  batch
```

这样 Metal 后续加入时：

```text
Logical IR 不变
Optimizer framework 不变
Physical graph 不变
```

只增加 capability + backend lowering。

---

# 13. Cost Model

第一版不要做复杂 ML-based/autotuning cost model。

数据 pipeline 更应该重点估计：

```text
bytes read
bytes written
allocation count
CPU/GPU transfer
kernel launches
synchronization
estimated compute
```

建议：

```rust
pub struct CostEstimate {
    pub host_bytes_read: u64,
    pub host_bytes_written: u64,
    pub device_bytes_read: u64,
    pub device_bytes_written: u64,
    pub transfer_bytes: u64,
    pub allocations: u32,
    pub kernel_launches: u32,
    pub sync_points: u32,
    pub compute_score: f64,
}
```

然后：

```rust
fn score(cost: &CostEstimate, machine: &MachineProfile) -> f64
```

---

# 14. Placement 不应逐 op 贪心

错误：

```text
Resize CPU 比 CUDA 快 -> CPU
Normalize CUDA 比 CPU 快 -> CUDA
Color CPU 比 CUDA 快 -> CPU
```

得到：

```text
CPU Resize
H2D
CUDA Normalize
D2H
CPU Color
H2D
CUDA sink
```

这是灾难。

正确目标：

> 尽量形成长的 same-device region。

例如：

```text
CPU:
  Decode
  Resize
  Color

boundary:
  H2D once

CUDA:
  Normalize
  Layout
  Output
```

或：

```text
CPU:
  Decode

H2D

CUDA:
  Resize
  Crop
  Flip
  Normalize
  Layout
```

所以 placement 应是 pipeline-level optimization。

---

# 15. Transfer 是显式 plan node

绝对不要把隐式 `to_device()` 藏在 CUDA kernel 内部。

Physical graph 必须看到：

```rust
pub enum TransferKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}
```

节点：

```rust
PhysNode::Transfer {
    input,
    from,
    to,
    kind,
}
```

原因：

```text
optimizer 才能计算 transfer cost
profiler 才能显示 transfer
memory planner 才能分配 pinned buffer
scheduler 才能 overlap H2D 和 compute
```

第一版建议限制：

```text
CPU* -> CUDA*
```

最多一个主要 H2D boundary。

不要一开始允许任意：

```text
CPU -> CUDA -> CPU -> CUDA
```

可以把反复 device crossing 设置成极高 cost。

---

# 34. MachineProfile

为了让 placement 不写死：

```rust
pub struct MachineProfile {
    pub cpu_threads: usize,
    pub devices: Vec<DeviceDescriptor>,
    pub h2d_bandwidth: Option<f64>,
    pub d2h_bandwidth: Option<f64>,
}
```

以后 benchmark/autotune 可以填更多：

```text
CPU resize throughput
CUDA resize throughput
H2D latency
kernel launch latency
```

第一版先用静态 heuristic 即可。

---

# 35. 第一个 placement heuristic

不用一开始上动态规划。

建议：

1. 找到最终 Sink device。
2. 从后往前确定 device-friendly region。
3. 尽可能减少 transfer boundary。
4. 对 CPU-only op 固定 CPU。
5. 对 GPU-friendly batch op 优先和 sink 同 device。
6. 大 resize/crop 若能显著减少 transfer bytes，可优先 CPU。
7. 避免 Device -> Host -> Device。

例如：

```text
Decode CPU-only
Resize CPU/CUDA
Normalize CUDA-friendly
Sink CUDA
```

如果原图很大：

```text
Decode CPU
Resize CPU
H2D small image
Normalize CUDA
Sink
```

如果 source 已经 GPU resident：

```text
Resize CUDA
Normalize CUDA
Sink
```

---

# 36. 长期 Placement 可变成图优化问题

每个 logical op 有多个 physical states：

```text
(op i, CPU)
(op i, CUDA)
(op i, Metal)
```

边 cost：

```text
same device -> compute cost
cross device -> transfer cost + compute cost
```

pipeline 是线性时可直接 dynamic programming 求最低 cost。

这是非常适合 Rivet 的一个实现。

例如：

```text
DP[i][device] =
  min_prev(
    DP[i-1][prev_device]
    + transfer(prev_device, device)
    + kernel_cost(i, device)
  )
```

等以后有 DAG/cache 分支，再扩展 heuristic。

---

# Phase 4：Physical Graph 与执行 Runtime

# 17. rivet-exec：Physical Graph

建议：

```rust
pub type PhysNodeId = usize;

pub struct PhysicalGraph {
    pub nodes: Vec<PhysNode>,
    pub root: PhysNodeId,
}
```

节点类型：

```rust
pub enum PhysNodeKind {
    Source,
    CpuKernel,
    DeviceKernel,
    Batch,
    Transfer,
    Cache,
    Sink,
}
```

不要让它知道 `Resize` 等 semantic name 必须怎么执行；具体 kernel payload 可以由 domain backend 提供。

---

# 18. Physical Kernel

建议 execution layer 看到的是：

```rust
pub enum PhysicalKernel {
    Cpu(CpuKernelHandle),
    Device(DeviceKernelHandle),
}
```

例如 vision lowering：

```text
Logical Normalize+Layout

CPU candidate
 -> CpuVisionKernel::NormalizeToChw

CUDA candidate
 -> CudaVisionKernel::NormalizeToNchw

Metal candidate
 -> MetalVisionKernel::NormalizeToNchw
```

---

# 19. Morsel：统一执行数据单位

建议 `rivet-exec` 引入类似 Polars streaming 的 morsel。

```rust
pub struct Morsel {
    pub sequence: SequenceId,
    pub data: ExecutionValue,
    pub metadata: MorselMetadata,
}
```

```rust
pub enum ExecutionValue {
    Samples(SampleBatch),
    Tensor(Tensor),
    Domain(Box<dyn Any + Send>),
}
```

第一版不要追求过度通用。

可以先：

```rust
pub enum ExecutionValue {
    EncodedSamples(Vec<EncodedImageSample>),
    DecodedSamples(Vec<DecodedSample>),
    Tensor(Tensor),
}
```

后面 text/audio 加入时再抽 domain payload。

核心理念：

```text
source
  -> morsel
transform
  -> morsel
batch
  -> morsel
transfer
  -> morsel
sink
```

---

# 20. Runtime Stage Graph

从当前：

```text
source.get_many
 -> worker pool
 -> batch builder
 -> batch ops
```

升级：

```text
Sampler
  |
  v
Source Stage
  |
  v
Decode Stage
  |
  v
CPU Transform Stage
  |
  v
Batch Stage
  |
  v
Transfer Stage
  |
  v
Device Transform Stage
  |
  v
Sink
```

每条边：

```text
bounded queue
```

形成 backpressure。

---

# 21. 为什么 Streaming Graph 比单 worker pool 更重要

真正高性能 pipeline 需要 overlap：

```text
CPU decode batch N+2
CPU resize batch N+1
H2D batch N
GPU normalize batch N-1
trainer batch N-2
```

而不是：

```text
read
wait
transform
wait
copy
wait
GPU
wait
```

因此 scheduler 的目标不是“启动更多线程”，而是：

> 让不同资源域同时工作，同时用 bounded queues 控制内存。

---

# 22. Execution Lanes

建议 scheduler 抽象：

```rust
pub enum ExecutionLaneKind {
    Io,
    Cpu,
    Transfer,
    Device,
}
```

future：

```text
CPU lane
CUDA transfer lane
CUDA compute lane
Metal command queue
```

而不是 runtime 到处直接写：

```text
CudaStream
```

---

# 23. CUDA stream 策略

你当前 `rivet-core::CudaDevice` 单 default stream 是合理基础。

pipeline engine 后续可以扩展：

```text
transfer stream
compute stream
```

形成：

```text
H2D(N+1) || compute(N)
```

通过 event：

```text
transfer done
   -> compute stream wait
```

Metal 对应：

```text
blit/transfer queue
compute queue
```

因此在 `rivet-exec` 用 `ExecutionLane` 抽象，不直接把 CUDA-specific API 泄露到 plan。

---

# 24. Pinned Host Memory

真正异步 H2D 需要 pinned memory。

建议未来 host memory property：

```rust
pub enum HostMemoryKind {
    Pageable,
    Pinned,
}
```

理想 pipeline：

```text
Decode
  -> pinned U8 output
  -> async H2D
  -> CUDA preprocessing
```

第一阶段可先：

```text
normal CPU output
 -> explicit pinned staging
 -> H2D
```

再优化为 decode 直接写 pinned destination。

---

# 25. CPU Parallelism

必须区分：

```text
inter-sample parallelism
intra-op parallelism
```

错误：

```text
16 workers
 x
每个 Resize 内部 Rayon 16 threads
```

造成 oversubscription。

建议 physical kernel metadata：

```rust
pub enum ParallelismClass {
    Serial,
    AcrossSamples,
    WithinBatch,
    Device,
}
```

scheduler 根据 stage 决定：

```text
sample transforms:
 outer worker parallelism
 kernel mostly single-thread

batch kernel:
 one stage invocation
 kernel internal parallelism
```

---

# 26. GPU Kernel 必须 batch-native

不要：

```text
for sample in batch:
    launch_cuda_resize(sample)
```

应该：

```text
one batch
 -> one/few GPU launches
```

例如 RandomHorizontalFlip：

```text
flip_flags[N]
 -> CudaHorizontalFlipBatch
```

RandomCrop：

```text
crop_x[N]
crop_y[N]
crop_w[N]
crop_h[N]
 -> CudaCropBatch
```

RandomResizedCrop：

```text
params[N]
 -> CudaCropResizeBatch
```

---

# 27. Randomness 与 backend 解耦

保留当前：

```text
global_seed
+ epoch
+ sample index
+ OpKey
```

不要改成 backend-local RNG。

建议：

```text
Semantic Random Op
   |
   v
Deterministic parameter generation
   |
   v
params[N]
   |
   +--> CPU kernel
   +--> CUDA kernel
   +--> Metal kernel
```

这样：

```text
CPU
CUDA
Metal
worker count
```

不会改变 augmentation 决策。

这对 regression test 非常重要。

---

# 28. Source Capability

当前：

```text
source.supports_batch_read()
```

建议升级：

```rust
pub struct SourceCapabilities {
    pub random_access: bool,
    pub batch_read: bool,
    pub zero_copy_batch: bool,
    pub parallel_read: bool,
    pub async_read: bool,
    pub fixed_shape: Option<ShapeProperty>,
    pub encoded: bool,
    pub device_read: Vec<DeviceClass>,
}
```

未来例子：

```text
Lance:
  batch_read = true
  random_access = true

Dense CPU cache:
  zero_copy_batch = true

GPU dataset cache:
  device_read = [CUDA]
```

optimizer 可以根据 capability 选择 source lowering。

---

# 29. Cache 作为 Plan Node

当前 cache 更像 source/runtime feature。

长期建议：

```rust
LogicalNodeKind::Cache {
    input,
    policy,
}
```

特别适合：

```text
Source
  -> Decode
  -> Cache
       |\
       | +--> View A
       +----> View B
```

例如 SimCLR / DINO 多 view。

可做 common subplan elimination：

```text
Decode once
reuse multiple branches
```

---

# 33. Physical Planner 的职责

输入：

```text
Optimized Logical IR
+ Property annotations
+ Placement annotations
+ MachineProfile
+ Backend capabilities
```

输出：

```text
PhysicalGraph
```

主要完成：

```text
选择 kernel implementation
选择 concrete device ordinal
插入 transfer
插入 batch/materialize
lower fusion groups
选择 execution lane
给 memory planner 提供 lifetime graph
```

---

# 47. 推荐 rivet-exec 目录

```text
rivet-exec/src/
├── lib.rs
├── physical/
│   ├── mod.rs
│   ├── graph.rs
│   ├── node.rs
│   └── planner.rs
├── runtime/
│   ├── mod.rs
│   ├── executor.rs
│   ├── scheduler.rs
│   ├── lane.rs
│   ├── morsel.rs
│   └── queue.rs
├── memory/
│   ├── mod.rs
│   ├── lifetime.rs
│   ├── cpu_pool.rs
│   ├── pinned_pool.rs
│   └── device_pool.rs
├── transfer/
│   ├── mod.rs
│   └── node.rs
├── cache/
│   └── mod.rs
├── profiler/
│   └── mod.rs
└── explain.rs
```

第一版不需要一次全部建立。

---

# 48. rivet-vision 迁移后目录

```text
rivet-vision/src/
├── pipeline/
│   ├── builder.rs
│   └── transform.rs
├── plan/
│   ├── op.rs
│   ├── property.rs
│   ├── rewrite.rs
│   └── fusion.rs
├── kernels/
│   ├── cpu/
│   ├── cuda/
│   └── metal/
├── source/
└── codec/
```

`compile.rs` 应逐渐被拆掉。

它目前的职责分别迁移：

```text
state validation
 -> rivet-vision property inference

rewrite/fusion
 -> rivet-vision optimizer hooks

sample/batch placement
 -> rivet-plan placement

ExecutionPlan construction
 -> rivet-exec physical planner

worker scheduling
 -> rivet-exec runtime
```

---

# Phase 5：Cache、内存规划与输出分配

# 30. Memory Planner

GPU pipeline 中 allocation 和 memory pressure 会非常关键。

建议 `rivet-exec` 在 physical graph 上做：

```text
consumer count
last-use analysis
buffer lifetime
reuse class
```

例如：

```text
A -> Resize -> B -> Normalize -> C
```

当 Resize 完成且 A 无其它 consumer：

```text
A buffer can be released
```

固定 shape 的 batch 更适合 pool：

```text
[256,224,224,3] U8
[256,3,224,224] F32
```

建议未来：

```text
CpuAlignedPool
PinnedHostPool
CudaBufferPool
MetalBufferPool
```

---

# 31. Direct Output Allocation

保持现有 CPU hot kernel 原则：

```text
不要 Vec intermediate
直接写最终 aligned output
```

GPU 更应该这样。

坏：

```text
Resize temp
 -> Normalize temp
 -> Layout temp
```

好：

```text
input U8 NHWC
 -> fused kernel
 -> final F32 NCHW buffer
```

这是 pipeline optimizer 最重要的收益之一。

---

# 32. Physical Memory Requirements

kernel capability 建议声明：

```rust
pub struct KernelMemoryRequirements {
    pub input_contiguous: bool,
    pub output_contiguous: bool,
    pub output_alignment: usize,
    pub temporary_bytes: usize,
    pub in_place: bool,
}
```

physical planner 可以据此插入：

```text
contiguous materialization
scratch allocation
```

而不是 kernel 自己偷偷处理。

---

# Phase 6：Backend 能力与设备支持

# 16. 为未来 Metal 做好的边界

planner 不应持有：

```text
CudaStream
CudaContext
CudaDevice
```

应只看到：

```rust
pub struct DeviceDescriptor {
    pub location: DeviceLocation,
    pub class: DeviceClass,
    pub memory_bytes: Option<u64>,
    pub async_copy: bool,
    pub supported_kernels: KernelSet,
}
```

例如：

```text
CPU
CUDA:0
CUDA:1
Metal:0
```

具体 CUDA stream 只存在 `rivet-exec` physical runtime。

---

# 44. CPU/CUDA/Metal 支持策略

## CPU

继续作为完整 fallback backend。

优点：

```text
所有 transform 尽量 CPU 可用
```

---

## CUDA

第一阶段不要求覆盖全部 vision ops。

优先：

```text
Convert + Normalize + Layout
Flip batch
Crop
Resize
CropResize
RandomResizedCrop
```

不支持的 op：

```text
继续留 CPU region
```

---

## Metal

未来遵循完全相同的 physical backend contract。

不修改：

```text
logical IR
optimizer core
runtime graph
```

只增加：

```text
Metal device backend
kernel capabilities
physical lowering
execution lane adapter
```

---

# 45. 不要过早做“万能 backend trait”

Rivet 已经通过 `Storage::{Cpu,Cuda,...}` 做 runtime enum dispatch，这个方向应保持。

对于 planning：

```text
trait 可以描述 capability
```

对于 hot Tensor runtime：

```text
enum dispatch
```

避免：

```text
Arc<dyn BackendStorage>
```

进入底层 hot path。

---

# 50. 推荐的第一条完整 CUDA physical pipeline

建议目标：

```text
Lance / ImageFolder
  |
  v
CPU Read
  |
  v
CPU JPEG Decode
  |
  v
CPU Resize / Crop
  |
  v
Batch U8 NHWC
  |
  v
H2D
  |
  v
CUDA Fused Convert + Normalize + NHWC->NCHW
  |
  v
Tensor F32 NCHW on CUDA
```

这个路径最适合当前 Rivet：

```text
CPU 保留成熟 image decode
GPU 处理最适合 batch/fusion 的部分
只发生一次 H2D
最终 Tensor 已在训练设备上
```

---

# Phase 7：迁移与实施顺序

# 40. 现有 ImageBatchBuilder 的改造

当前 `ImageBatchBuilder` 知道一些 vision-specific layout 假设。

短期：

```text
builder 从 pre-batch properties 获得 axis_order
```

而不是 hardcode HWC。

长期：

```text
rivet-exec::BatchNode
```

处理通用 batching contract，

vision 提供：

```text
如何 stack image samples
```

text 提供：

```text
pad / pack sequence
```

audio 提供：

```text
pad waveform / spectrogram
```

不要尝试所有 domain 完全共用同一个 stack 实现。

---

# 49. 分阶段迁移计划

## Phase 0：冻结当前行为

要求：

```text
现有 130+ tests 全部保留
CPU benchmark 建立基线
```

不要先改 runtime。

---

## Phase 1：创建 rivet-plan

只做：

```text
Arena
LogicalNode
LogicalPlan
Property container
Explain logical
```

现有 `ImagePipeline` lowering 到 LogicalPlan。

然后再 lowering 回现有 `ExecutionPlan`。

即：

```text
ImagePipeline
 -> rivet-plan IR
 -> legacy ExecutionPlan
```

功能不变。

这是最安全第一步。

---

## Phase 2：Property inference

把：

```text
PipelineImageState
```

迁移到 property inference。

至少支持：

```text
representation
dtype
shape
axis order
contiguity
granularity
```

Device 可先全部 CPU。

---

## Phase 3：Generic optimizer pass framework

实现：

```text
No-op elimination
Pushdown
Semantic rewrite
Fusion group
```

先只产生和当前 compiler 相同结果。

Normalize+Layout 现有 fusion 改造成正式 optimizer rule。

---

## Phase 4：Placement

把：

```rust
ImageOp::execution_kind()
```

逐步废除。

引入：

```text
kernel candidates
CPU/CUDA capability
cost
placement decision
```

第一阶段规则可以很简单：

```text
Decode CPU
most augmentation CPU
Normalize/Layout CUDA if output device CUDA
```

---

## Phase 5：创建 rivet-exec

先把现有：

```text
ExecutionPlan
loader
scheduler
batch builder
```

迁移到这里。

一开始 physical graph 可以仍然退化成：

```text
Source
 -> SampleStage
 -> Batch
 -> BatchStage
 -> Sink
```

行为和现在一致。

---

## Phase 6：Explicit Transfer Node

支持：

```text
CPU stage
 -> H2D
 -> CUDA batch stage
```

先只允许单 CUDA device、单 stream。

这是第一条真正 heterogeneous pipeline。

---

## Phase 7：GPU Batch Kernels

优先：

```text
NormalizeToNchw
Flip
CropResize
```

并确保：

```text
one batch -> few launches
```

---

## Phase 8：Persistent Stage Graph

把现有一次 batch 驱动升级为 persistent runtime：

```text
Source thread
CPU workers
Batch stage
Transfer stage
Device stage
```

bounded queues。

---

## Phase 9：Pinned memory + copy/compute overlap

加入：

```text
PinnedHostPool
CUDA transfer lane
CUDA compute lane
Events
```

实现：

```text
H2D(N+1) || compute(N)
```

---

## Phase 10：Memory Planner

加入：

```text
last-use
buffer reuse
cache lifetime
inflight byte budget
```

---

## Phase 11：Metal

如果此前边界正确，主要工作应集中：

```text
rivet-core Metal storage/device
rivet-vision Metal kernels
rivet-exec Metal lane
capability registration
```

而不是修改 optimizer architecture。

---

# 55. 当前最应该开始实现的代码顺序

建议接下来实际代码工作按：

```text
Step 1
创建 rivet-plan crate

Step 2
定义 Arena / NodeId / LogicalPlan

Step 3
ImagePipeline -> LogicalPlan lowering

Step 4
Properties + vision inference

Step 5
Optimizer pass framework

Step 6
把现有 Normalize+Layout fusion 搬成 rewrite/fusion rule

Step 7
KernelCapability + PlacementCandidate

Step 8
CPU/CUDA placement

Step 9
创建 rivet-exec crate

Step 10
把现有 ExecutionPlan/runtime 迁过去

Step 11
显式 Transfer node

Step 12
第一条 CUDA fused NormalizeToNchw physical kernel

Step 13
explain_physical + profiler

Step 14
persistent stage graph
```

这条迁移路线尽量保证每一步都可以编译、测试、benchmark，而不是大爆炸式重写。

---

# Phase 8：Explain、性能与正确性验证

# 37. Explain 系统

必须尽早实现。

建议：

```rust
pipeline.explain()
pipeline.explain_optimized()
pipeline.explain_physical()
```

逻辑计划：

```text
Sink(CUDA)
└── Batch(256)
    └── Layout(NCHW)
        └── Normalize
            └── Resize(224,224)
                └── Decode
                    └── LanceSource
```

优化后：

```text
Sink(CUDA)
└── Batch(256)
    └── FusionGroup[Normalize, Layout]
        └── Resize
            └── Decode
                └── LanceSource
```

physical：

```text
CudaSink[cuda:0]
└── CudaNormalizeToNchwBatch
    └── AsyncH2D[pinned]
        └── CpuBatchStack
            └── CpuResize[8 workers]
                └── CpuJpegDecode[8 workers]
                    └── LanceBatchRead
```

必须显示：

```text
chosen device
chosen kernel
fusion
transfer
expected shape/dtype/layout
parallelism
```

---

# 38. Runtime Profiler

建议每个 PhysNode 记录：

```text
calls
items
input bytes
output bytes
queue wait
action time
allocation bytes
H2D/D2H bytes
kernel launches
```

输出：

```text
LanceBatchRead          1.2 ms
JPEG Decode             4.9 ms
Resize CPU              2.7 ms
Batch Stack             0.6 ms
H2D                     0.5 ms
NormalizeToNchw CUDA    0.2 ms
```

这样才能知道 bottleneck。

---

# 39. Scheduler Backpressure

不要只用：

```text
prefetch_batches = N
```

更高级可以支持 byte budget：

```text
max_inflight_bytes
```

因为：

```text
encoded sample size
raw decoded image size
F32 batch size
```

差异巨大。

建议每条 queue：

```rust
pub struct QueueBudget {
    pub max_items: usize,
    pub max_bytes: Option<usize>,
}
```

---

# 51. 性能优化优先级

按收益优先排序：

```text
1. 避免重复 CPU<->GPU transfer
2. 先缩小空间尺寸，再做 dtype promotion
3. 减少 full-image memory passes
4. operator fusion
5. GPU batch-native kernels
6. CPU/GPU stage overlap
7. pinned memory async transfer
8. buffer reuse / memory pool
9. source batch-native read
10. 单 kernel 微优化
```

不要一开始把大量时间投入单 kernel 极限调优，而 pipeline 仍然串行。

---

# 52. Benchmark 体系

必须分层 benchmark。

## Kernel benchmark

```text
CPU Resize
CUDA Resize
Normalize
NormalizeToNchw
CropResize
```

## Transfer benchmark

```text
H2D pageable
H2D pinned
D2H
不同 tensor 大小
```

## Stage benchmark

```text
Decode throughput
CPU transform throughput
GPU transform throughput
```

## End-to-end

```text
images/sec
samples/sec
trainer wait ratio
GPU utilization
CPU utilization
H2D GB/s
peak host memory
peak GPU memory
```

最终应该以：

> trainer 是否更少等待数据

作为 pipeline 真实性能指标。

---

# 53. Testing

必须保留三类测试。

## Semantic correctness

CPU / CUDA / Metal 输出在容忍误差内一致。

## Determinism

同：

```text
seed
epoch
sample index
OpKey
```

在不同 worker count / backend 下 augmentation decision 一致。

## Plan correctness

snapshot：

```text
logical plan
optimized plan
physical plan
```

例如测试：

```text
Normalize + Layout
 -> fused

CUDA sink
 -> exactly one H2D

CPU-only op
 -> not placed on CUDA
```

---

# Phase 9：边界约束与目标架构收束

# 54. 不建议现在做的事

暂时不要：

```text
1. 一次性把所有 vision op 写 CUDA
2. 一次性实现完全通用 DAG scheduler
3. 一开始就多 CUDA stream + 多 GPU
4. 一开始上复杂 cost model/autotuner
5. 提前为 text/audio 设计过度抽象的 Any/trait object 系统
6. 把 CUDA/Metal variant 塞进 Logical IR
7. 把 to_device 隐藏在 kernel 内
8. GPU 每 sample 单独 launch
```

---

# 56. 最终架构图

```text
                   User API
                     |
         +-----------+-----------+
         |           |           |
         v           v           v
   rivet-vision  rivet-text  rivet-audio
         |           |           |
         +-----------+-----------+
                     |
                     v
                 rivet-plan
        +---------------------------+
        | Logical IR                |
        | Arena                     |
        | Property inference hooks  |
        | Optimizer                 |
        | Fusion                    |
        | Cost model                |
        | Device placement          |
        +---------------------------+
                     |
                     v
                 rivet-exec
        +---------------------------+
        | Physical planner          |
        | Physical graph            |
        | Morsel                    |
        | Scheduler                 |
        | Execution lanes           |
        | Transfer                  |
        | Cache                     |
        | Memory planner            |
        | Profiler                  |
        +---------------------------+
                     |
                     v
                 rivet-core
        +---------------------------+
        | Tensor / Layout / DType   |
        | Storage                   |
        | CPU backend               |
        | CUDA backend              |
        | Metal backend             |
        +---------------------------+
```

---

# 58. 核心设计结论

最终必须守住以下边界：

```text
Domain DSL
    描述“用户想做什么”

rivet-plan
    决定“哪些操作可以重排、融合、放在哪个设备”

rivet-exec
    决定“physical node 怎么组成执行图、何时运行、如何 overlap、如何管理内存”

rivet-core
    提供“某个设备上的 Tensor 与 kernel 基础能力”
```

因此对于 CPU / CUDA / Metal：

```text
不是三套 pipeline
```

而是：

```text
一套 logical plan
一套 optimizer
一套 execution runtime
多个 physical backend
```

对于 vision / text / audio：

```text
不是三套 scheduler
```

而是：

```text
三套 domain semantics
共享 rivet-plan + rivet-exec
```

这应该成为 Rivet 后续 pipeline 架构的核心方向。

# Phase 10：当前 IR 编译器重构细则

# 57. 当前实现的 IR 编译器重构指南（不考虑前向兼容）

本节基于当前代码实际状态给出新的实施方案。前提是：

```text
不要求保留旧 ExecutionPlan / compile_image_ops / sample_ops / batch_ops API
不要求兼容旧的内部编译路径
允许直接删除重复抽象
允许修改 logical / physical / runtime 数据结构
```

目标不是“在旧 pipeline 上再加一层 IR”，而是让 IR 成为唯一编译入口和唯一执行真相来源。

最终编译链必须收敛为：

```text
Domain DSL
    ↓
Logical IR
    ↓
Semantic Analysis
    ↓
Semantic Rewrite Fixed Point
    ↓
Kernel Candidate Enumeration
    ↓
Global Placement
    ↓
Physical Fusion
    ↓
Physical IR
    ↓
Memory Planning
    ↓
Runtime Specialization
    ↓
Executable Pipeline
```

必须删除当前这种回路：

```text
LogicalPlan
→ optimizer
→ placement
→ PhysicalGraph
→ from_logical_plan()
→ compile_image_ops()
→ legacy ExecutionPlan
→ runtime
```

因为这会导致 planner 与真正执行路径拥有两套决策逻辑。

## 57.1 第一原则：Optimized LogicalPlan 是唯一 source of truth

当前最重要的重构不是增加更多 rewrite rule，而是消除双重编译。

应直接删除或逐步替换以下职责：

```text
compile_legacy_from_physical
compile_image_ops
ExecutionKind::Sample / Batch 作为最终 placement 决策
CompiledSampleOp
BatchKernel 作为第二套 compiler 输出
ExecutionPlan.sample_ops
ExecutionPlan.batch_ops
```

新的 compile() 应该近似：

```rust
pub fn compile(self) -> RivetResult<ImageDataLoader> {
    let mut logical = self.to_logical_plan();

    let optimized = rivet_plan::compile_logical(
        &mut logical,
        &vision_registry(),
        &machine_profile(&self.runtime),
    )?;

    let physical = rivet_plan::lower_physical(
        &optimized,
        &vision_registry(),
    )?;

    let executable = rivet_exec::compile(
        physical,
        self.runtime.into(),
    )?;

    ImageDataLoader::new(executable)
}
```

Domain DSL 在 `to_logical_plan()` 后即退出编译流程。之后不能再重新读取 `self.ops` 来决定 sample/batch/fusion。

## 57.2 将 Logical IR 与 Physical IR 的职责彻底分离

Logical IR 只描述语义：

```text
Decode
Resize
RandomCrop
RandomHorizontalFlip
Normalize
Layout
Batch
Cache
Sink
```

不应包含 CPU/CUDA/Metal kernel、worker 数量、stream、launch configuration 或具体 batch kernel 名称。

同一个逻辑图：

```text
Source → Decode → RandomCrop → Flip → Normalize → Layout(CHW) → Batch → Sink
```

可以 lower 成不同 physical plan：

```text
CPU:
CpuDecodeChunk
→ CpuFusedAugmentIntoBatch
→ Sink

CUDA:
CpuDecodeChunk
→ H2D
→ CudaAugmentNormalizeNchw
→ Sink
```

## 57.3 Semantic Fusion 与 Physical Fusion 必须分开

当前 `FusionGroupPayload` 同时承担逻辑融合和实现融合职责，应拆开。

Semantic Fusion 只用于真正的语义等价合并；Physical Fusion 只用于减少 allocation、memory pass、kernel launch、queue boundary。

例如：

```text
RandomCrop → Flip → Normalize → Layout
```

Logical IR 仍保留四个 op；CPU 可以选择 `CpuFusedCropFlipNormalizeNchw`，CUDA 可以选择 `CudaAugmentNormalizeNchw`。

建议：

```rust
pub struct PhysicalFusionCandidate {
    pub logical_nodes: SmallVec<[NodeId; 8]>,
    pub kernel: KernelId,
    pub device: DeviceClass,
    pub input: ValueId,
    pub output: ValueId,
    pub cost: CostEstimate,
}
```

## 57.4 删除 `VisionPropertyInference::num_workers`

Property inference 必须重新定义为纯语义分析。

当前 `VisionPropertyInference { num_workers }` 与 `infer_image_op_with_workers()` 说明 semantic properties 依赖 runtime worker 配置，这是层次耦合。

Resize 的 dtype/shape/layout 不应该随 workers=0/4/24 改变。

因此从 `ValueProperties` 删除 execution policy，尤其是：

```text
OperatorProperties.stage
OperatorProperties.sample_stage_has_work
```

## 57.5 将 ValueProperties 与 ExecutionProperties 拆开

新的 `ValueProperties`：

```rust
pub struct ValueProperties {
    pub representation: Representation,
    pub dtype: Option<DataType>,
    pub shape: ValueShape,
    pub axis_order: Option<AxisOrder>,
    pub contiguity: Contiguity,
    pub residency: Residency,
    pub granularity: SemanticGranularity,
    pub materialization: Materialization,
    pub ownership: Ownership,
}
```

新增：

```rust
pub enum Materialization { View, Materialized, Either, Unknown }
pub enum Ownership { Borrowed, Shared, Unique, Unknown }
```

Execution properties 单独分析：

```rust
pub struct ExecutionProperties {
    pub legal_granularities: SmallVec<[ExecutionGranularity; 3]>,
    pub parallelism: ParallelismClass,
    pub statefulness: Statefulness,
    pub estimated_work: WorkEstimate,
}
```

并明确：

```text
semantic sample operation != physical per-sample execution
```

RandomHorizontalFlip 语义上按 sample RNG，但可以 lower 到 batch GPU kernel。

## 57.6 删除 `OperatorStage::Sample/Batch` 作为逻辑约束

保留 `Batch` 作为语义 barrier，但不再把所有 op 固定分类到 sample/batch stage。

Normalize 可以注册多个 physical implementation：

```text
CpuNormalizeSample
CpuNormalizeChunk
CpuNormalizeBatch
CpuNormalizeToNchwBatch
CudaNormalizeBatch
CudaNormalizeToNchwBatch
MetalNormalizeBatch
```

由 planner 选择。

## 57.7 统一 KernelCapabilities 与 PhysicalCandidateProvider

当前两套机制长期会重复，应合并成唯一 `KernelRegistry`：

```rust
pub struct KernelRegistration {
    pub id: KernelId,
    pub op: OpPattern,
    pub backend: BackendId,
    pub device: DeviceClass,
    pub requirements: KernelRequirements,
    pub execution: ExecutionProperties,
    pub output: KernelOutputRule,
    pub cost_model: KernelCostModel,
    pub lowering: KernelLowering,
}
```

一个 registry 同时负责 candidate discovery、compatibility、placement、cost、lowering 和 explain。

## 57.8 编译器内部 identity 不再使用 String

把：

```rust
operator: String
name: String
backend: String
fusion_tags: Vec<String>
```

替换成：

```rust
pub struct DomainId(u16);
pub struct OpId(u32);
pub struct KernelId(u32);
pub struct BackendId(u16);
pub struct FusionClassId(u32);
```

字符串只用于 explain/debug。

## 57.9 PlanPayload 升级成真正的 DomainOp

当前 `PlanPayload` 太弱。建议替换为：

```rust
pub trait DomainOp: Any + Send + Sync {
    fn op_id(&self) -> OpId;
    fn effects(&self) -> OpEffects;
    fn infer(&self, inputs: &[ValueProperties]) -> Result<ValueProperties, PlanError>;
    fn as_any(&self) -> &dyn Any;
}
```

并增加：

```rust
pub struct OpEffects {
    pub deterministic: bool,
    pub random: RandomEffect,
    pub order_sensitive: bool,
    pub stateful: bool,
}
```

这样 rewrite/fusion legality 可以显式依据 effect system 判断。

## 57.10 Semantic rewrite 改成规则系统

当前 `SemanticRewrite` 同时处理 identity crop、identity dtype、late dtype promotion。继续扩张后会变成大型 match。

建议：

```rust
pub trait RewriteRule {
    fn match_node(
        &self,
        plan: &LogicalPlan,
        node: NodeId,
        analyses: &AnalysisManager,
    ) -> Option<RewriteMatch>;

    fn rewrite(
        &self,
        plan: &mut LogicalPlan,
        matched: RewriteMatch,
    ) -> Result<RewriteResult>;
}
```

拆成独立规则：

```text
EliminateIdentityCrop
EliminateIdentityLayout
EliminateIdentityCast
SinkCastIntoNormalize
PushIndexIntoSource
SimplifyNestedRandomApply
```

然后跑 fixed point，而不是巨型 pass 内部不断加分支。

## 57.11 Pass Manager 不再依赖 pass name 字符串

当前 `if pass.name() == "fusion-discovery"` 必须删除。

Compiler protocol 应改成明确阶段：

```rust
pub struct Compiler {
    canonicalization: Vec<Arc<dyn RewriteRule>>,
    semantic_rewrites: Vec<Arc<dyn RewriteRule>>,
    analyses: AnalysisManager,
    kernel_registry: KernelRegistry,
    placement: PlacementPlanner,
    physical_optimizer: PhysicalOptimizer,
}
```

执行：

```text
canonicalize
→ semantic fixed point
→ analysis
→ enumerate candidates
→ placement
→ physical fusion
→ memory planning
→ runtime specialization
```

## 57.12 引入 AnalysisManager 与 invalidation

当前 optimizer 多次显式运行 `InferProperties`。应改成按需分析缓存：

```rust
pub struct AnalysisManager {
    value_properties: Option<PropertyAnnotations>,
    use_def: Option<UseDefGraph>,
    source_capabilities: Option<SourceAnalysis>,
    effects: Option<EffectAnalysis>,
    aliasing: Option<AliasAnalysis>,
}
```

rewrite 返回：

```rust
pub struct RewriteResult {
    pub changed: bool,
    pub invalidates: AnalysisSet,
}
```

这样 pass manager 自动决定重新计算哪些分析，不需要手工插入多次 property inference。

## 57.13 SourceProperties 必须更强

Source 应显式提供：

```rust
pub struct SourceProperties {
    pub access_pattern: AccessPattern,
    pub shape: Option<ValueShape>,
    pub dtype: Option<DataType>,
    pub axis_order: Option<AxisOrder>,
    pub fixed_shape: bool,
    pub dense: bool,
    pub batched_reads: bool,
    pub zero_copy: bool,
    pub parallel_reads: bool,
    pub async_reads: bool,
    pub read_device: SmallVec<[DeviceClass; 2]>,
}
```

例如 decoded CIFAR memory source 应能告诉 planner：

```text
shape=[32,32,3]
dtype=U8
fixed_shape=true
dense=true
batch_read=true
```

这会直接帮助 chunk sizing、direct batch write、allocation planning、fusion legality 与 transfer costing。

## 57.14 Placement 从 greedy 改成全局 DP

当前 `place()` 只看 previous device，属于顺序 greedy。

对于线性 pipeline，直接改成 dynamic programming：

```text
dp[i][k] = kernel_cost(i,k)
         + min_j(dp[i-1][j] + edge_cost(j,k))
```

最后 backtrace 得到全局最低 cost 路径。

复杂度 `O(N*K^2)`，对数据 pipeline 足够便宜。

## 57.15 删除 `TooManyTransferBoundaries`

不要在 compiler 层硬限制只能一次 CPU/device boundary。

允许任意 boundary，由真实 cost model 自动淘汰昂贵的 CPU→GPU→CPU→GPU 路径。

## 57.16 删除 1e9 device transition penalty

当前 device switch 的超大常数 penalty 会掩盖真实 transfer cost。

应只使用：

```text
fixed transfer latency
transfer_bytes / bandwidth
allocation latency
kernel launch latency
synchronization latency
memory traffic
compute estimate
```

MachineProfile 推荐加入：

```rust
pub struct MachineProfile {
    pub h2d_bandwidth: f64,
    pub d2h_bandwidth: f64,
    pub d2d_bandwidth: f64,
    pub h2d_latency: f64,
    pub d2h_latency: f64,
    pub kernel_launch_latency: f64,
    pub cpu_mem_bandwidth: f64,
    pub device_mem_bandwidth: f64,
}
```

## 57.17 Cost Model 改成 memory-traffic aware

当前 `compute_score ≈ output_bytes / parallelism` 太粗。

建议：

```rust
pub struct KernelCostModel {
    pub input_reads: ByteExpr,
    pub output_writes: ByteExpr,
    pub temporary_bytes: ByteExpr,
    pub allocation_count: u32,
    pub launch_count: u32,
    pub sync_count: u32,
    pub compute_ops: WorkExpr,
}
```

例如 Normalize U8→F32：

```text
read = N
write = 4N
```

Normalize + Layout 未融合约 13N memory traffic；融合后约 5N。这样 fusion 会因为真实 memory traffic 而被自然选择。

## 57.18 Candidate Enumeration 要支持 region candidate

不能只按单 node 注册 kernel。

推荐：

```rust
pub enum CandidatePattern {
    Node(OpId),
    Chain(SmallVec<[OpId; 8]>),
}
```

例如 `Normalize + Layout` 同时拥有独立 candidate 和 fused candidate；`RandomCrop + Flip + Normalize + Layout` 也可以注册 CPU/CUDA/Metal region candidate。

## 57.19 Batch 升级为 allocation boundary

新的 physical planning 中，Batch 不只是 barrier，还应定义最终 batch storage：

```rust
pub struct BatchAllocationPlan {
    pub dtype: DataType,
    pub shape: ValueShape,
    pub axis_order: AxisOrder,
    pub device: DeviceClass,
    pub alignment: usize,
}
```

planner 可以先分配 `[B,C,H,W] F32`，sample worker 直接写 `batch[position]`。

## 57.20 Physical IR 表达 output-directed execution

建议：

```rust
pub enum PhysicalOp {
    SourceRead(SourceExec),
    Kernel(KernelExec),
    BatchAllocate(BatchAllocationPlan),
    KernelInto {
        kernel: KernelId,
        destination: BufferId,
    },
    Transfer(TransferPlan),
    Cache(CachePlan),
    Sink(SinkPlan),
}
```

例如 logical：

```text
RandomCrop → Flip → Normalize → Layout → Batch
```

physical 可以变成：

```text
BatchAllocate [B,C,H,W] F32
        ↓
CpuFusedCropFlipNormalizeIntoBatch
```

中间 sample Tensor 不再 materialize。

## 57.21 Memory Planner 成为正式编译阶段

流程：

```text
PhysicalGraph
    ↓
MemoryPlanner
    ↓
ExecutableGraph
```

至少处理：

```text
value liveness
last use
buffer size
alignment
device
alias
view/materialized
reuse
batch destination
pinned host memory
```

建议：

```rust
pub struct BufferPlan {
    pub id: BufferId,
    pub device: DeviceClass,
    pub bytes: usize,
    pub alignment: usize,
    pub lifetime: LiveRange,
    pub kind: BufferKind,
}

pub enum BufferKind {
    Cpu,
    PinnedHost,
    Device,
    External,
    Alias(BufferId),
}
```

目标是：能 view 就不 materialize，能 reuse 就不重新 allocate，能直接写 final output 就不创建 intermediate。

## 57.22 Runtime Compiler 增加 specialization

不要直接解释执行 PhysicalGraph。

应变成：

```text
PhysicalGraph
→ RuntimeCompiler
→ ExecutablePipeline
```

例如：

```rust
pub enum ExecutablePipeline {
    CpuInline(CpuInlineExecutor),
    CpuChunked(CpuChunkedExecutor),
    CpuDirectBatch(CpuDirectBatchExecutor),
    BatchNative(BatchNativeExecutor),
    HybridCuda(HybridCudaExecutor),
    HybridMetal(HybridMetalExecutor),
}
```

PhysicalGraph 继续服务 explain/profile/debug，但 steady-state runtime 只跑 specialized executor。

## 57.23 Runtime stage 与 Physical node 不一一对应

明确：

```text
Physical node != runtime thread/stage
```

Stage coalescing 作为 runtime compile pass：

```text
CPU Source → Decode → Crop → Flip
```

可以合并成一个 `CpuWorkerStage`。

只有真正可 overlap 的边界才产生 queue，例如 CPU preprocess→H2D→CUDA compute。

## 57.24 CUDA / Metal 建立统一 queue abstraction

不要让 runtime 核心写死 CUDA stream。

```rust
pub enum ExecutionLane {
    Cpu,
    Io,
    DeviceCompute(DeviceId),
    DeviceTransfer(DeviceId),
}
```

CUDA 映射到 compute/copy stream，Metal 映射到 command queue/blit path。

## 57.25 Randomness 进入 compiler-visible effect system

保留现有 `seed + epoch + sample identity + OpKey` 语义，并把它正式写进 IR effect。

```rust
pub struct RandomEffect {
    pub keyed_by_sample: bool,
    pub key: RandomKeyKind,
    pub occurrence_sensitive: bool,
}
```

这样 sample-level stochastic op 可以安全 lower 到 batch CPU/CUDA/Metal kernel，同时 fusion 不能改变 RNG key 顺序。

## 57.26 Domain plugin 最终只注册四类能力

`rivet-vision / rivet-text / rivet-audio` 最终只负责：

```text
1. DomainOp definitions
2. semantic property inference
3. semantic rewrite rules
4. kernel registrations
```

不再自己拥有 scheduler/runtime。

## 57.27 推荐新的 crate 边界

```text
rivet-core
  tensor/storage/device/backend kernels

rivet-plan
  logical IR
  domain op protocol
  property/effect/alias analysis
  semantic optimizer
  kernel registry
  candidate enumeration
  global placement
  physical fusion
  physical IR
  memory planning

rivet-exec
  runtime compiler
  executable plans
  worker pools/queues
  device lanes
  transfer runtime
  buffer pools
  profiling

rivet-vision / rivet-text / rivet-audio
  domain ops
  semantic rules
  kernel registrations
```

关键依赖规则：`rivet-plan` 和 `rivet-exec` 都不能依赖 vision/text/audio。

## 57.28 直接删除的旧设计

因为不要求前向兼容，建议直接移除：

```text
ExecutionKind::Sample / Batch
OperatorStage::Sample / Batch
OperatorProperties.sample_stage_has_work
VisionPropertyInference::num_workers
compile_image_ops 的 placement/fusion 职责
compile_legacy_from_physical
legacy sample_ops / batch_ops 二次编译
PhysicalCandidateProvider
pass.name() == "fusion-discovery"
TooManyTransferBoundaries
1e9 device-switch penalty
字符串形式的 compiler identity
```

旧测试如果依赖这些内部结构，直接改测试，不要保留双轨实现。

## 57.29 新 compiler 的固定阶段

建议最终固定为：

```text
Phase 0  DSL lowering
Phase 1  Logical canonicalization
Phase 2  Semantic property inference
Phase 3  Semantic rewrite fixed point
Phase 4  Effect / alias / ownership analysis
Phase 5  Source pushdown
Phase 6  Kernel candidate enumeration
Phase 7  Global placement DP
Phase 8  Physical fusion
Phase 9  Physical IR lowering
Phase 10 Memory planning
Phase 11 Stage coalescing
Phase 12 Runtime specialization
Phase 13 Executable pipeline construction
```

每一阶段必须有明确输入和输出，禁止后续阶段重新做前面阶段已经做过的决策。

## 57.30 推荐实施顺序

### Step 1：先消灭双重编译

定义新的 `ExecutablePlan`：

```rust
pub struct ExecutablePlan {
    pub source: SourceExec,
    pub sampler: SamplerExec,
    pub stages: Vec<ExecutableStage>,
    pub buffers: Vec<BufferPlan>,
}
```

直接从 optimized LogicalPlan / PhysicalPlan 生成它。删除 `from_logical_plan() -> ImagePipeline -> compile_image_ops()` 回路。

这是第一里程碑，也是优先级最高的一步。

### Step 2：删除 worker-dependent property inference

移除 `num_workers / OperatorStage / sample_stage_has_work`。所有 runtime policy 转到 physical planner。

### Step 3：建立唯一 KernelRegistry

合并 KernelCapabilities 与 PhysicalCandidateProvider，统一 candidate discovery、cost、placement 和 lowering。

### Step 4：把 NormalizeLayout fusion 移到 PhysicalFusion

LogicalPlan 保留 Normalize + Layout；CPU/CUDA/Metal 分别注册自己的 fused region candidate。

### Step 5：greedy placement → DP placement

先支持当前主流单输入线性 pipeline，删除 single-transfer restriction 和 1e9 penalty。

### Step 6：memory-traffic cost model

第一版只需要可靠支持：

```text
read bytes
write bytes
temporary bytes
allocations
transfers
launches
syncs
```

### Step 7：实现 BatchAllocation + KernelInto

让 physical plan 能表达 direct batch write。

### Step 8：实现 MemoryPlanner

先做 fixed-shape CPU buffer planning / reuse，再扩 pinned/device pools。

### Step 9：实现 CpuDirectBatchExecutor

将 decoded/fixed-shape CIFAR pipeline 编译成直接写最终 batch 的 fast path。

### Step 10：实现 HybridCudaExecutor

目标路径：

```text
CPU decode/augment
→ pinned batch buffer
→ async H2D
→ CUDA fused batch kernel
→ sink
```

### Step 11：再接 Metal

当上述抽象稳定后，Metal 只需要新增 backend kernel registrations、device queue/runtime，不应修改 logical optimizer。

## 57.31 第一批 semantic rewrites

优先实现：

```text
identity crop elimination
identity layout elimination
identity dtype conversion
late dtype promotion
index/source pushdown
redundant materialize elimination
redundant contiguous elimination
```

原则：只做 backend-independent 的等价变换。

## 57.32 第一批 physical fusions

Vision：

```text
Normalize + Layout
Crop + Normalize + Layout
RandomCrop + Flip + Normalize + Layout
RandomResizedCrop + Flip + Normalize + Layout
```

Text：

```text
Template + Tokenize
Tokenize + Truncate
Pad + AttentionMask
```

Audio：

```text
STFT + Mel
Resample + Normalize
```

这些 fusion 都应允许 CPU/CUDA/Metal 各自注册实现，而不是修改 logical IR。

## 57.33 第一批 specialized executors

推荐顺序：

```text
1. CpuInlineExecutor
2. CpuChunkedExecutor
3. CpuDirectBatchExecutor
4. BatchNativeExecutor
5. HybridCudaExecutor
6. HybridMetalExecutor
```

不要继续扩张一个万能 generic graph executor 来承担所有稳态执行。

## 57.34 Explain 必须分层

建议暴露：

```text
pipeline.explain_logical()
pipeline.explain_optimized()
pipeline.explain_placement()
pipeline.explain_physical()
pipeline.explain_executable()
```

例如：

```text
Logical:
RandomCrop → Flip → Normalize → Layout → Batch

Optimized:
RandomCrop → Flip → Normalize → Layout → Batch

Placement:
CPU region selected

Physical:
BatchAllocate [128,3,32,32] F32
FusedCropFlipNormalizeNchwIntoBatch

Executable:
CpuDirectBatchExecutor
workers=24
chunk=8
buffer_pool=enabled
```

这样可以直接验证 optimizer 是否真正改变数据路径。

## 57.35 Testing 策略

因为不考虑前向兼容，测试重点从“旧 ExecutionPlan 内部结构一致”改成：

```text
semantic equivalence
RNG equivalence
property inference correctness
rewrite legality
placement optimality
physical fusion correctness
memory alias safety
executor output correctness
```

特别是 stochastic pipeline，必须验证：

```text
workers=0/4/24
fused/unfused
CPU/CUDA
```

在相同 seed + epoch + sample index + OpKey 下生成相同 augmentation 参数。

## 57.36 Benchmark gate

每个 compiler feature 都应该 A/B：

```text
optimization disabled
optimization enabled
```

至少覆盖：

```text
CIFAR decoded
CIFAR encoded
ImageNet encoded
CPU-only
CUDA
```

观测：

```text
images/s
batch p50/p95
allocations/sample
memory read/write bytes
CPU utilization
H2D bytes
kernel launch count
```

Direct batch write 的验收条件应明确：

```text
per-sample Tensor allocation 显著下降
Tensor::stack = 0
完整 batch copy = 0
full-image memory pass 数下降
```

## 57.37 最终目标

Rivet 最终不应是：

```text
DataLoader + graph representation
```

而应是：

```text
ML data pipeline compiler
```

IR 的价值必须最终体现为：

```text
更少的数据搬运
更少的 allocation
更少的 materialization
更少的 kernel launch
更长的 same-device region
更大的 fused region
更低的 runtime abstraction cost
```

最终路径：

```text
User Pipeline
   ↓
Domain Logical IR
   ↓
Semantic Optimization
   ↓
Global Physical Planning
   ↓
Memory-aware Fusion
   ↓
Buffer Planning
   ↓
Runtime Specialization
   ↓
CPU / CUDA / Metal Executable Pipeline
```

这应该完全替代当前“logical IR + legacy compiler”双轨架构。

---
