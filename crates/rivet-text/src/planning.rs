//! Text plan payloads contain only semantic metadata, never tokenizer handles
//! or dataset buffers. The initial compiler validates this fixed CPU chain.
use crate::errors::{TextError, TextResult};
use crate::{EncodeOptions, LabelMode, Override, PaddingParams, PaddingStrategy, TextPipeline};
use rivet_plan::{
    Contiguity, DataType, DomainId, LogicalNode, LogicalPlan, Mutability, NodeKind,
    OperatorProperties, OperatorStage, PlanPayload, PropertyInference, Representation, Residency,
    ShapeDim, ValueGranularity, ValueProperties, ValueShape,
};
use std::sync::Arc;

pub const TEXT_DOMAIN_ID: DomainId = DomainId::new(2);
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TextPlanOp {
    Source {
        rows: usize,
        identity: u64,
        indices: Option<Vec<usize>>,
        shuffle_seed: Option<u64>,
        epoch: u64,
    },
    Tokenize {
        tokenizer_identity: u64,
        options: EncodeOptions,
    },
    Collate {
        batch_size: usize,
        drop_last: bool,
        padding: Option<PaddingParams>,
        labels: LabelMode,
    },
    Sink,
}
impl PlanPayload for TextPlanOp {
    fn domain(&self) -> &'static str {
        "text"
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Source { .. } => "TextSource",
            Self::Tokenize { .. } => "Tokenize",
            Self::Collate { .. } => "Collate",
            Self::Sink => "TextSink",
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn domain_id(&self) -> Option<DomainId> {
        Some(TEXT_DOMAIN_ID)
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct TextPropertyInference;
impl PropertyInference for TextPropertyInference {
    fn domain_id(&self) -> Option<DomainId> {
        Some(TEXT_DOMAIN_ID)
    }
    fn infer_node(
        &self,
        node: &LogicalNode,
        inputs: &[ValueProperties],
    ) -> Result<ValueProperties, String> {
        let op = node
            .payload_as::<TextPlanOp>()
            .ok_or("text node has no TextPlanOp payload")?;
        if matches!(op, TextPlanOp::Source { .. }) {
            if !inputs.is_empty() || node.kind() != NodeKind::Source {
                return Err("text source requires no inputs and Source kind".into());
            }
            return Ok(ValueProperties {
                representation: Some(Representation::Other(Arc::from("text"))),
                granularity: Some(ValueGranularity::Sample),
                residency: Some(Residency::Host),
                mutability: Some(Mutability::Immutable),
                operator: Some(OperatorProperties {
                    stage: OperatorStage::Source,
                    sample_stage_has_work: false,
                }),
                ..Default::default()
            });
        }
        let [input] = inputs else {
            return Err("text pipeline nodes require exactly one input".into());
        };
        let mut output = input.clone();
        match op {
            TextPlanOp::Tokenize { .. } => {
                if input.granularity != Some(ValueGranularity::Sample)
                    || node.kind() != NodeKind::Op
                    || input.representation != Some(Representation::Other(Arc::from("text")))
                {
                    return Err("Tokenize requires raw text".into());
                }
                output.representation = Some(Representation::Other(Arc::from("text_encoding")));
                output.dtype = Some(DataType::U32);
                output.shape = Some(ValueShape(vec![ShapeDim::Dynamic]));
                output.operator = Some(OperatorProperties {
                    stage: OperatorStage::Sample,
                    sample_stage_has_work: true,
                });
            }
            TextPlanOp::Collate {
                batch_size,
                drop_last,
                padding,
                ..
            } => {
                if input.granularity != Some(ValueGranularity::Sample)
                    || *batch_size == 0
                    || node.kind() != NodeKind::Batch
                    || input.representation
                        != Some(Representation::Other(Arc::from("text_encoding")))
                {
                    return Err("Collate requires token encodings and a positive batch size".into());
                }
                output.representation = Some(Representation::Tensor);
                output.dtype = Some(DataType::I64);
                let padded_length = match padding
                    .as_ref()
                    .map(|p| (&p.strategy, p.pad_to_multiple_of))
                {
                    Some((PaddingStrategy::Fixed(length), multiple)) => Some(
                        crate::tokenizer::padded_length(*length, multiple)
                            .map_err(|e| e.to_string())?,
                    ),
                    _ => None,
                };
                output.shape = Some(ValueShape(vec![
                    if *drop_last {
                        ShapeDim::Known(*batch_size)
                    } else {
                        ShapeDim::Dynamic
                    },
                    padded_length.map_or(ShapeDim::Dynamic, ShapeDim::Known),
                ]));
                output.granularity = Some(ValueGranularity::Batch);
                output.contiguity = Some(Contiguity::Contiguous);
                output.operator = Some(OperatorProperties {
                    stage: OperatorStage::Batch,
                    sample_stage_has_work: true,
                });
            }
            TextPlanOp::Sink => {
                if node.kind() != NodeKind::Sink
                    || input.granularity != Some(ValueGranularity::Batch)
                {
                    return Err("text sink requires a batch".into());
                }
            }
            TextPlanOp::Source { .. } => unreachable!(),
        }
        Ok(output)
    }
}
impl TextPipeline {
    pub fn to_logical_plan(&self) -> TextResult<LogicalPlan> {
        self.tokenizer.validate_options(&self.options)?;
        let padding = self.tokenizer.padding(&self.options).cloned();
        let mut token_options = self.options.clone();
        token_options.padding = Override::Off;
        token_options.truncation = self
            .tokenizer
            .truncation(&self.options)
            .cloned()
            .map_or(Override::Off, Override::With);
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(
            NodeKind::Source,
            [],
            Some(Arc::new(TextPlanOp::Source {
                rows: self.source.len(),
                identity: self.source_identity,
                indices: self.indices.clone(),
                shuffle_seed: self.shuffle_seed,
                epoch: self.epoch,
            })),
        ));
        let tokenize = plan.add_node(LogicalNode::new(
            NodeKind::Op,
            [source],
            Some(Arc::new(TextPlanOp::Tokenize {
                tokenizer_identity: self.tokenizer.identity(),
                options: token_options,
            })),
        ));
        let batch = plan.add_node(LogicalNode::new(
            NodeKind::Batch,
            [tokenize],
            Some(Arc::new(TextPlanOp::Collate {
                batch_size: self.batch_size,
                drop_last: self.drop_last,
                padding,
                labels: self.label_mode,
            })),
        ));
        let sink = plan.add_node(LogicalNode::new(
            NodeKind::Sink,
            [batch],
            Some(Arc::new(TextPlanOp::Sink)),
        ));
        plan.set_root(sink)
            .map_err(|e| TextError::Plan(e.to_string()))?;
        Ok(plan)
    }
    pub fn explain(&self) -> TextResult<String> {
        let plan = self.to_logical_plan()?;
        plan.infer_properties(&TextPropertyInference)
            .map_err(|e| TextError::Plan(e.to_string()))?;
        plan.explain().map_err(|e| TextError::Plan(e.to_string()))
    }
}
