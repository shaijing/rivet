//! Explicit, backend-neutral logical optimization passes.

use std::sync::Arc;

use thiserror::Error;

use crate::{LogicalNode, LogicalPlan, NodeId, PropertyAnnotations, PropertyInference};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: &'static str,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PassResult {
    pub changed: bool,
    pub diagnostics: Vec<Diagnostic>,
}

impl PassResult {
    pub fn unchanged() -> Self {
        Self::default()
    }
    pub fn changed() -> Self {
        Self {
            changed: true,
            diagnostics: Vec::new(),
        }
    }
    pub fn diagnostic(mut self, code: &'static str, message: impl Into<String>) -> Self {
        self.diagnostics.push(Diagnostic {
            code,
            message: message.into(),
        });
        self
    }
}

#[derive(Debug, Error)]
pub enum OptimizerError {
    #[error("optimizer pass `{pass}` failed: {message}")]
    Pass { pass: String, message: String },
    #[error("optimizer did not converge after {iterations} iterations")]
    NonConvergent { iterations: usize },
}

/// Shared state is deliberately supplied to every pass so passes can inspect
/// and update inferred properties without coupling the core IR to a domain.
#[derive(Default)]
pub struct OptimizerContext {
    annotations: PropertyAnnotations,
    pub diagnostics: Vec<Diagnostic>,
    pub snapshots: Vec<(String, String)>,
}

impl OptimizerContext {
    pub fn annotations(&self) -> &PropertyAnnotations {
        &self.annotations
    }
    pub fn annotations_mut(&mut self) -> &mut PropertyAnnotations {
        &mut self.annotations
    }
    pub fn set_annotations(&mut self, annotations: PropertyAnnotations) {
        self.annotations = annotations;
    }
    pub fn clear_annotations(&mut self) {
        self.annotations = PropertyAnnotations::default();
    }
}

pub trait OptimizerPass: Send + Sync {
    fn name(&self) -> &'static str;
    fn run(
        &self,
        plan: &mut LogicalPlan,
        context: &mut OptimizerContext,
    ) -> Result<PassResult, String>;
}

/// Domain hooks are registered by the application composition root. A plugin
/// may contribute inference and ordered optimizer passes without global state.
pub trait PlanPlugin: Send + Sync {
    fn name(&self) -> &'static str;
    fn property_inference(&self) -> Option<Arc<dyn PropertyInference>> {
        None
    }
    fn optimizer_passes(&self) -> Vec<Arc<dyn OptimizerPass>> {
        Vec::new()
    }
    fn fusion_rules(&self) -> Vec<Arc<dyn FusionRule>> {
        Vec::new()
    }
    fn physical_candidates(&self) -> Vec<Arc<dyn PhysicalCandidateProvider>> {
        Vec::new()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusionCandidate {
    pub rule: &'static str,
    pub nodes: Vec<NodeId>,
    pub name: String,
    pub legal: bool,
    pub reason: String,
}

pub trait FusionRule: Send + Sync {
    fn name(&self) -> &'static str;
    fn discover(
        &self,
        plan: &LogicalPlan,
        annotations: &PropertyAnnotations,
    ) -> Result<Vec<FusionCandidate>, String>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalCandidate {
    pub name: String,
    pub backend: String,
}

pub trait PhysicalCandidateProvider: Send + Sync {
    fn candidates(
        &self,
        node: &LogicalNode,
        properties: Option<&crate::ValueProperties>,
    ) -> Vec<PhysicalCandidate>;
}

#[derive(Default)]
pub struct PlanRegistry {
    inference: Option<Arc<dyn PropertyInference>>,
    passes: Vec<Arc<dyn OptimizerPass>>,
    fusion_rules: Vec<Arc<dyn FusionRule>>,
    physical_candidates: Vec<Arc<dyn PhysicalCandidateProvider>>,
}

impl PlanRegistry {
    pub fn register_plugin(&mut self, plugin: &dyn PlanPlugin) {
        if let Some(inference) = plugin.property_inference() {
            self.inference = Some(inference);
        }
        self.passes.extend(plugin.optimizer_passes());
        self.fusion_rules.extend(plugin.fusion_rules());
        self.physical_candidates
            .extend(plugin.physical_candidates());
    }
    pub fn register_pass(&mut self, pass: Arc<dyn OptimizerPass>) {
        self.passes.push(pass);
    }
    pub fn passes(&self) -> &[Arc<dyn OptimizerPass>] {
        &self.passes
    }
    pub fn inference(&self) -> Option<&Arc<dyn PropertyInference>> {
        self.inference.as_ref()
    }
    pub fn fusion_rules(&self) -> &[Arc<dyn FusionRule>] {
        &self.fusion_rules
    }
    pub fn register_fusion_rule(&mut self, rule: Arc<dyn FusionRule>) {
        self.fusion_rules.push(rule);
    }
    pub fn register_physical_candidates(&mut self, provider: Arc<dyn PhysicalCandidateProvider>) {
        self.physical_candidates.push(provider);
    }
}

#[derive(Clone, Debug, Default)]
pub struct OptimizationReport {
    pub passes_run: Vec<&'static str>,
    pub diagnostics: Vec<Diagnostic>,
    pub iterations: usize,
    pub fusion_candidates: Vec<FusionCandidate>,
}

/// Run the ordered pass list once. Each pass is followed by structural
/// validation and a snapshot, making rewrites reviewable in `explain` output.
pub fn optimize(
    plan: &mut LogicalPlan,
    registry: &PlanRegistry,
) -> Result<(OptimizerContext, OptimizationReport), OptimizerError> {
    let mut context = OptimizerContext::default();
    let mut report = OptimizationReport::default();
    for pass in registry.passes() {
        let result = pass
            .run(plan, &mut context)
            .map_err(|message| OptimizerError::Pass {
                pass: pass.name().into(),
                message,
            })?;
        plan.validate().map_err(|error| OptimizerError::Pass {
            pass: pass.name().into(),
            message: error.to_string(),
        })?;
        report.passes_run.push(pass.name());
        context.diagnostics.extend(result.diagnostics);
        let mut fusion_snapshot = String::new();
        if pass.name() == "fusion-discovery" {
            for rule in &registry.fusion_rules {
                let candidates = rule
                    .discover(plan, context.annotations())
                    .map_err(|message| OptimizerError::Pass {
                        pass: rule.name().into(),
                        message,
                    })?;
                for candidate in &candidates {
                    fusion_snapshot.push_str(&format!(
                        "  FusionCandidate {} {:?} {}: {} ({})\n",
                        candidate.rule,
                        candidate.nodes,
                        candidate.name,
                        if candidate.legal { "legal" } else { "illegal" },
                        candidate.reason,
                    ));
                    context.diagnostics.push(Diagnostic {
                        code: if candidate.legal {
                            "fusion.legal"
                        } else {
                            "fusion.illegal"
                        },
                        message: format!(
                            "{}: {} ({})",
                            candidate.rule, candidate.name, candidate.reason
                        ),
                    });
                }
                report.fusion_candidates.extend(candidates);
            }
        }
        let mut snapshot = plan.explain().map_err(|error| OptimizerError::Pass {
            pass: pass.name().into(),
            message: error.to_string(),
        })?;
        snapshot.push_str(&fusion_snapshot);
        context.snapshots.push((pass.name().to_string(), snapshot));
    }
    report.diagnostics = context.diagnostics.clone();
    report.iterations = 1;
    Ok((context, report))
}

/// Iterate a rewrite pass to a fixed point, stopping on the first unchanged
/// iteration and returning an explicit diagnostic on the configured limit.
pub fn run_fixed_point(
    plan: &mut LogicalPlan,
    pass: &dyn OptimizerPass,
    context: &mut OptimizerContext,
    max_iterations: usize,
) -> Result<usize, OptimizerError> {
    for iteration in 1..=max_iterations {
        let result = pass
            .run(plan, context)
            .map_err(|message| OptimizerError::Pass {
                pass: pass.name().into(),
                message,
            })?;
        context.diagnostics.extend(result.diagnostics);
        plan.validate().map_err(|error| OptimizerError::Pass {
            pass: pass.name().into(),
            message: error.to_string(),
        })?;
        if !result.changed {
            return Ok(iteration);
        }
    }
    Err(OptimizerError::NonConvergent {
        iterations: max_iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LogicalNode, NodeKind};

    struct CountDown(usize);
    impl OptimizerPass for CountDown {
        fn name(&self) -> &'static str {
            "count-down"
        }
        fn run(
            &self,
            _plan: &mut LogicalPlan,
            _context: &mut OptimizerContext,
        ) -> Result<PassResult, String> {
            // Interior mutation is unnecessary here; test the no-change fast path.
            let _ = self.0;
            Ok(PassResult::unchanged())
        }
    }

    struct AlwaysChanges;
    impl OptimizerPass for AlwaysChanges {
        fn name(&self) -> &'static str {
            "always-changes"
        }
        fn run(
            &self,
            _plan: &mut LogicalPlan,
            _context: &mut OptimizerContext,
        ) -> Result<PassResult, String> {
            Ok(PassResult::changed())
        }
    }

    #[test]
    fn fixed_point_stops_when_pass_reports_unchanged() {
        let mut plan = LogicalPlan::new();
        let root = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        plan.set_root(root).unwrap();
        assert_eq!(
            run_fixed_point(
                &mut plan,
                &CountDown(0),
                &mut OptimizerContext::default(),
                4
            )
            .unwrap(),
            1
        );
    }

    #[test]
    fn snapshots_include_pass_name_and_plan_explain() {
        let mut plan = LogicalPlan::new();
        let root = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        plan.set_root(root).unwrap();
        let mut registry = PlanRegistry::default();
        registry.register_pass(Arc::new(CountDown(0)));
        let (context, report) = optimize(&mut plan, &registry).unwrap();
        assert_eq!(report.passes_run, ["count-down"]);
        assert!(context.snapshots[0].1.contains("LogicalPlan(root="));
    }

    #[test]
    fn fixed_point_reports_iteration_limit() {
        let mut plan = LogicalPlan::new();
        let root = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        plan.set_root(root).unwrap();
        let error = run_fixed_point(
            &mut plan,
            &AlwaysChanges,
            &mut OptimizerContext::default(),
            3,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            OptimizerError::NonConvergent { iterations: 3 }
        ));
    }
}
