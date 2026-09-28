//! Explicit, backend-neutral logical optimization passes.

use std::sync::Arc;

use thiserror::Error;

use crate::{DomainId, LogicalNode, LogicalPlan, NodeId, PropertyAnnotations, PropertyInference};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: &'static str,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PassResult {
    pub changed: bool,
    pub diagnostics: Vec<Diagnostic>,
    /// Request a restart from canonicalization after this pass changes a plan
    /// that has already passed through placement.
    pub replan: bool,
}

impl PassResult {
    pub fn unchanged() -> Self {
        Self::default()
    }
    pub fn changed() -> Self {
        Self {
            changed: true,
            diagnostics: Vec::new(),
            replan: false,
        }
    }
    pub fn diagnostic(mut self, code: &'static str, message: impl Into<String>) -> Self {
        self.diagnostics.push(Diagnostic {
            code,
            message: message.into(),
        });
        self
    }
    pub fn request_replan(mut self) -> Self {
        self.replan = true;
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
    placement: Option<crate::PlacementPlan>,
    pub diagnostics: Vec<Diagnostic>,
    pub snapshots: Vec<(String, String)>,
}

impl OptimizerContext {
    pub fn annotations(&self) -> &PropertyAnnotations {
        &self.annotations
    }
    pub fn annotations_mut(&mut self) -> &mut PropertyAnnotations {
        self.placement = None;
        &mut self.annotations
    }
    pub fn set_annotations(&mut self, annotations: PropertyAnnotations) {
        self.annotations = annotations;
        self.placement = None;
    }
    pub fn clear_annotations(&mut self) {
        self.annotations = PropertyAnnotations::default();
        self.placement = None;
    }
    pub fn set_placement(&mut self, placement: crate::PlacementPlan) {
        self.placement = Some(placement);
    }
    pub fn placement(&self) -> Option<&crate::PlacementPlan> {
        self.placement.as_ref()
    }
    pub fn take_placement(&mut self) -> Option<crate::PlacementPlan> {
        self.placement.take()
    }
}

/// Plan snapshots are optional; validation always runs after each pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OptimizerOptions {
    pub record_snapshots: bool,
}

pub trait OptimizerPass: Send + Sync {
    fn name(&self) -> &'static str;
    /// Marks the point where registered semantic fusion rules are discovered.
    fn is_fusion_discovery(&self) -> bool {
        false
    }
    /// Marks the first placement or later physical-planning pass. Earlier
    /// passes are iterated to a semantic fixed point before this boundary.
    fn is_placement_boundary(&self) -> bool {
        false
    }
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
    fn domain_id(&self) -> Option<DomainId> {
        None
    }
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
    fn apply(
        &self,
        _plan: &mut LogicalPlan,
        _candidate: &FusionCandidate,
        _context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        Ok(PassResult::unchanged())
    }
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
    inference: Vec<(Option<DomainId>, Arc<dyn PropertyInference>)>,
    passes: Vec<Arc<dyn OptimizerPass>>,
    fusion_rules: Vec<Arc<dyn FusionRule>>,
    physical_candidates: Vec<Arc<dyn PhysicalCandidateProvider>>,
}

impl PlanRegistry {
    pub fn register_plugin(&mut self, plugin: &dyn PlanPlugin) {
        if let Some(inference) = plugin.property_inference() {
            let domain = plugin.domain_id().or_else(|| inference.domain_id());
            self.inference.push((domain, inference));
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
        self.inference.first().map(|(_, inference)| inference)
    }
    pub fn inference_for_domain(&self, domain: DomainId) -> Option<&Arc<dyn PropertyInference>> {
        self.inference
            .iter()
            .find_map(|(registered, inference)| (*registered == Some(domain)).then_some(inference))
    }
    pub fn inferences(
        &self,
    ) -> impl Iterator<Item = (Option<DomainId>, &Arc<dyn PropertyInference>)> {
        self.inference
            .iter()
            .map(|(domain, inference)| (*domain, inference))
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
    pub fn physical_candidates_for(
        &self,
        node: &LogicalNode,
        properties: Option<&crate::ValueProperties>,
    ) -> Vec<PhysicalCandidate> {
        self.physical_candidates
            .iter()
            .flat_map(|provider| provider.candidates(node, properties))
            .collect()
    }
}

#[derive(Clone, Debug, Default)]
pub struct OptimizationReport {
    pub passes_run: Vec<&'static str>,
    pub diagnostics: Vec<Diagnostic>,
    pub iterations: usize,
    pub fusion_candidates: Vec<FusionCandidate>,
}

/// Run ordered passes until they complete without requesting a placement
/// restart. Each pass is followed by structural validation and a snapshot.
pub fn optimize(
    plan: &mut LogicalPlan,
    registry: &PlanRegistry,
) -> Result<(OptimizerContext, OptimizationReport), OptimizerError> {
    // Retain snapshot behavior of the original API. Production callers may
    // opt out without changing rewrite rules or structural validation.
    optimize_with_options(
        plan,
        registry,
        OptimizerOptions {
            record_snapshots: true,
        },
    )
}

pub fn optimize_with_options(
    plan: &mut LogicalPlan,
    registry: &PlanRegistry,
    options: OptimizerOptions,
) -> Result<(OptimizerContext, OptimizationReport), OptimizerError> {
    let mut context = OptimizerContext::default();
    let mut report = OptimizationReport::default();
    const MAX_REPLANS: usize = 8;
    const MAX_SEMANTIC_ITERATIONS: usize = 32;
    let placement_boundary = registry
        .passes()
        .iter()
        .position(|pass| pass.is_placement_boundary())
        .unwrap_or(registry.passes().len());
    let (semantic_passes, placement_passes) = registry.passes().split_at(placement_boundary);

    for iteration in 1..=MAX_REPLANS {
        context.clear_annotations();
        let mut requested_replan = false;
        let mut converged = false;

        for _ in 0..MAX_SEMANTIC_ITERATIONS {
            let mut changed = false;
            for pass in semantic_passes {
                let result = run_optimizer_pass(
                    plan,
                    registry,
                    pass.as_ref(),
                    &mut context,
                    &mut report,
                    options,
                )?;
                changed |= result.changed;
                if result.replan {
                    request_full_replan(
                        pass.name(),
                        &mut context,
                        &mut report,
                        &mut requested_replan,
                    );
                    break;
                }
            }
            if requested_replan {
                break;
            }
            if !changed {
                converged = true;
                break;
            }
        }

        if requested_replan {
            report.iterations = iteration;
            continue;
        }
        if !converged {
            return Err(OptimizerError::NonConvergent {
                iterations: MAX_SEMANTIC_ITERATIONS,
            });
        }

        for pass in placement_passes {
            let result = run_optimizer_pass(
                plan,
                registry,
                pass.as_ref(),
                &mut context,
                &mut report,
                options,
            )?;
            if result.replan {
                request_full_replan(
                    pass.name(),
                    &mut context,
                    &mut report,
                    &mut requested_replan,
                );
                break;
            }
        }
        if requested_replan {
            report.iterations = iteration;
            continue;
        }

        report.iterations = iteration;
        report.diagnostics = context.diagnostics.clone();
        return Ok((context, report));
    }
    Err(OptimizerError::NonConvergent {
        iterations: MAX_REPLANS,
    })
}

fn run_optimizer_pass(
    plan: &mut LogicalPlan,
    registry: &PlanRegistry,
    pass: &dyn OptimizerPass,
    context: &mut OptimizerContext,
    report: &mut OptimizationReport,
    options: OptimizerOptions,
) -> Result<PassResult, OptimizerError> {
    let mut result = pass
        .run(plan, context)
        .map_err(|message| OptimizerError::Pass {
            pass: pass.name().into(),
            message,
        })?;
    let mut fusion_snapshot = String::new();
    if pass.is_fusion_discovery() {
        for rule in registry.fusion_rules() {
            let candidates = rule
                .discover(plan, context.annotations())
                .map_err(|message| OptimizerError::Pass {
                    pass: rule.name().into(),
                    message,
                })?;
            for candidate in &candidates {
                if options.record_snapshots {
                    fusion_snapshot.push_str(&format!(
                        "  FusionCandidate {} {:?} {}: {} ({})\n",
                        candidate.rule,
                        candidate.nodes,
                        candidate.name,
                        if candidate.legal { "legal" } else { "illegal" },
                        candidate.reason,
                    ));
                }
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
                report.fusion_candidates.push(candidate.clone());
                if candidate.legal {
                    let applied = rule.apply(plan, candidate, context).map_err(|message| {
                        OptimizerError::Pass {
                            pass: rule.name().into(),
                            message,
                        }
                    })?;
                    result.changed |= applied.changed;
                    result.replan |= applied.replan;
                    context.diagnostics.extend(applied.diagnostics);
                }
            }
        }
    }
    plan.validate().map_err(|error| OptimizerError::Pass {
        pass: pass.name().into(),
        message: error.to_string(),
    })?;
    report.passes_run.push(pass.name());
    context
        .diagnostics
        .extend(result.diagnostics.iter().cloned());
    if options.record_snapshots {
        let mut snapshot = plan.explain().map_err(|error| OptimizerError::Pass {
            pass: pass.name().into(),
            message: error.to_string(),
        })?;
        snapshot.push_str(&fusion_snapshot);
        for diagnostic in &context.diagnostics {
            snapshot.push_str(&format!(
                "  Diagnostic {}: {}\n",
                diagnostic.code, diagnostic.message
            ));
        }
        context.snapshots.push((pass.name().to_string(), snapshot));
    }
    Ok(result)
}

fn request_full_replan(
    pass_name: &str,
    context: &mut OptimizerContext,
    report: &mut OptimizationReport,
    requested_replan: &mut bool,
) {
    *requested_replan = true;
    context.clear_annotations();
    // Snapshots retain intermediate history; diagnostics and fusion candidates
    // returned to callers describe only the final planning attempt.
    context.diagnostics.clear();
    report.fusion_candidates.clear();
    context.diagnostics.push(Diagnostic {
        code: "optimizer.replan",
        message: format!("pass `{pass_name}` requested a full planning restart"),
    });
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
    use std::sync::atomic::{AtomicBool, Ordering};

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

    struct ReplanOnce(AtomicBool);
    impl OptimizerPass for ReplanOnce {
        fn name(&self) -> &'static str {
            "late-semantic-rewrite"
        }
        fn run(
            &self,
            _plan: &mut LogicalPlan,
            context: &mut OptimizerContext,
        ) -> Result<PassResult, String> {
            if !self.0.swap(true, Ordering::SeqCst) {
                context.diagnostics.push(Diagnostic {
                    code: "stale-placement",
                    message: "discard after restart".to_owned(),
                });
                Ok(PassResult::changed().request_replan())
            } else {
                Ok(PassResult::unchanged())
            }
        }
    }

    struct NamedNoop(&'static str);
    impl OptimizerPass for NamedNoop {
        fn name(&self) -> &'static str {
            self.0
        }
        fn run(
            &self,
            _plan: &mut LogicalPlan,
            _context: &mut OptimizerContext,
        ) -> Result<PassResult, String> {
            Ok(PassResult::unchanged())
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
    fn optional_snapshots_do_not_change_passes_or_plan() {
        let mut plan = LogicalPlan::new();
        let root = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        plan.set_root(root).unwrap();
        let mut registry = PlanRegistry::default();
        registry.register_pass(Arc::new(CountDown(0)));
        let mut recorded = plan.clone();
        let (context, report) =
            optimize_with_options(&mut plan, &registry, OptimizerOptions::default()).unwrap();
        let (recorded_context, recorded_report) = optimize(&mut recorded, &registry).unwrap();
        assert!(context.snapshots.is_empty());
        assert_eq!(recorded_context.snapshots.len(), 1);
        assert_eq!(report.passes_run, recorded_report.passes_run);
        assert_eq!(report.diagnostics, recorded_report.diagnostics);
        assert_eq!(plan.explain().unwrap(), recorded.explain().unwrap());
    }

    #[test]
    fn annotation_invalidation_discards_cached_placement() {
        let mut context = OptimizerContext::default();
        context.set_placement(crate::PlacementPlan::default());
        assert!(context.placement().is_some());
        context.set_annotations(PropertyAnnotations::default());
        assert!(context.placement().is_none());
        context.set_placement(crate::PlacementPlan::default());
        context.annotations_mut();
        assert!(context.placement().is_none());
        context.set_placement(crate::PlacementPlan::default());
        context.clear_annotations();
        assert!(context.placement().is_none());
    }

    #[test]
    fn optimizer_restarts_pass_sequence_when_a_pass_requests_replan() {
        let mut plan = LogicalPlan::new();
        let root = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        plan.set_root(root).unwrap();
        let mut registry = PlanRegistry::default();
        registry.register_pass(Arc::new(NamedNoop("placement-boundary")));
        registry.register_pass(Arc::new(ReplanOnce(AtomicBool::new(false))));

        let (context, report) = optimize(&mut plan, &registry).unwrap();
        assert_eq!(report.iterations, 2);
        assert_eq!(
            report.passes_run,
            [
                "placement-boundary",
                "late-semantic-rewrite",
                "placement-boundary",
                "late-semantic-rewrite"
            ]
        );
        assert!(
            context
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "optimizer.replan")
        );
        assert!(
            !context
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "stale-placement")
        );
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
