//! Explicit, backend-neutral logical optimization passes.
//!
//! Mutation protocol:
//! - LogicalPlan mutators change revision only for an effective edit. Erased
//!   payloads use Arc identity; newly allocated payloads conservatively count as
//!   edits. Payload semantics must be immutable after insertion.
//! - The driver synchronizes analyses before every pass and invalidates stale
//!   results after an edit, even if the pass reports `changed = false`.
//!   Analysis recomputation alone never counts toward fixed-point convergence.
//! - Passes declare `requires_properties`; the registry dispatches inference by
//!   domain whenever those prerequisites are stale. Standalone run_fixed_point
//!   requires the caller to supply current prerequisites instead.
//! - A pass may give a preservation proof and update all affected annotations.
//!   Compaction invalidates the NodeId mapping and cannot use this shortcut.
//!   Placement and candidate revisions must still match independently.
//! - Stages label canonicalization, semantic rewrites, physical selection and
//!   execution preparation in reports. Plugin order remains authoritative:
//!   pre-placement rewrites iterate together, and is_placement_boundary marks
//!   final selection. Late semantic changes request a full replan.
//! - Outside the driver, call OptimizerContext::synchronize before reading
//!   cached results. Context reuse assumes the same inference configuration;
//!   clear annotations when changing provider configuration.
//! - Full snapshots are opt-in; nonconvergence retains only the latest changing
//!   pass and a bounded graph summary by default.

use std::sync::Arc;

use thiserror::Error;

use crate::{
    DomainId, LogicalNode, LogicalPlan, NodeId, PlanRevision, PropertyAnnotations,
    PropertyInference,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: &'static str,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PassResult {
    /// A hint for direct callers; the driver derives actual IR change from revision.
    pub changed: bool,
    pub diagnostics: Vec<Diagnostic>,
    /// Request a restart from canonicalization after this pass changes a plan
    /// that has already passed through placement.
    pub replan: bool,
    /// Descriptive change scope; revision tracking remains authoritative.
    pub changed_nodes: Vec<NodeId>,
    pub topology_changed: bool,
    /// Explicit proof that all affected annotations were updated. Default:
    /// conservatively invalidate all analyses after an IR mutation.
    pub preserves_properties: Option<&'static str>,
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
            ..Self::default()
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
    #[error(
        "optimizer did not converge after {iterations} iterations; last changing pass={last_pass:?}; {summary}"
    )]
    NonConvergent {
        iterations: usize,
        last_pass: Option<String>,
        summary: String,
    },
}

/// Shared state is deliberately supplied to every pass so passes can inspect
/// and update inferred properties without coupling the core IR to a domain.
#[derive(Default)]
pub struct OptimizerContext {
    annotations: PropertyAnnotations,
    placement: Option<crate::PlacementPlan>,
    pub diagnostics: Vec<Diagnostic>,
    pub snapshots: Vec<(String, String)>,
    last_changing_pass: Option<String>,
}

impl OptimizerContext {
    /// Call before consuming analysis outside the optimizer driver too.
    pub fn synchronize(&mut self, plan: &LogicalPlan) {
        if !self.annotations.is_valid_for(plan) {
            self.clear_annotations();
        }
        if self
            .placement
            .as_ref()
            .is_some_and(|p| !p.is_valid_for(plan))
        {
            self.placement = None;
        }
    }

    fn finish_change(
        &mut self,
        plan: &LogicalPlan,
        before: PlanRevision,
        result: &mut PassResult,
        pass: &str,
    ) {
        result.changed = before != plan.revision();
        if result.changed {
            self.last_changing_pass = Some(pass.to_owned());
            // A freshly inferred result already carries the new revision.
            if result.preserves_properties.is_some() {
                self.annotations.certify_preserved(plan);
            }
            self.synchronize(plan);
        }
    }

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

/// Stages describe the semantic/physical boundary, not analysis mutations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizerStage {
    Canonicalization,
    #[default]
    Semantic,
    PhysicalSelection,
    ExecutionPreparation,
}

pub trait OptimizerPass: Send + Sync {
    fn stage(&self) -> OptimizerStage {
        if self.is_placement_boundary() || self.is_fusion_discovery() {
            OptimizerStage::PhysicalSelection
        } else {
            OptimizerStage::Semantic
        }
    }
    /// The driver recomputes stale facts using registered domain inference.
    fn requires_properties(&self) -> bool {
        false
    }

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
    pub revision: PlanRevision,
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
    pub revision: Option<PlanRevision>,
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
        plan: &LogicalPlan,
        node: &LogicalNode,
        properties: Option<&crate::ValueProperties>,
    ) -> Vec<PhysicalCandidate> {
        self.physical_candidates
            .iter()
            .flat_map(|provider| provider.candidates(node, properties))
            .map(|mut candidate| {
                candidate.revision = Some(plan.revision());
                candidate
            })
            .collect()
    }
}

#[derive(Clone, Debug, Default)]
pub struct OptimizationReport {
    pub passes_run: Vec<&'static str>,
    pub stages_run: Vec<OptimizerStage>,
    pub diagnostics: Vec<Diagnostic>,
    pub iterations: usize,
    pub fusion_candidates: Vec<FusionCandidate>,
}

/// Run ordered passes until they complete without requesting a placement
/// restart. Validate after each pass; full snapshots require explicit options.
pub fn optimize(
    plan: &mut LogicalPlan,
    registry: &PlanRegistry,
) -> Result<(OptimizerContext, OptimizationReport), OptimizerError> {
    optimize_with_options(plan, registry, OptimizerOptions::default())
}

pub fn optimize_with_options(
    plan: &mut LogicalPlan,
    registry: &PlanRegistry,
    options: OptimizerOptions,
) -> Result<(OptimizerContext, OptimizationReport), OptimizerError> {
    let mut context = OptimizerContext::default();
    let report = optimize_with_context(plan, registry, options, &mut context)?;
    Ok((context, report))
}

/// Reuse certified analysis across repeated optimization of an unchanged plan.
/// Diagnostics and snapshots describe this invocation only.
pub fn optimize_with_context(
    plan: &mut LogicalPlan,
    registry: &PlanRegistry,
    options: OptimizerOptions,
    context: &mut OptimizerContext,
) -> Result<OptimizationReport, OptimizerError> {
    context.diagnostics.clear();
    context.snapshots.clear();
    context.last_changing_pass = None;
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
        context.synchronize(plan);
        let mut requested_replan = false;
        let mut converged = false;

        for _ in 0..MAX_SEMANTIC_ITERATIONS {
            let mut changed = false;
            for pass in semantic_passes {
                let result = run_optimizer_pass(
                    plan,
                    registry,
                    pass.as_ref(),
                    context,
                    &mut report,
                    options,
                )?;
                changed |= result.changed;
                if result.replan {
                    request_full_replan(pass.name(), context, &mut report, &mut requested_replan);
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
            return Err(non_convergent(plan, context, MAX_SEMANTIC_ITERATIONS));
        }

        for pass in placement_passes {
            let result =
                run_optimizer_pass(plan, registry, pass.as_ref(), context, &mut report, options)?;
            if result.replan {
                request_full_replan(pass.name(), context, &mut report, &mut requested_replan);
                break;
            }
        }
        if requested_replan {
            report.iterations = iteration;
            continue;
        }

        report.iterations = iteration;
        report.diagnostics = context.diagnostics.clone();
        return Ok(report);
    }
    Err(non_convergent(plan, context, MAX_REPLANS))
}

fn ensure_properties(
    plan: &LogicalPlan,
    registry: &PlanRegistry,
    context: &mut OptimizerContext,
    pass_name: &str,
) -> Result<(), OptimizerError> {
    if !context.annotations().is_valid_for(plan) {
        struct RegistryInference<'a>(&'a PlanRegistry);
        impl PropertyInference for RegistryInference<'_> {
            fn infer_node(
                &self,
                node: &LogicalNode,
                inputs: &[crate::ValueProperties],
            ) -> Result<crate::ValueProperties, String> {
                let provider = self
                    .0
                    .inferences()
                    .find_map(|(domain, provider)| {
                        (domain.is_none() || domain == node.payload().and_then(|p| p.domain_id()))
                            .then_some(provider)
                    })
                    .ok_or_else(|| {
                        "pass requires properties but no matching inference provider is registered"
                            .to_owned()
                    })?;
                provider.infer_node(node, inputs)
            }
        }
        let properties = plan
            .infer_properties(&RegistryInference(registry))
            .map_err(|e| OptimizerError::Pass {
                pass: pass_name.into(),
                message: e.to_string(),
            })?;
        context.set_annotations(properties);
    }
    Ok(())
}

fn run_optimizer_pass(
    plan: &mut LogicalPlan,
    registry: &PlanRegistry,
    pass: &dyn OptimizerPass,
    context: &mut OptimizerContext,
    report: &mut OptimizationReport,
    options: OptimizerOptions,
) -> Result<PassResult, OptimizerError> {
    context.synchronize(plan);
    if pass.requires_properties() {
        ensure_properties(plan, registry, context, pass.name())?;
    }
    let before = plan.revision();
    let mut result = pass.run(plan, context).map_err(|message| {
        context.synchronize(plan);
        OptimizerError::Pass {
            pass: pass.name().into(),
            message,
        }
    })?;
    context.finish_change(plan, before, &mut result, pass.name());
    let mut fusion_snapshot = String::new();
    if pass.is_fusion_discovery() {
        for rule in registry.fusion_rules() {
            ensure_properties(plan, registry, context, rule.name())?;
            let candidates = rule
                .discover(plan, context.annotations())
                .map_err(|message| OptimizerError::Pass {
                    pass: rule.name().into(),
                    message,
                })?;
            for candidate in &candidates {
                if candidate.revision != plan.revision() {
                    context.diagnostics.push(Diagnostic {
                        code: "fusion.stale-revision",
                        message: format!(
                            "{}: skipped candidate from {:?}; current {:?}",
                            candidate.rule,
                            candidate.revision,
                            plan.revision()
                        ),
                    });
                    continue;
                }
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
                    let before = plan.revision();
                    let mut applied = rule.apply(plan, candidate, context).map_err(|message| {
                        context.synchronize(plan);
                        OptimizerError::Pass {
                            pass: rule.name().into(),
                            message,
                        }
                    })?;
                    context.finish_change(plan, before, &mut applied, rule.name());
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
    report
        .fusion_candidates
        .retain(|candidate| candidate.revision == plan.revision());
    report.passes_run.push(pass.name());
    report.stages_run.push(pass.stage());
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
        context.synchronize(plan);
        if pass.requires_properties() && !context.annotations().is_valid_for(plan) {
            return Err(OptimizerError::Pass { pass: pass.name().into(), message: "run_fixed_point requires current properties; use the registry optimizer for automatic inference".into() });
        }
        let before = plan.revision();
        let mut result = pass.run(plan, context).map_err(|message| {
            context.synchronize(plan);
            OptimizerError::Pass {
                pass: pass.name().into(),
                message,
            }
        })?;
        context.finish_change(plan, before, &mut result, pass.name());
        context.diagnostics.extend(result.diagnostics);
        plan.validate().map_err(|error| OptimizerError::Pass {
            pass: pass.name().into(),
            message: error.to_string(),
        })?;
        if !result.changed {
            return Ok(iteration);
        }
    }
    Err(non_convergent(plan, context, max_iterations))
}

fn non_convergent(
    plan: &LogicalPlan,
    context: &OptimizerContext,
    iterations: usize,
) -> OptimizerError {
    OptimizerError::NonConvergent {
        iterations,
        last_pass: context.last_changing_pass.clone(),
        summary: format!(
            "revision={:?}, nodes={}, root={:?}",
            plan.revision(),
            plan.nodes().len(),
            plan.root().ok()
        ),
    }
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
            plan: &mut LogicalPlan,
            _context: &mut OptimizerContext,
        ) -> Result<PassResult, String> {
            plan.add_node(LogicalNode::new(NodeKind::Op, [], None));
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
        let (context, report) = optimize_with_options(
            &mut plan,
            &registry,
            OptimizerOptions {
                record_snapshots: true,
            },
        )
        .unwrap();
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
        let (recorded_context, recorded_report) = optimize_with_options(
            &mut recorded,
            &registry,
            OptimizerOptions {
                record_snapshots: true,
            },
        )
        .unwrap();
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
            OptimizerError::NonConvergent { iterations: 3, .. }
        ));
    }
    struct CountInference(Arc<std::sync::atomic::AtomicUsize>);
    impl PropertyInference for CountInference {
        fn infer_node(
            &self,
            _node: &LogicalNode,
            inputs: &[crate::ValueProperties],
        ) -> Result<crate::ValueProperties, String> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(inputs.first().cloned().unwrap_or_default())
        }
    }
    struct NeedsProperties;
    impl OptimizerPass for NeedsProperties {
        fn name(&self) -> &'static str {
            "needs-properties"
        }
        fn requires_properties(&self) -> bool {
            true
        }
        fn run(
            &self,
            plan: &mut LogicalPlan,
            context: &mut OptimizerContext,
        ) -> Result<PassResult, String> {
            assert!(context.annotations().is_valid_for(plan));
            // Analysis-only work must not prolong the semantic fixed point.
            Ok(PassResult::changed())
        }
    }

    #[test]
    fn analysis_reuses_revision_and_does_not_prolong_fixed_point() {
        let mut plan = LogicalPlan::new();
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        plan.set_root(source).unwrap();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut registry = PlanRegistry::default();
        registry
            .inference
            .push((None, Arc::new(CountInference(calls.clone()))));
        registry.register_pass(Arc::new(NeedsProperties));
        let (mut context, first) =
            optimize_with_options(&mut plan, &registry, OptimizerOptions::default()).unwrap();
        let revision = plan.revision();
        let second = optimize_with_context(
            &mut plan,
            &registry,
            OptimizerOptions::default(),
            &mut context,
        )
        .unwrap();
        assert_eq!(first.passes_run, second.passes_run);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(revision, plan.revision());
    }

    #[test]
    fn every_graph_mutation_invalidates_properties_and_placement() {
        let mut plan = LogicalPlan::new();
        let dead = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let root = plan.add_node(LogicalNode::new(NodeKind::Op, [source], None));
        plan.set_root(root).unwrap();
        let inference = CountInference(Arc::new(std::sync::atomic::AtomicUsize::new(0)));
        let mut context = OptimizerContext::default();
        for mutation in 0..6 {
            context.set_annotations(plan.infer_properties(&inference).unwrap());
            context.set_placement(crate::PlacementPlan {
                revision: Some(plan.revision()),
                ..Default::default()
            });
            let revision = plan.revision();
            match mutation {
                0 => {
                    plan.replace_node(
                        root,
                        LogicalNode::new(NodeKind::Op, [source], None).with_semantic_identity(5),
                    )
                    .unwrap();
                }
                1 => {
                    plan.replace_node(root, LogicalNode::new(NodeKind::Op, [dead], None))
                        .unwrap();
                }
                2 => {
                    plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
                }
                3 => {
                    plan.set_root(source).unwrap();
                }
                4 => {
                    plan.prune_unreachable().unwrap();
                }
                _ => {
                    plan.set_context(Arc::new(crate::DeviceCut::new(crate::DeviceTarget::metal(
                        0,
                    ))));
                }
            }
            assert_ne!(plan.revision(), revision);
            assert!(!context.annotations().is_valid_for(&plan));
            context.synchronize(&plan);
            assert_eq!(context.annotations().iter().count(), 0);
            assert!(context.placement().is_none());
        }
    }

    #[test]
    fn no_op_mutations_and_clone_branches_have_reliable_revisions() {
        let mut plan = LogicalPlan::new();
        let root = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        plan.set_root(root).unwrap();
        let revision = plan.revision();
        plan.set_root(root).unwrap();
        plan.replace_node(root, plan.node(root).unwrap().clone())
            .unwrap();
        plan.redirect_uses(root, root).unwrap();
        plan.prune_unreachable().unwrap();
        assert_eq!(revision, plan.revision());
        let mut other = plan.clone();
        plan.add_node(LogicalNode::new(NodeKind::Op, [], None));
        other.add_node(LogicalNode::new(NodeKind::Sink, [], None));
        assert_ne!(plan.revision(), other.revision());
    }

    #[test]
    fn unreported_ir_change_still_invalidates_and_iterates() {
        struct SilentRewrite;
        impl OptimizerPass for SilentRewrite {
            fn name(&self) -> &'static str {
                "silent-rewrite"
            }
            fn run(
                &self,
                plan: &mut LogicalPlan,
                _context: &mut OptimizerContext,
            ) -> Result<PassResult, String> {
                let root = plan.root().unwrap();
                let node = plan.node(root).unwrap().clone();
                if node.semantic_identity().is_none() {
                    plan.replace_node(root, node.with_semantic_identity(1))
                        .unwrap();
                }
                Ok(PassResult::unchanged())
            }
        }
        let mut plan = LogicalPlan::new();
        let root = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        plan.set_root(root).unwrap();
        assert_eq!(
            run_fixed_point(
                &mut plan,
                &SilentRewrite,
                &mut OptimizerContext::default(),
                3
            )
            .unwrap(),
            2
        );
    }

    #[test]
    fn compaction_rejects_property_preservation_with_old_node_ids() {
        let mut plan = LogicalPlan::new();
        plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        let source = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        plan.set_root(source).unwrap();
        let inference = CountInference(Arc::new(std::sync::atomic::AtomicUsize::new(0)));
        let mut facts = plan.infer_properties(&inference).unwrap();
        let mapping = plan.prune_unreachable().unwrap();
        assert_eq!(mapping[source.index()].unwrap().index(), 0);
        assert!(!facts.certify_preserved(&plan));
        assert!(!facts.is_valid_for(&plan));
        assert_eq!(facts.iter().count(), 0);
    }

    #[test]
    fn property_preservation_requires_an_explicit_proof() {
        struct Preserve;
        impl OptimizerPass for Preserve {
            fn name(&self) -> &'static str {
                "preserve-semantic-identity"
            }
            fn run(
                &self,
                plan: &mut LogicalPlan,
                _context: &mut OptimizerContext,
            ) -> Result<PassResult, String> {
                let root = plan.root().unwrap();
                let node = plan.node(root).unwrap().clone().with_semantic_identity(3);
                plan.replace_node(root, node).unwrap();
                Ok(PassResult {
                    changed_nodes: vec![root],
                    preserves_properties: Some(
                        "this deterministic source's identity does not change inferred value properties",
                    ),
                    ..Default::default()
                })
            }
        }
        let mut plan = LogicalPlan::new();
        let root = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        plan.set_root(root).unwrap();
        let inference = CountInference(Arc::new(std::sync::atomic::AtomicUsize::new(0)));
        let mut context = OptimizerContext::default();
        context.set_annotations(plan.infer_properties(&inference).unwrap());
        context.set_placement(crate::PlacementPlan {
            revision: Some(plan.revision()),
            ..Default::default()
        });
        assert_eq!(
            run_fixed_point(&mut plan, &Preserve, &mut context, 3).unwrap(),
            2
        );
        assert!(context.annotations().is_valid_for(&plan));
        assert!(context.placement().is_none());
    }

    #[test]
    fn nonconvergence_reports_last_pass_and_bounded_graph_summary() {
        let mut plan = LogicalPlan::new();
        let root = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        plan.set_root(root).unwrap();
        let error = run_fixed_point(
            &mut plan,
            &AlwaysChanges,
            &mut OptimizerContext::default(),
            2,
        )
        .unwrap_err();
        assert!(
            matches!(error, OptimizerError::NonConvergent { iterations: 2, ref last_pass, ref summary } if last_pass.as_deref() == Some("always-changes") && summary.contains("nodes=3"))
        );
        assert!(error.to_string().contains("revision="));
    }
    #[test]
    fn stale_fusion_candidates_are_not_applied_or_returned() {
        struct Discovery;
        impl OptimizerPass for Discovery {
            fn name(&self) -> &'static str {
                "discover"
            }
            fn is_fusion_discovery(&self) -> bool {
                true
            }
            fn run(
                &self,
                _plan: &mut LogicalPlan,
                _context: &mut OptimizerContext,
            ) -> Result<PassResult, String> {
                Ok(PassResult::unchanged())
            }
        }
        struct StaleRule;
        impl FusionRule for StaleRule {
            fn name(&self) -> &'static str {
                "stale"
            }
            fn discover(
                &self,
                plan: &LogicalPlan,
                _annotations: &PropertyAnnotations,
            ) -> Result<Vec<FusionCandidate>, String> {
                Ok(vec![FusionCandidate {
                    revision: PlanRevision::default(),
                    rule: "stale",
                    nodes: vec![plan.root().unwrap()],
                    name: "stale".into(),
                    legal: true,
                    reason: "old revision".into(),
                }])
            }
            fn apply(
                &self,
                _plan: &mut LogicalPlan,
                _candidate: &FusionCandidate,
                _context: &mut OptimizerContext,
            ) -> Result<PassResult, String> {
                panic!("must not apply stale candidate")
            }
        }
        let mut plan = LogicalPlan::new();
        let root = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        plan.set_root(root).unwrap();
        let mut registry = PlanRegistry::default();
        registry.inference.push((
            None,
            Arc::new(CountInference(Arc::new(
                std::sync::atomic::AtomicUsize::new(0),
            ))),
        ));
        registry.register_pass(Arc::new(Discovery));
        registry.register_fusion_rule(Arc::new(StaleRule));
        let (_, report) = optimize(&mut plan, &registry).unwrap();
        assert!(report.fusion_candidates.is_empty());
    }

    #[test]
    fn placement_rejects_stale_properties_and_explain_results() {
        let mut plan = LogicalPlan::new();
        let root = plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        plan.set_root(root).unwrap();
        let inference = CountInference(Arc::new(std::sync::atomic::AtomicUsize::new(0)));
        let facts = plan.infer_properties(&inference).unwrap();
        let placement = crate::PlacementPlan {
            revision: Some(plan.revision()),
            ..Default::default()
        };
        plan.add_node(LogicalNode::new(NodeKind::Source, [], None));
        assert!(matches!(
            crate::place(
                &plan,
                &facts,
                &crate::KernelCapabilities::default(),
                &crate::MachineProfile::default()
            ),
            Err(crate::PlacementError::StaleAnalysis)
        ));
        assert!(matches!(
            placement.explain(&plan, &facts),
            Err(crate::PlacementError::StaleAnalysis)
        ));
    }
}
