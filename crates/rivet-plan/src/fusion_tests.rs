use super::*;
use crate::{LogicalNode, NodeKind, PlanPayload, ValueProperties};
use std::sync::Mutex;

#[derive(Clone)]
struct Add {
    amount: u64,
    fused: bool,
}
impl PlanPayload for Add {
    fn domain(&self) -> &'static str {
        "test"
    }
    fn name(&self) -> &'static str {
        "add"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
struct Inference;
impl PropertyInference for Inference {
    fn infer_node(
        &self,
        _node: &LogicalNode,
        _inputs: &[ValueProperties],
    ) -> Result<ValueProperties, String> {
        Ok(ValueProperties::unknown())
    }
}
#[derive(Clone, Copy)]
enum Action {
    Fuse,
    Noop,
    Fail,
    Cycle,
    Escape,
    DamageShared,
    Compact,
}
struct Rule {
    name: &'static str,
    priority: i32,
    sharing: FusionSharing,
    action: Action,
    reject: bool,
    outer: Option<u64>,
    calls: Arc<Mutex<Vec<(&'static str, PlanRevision)>>>,
}
impl Rule {
    fn new(name: &'static str, calls: &Arc<Mutex<Vec<(&'static str, PlanRevision)>>>) -> Self {
        Self {
            name,
            priority: 0,
            sharing: FusionSharing::Exclusive,
            action: Action::Fuse,
            reject: false,
            outer: None,
            calls: calls.clone(),
        }
    }
}
impl FusionRule for Rule {
    fn name(&self) -> &'static str {
        self.name
    }
    fn priority(&self) -> i32 {
        self.priority
    }
    fn discover(
        &self,
        plan: &LogicalPlan,
        annotations: &PropertyAnnotations,
    ) -> Result<Vec<FusionCandidate>, String> {
        let mut result = vec![];
        // Deliberately emit reverse graph order: framework sorting is explicit.
        for id in plan.preorder().map_err(|e| e.to_string())? {
            let node = plan.node(id).unwrap();
            if node.kind() != NodeKind::Op {
                continue;
            }
            let Some(outer) = node.payload_as::<Add>() else {
                continue;
            };
            if outer.fused || self.outer.is_some_and(|wanted| wanted != outer.amount) {
                continue;
            }
            let Some(inner) = node.inputs().get(0) else {
                continue;
            };
            if plan.node(inner).unwrap().kind() != NodeKind::Op
                || plan.node(inner).unwrap().payload_as::<Add>().is_none()
            {
                continue;
            }
            let mut candidate = FusionCandidate::new(
                plan,
                annotations,
                self.name,
                "add-pair",
                vec![inner, id],
                vec![id],
            )?;
            candidate.sharing = self.sharing;
            result.push(candidate);
        }
        Ok(result)
    }
    fn validate(
        &self,
        _plan: &LogicalPlan,
        _candidate: &FusionCandidate,
        _annotations: &PropertyAnnotations,
    ) -> Result<(), String> {
        if self.reject {
            Err("domain precondition rejected".into())
        } else {
            Ok(())
        }
    }
    fn apply(
        &self,
        plan: &mut LogicalPlan,
        candidate: &FusionCandidate,
        context: &mut OptimizerContext,
    ) -> Result<PassResult, String> {
        self.calls
            .lock()
            .unwrap()
            .push((self.name, candidate.revision));
        assert_eq!(candidate.revision, plan.revision());
        context.diagnostics.push(Diagnostic {
            code: "test.trial",
            message: "transaction-local diagnostic".into(),
        });
        if matches!(self.action, Action::Noop) {
            context.clear_annotations();
            return Ok(PassResult::changed());
        }
        let inner = candidate.nodes[0];
        let outer = candidate.nodes[1];
        let amount = plan
            .node(inner)
            .unwrap()
            .payload_as::<Add>()
            .unwrap()
            .amount
            + plan
                .node(outer)
                .unwrap()
                .payload_as::<Add>()
                .unwrap()
                .amount;
        let input = plan.node(inner).unwrap().inputs().get(0).unwrap();
        plan.replace_node(
            outer,
            LogicalNode::new(
                NodeKind::Op,
                [input],
                Some(Arc::new(Add {
                    amount,
                    fused: true,
                })),
            ),
        )
        .unwrap();
        match self.action {
            Action::Fail => {
                context.clear_annotations();
                return Err("failed after editing trial".into());
            }
            Action::Cycle => {
                plan.replace_node(outer, LogicalNode::new(NodeKind::Op, [outer], None))
                    .unwrap();
            }
            Action::Escape => {
                let source = plan.node(input).unwrap().clone().with_semantic_identity(99);
                plan.replace_node(input, source).unwrap();
            }
            Action::DamageShared => {
                plan.replace_node(inner, LogicalNode::new(NodeKind::Op, [input], None))
                    .unwrap();
            }
            Action::Compact => {
                plan.prune_unreachable().unwrap();
            }
            _ => {}
        }
        Ok(PassResult::changed())
    }
}
fn source(plan: &mut LogicalPlan) -> NodeId {
    plan.add_node(
        LogicalNode::new(
            NodeKind::Source,
            [],
            Some(Arc::new(Add {
                amount: 1,
                fused: false,
            })),
        )
        .with_semantic_identity(42),
    )
}
fn add(plan: &mut LogicalPlan, input: NodeId, amount: u64) -> NodeId {
    plan.add_node(LogicalNode::new(
        NodeKind::Op,
        [input],
        Some(Arc::new(Add {
            amount,
            fused: false,
        })),
    ))
}
fn chain() -> LogicalPlan {
    let mut plan = LogicalPlan::new();
    let s = source(&mut plan);
    let a = add(&mut plan, s, 2);
    let b = add(&mut plan, a, 3);
    let c = add(&mut plan, b, 5);
    plan.set_root(c).unwrap();
    plan
}
fn diamond() -> (LogicalPlan, NodeId, NodeId) {
    let mut plan = LogicalPlan::new();
    let s = source(&mut plan);
    let shared = add(&mut plan, s, 2);
    let left = add(&mut plan, shared, 3);
    let right = add(&mut plan, shared, 7);
    let root = plan.add_node(LogicalNode::new(NodeKind::Sink, [left, right], None));
    plan.set_root(root).unwrap();
    (plan, shared, right)
}
fn value(plan: &LogicalPlan, id: NodeId) -> u64 {
    let node = plan.node(id).unwrap();
    node.payload_as::<Add>().map_or(0, |op| op.amount)
        + node
            .inputs()
            .iter()
            .map(|input| value(plan, input))
            .sum::<u64>()
}
fn registry(rules: Vec<Rule>) -> PlanRegistry {
    let mut registry = PlanRegistry::default();
    registry.inference.push((None, Arc::new(Inference)));
    for rule in rules {
        registry.register_fusion_rule(Arc::new(rule));
    }
    registry
}
fn run(
    plan: &mut LogicalPlan,
    registry: &PlanRegistry,
    context: &mut OptimizerContext,
) -> Result<PassResult, OptimizerError> {
    run_fusion_rules(
        plan,
        registry,
        context,
        &mut OptimizationReport::default(),
        &mut String::new(),
        OptimizerOptions::default(),
    )
}
#[test]
fn overlapping_candidates_are_rediscovered_and_optimization_is_idempotent() {
    let calls = Arc::new(Mutex::new(vec![]));
    let registry = registry(vec![Rule::new("pairs", &calls)]);
    let mut plan = chain();
    let expected = value(&plan, plan.root().unwrap());
    let mut context = OptimizerContext::default();
    assert!(run(&mut plan, &registry, &mut context).unwrap().changed);
    assert_eq!(value(&plan, plan.root().unwrap()), expected);
    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_ne!(calls[0].1, calls[1].1);
    drop(calls);
    let revision = plan.revision();
    let structure = plan.explain().unwrap();
    assert!(!run(&mut plan, &registry, &mut context).unwrap().changed);
    assert_eq!(revision, plan.revision());
    assert_eq!(structure, plan.explain().unwrap());
}
#[test]
fn competition_uses_priority_and_name_instead_of_registration_order() {
    for reverse in [false, true] {
        for tie in [false, true] {
            let calls = Arc::new(Mutex::new(vec![]));
            let winner = if tie { "alpha" } else { "zeta" };
            let loser = if tie { "zeta" } else { "alpha" };
            let mut high = Rule::new(winner, &calls);
            high.priority = if tie { 0 } else { 10 };
            let low = Rule::new(loser, &calls);
            let registry = registry(if reverse {
                vec![low, high]
            } else {
                vec![high, low]
            });
            let mut plan = chain();
            run(&mut plan, &registry, &mut OptimizerContext::default()).unwrap();
            assert_eq!(calls.lock().unwrap().len(), 2);
            assert!(
                calls
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|(name, _)| *name == winner)
            );
        }
    }
}
#[test]
fn exclusive_fusion_rejects_shared_intermediates() {
    let calls = Arc::new(Mutex::new(vec![]));
    let registry = registry(vec![Rule::new("pairs", &calls)]);
    let (mut plan, _, _) = diamond();
    let revision = plan.revision();
    let mut context = OptimizerContext::default();
    assert!(!run(&mut plan, &registry, &mut context).unwrap().changed);
    assert_eq!(revision, plan.revision());
    assert!(calls.lock().unwrap().is_empty());
    assert!(
        context
            .diagnostics
            .iter()
            .any(|d| d.code == "fusion.rejected" && d.message.contains("shared intermediate"))
    );
}
#[test]
fn explicit_shared_fusion_preserves_other_consumers_and_value() {
    let calls = Arc::new(Mutex::new(vec![]));
    let mut rule = Rule::new("pairs", &calls);
    rule.sharing = FusionSharing::PreserveShared;
    rule.outer = Some(3);
    let registry = registry(vec![rule]);
    let (mut plan, shared, right) = diamond();
    let expected = value(&plan, plan.root().unwrap());
    let original = plan.node(shared).unwrap().clone();
    run(&mut plan, &registry, &mut OptimizerContext::default()).unwrap();
    assert_eq!(value(&plan, plan.root().unwrap()), expected);
    assert!(original.same_definition(plan.node(shared).unwrap()));
    assert_eq!(plan.node(right).unwrap().inputs().get(0), Some(shared));
}
#[test]
fn failed_rule_does_not_commit_partial_graph_or_context_edits() {
    let calls = Arc::new(Mutex::new(vec![]));
    let mut rule = Rule::new("fail", &calls);
    rule.action = Action::Fail;
    let registry = registry(vec![rule]);
    let mut plan = chain();
    let revision = plan.revision();
    let structure = plan.explain().unwrap();
    let mut context = OptimizerContext::default();
    context.set_annotations(plan.infer_properties(&Inference).unwrap());
    assert!(run(&mut plan, &registry, &mut context).is_err());
    assert_eq!(revision, plan.revision());
    assert_eq!(structure, plan.explain().unwrap());
    assert!(context.annotations().is_valid_for(&plan));
    assert!(!context.diagnostics.iter().any(|d| d.code == "test.trial"));
}
#[test]
fn invalid_graph_boundary_edits_and_compaction_are_not_committed() {
    for action in [Action::Cycle, Action::Escape, Action::Compact] {
        let calls = Arc::new(Mutex::new(vec![]));
        let mut rule = Rule::new("invalid", &calls);
        rule.action = action;
        let registry = registry(vec![rule]);
        let mut plan = chain();
        let revision = plan.revision();
        let mut context = OptimizerContext::default();
        assert!(!run(&mut plan, &registry, &mut context).unwrap().changed);
        assert_eq!(revision, plan.revision());
        assert!(
            context
                .diagnostics
                .iter()
                .any(|d| d.code == "fusion.rejected")
        );
        assert!(!context.diagnostics.iter().any(|d| d.code == "test.trial"));
    }
}
#[test]
fn preserve_shared_does_not_authorize_destroying_the_shared_value() {
    let calls = Arc::new(Mutex::new(vec![]));
    let mut rule = Rule::new("invalid", &calls);
    rule.action = Action::DamageShared;
    rule.outer = Some(3);
    rule.sharing = FusionSharing::PreserveShared;
    let registry = registry(vec![rule]);
    let (mut plan, _, _) = diamond();
    let revision = plan.revision();
    assert!(
        !run(&mut plan, &registry, &mut OptimizerContext::default())
            .unwrap()
            .changed
    );
    assert_eq!(revision, plan.revision());
}
#[test]
fn no_op_and_domain_rejection_allow_lower_priority_candidate_to_run() {
    for no_op in [false, true] {
        let calls = Arc::new(Mutex::new(vec![]));
        let mut high = Rule::new("alpha", &calls);
        high.priority = 10;
        if no_op {
            high.action = Action::Noop;
        } else {
            high.reject = true;
        }
        let registry = registry(vec![high, Rule::new("zeta", &calls)]);
        let mut plan = chain();
        let mut context = OptimizerContext::default();
        run(&mut plan, &registry, &mut context).unwrap();
        assert!(
            calls
                .lock()
                .unwrap()
                .iter()
                .any(|(name, _)| *name == "zeta")
        );
        assert!(context.diagnostics.iter().any(|d| d.code
            == if no_op {
                "fusion.unchanged"
            } else {
                "fusion.rejected"
            }));
    }
}
#[test]
fn no_op_candidates_stop_without_claiming_changes() {
    let calls = Arc::new(Mutex::new(vec![]));
    let mut rule = Rule::new("noop", &calls);
    rule.action = Action::Noop;
    let registry = registry(vec![rule]);
    let mut plan = chain();
    let mut context = OptimizerContext::default();
    assert!(!run(&mut plan, &registry, &mut context).unwrap().changed);
    assert_eq!(calls.lock().unwrap().len(), 2);
    assert!(context.annotations().is_valid_for(&plan));
    assert!(!context.diagnostics.iter().any(|d| d.code == "test.trial"));
}
#[test]
fn matching_revision_cannot_hide_changed_edges_or_properties() {
    let plan = chain();
    let annotations = plan.infer_properties(&Inference).unwrap();
    let calls = Arc::new(Mutex::new(vec![]));
    let rule = Rule::new("pairs", &calls);
    let candidate = rule.discover(&plan, &annotations).unwrap().remove(0);
    assert!(candidate.validate(&plan, &annotations).is_ok());
    let mut wrong_edges = candidate.clone();
    wrong_edges.inputs[0].1.clear();
    assert!(
        wrong_edges
            .validate(&plan, &annotations)
            .unwrap_err()
            .contains("input edges")
    );
    let mut wrong_properties = candidate;
    wrong_properties.properties[0].1.dtype = Some(crate::DataType::F32);
    assert!(
        wrong_properties
            .validate(&plan, &annotations)
            .unwrap_err()
            .contains("property prerequisites")
    );
}
#[test]
fn duplicate_rule_names_are_rejected() {
    let calls = Arc::new(Mutex::new(vec![]));
    let registry = registry(vec![Rule::new("same", &calls), Rule::new("same", &calls)]);
    assert!(
        run(&mut chain(), &registry, &mut OptimizerContext::default())
            .unwrap_err()
            .to_string()
            .contains("unique")
    );
}

#[test]
fn semantic_inference_failure_does_not_publish_the_trial() {
    struct RejectFused;
    impl PropertyInference for RejectFused {
        fn infer_node(
            &self,
            node: &LogicalNode,
            _inputs: &[ValueProperties],
        ) -> Result<ValueProperties, String> {
            if node.payload_as::<Add>().is_some_and(|op| op.fused) {
                Err("fused output is not supported".into())
            } else {
                Ok(ValueProperties::unknown())
            }
        }
    }
    let calls = Arc::new(Mutex::new(vec![]));
    let mut registry = registry(vec![Rule::new("pairs", &calls)]);
    registry.inference = vec![(None, Arc::new(RejectFused))];
    let mut plan = chain();
    let revision = plan.revision();
    let mut context = OptimizerContext::default();
    context.set_annotations(plan.infer_properties(&Inference).unwrap());
    assert!(
        run(&mut plan, &registry, &mut context)
            .unwrap_err()
            .to_string()
            .contains("fused output is not supported")
    );
    assert_eq!(revision, plan.revision());
    assert!(context.annotations().is_valid_for(&plan));
}

#[test]
fn disjoint_matches_are_rediscovered_after_each_commit() {
    let calls = Arc::new(Mutex::new(vec![]));
    let registry = registry(vec![Rule::new("pairs", &calls)]);
    let mut plan = LogicalPlan::new();
    let s = source(&mut plan);
    let a = add(&mut plan, s, 2);
    let left = add(&mut plan, a, 3);
    let b = add(&mut plan, s, 7);
    let right = add(&mut plan, b, 11);
    let root = plan.add_node(LogicalNode::new(NodeKind::Sink, [right, left], None));
    plan.set_root(root).unwrap();
    let expected = value(&plan, root);
    run(&mut plan, &registry, &mut OptimizerContext::default()).unwrap();
    assert_eq!(value(&plan, root), expected);
    assert_eq!(calls.lock().unwrap().len(), 2);
    let applied = calls.lock().unwrap();
    assert_ne!(applied[0].1, applied[1].1);
}
#[test]
fn full_optimizer_converges_with_noops_competition_and_rediscovery() {
    struct Discover;
    impl OptimizerPass for Discover {
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
    let calls = Arc::new(Mutex::new(vec![]));
    let mut noop = Rule::new("noop", &calls);
    noop.priority = 5;
    noop.action = Action::Noop;
    let mut registry = registry(vec![noop, Rule::new("pairs", &calls)]);
    registry.register_pass(Arc::new(Discover));
    let mut plan = chain();
    let expected = value(&plan, plan.root().unwrap());
    let (_, report) = optimize(&mut plan, &registry).unwrap();
    assert!(
        report
            .fusion_candidates
            .iter()
            .all(|c| c.revision == plan.revision())
    );
    assert_eq!(value(&plan, plan.root().unwrap()), expected);
    let revision = plan.revision();
    let explain = plan.explain().unwrap();
    optimize(&mut plan, &registry).unwrap();
    assert_eq!(revision, plan.revision());
    assert_eq!(explain, plan.explain().unwrap());
}
