use loka_agent::permissions::{ApprovalDecision, ApprovalPolicy, PermissionMode};
use loka_agent::tools::{ToolAccess, ToolRegistry};

#[test]
fn registry_contains_core_harness_tools() {
    let registry = ToolRegistry::built_in();
    let names = registry
        .list()
        .into_iter()
        .map(|tool| tool.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        names,
        vec![
            "git_status",
            "learn_session",
            "memory_propose",
            "memory_search",
            "read_file",
            "search_files",
            "session_list",
            "session_search",
            "shell",
        ]
    );

    assert_eq!(
        registry.get("shell").expect("shell").access,
        ToolAccess::Execute
    );
    assert!(
        registry
            .get("session_search")
            .expect("search")
            .read_only_hint()
    );
}

#[test]
fn approval_policy_auto_allows_read_tools_and_asks_for_writes() {
    let registry = ToolRegistry::built_in();
    let policy = ApprovalPolicy::new(PermissionMode::AutoRead);

    assert_eq!(
        policy.evaluate(&registry, "session_search"),
        ApprovalDecision::Allow
    );
    assert_eq!(
        policy.evaluate(&registry, "memory_propose"),
        ApprovalDecision::Ask
    );
    assert_eq!(policy.evaluate(&registry, "shell"), ApprovalDecision::Ask);
}

#[test]
fn approval_policy_denies_before_allowing() {
    let registry = ToolRegistry::built_in();
    let policy = ApprovalPolicy::new(PermissionMode::Bypass)
        .with_allowed(["shell"])
        .with_denied(["shell"]);

    assert!(matches!(
        policy.evaluate(&registry, "shell"),
        ApprovalDecision::Deny { .. }
    ));
}

#[test]
fn approval_policy_plan_mode_blocks_even_allowed_tools() {
    let registry = ToolRegistry::built_in();
    let policy = ApprovalPolicy::new(PermissionMode::Plan).with_allowed(["session_search"]);

    assert!(matches!(
        policy.evaluate(&registry, "session_search"),
        ApprovalDecision::Deny { .. }
    ));
}

#[test]
fn approval_policy_denies_unknown_tools() {
    let registry = ToolRegistry::built_in();
    let policy = ApprovalPolicy::default();

    assert!(matches!(
        policy.evaluate(&registry, "unknown"),
        ApprovalDecision::Deny { .. }
    ));
}
