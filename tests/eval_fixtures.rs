use loka_agent::evals::{
    AskScenario, EvalExpectations, EvalFixture, EvalScenario, load_fixtures, validate_fixture,
    validate_fixtures,
};
use std::collections::BTreeSet;
use std::path::PathBuf;

#[test]
fn bundled_eval_fixtures_cover_product_flows() {
    let fixtures = load_fixtures(fixture_dir()).expect("load fixtures");
    validate_fixtures(&fixtures).expect("valid fixtures");

    let kinds = fixtures
        .iter()
        .map(|fixture| fixture.kind().as_str())
        .collect::<BTreeSet<_>>();

    assert_eq!(
        kinds,
        BTreeSet::from(["ask", "chat", "learning", "multi-agent", "skill-creation"])
    );
}

#[test]
fn multi_agent_fixture_has_worker_expectations() {
    let fixtures = load_fixtures(fixture_dir()).expect("load fixtures");
    let fixture = fixtures
        .iter()
        .find(|fixture| matches!(fixture.scenario, EvalScenario::MultiAgent(_)))
        .expect("multi-agent fixture");
    let EvalScenario::MultiAgent(scenario) = &fixture.scenario else {
        unreachable!("fixture matched multi-agent");
    };

    assert_eq!(scenario.workers.len(), 2);
    assert_eq!(fixture.expectations.workers.len(), 2);
}

#[test]
fn fixture_without_expectations_is_rejected() {
    let fixture = EvalFixture {
        id: "ask.empty-expectations".to_string(),
        title: "Empty expectations".to_string(),
        tags: vec!["ask".to_string()],
        scenario: EvalScenario::Ask(AskScenario {
            prompt: "what next?".to_string(),
            recall: false,
            system_message: None,
            memory_context: Vec::new(),
        }),
        expectations: EvalExpectations::default(),
    };

    let error = validate_fixture(&fixture).expect_err("expectations should be required");
    assert!(
        error
            .to_string()
            .contains("expectations must define at least one assertion")
    );
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("evals/fixtures")
}
