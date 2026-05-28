use httpmock::Mock;
use httpmock::prelude::*;
use loka_agent::config::AppConfig;
use loka_agent::multi_agent::{
    AgentProfile, MultiAgentRunRequest, MultiAgentRuntime, TaskGraphStore, TaskStatus, WorkerSpec,
};
use loka_agent::session::SessionStore;
use serde_json::json;
use std::path::PathBuf;

#[tokio::test]
async fn multi_agent_run_persists_isolated_worker_sessions_and_synthesizes() {
    let llm = MockServer::start();
    let wiki = MockServer::start();
    let state = tempfile::tempdir().expect("state");
    let sessions = SessionStore::open(state.path()).expect("sessions");
    let tasks = TaskGraphStore::open(state.path()).expect("tasks");

    let planner = mock_planner(&llm);
    let reviewer = mock_reviewer(&llm);
    let supervisor = mock_supervisor(&llm);

    let runtime = MultiAgentRuntime::new(app_config(&llm, &wiki), sessions, tasks);

    let output = runtime
        .run(two_worker_request())
        .await
        .expect("multi-agent run");

    planner.assert();
    reviewer.assert();
    supervisor.assert();

    assert_eq!(output.synthesis, "Final supervisor answer.");
    assert_eq!(output.total_tokens, 43);
    assert_eq!(output.workers.len(), 2);
    assert_ne!(output.supervisor_session_id, output.workers[0].session_id);
    assert_ne!(output.workers[0].session_id, output.workers[1].session_id);

    let sessions = SessionStore::open(state.path()).expect("sessions");
    let private_hits = sessions
        .search("worker-private-detail-123", 10)
        .expect("private search");
    assert_eq!(private_hits.len(), 1);
    assert_eq!(private_hits[0].session_id, output.workers[0].session_id);

    assert_persisted_run(state.path(), &output.run_id, &output.supervisor_session_id);
}

#[tokio::test]
async fn multi_agent_run_requires_explicit_worker_contracts() {
    let llm = MockServer::start();
    let wiki = MockServer::start();
    let sessions = SessionStore::in_memory().expect("sessions");
    let tasks = TaskGraphStore::in_memory().expect("tasks");
    let runtime = MultiAgentRuntime::new(app_config(&llm, &wiki), sessions, tasks);

    let error = runtime
        .run(MultiAgentRunRequest {
            objective: "ship multi-agent runtime".to_string(),
            recall: false,
            max_workers: 1,
            workers: vec![WorkerSpec {
                profile: AgentProfile::Coder,
                objective: "implement it".to_string(),
                output_format: String::new(),
                tools_allowed: Vec::new(),
                max_iterations: 0,
                max_tokens: 0,
                timeout_seconds: 0,
                justification: String::new(),
            }],
        })
        .await
        .expect_err("invalid worker contract should fail");

    assert!(error.to_string().contains("output format"));
}

fn app_config(llm: &MockServer, wiki: &MockServer) -> AppConfig {
    AppConfig {
        pengepul_base_url: llm.base_url(),
        pengepul_api_key: "sk-test".to_string(),
        wiki_base_url: wiki.base_url(),
        model: "gpt-5".to_string(),
        agent_id: "loka-agent".to_string(),
        provider_id: "pengepul".to_string(),
        working_dir: PathBuf::from("/tmp"),
        state_dir: PathBuf::from(".test-state"),
    }
}

fn mock_planner(llm: &MockServer) -> Mock<'_> {
    llm.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("Worker profile: planner")
            .body_includes("worker-private-detail-123")
            .body_includes("Return format: concise plan with risks");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "content": "Planner summary only." } }
                ],
                "usage": { "prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16 }
            }));
    })
}

fn mock_reviewer(llm: &MockServer) -> Mock<'_> {
    llm.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("Worker profile: reviewer")
            .body_includes("Return format: review findings");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "content": "Reviewer summary only." } }
                ],
                "usage": { "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10 }
            }));
    })
}

fn mock_supervisor(llm: &MockServer) -> Mock<'_> {
    llm.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("Supervisor synthesis")
            .body_includes("Planner summary only.")
            .body_includes("Reviewer summary only.")
            .body_includes("Do not include private worker scratchpads");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "content": "Final supervisor answer." } }
                ],
                "usage": { "prompt_tokens": 13, "completion_tokens": 4, "total_tokens": 17 }
            }));
    })
}

fn two_worker_request() -> MultiAgentRunRequest {
    MultiAgentRunRequest {
        objective: "ship multi-agent runtime".to_string(),
        recall: false,
        max_workers: 4,
        workers: vec![
            WorkerSpec {
                profile: AgentProfile::Planner,
                objective: "plan the work; worker-private-detail-123".to_string(),
                output_format: "concise plan with risks".to_string(),
                tools_allowed: vec!["session_search".to_string()],
                max_iterations: 3,
                max_tokens: 2_000,
                timeout_seconds: 10,
                justification: "planning decomposes the work before implementation".to_string(),
            },
            WorkerSpec {
                profile: AgentProfile::Reviewer,
                objective: "review the design".to_string(),
                output_format: "review findings".to_string(),
                tools_allowed: vec!["read_file".to_string(), "search_files".to_string()],
                max_iterations: 2,
                max_tokens: 1_500,
                timeout_seconds: 10,
                justification: "reviewer catches architecture and safety issues".to_string(),
            },
        ],
    }
}

fn assert_persisted_run(state_path: &std::path::Path, run_id: &str, supervisor_session_id: &str) {
    let tasks = TaskGraphStore::open(state_path).expect("tasks");
    let run = tasks.run(run_id).expect("run").expect("stored run");
    assert_eq!(run.status, TaskStatus::Succeeded);
    assert_eq!(
        run.supervisor_session_id,
        Some(supervisor_session_id.to_string())
    );
    assert_eq!(run.tasks.len(), 2);
    assert!(
        run.tasks
            .iter()
            .all(|task| task.status == TaskStatus::Succeeded)
    );
}
