use httpmock::Mock;
use httpmock::prelude::*;
use loka_agent::config::AppConfig;
use loka_agent::multi_agent::{
    AgentProfile, MultiAgentRunRequest, MultiAgentRuntime, TaskGraphStore, TaskStatus, WorkerSpec,
};
use loka_agent::session::SessionStore;
use loka_agent::tokens::TokenScope;
use serde_json::json;
use std::path::PathBuf;

#[tokio::test]
async fn multi_agent_run_persists_isolated_worker_sessions_and_synthesizes() {
    let model_client = MockServer::start();
    let memory = MockServer::start();
    let state = tempfile::tempdir().expect("state");
    let sessions = SessionStore::open(state.path()).expect("sessions");
    let tasks = TaskGraphStore::open(state.path()).expect("tasks");

    let planner = mock_planner(&model_client);
    let reviewer = mock_reviewer(&model_client);
    let supervisor = mock_supervisor(&model_client);

    let runtime = MultiAgentRuntime::new(app_config(&model_client, &memory), sessions, tasks);

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
    let worker_tokens = sessions
        .session_token_usage_records(&output.workers[0].session_id)
        .expect("worker token records");
    assert!(worker_tokens.iter().any(|record| {
        record.scope == TokenScope::Worker
            && record.source == "worker:planner"
            && record.total_tokens == 16
    }));

    assert_persisted_run(state.path(), &output.run_id, &output.supervisor_session_id);
}

#[tokio::test]
async fn multi_agent_run_requires_explicit_worker_contracts() {
    let model_client = MockServer::start();
    let memory = MockServer::start();
    let sessions = SessionStore::in_memory().expect("sessions");
    let tasks = TaskGraphStore::in_memory().expect("tasks");
    let runtime = MultiAgentRuntime::new(app_config(&model_client, &memory), sessions, tasks);

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

#[tokio::test]
async fn multi_agent_worker_over_token_budget_is_failed_and_persisted() {
    let model_client = MockServer::start();
    let memory = MockServer::start();
    let state = tempfile::tempdir().expect("state");
    let sessions = SessionStore::open(state.path()).expect("sessions");
    let tasks = TaskGraphStore::open(state.path()).expect("tasks");

    let worker = model_client.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("Worker profile: planner")
            .body_includes("Max tokens: 5");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "content": "This worker spent too much." } }
                ],
                "usage": { "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10 }
            }));
    });
    let supervisor = model_client.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("failed:")
            .body_includes("exceeded token budget");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "content": "Supervisor handled worker budget failure." } }
                ],
                "usage": { "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6 }
            }));
    });

    let runtime = MultiAgentRuntime::new(app_config(&model_client, &memory), sessions, tasks);
    let output = runtime
        .run(MultiAgentRunRequest {
            objective: "budgeted run".to_string(),
            recall: false,
            max_workers: 1,
            workers: vec![WorkerSpec {
                profile: AgentProfile::Planner,
                objective: "plan within budget".to_string(),
                output_format: "short answer".to_string(),
                tools_allowed: vec!["session_search".to_string()],
                max_iterations: 1,
                max_tokens: 5,
                timeout_seconds: 10,
                justification: "planning should stay bounded".to_string(),
            }],
        })
        .await
        .expect("multi-agent run");

    worker.assert();
    supervisor.assert();
    assert_eq!(
        output.synthesis,
        "Supervisor handled worker budget failure."
    );
    assert_eq!(output.total_tokens, 16);
    assert_eq!(output.workers.len(), 1);
    assert_eq!(output.workers[0].status, TaskStatus::Failed);
    assert_eq!(output.workers[0].tokens_used, 10);
    assert!(
        output.workers[0]
            .error
            .as_deref()
            .expect("error")
            .contains("exceeded token budget")
    );

    let tasks = TaskGraphStore::open(state.path()).expect("tasks");
    let run = tasks.run(&output.run_id).expect("run").expect("stored run");
    assert_eq!(run.tasks.len(), 1);
    assert_eq!(run.tasks[0].status, TaskStatus::Failed);
    assert_eq!(run.tasks[0].tokens_used, 10);
}

fn app_config(model_client: &MockServer, memory: &MockServer) -> AppConfig {
    AppConfig {
        model_base_url: model_client.base_url(),
        model_api_key: "sk-test".to_string(),
        memory_base_url: memory.base_url(),
        model: "gpt-5.5".to_string(),
        agent_id: "loka-agent".to_string(),
        model_protocol: loka_agent::config::ModelProtocol::OpenAiCompatible,
        memory_lifecycle: loka_agent::config::MemoryLifecycleMode::Off,
        working_dir: PathBuf::from("/tmp"),
        state_dir: PathBuf::from(".test-state"),
    }
}

fn mock_planner(model_client: &MockServer) -> Mock<'_> {
    model_client.mock(|when, then| {
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

fn mock_reviewer(model_client: &MockServer) -> Mock<'_> {
    model_client.mock(|when, then| {
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

fn mock_supervisor(model_client: &MockServer) -> Mock<'_> {
    model_client.mock(|when, then| {
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
