use httpmock::prelude::*;
use loka_agent::agent::{Agent, ChatSessionRequest};
use loka_agent::config::AppConfig;
use loka_agent::session::SessionStore;
use serde_json::json;
use std::path::PathBuf;

#[tokio::test]
async fn chat_reuses_one_session_and_sends_prior_turns() {
    let memory = MockServer::start();
    let model_client = MockServer::start();
    let state = tempfile::tempdir().expect("state");
    let sessions = SessionStore::open(state.path()).expect("sessions");

    let first = model_client.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("hello")
            .body_excludes("First answer");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "content": "First answer." } }
                ]
            }));
    });
    let second = model_client.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("First answer.")
            .body_includes("follow up");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "content": "Second answer." } }
                ]
            }));
    });

    let agent = Agent::with_session_store(app_config(&model_client, &memory), sessions);
    let output = agent
        .chat(ChatSessionRequest {
            messages: vec!["hello".to_string(), "follow up".to_string()],
            recall: false,
        })
        .await
        .expect("chat should succeed");

    first.assert();
    second.assert();
    assert_eq!(output.answers, vec!["First answer.", "Second answer."]);
    assert_eq!(output.summary_proposal_id, None);

    let session_id = output.session_id;
    let sessions = SessionStore::open(state.path()).expect("sessions");
    let turns = sessions.session_turns(&session_id).expect("turns");
    assert_eq!(turns.len(), 4);
    assert!(turns.iter().all(|turn| turn.content != "system prompt"));
    assert_eq!(turns[0].content, "hello");
    assert_eq!(turns[1].content, "First answer.");
    assert_eq!(turns[2].content, "follow up");
    assert_eq!(turns[3].content, "Second answer.");
}

#[tokio::test]
async fn chat_summarizes_long_session_as_proposal() {
    let memory = MockServer::start();
    let model_client = MockServer::start();
    let state = tempfile::tempdir().expect("state");
    let sessions = SessionStore::open(state.path()).expect("sessions");

    let chat_completion = model_client.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_excludes("Summarize this Loka session")
            .body_includes("turn");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "content": "chat answer" } }
                ]
            }));
    });
    let summary_completion = model_client.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("Summarize this Loka session")
            .body_includes("turn 6");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "content": "- durable chat summary" } }
                ]
            }));
    });
    let proposal = memory.mock(|when, then| {
        when.method(POST)
            .path("/api/notes")
            .body_includes("- durable chat summary")
            .body_includes("loka-agent")
            .body_includes("summary")
            .body_includes("session")
            .body_includes("propose");

        then.status(201)
            .header("content-type", "application/json")
            .json_body(json!({
                "mode": "propose",
                "proposal": { "id": "proposal-auto-summary-1" }
            }));
    });

    let agent = Agent::with_session_store(app_config(&model_client, &memory), sessions);
    let output = agent
        .chat(ChatSessionRequest {
            messages: (1..=6).map(|index| format!("turn {index}")).collect(),
            recall: false,
        })
        .await
        .expect("chat should succeed");

    assert_eq!(chat_completion.calls(), 6);
    summary_completion.assert();
    proposal.assert();
    assert_eq!(output.answers.len(), 6);
    assert_eq!(
        output.summary_proposal_id,
        Some("proposal-auto-summary-1".to_string())
    );

    let sessions = SessionStore::open(state.path()).expect("sessions");
    let turns = sessions.session_turns(&output.session_id).expect("turns");
    assert_eq!(turns.len(), 12);
}

#[tokio::test]
async fn chat_requires_at_least_one_message() {
    let memory = MockServer::start();
    let model_client = MockServer::start();
    let sessions = SessionStore::in_memory().expect("sessions");
    let agent = Agent::with_session_store(app_config(&model_client, &memory), sessions);

    let error = agent
        .chat(ChatSessionRequest {
            messages: Vec::new(),
            recall: false,
        })
        .await
        .expect_err("empty chat should fail");

    assert!(error.to_string().contains("at least one message"));
}

fn app_config(model_client: &MockServer, memory: &MockServer) -> AppConfig {
    AppConfig {
        model_base_url: model_client.base_url(),
        model_api_key: "sk-test".to_string(),
        memory_base_url: memory.base_url(),
        model: "gpt-5.5".to_string(),
        agent_id: "loka-agent".to_string(),
        model_protocol: loka_agent::config::ModelProtocol::OpenAiCompatible,
        working_dir: PathBuf::from("/tmp"),
        state_dir: PathBuf::from(".test-state"),
    }
}
