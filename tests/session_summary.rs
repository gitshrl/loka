use httpmock::prelude::*;
use loka_agent::config::AppConfig;
use loka_agent::messages::Role;
use loka_agent::session::SessionStore;
use loka_agent::session_summary::{
    SessionSummaryEngine, SessionSummaryOutput, SessionSummaryRequest,
};
use serde_json::json;
use std::path::PathBuf;

#[tokio::test]
async fn summarize_session_writes_proposal_first_memory_note() {
    let memory = MockServer::start();
    let model_client = MockServer::start();
    let sessions = SessionStore::in_memory().expect("sessions");
    let session_id = sessions.create_session("runtime design").expect("session");
    sessions
        .append_turn(
            &session_id,
            Role::User,
            "We need Docker and SSH runtime support.",
        )
        .expect("user turn");
    sessions
        .append_turn(
            &session_id,
            Role::Assistant,
            "Decision: implement Docker first, then SSH and cloud VM.",
        )
        .expect("assistant turn");
    let tool_call_id = sessions
        .record_tool_call_started(&session_id, "shell", &json!({ "command": "docker ps" }))
        .expect("tool call");
    sessions
        .record_tool_call_failed(&tool_call_id, "docker daemon unavailable")
        .expect("failed tool call");

    let completion = model_client.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("Summarize this Loka session")
            .body_includes("Docker and SSH runtime support")
            .body_includes("shell")
            .body_includes("docker daemon unavailable");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "role": "assistant", "content": "- Runtime plan: Docker first, then SSH and cloud VM." } }
                ]
            }));
    });

    let proposal = memory.mock(|when, then| {
        when.method(POST).path("/api/notes").json_body(json!({
            "title": format!("Session summary: {session_id}"),
            "body": "- Runtime plan: Docker first, then SSH and cloud VM.",
            "kind": "note",
            "agentId": "loka-agent",
            "tags": ["summary", "session"],
            "mode": "propose"
        }));

        then.status(201)
            .header("content-type", "application/json")
            .json_body(json!({
                "mode": "propose",
                "proposal": { "id": "proposal-summary-1" }
            }));
    });

    let engine = SessionSummaryEngine::new(app_config(&model_client, &memory), sessions);
    let output = engine
        .summarize(SessionSummaryRequest {
            session_id: session_id.clone(),
            min_turns: 2,
        })
        .await
        .expect("summary");

    completion.assert();
    proposal.assert();
    assert_eq!(
        output,
        SessionSummaryOutput::ProposalCreated {
            proposal_id: "proposal-summary-1".to_string()
        }
    );
}

#[tokio::test]
async fn summarize_session_skips_short_sessions() {
    let memory = MockServer::start();
    let model_client = MockServer::start();
    let sessions = SessionStore::in_memory().expect("sessions");
    let session_id = sessions.create_session("short").expect("session");
    sessions
        .append_turn(&session_id, Role::User, "hello")
        .expect("turn");

    let model_client_call = model_client.mock(|when, then| {
        when.method(POST).path("/v1/chat/completions");
        then.status(500);
    });
    let memory_call = memory.mock(|when, then| {
        when.method(POST).path("/api/notes");
        then.status(500);
    });

    let engine = SessionSummaryEngine::new(app_config(&model_client, &memory), sessions);
    let output = engine
        .summarize(SessionSummaryRequest {
            session_id,
            min_turns: 2,
        })
        .await
        .expect("summary");

    assert_eq!(output, SessionSummaryOutput::TooShort { turn_count: 1 });
    assert_eq!(model_client_call.calls(), 0);
    assert_eq!(memory_call.calls(), 0);
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
