use httpmock::prelude::*;
use loka_agent::config::AppConfig;
use loka_agent::learning::{LearnSessionOutput, LearnSessionRequest, LearningEngine};
use loka_agent::messages::Role;
use loka_agent::session::SessionStore;
use serde_json::json;
use std::path::PathBuf;

#[tokio::test]
async fn learn_session_extracts_durable_note_and_writes_proposal() {
    let memory = MockServer::start();
    let model_client = MockServer::start();
    let sessions = SessionStore::in_memory().expect("sessions");
    let session_id = sessions
        .create_session("agent architecture")
        .expect("session");
    sessions
        .append_turn(
            &session_id,
            Role::User,
            "We decided Loka should use Rust as the control plane.",
        )
        .expect("user turn");
    sessions
        .append_turn(
            &session_id,
            Role::Assistant,
            "Decision captured: Rust owns orchestration, memory stays in memory API.",
        )
        .expect("assistant turn");
    let tool_call_id = sessions
        .record_tool_call_started(&session_id, "shell", &json!({ "command": "cargo clippy" }))
        .expect("tool call");
    sessions
        .record_tool_call_failed(&tool_call_id, "clippy failed on unwrap")
        .expect("failed tool call");

    let extraction = model_client.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("We decided Loka should use Rust")
            .body_includes("clippy failed on unwrap")
            .body_includes("Extract only durable knowledge");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "role": "assistant", "content": "- Decision: Loka uses Rust as the control plane." } }
                ]
            }));
    });

    let proposal = memory.mock(|when, then| {
        when.method(POST).path("/api/notes").json_body(json!({
            "title": format!("Session learning: {session_id}"),
            "body": "- Decision: Loka uses Rust as the control plane.",
            "kind": "note",
            "agentId": "loka-agent",
            "tags": ["learning", "session"],
            "mode": "propose"
        }));

        then.status(201)
            .header("content-type", "application/json")
            .json_body(json!({
                "mode": "propose",
                "proposal": { "id": "proposal-learning-1" }
            }));
    });

    let learning = LearningEngine::new(
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
        },
        sessions,
    );

    let output = learning
        .learn_session(LearnSessionRequest {
            session_id: session_id.clone(),
        })
        .await
        .expect("learning should succeed");

    extraction.assert();
    proposal.assert();
    assert_eq!(
        output,
        LearnSessionOutput::ProposalCreated {
            proposal_id: "proposal-learning-1".to_string()
        }
    );
}

#[tokio::test]
async fn learn_session_skips_memory_write_when_model_returns_none() {
    let memory = MockServer::start();
    let model_client = MockServer::start();
    let sessions = SessionStore::in_memory().expect("sessions");
    let session_id = sessions.create_session("casual chat").expect("session");
    sessions
        .append_turn(&session_id, Role::User, "hello")
        .expect("turn");

    model_client.mock(|when, then| {
        when.method(POST).path("/v1/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "role": "assistant", "content": "NONE" } }
                ]
            }));
    });

    let note = memory.mock(|when, then| {
        when.method(POST).path("/api/notes");
        then.status(500);
    });

    let learning = LearningEngine::new(
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
        },
        sessions,
    );

    let output = learning
        .learn_session(LearnSessionRequest { session_id })
        .await
        .expect("learning should succeed");

    assert_eq!(output, LearnSessionOutput::NoDurableKnowledge);
    assert_eq!(note.calls(), 0);
}
