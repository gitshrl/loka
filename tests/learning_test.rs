use httpmock::prelude::*;
use loka_agent::config::AppConfig;
use loka_agent::learning::{LearnSessionOutput, LearnSessionRequest, LearningEngine};
use loka_agent::messages::Role;
use loka_agent::session::SessionStore;
use serde_json::json;
use std::path::PathBuf;

#[tokio::test]
async fn learn_session_extracts_durable_note_and_writes_proposal() {
    let wiki = MockServer::start();
    let llm = MockServer::start();
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
            "Decision captured: Rust owns orchestration, memory stays in personal-wiki.",
        )
        .expect("assistant turn");

    let extraction = llm.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("We decided Loka should use Rust")
            .body_includes("Extract only durable knowledge");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "role": "assistant", "content": "- Decision: Loka uses Rust as the control plane." } }
                ]
            }));
    });

    let proposal = wiki.mock(|when, then| {
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
            pengepul_base_url: llm.base_url(),
            pengepul_api_key: "sk-test".to_string(),
            wiki_base_url: wiki.base_url(),
            model: "gpt-5".to_string(),
            agent_id: "loka-agent".to_string(),
            provider_id: "pengepul".to_string(),
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
async fn learn_session_skips_wiki_write_when_model_returns_none() {
    let wiki = MockServer::start();
    let llm = MockServer::start();
    let sessions = SessionStore::in_memory().expect("sessions");
    let session_id = sessions.create_session("casual chat").expect("session");
    sessions
        .append_turn(&session_id, Role::User, "hello")
        .expect("turn");

    llm.mock(|when, then| {
        when.method(POST).path("/v1/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "role": "assistant", "content": "NONE" } }
                ]
            }));
    });

    let note = wiki.mock(|when, then| {
        when.method(POST).path("/api/notes");
        then.status(500);
    });

    let learning = LearningEngine::new(
        AppConfig {
            pengepul_base_url: llm.base_url(),
            pengepul_api_key: "sk-test".to_string(),
            wiki_base_url: wiki.base_url(),
            model: "gpt-5".to_string(),
            agent_id: "loka-agent".to_string(),
            provider_id: "pengepul".to_string(),
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
