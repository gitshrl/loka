use httpmock::prelude::*;
use loka_agent::agent::{Agent, ChatSessionRequest};
use loka_agent::config::AppConfig;
use loka_agent::session::SessionStore;
use serde_json::json;
use std::path::PathBuf;

#[tokio::test]
async fn chat_reuses_one_session_and_sends_prior_turns() {
    let wiki = MockServer::start();
    let llm = MockServer::start();
    let state = tempfile::tempdir().expect("state");
    let sessions = SessionStore::open(state.path()).expect("sessions");

    let first = llm.mock(|when, then| {
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
    let second = llm.mock(|when, then| {
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

    let agent = Agent::with_session_store(app_config(&llm, &wiki), sessions);
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
async fn chat_requires_at_least_one_message() {
    let wiki = MockServer::start();
    let llm = MockServer::start();
    let sessions = SessionStore::in_memory().expect("sessions");
    let agent = Agent::with_session_store(app_config(&llm, &wiki), sessions);

    let error = agent
        .chat(ChatSessionRequest {
            messages: Vec::new(),
            recall: false,
        })
        .await
        .expect_err("empty chat should fail");

    assert!(error.to_string().contains("at least one message"));
}

fn app_config(llm: &MockServer, wiki: &MockServer) -> AppConfig {
    AppConfig {
        pengepul_base_url: llm.base_url(),
        pengepul_api_key: "sk-test".to_string(),
        wiki_base_url: wiki.base_url(),
        model: "gpt-5.5".to_string(),
        agent_id: "loka-agent".to_string(),
        provider_id: "pengepul".to_string(),
        working_dir: PathBuf::from("/tmp"),
        state_dir: PathBuf::from(".test-state"),
    }
}
