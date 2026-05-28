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
async fn summarize_session_writes_proposal_first_wiki_note() {
    let wiki = MockServer::start();
    let llm = MockServer::start();
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

    let completion = llm.mock(|when, then| {
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

    let proposal = wiki.mock(|when, then| {
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

    let engine = SessionSummaryEngine::new(app_config(&llm, &wiki), sessions);
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
    let wiki = MockServer::start();
    let llm = MockServer::start();
    let sessions = SessionStore::in_memory().expect("sessions");
    let session_id = sessions.create_session("short").expect("session");
    sessions
        .append_turn(&session_id, Role::User, "hello")
        .expect("turn");

    let llm_call = llm.mock(|when, then| {
        when.method(POST).path("/v1/chat/completions");
        then.status(500);
    });
    let wiki_call = wiki.mock(|when, then| {
        when.method(POST).path("/api/notes");
        then.status(500);
    });

    let engine = SessionSummaryEngine::new(app_config(&llm, &wiki), sessions);
    let output = engine
        .summarize(SessionSummaryRequest {
            session_id,
            min_turns: 2,
        })
        .await
        .expect("summary");

    assert_eq!(output, SessionSummaryOutput::TooShort { turn_count: 1 });
    assert_eq!(llm_call.calls(), 0);
    assert_eq!(wiki_call.calls(), 0);
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
