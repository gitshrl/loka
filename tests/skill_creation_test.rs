use httpmock::prelude::*;
use loka_agent::config::AppConfig;
use loka_agent::messages::Role;
use loka_agent::session::SessionStore;
use loka_agent::skill_creation::{
    ProposeSkillFromSessionOutput, ProposeSkillFromSessionRequest, SkillCreationEngine,
};
use loka_agent::skills::{SkillStatus, SkillStore};
use serde_json::json;
use std::path::PathBuf;

#[tokio::test]
async fn propose_from_session_creates_memory_and_local_skill_proposals() {
    let memory = MockServer::start();
    let model_client = MockServer::start();
    let sessions = SessionStore::in_memory().expect("sessions");
    let skills = SkillStore::in_memory().expect("skills");
    let session_id = sessions
        .create_session("rust review workflow")
        .expect("session");
    sessions
        .append_turn(
            &session_id,
            Role::User,
            "When reviewing Rust, I always run cargo fmt, cargo test, and clippy.",
        )
        .expect("user turn");
    sessions
        .append_turn(
            &session_id,
            Role::Assistant,
            "Repeated workflow: inspect diffs, run fmt/test/clippy, then report risks.",
        )
        .expect("assistant turn");

    let extraction = model_client.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("Create exactly one Loka skill proposal")
            .body_includes("When reviewing Rust")
            .body_includes("required_tools");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json!({
                                "name": "Rust review",
                                "trigger": "rust review",
                                "instructions": "Inspect the diff, run cargo fmt, cargo test, and clippy, then report correctness risks first.",
                                "required_tools": ["read_file", "search_files"],
                                "safety_notes": ["Do not run destructive git commands."],
                                "examples": ["rust review src/main.rs"]
                            }).to_string()
                        }
                    }
                ]
            }));
    });

    let proposal = memory.mock(|when, then| {
        when.method(POST)
            .path("/api/notes")
            .body_includes("Skill proposal: Rust review")
            .body_includes("\"kind\":\"note\"")
            .body_includes("\"agentId\":\"loka-agent\"")
            .body_includes("\"tags\":[\"skill\",\"proposal\",\"session\"]")
            .body_includes("\"mode\":\"propose\"")
            .body_includes("Trigger: rust review")
            .body_includes("Inspect the diff");

        then.status(201)
            .header("content-type", "application/json")
            .json_body(json!({
                "mode": "propose",
                "proposal": { "id": "proposal-skill-1" }
            }));
    });

    let engine = SkillCreationEngine::new(app_config(&model_client, &memory), sessions, skills);
    let output = engine
        .propose_from_session(ProposeSkillFromSessionRequest {
            session_id: session_id.clone(),
        })
        .await
        .expect("skill proposal");

    extraction.assert();
    proposal.assert();
    let ProposeSkillFromSessionOutput::ProposalCreated {
        skill,
        memory_proposal_id,
    } = output
    else {
        panic!("expected skill proposal");
    };
    assert_eq!(memory_proposal_id, "proposal-skill-1");
    assert_eq!(skill.name, "Rust review");
    assert_eq!(skill.status, SkillStatus::Proposed);
    assert_eq!(skill.required_tools, vec!["read_file", "search_files"]);
}

#[tokio::test]
async fn propose_from_session_skips_when_no_reusable_workflow_exists() {
    let memory = MockServer::start();
    let model_client = MockServer::start();
    let sessions = SessionStore::in_memory().expect("sessions");
    let skills = SkillStore::in_memory().expect("skills");
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

    let proposal = memory.mock(|when, then| {
        when.method(POST).path("/api/notes");
        then.status(500);
    });

    let engine = SkillCreationEngine::new(app_config(&model_client, &memory), sessions, skills);
    let output = engine
        .propose_from_session(ProposeSkillFromSessionRequest { session_id })
        .await
        .expect("skill proposal");

    assert_eq!(output, ProposeSkillFromSessionOutput::NoReusableWorkflow);
    assert_eq!(proposal.calls(), 0);
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
