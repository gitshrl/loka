use httpmock::prelude::*;
use loka_agent::agent::{Agent, AskRequest};
use loka_agent::config::AppConfig;
use loka_agent::session::SessionStore;
use loka_agent::skills::{SkillDraft, SkillStore};
use serde_json::json;
use std::path::PathBuf;

#[tokio::test]
async fn ask_with_recall_injects_memory_through_volatile_prompt_layer() {
    let wiki = MockServer::start();
    let llm = MockServer::start();

    wiki.mock(|when, then| {
        when.method(POST).path("/api/rag").json_body(json!({
            "query": "what next",
            "limit": 6,
            "depth": 1
        }));

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "mode": "fts",
                "markdown": "# Wiki Context\n- build the Rust platform spine"
            }));
    });

    let completion = llm.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .header("authorization", "Bearer sk-test")
            .body_includes("# Loka Identity")
            .body_includes("# Runtime State")
            .body_includes("# Memory Recall")
            .body_includes("Session ID: session-1")
            .body_includes("build the Rust platform spine")
            .body_includes("what next");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "role": "assistant", "content": "Build the ask --recall command." } }
                ]
            }));
    });

    let agent = Agent::new(AppConfig {
        pengepul_base_url: llm.base_url(),
        pengepul_api_key: "sk-test".to_string(),
        wiki_base_url: wiki.base_url(),
        model: "gpt-5.5".to_string(),
        agent_id: "loka-agent".to_string(),
        provider_id: "pengepul".to_string(),
        working_dir: PathBuf::from("/tmp"),
        state_dir: PathBuf::from(".test-state"),
    });

    let answer = agent
        .ask(AskRequest {
            prompt: "what next".to_string(),
            recall: true,
            session_id: Some("session-1".to_string()),
            system_message: None,
        })
        .await
        .expect("ask should succeed");

    completion.assert();
    assert_eq!(answer.answer, "Build the ask --recall command.");
}

#[tokio::test]
async fn ask_without_recall_does_not_call_personal_wiki() {
    let wiki = MockServer::start();
    let llm = MockServer::start();

    let wiki_rag = wiki.mock(|when, then| {
        when.method(POST).path("/api/rag");
        then.status(500);
    });

    llm.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("# Loka Identity")
            .body_includes("# Session Context")
            .body_includes("Prefer terse answers.")
            .body_excludes("# Memory Recall")
            .body_includes("what next");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "role": "assistant", "content": "Answer without recall." } }
                ]
            }));
    });

    let agent = Agent::new(AppConfig {
        pengepul_base_url: llm.base_url(),
        pengepul_api_key: "sk-test".to_string(),
        wiki_base_url: wiki.base_url(),
        model: "gpt-5.5".to_string(),
        agent_id: "loka-agent".to_string(),
        provider_id: "pengepul".to_string(),
        working_dir: PathBuf::from("/tmp"),
        state_dir: PathBuf::from(".test-state"),
    });

    let answer = agent
        .ask(AskRequest {
            prompt: "what next".to_string(),
            recall: false,
            session_id: None,
            system_message: Some("Prefer terse answers.".to_string()),
        })
        .await
        .expect("ask should succeed");

    assert_eq!(wiki_rag.calls(), 0);
    assert_eq!(answer.answer, "Answer without recall.");
}

#[tokio::test]
async fn ask_with_session_store_persists_user_and_assistant_turns() {
    let wiki = MockServer::start();
    let llm = MockServer::start();
    let tempdir = tempfile::tempdir().expect("tempdir");
    let sessions = SessionStore::open(tempdir.path()).expect("session store");

    llm.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("persist this");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "role": "assistant", "content": "Persisted." } }
                ]
            }));
    });

    let agent = Agent::with_session_store(
        AppConfig {
            pengepul_base_url: llm.base_url(),
            pengepul_api_key: "sk-test".to_string(),
            wiki_base_url: wiki.base_url(),
            model: "gpt-5.5".to_string(),
            agent_id: "loka-agent".to_string(),
            provider_id: "pengepul".to_string(),
            working_dir: PathBuf::from("/tmp"),
            state_dir: PathBuf::from(".test-state"),
        },
        sessions,
    );

    let output = agent
        .ask(AskRequest {
            prompt: "persist this".to_string(),
            recall: false,
            session_id: None,
            system_message: None,
        })
        .await
        .expect("ask should succeed");

    let session_id = output.session_id.expect("session id");
    let sessions = SessionStore::open(tempdir.path()).expect("session store");
    let hits = sessions.search("persisted", 10).expect("search");

    assert!(
        sessions
            .session_exists(&session_id)
            .expect("session exists")
    );
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].session_id, session_id);
    assert_eq!(hits[0].content, "Persisted.");
}

#[tokio::test]
async fn ask_stream_persists_accumulated_assistant_answer() {
    let wiki = MockServer::start();
    let llm = MockServer::start();
    let tempdir = tempfile::tempdir().expect("tempdir");
    let sessions = SessionStore::open(tempdir.path()).expect("sessions");

    llm.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("\"stream\":true")
            .body_includes("stream this");

        then.status(200)
            .header("content-type", "text/event-stream")
            .body(concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"streamed\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\" answer\"}}]}\n\n",
                "data: [DONE]\n\n"
            ));
    });

    let agent = Agent::with_session_store(
        AppConfig {
            pengepul_base_url: llm.base_url(),
            pengepul_api_key: "sk-test".to_string(),
            wiki_base_url: wiki.base_url(),
            model: "gpt-5.5".to_string(),
            agent_id: "loka-agent".to_string(),
            provider_id: "pengepul".to_string(),
            working_dir: PathBuf::from("/tmp"),
            state_dir: PathBuf::from(".test-state"),
        },
        sessions,
    );

    let mut deltas = Vec::new();
    let output = agent
        .ask_stream(
            AskRequest {
                prompt: "stream this".to_string(),
                recall: false,
                session_id: None,
                system_message: None,
            },
            |delta| {
                deltas.push(delta.to_string());
                Ok(())
            },
        )
        .await
        .expect("stream should succeed");

    assert_eq!(deltas, vec!["streamed", " answer"]);
    assert_eq!(output.answer, "streamed answer");
    let session_id = output.session_id.expect("session id");
    let hits = SessionStore::open(tempdir.path())
        .expect("sessions")
        .search("streamed", 10)
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].session_id, session_id);
}

#[tokio::test]
async fn ask_injects_enabled_matching_skill_context() {
    let wiki = MockServer::start();
    let llm = MockServer::start();
    let sessions = SessionStore::in_memory().expect("sessions");
    let skills = SkillStore::in_memory().expect("skills");
    let skill = skills
        .propose(&SkillDraft {
            name: "Rust review".to_string(),
            trigger: "rust review".to_string(),
            instructions: "Apply strict Rust review rules before answering.".to_string(),
            required_tools: vec!["read_file".to_string()],
            safety_notes: vec!["Do not execute shell commands.".to_string()],
            examples: vec![],
        })
        .expect("propose");
    skills.enable(&skill.id).expect("enable");

    let completion = llm.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("Enabled skill available")
            .body_includes("Apply strict Rust review rules")
            .body_includes("rust review this module");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "role": "assistant", "content": "Skill context applied." } }
                ]
            }));
    });

    let agent = Agent::with_stores(
        AppConfig {
            pengepul_base_url: llm.base_url(),
            pengepul_api_key: "sk-test".to_string(),
            wiki_base_url: wiki.base_url(),
            model: "gpt-5.5".to_string(),
            agent_id: "loka-agent".to_string(),
            provider_id: "pengepul".to_string(),
            working_dir: PathBuf::from("/tmp"),
            state_dir: PathBuf::from(".test-state"),
        },
        sessions,
        skills,
    );

    let output = agent
        .ask(AskRequest {
            prompt: "rust review this module".to_string(),
            recall: false,
            session_id: None,
            system_message: None,
        })
        .await
        .expect("ask should succeed");

    completion.assert();
    assert_eq!(output.answer, "Skill context applied.");
}
