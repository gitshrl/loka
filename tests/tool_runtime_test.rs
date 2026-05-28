use httpmock::prelude::*;
use loka_agent::config::AppConfig;
use loka_agent::memory::MemoryClient;
use loka_agent::messages::Role;
use loka_agent::session::{SessionStore, ToolCallStatus};
use loka_agent::tool_runtime::{ToolCall, ToolRuntime};
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[tokio::test]
async fn tool_runtime_executes_session_search() {
    let sessions = SessionStore::in_memory().expect("sessions");
    let session_id = sessions.create_session("tool runtime").expect("session");
    sessions
        .append_turn(&session_id, Role::User, "find approval policy")
        .expect("turn");

    let runtime = ToolRuntime::new(sessions);
    let result = runtime
        .execute(ToolCall {
            name: "session_search".to_string(),
            input: json!({ "query": "approval", "limit": 10 }),
        })
        .await
        .expect("tool call should succeed");

    let hits = result.output["hits"].as_array().expect("hits");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0]["session_id"], session_id);
}

#[tokio::test]
async fn tool_runtime_persists_completed_tool_call_transcript() {
    let state = tempfile::tempdir().expect("state");
    let database = state.path().join("sessions.sqlite3");
    let sessions = SessionStore::open_database(&database).expect("sessions");
    let session_id = sessions.create_session("tool transcript").expect("session");
    sessions
        .append_turn(&session_id, Role::User, "find approval policy")
        .expect("turn");

    let runtime = ToolRuntime::new(sessions);
    let result = runtime
        .execute_in_session(
            &session_id,
            ToolCall {
                name: "session_search".to_string(),
                input: json!({ "query": "approval", "limit": 10 }),
            },
        )
        .await
        .expect("tool call should succeed");
    assert_eq!(result.output["hits"][0]["session_id"], session_id);

    let inspector = SessionStore::open_database(&database).expect("inspector");
    let calls = inspector
        .session_tool_calls(&session_id)
        .expect("tool calls");

    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "session_search");
    assert_eq!(calls[0].status, ToolCallStatus::Completed);
    assert_eq!(calls[0].input, json!({ "query": "approval", "limit": 10 }));
    assert_eq!(calls[0].output, Some(result.output));
    assert_eq!(calls[0].error, None);
    assert!(calls[0].completed_at.is_some());
}

#[tokio::test]
async fn tool_runtime_persists_failed_tool_call_transcript() {
    let state = tempfile::tempdir().expect("state");
    let database = state.path().join("sessions.sqlite3");
    let sessions = SessionStore::open_database(&database).expect("sessions");
    let session_id = sessions.create_session("failed tool").expect("session");

    let runtime = ToolRuntime::new(sessions);
    let error = runtime
        .execute_in_session(
            &session_id,
            ToolCall {
                name: "shell".to_string(),
                input: json!({ "command": "printf no-workspace" }),
            },
        )
        .await
        .expect_err("host workspace is required");
    assert!(error.to_string().contains("host workspace"));

    let inspector = SessionStore::open_database(&database).expect("inspector");
    let calls = inspector
        .session_tool_calls(&session_id)
        .expect("tool calls");

    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "shell");
    assert_eq!(calls[0].status, ToolCallStatus::Failed);
    assert_eq!(calls[0].input, json!({ "command": "printf no-workspace" }));
    assert_eq!(calls[0].output, None);
    assert!(
        calls[0]
            .error
            .as_deref()
            .expect("error")
            .contains("host workspace")
    );
    assert!(calls[0].completed_at.is_some());
}

#[tokio::test]
async fn tool_runtime_executes_memory_search() {
    let memory = MockServer::start();
    let rag = memory.mock(|when, then| {
        when.method(POST).path("/api/rag").json_body(json!({
            "query": "runtime",
            "limit": 6,
            "depth": 1
        }));

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "mode": "fts",
                "markdown": "# Context\nRuntime notes"
            }));
    });

    let runtime = ToolRuntime::new(SessionStore::in_memory().expect("sessions"))
        .with_memory(MemoryClient::new(memory.base_url()), "loka-agent");
    let result = runtime
        .execute(ToolCall {
            name: "memory_search".to_string(),
            input: json!({ "query": "runtime" }),
        })
        .await
        .expect("tool call should succeed");

    rag.assert();
    assert_eq!(
        result.output["context"]["markdown"],
        "# Context\nRuntime notes"
    );
}

#[tokio::test]
async fn tool_runtime_executes_memory_propose_in_proposal_mode() {
    let memory = MockServer::start();
    let note = memory.mock(|when, then| {
        when.method(POST).path("/api/notes").json_body(json!({
            "title": "Tool note",
            "body": "Tool runtime writes proposal-first.",
            "kind": "note",
            "agentId": "loka-agent",
            "tags": ["tool"],
            "mode": "propose"
        }));

        then.status(201)
            .header("content-type", "application/json")
            .json_body(json!({
                "mode": "propose",
                "proposal": { "id": "proposal-tool-1" }
            }));
    });

    let runtime = ToolRuntime::new(SessionStore::in_memory().expect("sessions"))
        .with_memory(MemoryClient::new(memory.base_url()), "loka-agent");
    let result = runtime
        .execute(ToolCall {
            name: "memory_propose".to_string(),
            input: json!({
                "title": "Tool note",
                "body": "Tool runtime writes proposal-first.",
                "tags": ["tool"]
            }),
        })
        .await
        .expect("tool call should succeed");

    note.assert();
    assert_eq!(result.output["proposal_id"], "proposal-tool-1");
}

#[tokio::test]
async fn tool_runtime_executes_learn_session() {
    let memory = MockServer::start();
    let model_client = MockServer::start();
    let sessions = SessionStore::in_memory().expect("sessions");
    let session_id = sessions
        .create_session("runtime learning")
        .expect("session");
    sessions
        .append_turn(
            &session_id,
            Role::User,
            "Decision: keep durable memory in memory API.",
        )
        .expect("turn");

    let completion = model_client.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("Extract only durable knowledge")
            .body_includes("durable memory in memory API");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "content": "- Decision: durable memory stays in memory API." } }
                ]
            }));
    });
    let proposal = memory.mock(|when, then| {
        when.method(POST).path("/api/notes").json_body(json!({
            "title": format!("Session learning: {session_id}"),
            "body": "- Decision: durable memory stays in memory API.",
            "kind": "note",
            "agentId": "loka-agent",
            "tags": ["learning", "session"],
            "mode": "propose"
        }));

        then.status(201)
            .header("content-type", "application/json")
            .json_body(json!({
                "mode": "propose",
                "proposal": { "id": "proposal-learning-tool-1" }
            }));
    });

    let runtime =
        ToolRuntime::new(sessions).with_learning_config(app_config(&model_client, &memory));
    let result = runtime
        .execute(ToolCall {
            name: "learn_session".to_string(),
            input: json!({ "session_id": session_id }),
        })
        .await
        .expect("learn_session tool should succeed");

    completion.assert();
    proposal.assert();
    assert_eq!(result.output["proposal_id"], "proposal-learning-tool-1");
}

#[tokio::test]
async fn tool_runtime_rejects_learn_session_without_learning_config() {
    let runtime = ToolRuntime::new(SessionStore::in_memory().expect("sessions"));
    let error = runtime
        .execute(ToolCall {
            name: "learn_session".to_string(),
            input: json!({ "session_id": "missing" }),
        })
        .await
        .expect_err("learning config is required");

    assert!(error.to_string().contains("learning configuration"));
}

#[tokio::test]
async fn tool_runtime_rejects_host_tool_without_workspace() {
    let runtime = ToolRuntime::new(SessionStore::in_memory().expect("sessions"));
    let error = runtime
        .execute(ToolCall {
            name: "shell".to_string(),
            input: json!({ "command": "echo no" }),
        })
        .await
        .expect_err("shell executor requires a workspace");

    assert!(error.to_string().contains("host workspace"));
}

#[tokio::test]
async fn host_runtime_reads_file_inside_workspace() {
    let workspace = tempfile::tempdir().expect("workspace");
    fs::write(workspace.path().join("notes.txt"), "agent harness\n").expect("write file");

    let runtime = ToolRuntime::new(SessionStore::in_memory().expect("sessions"))
        .with_host_workspace(workspace.path())
        .expect("host runtime");
    let result = runtime
        .execute(ToolCall {
            name: "read_file".to_string(),
            input: json!({ "path": "notes.txt" }),
        })
        .await
        .expect("read file");

    assert_eq!(result.output["path"], "notes.txt");
    assert_eq!(result.output["content"], "agent harness\n");
}

#[tokio::test]
async fn host_runtime_blocks_path_escape() {
    let workspace = tempfile::tempdir().expect("workspace");
    let outside = tempfile::NamedTempFile::new().expect("outside");

    let runtime = ToolRuntime::new(SessionStore::in_memory().expect("sessions"))
        .with_host_workspace(workspace.path())
        .expect("host runtime");
    let error = runtime
        .execute(ToolCall {
            name: "read_file".to_string(),
            input: json!({ "path": outside.path().display().to_string() }),
        })
        .await
        .expect_err("outside read should fail");

    assert!(error.to_string().contains("escapes workspace"));
}

#[tokio::test]
async fn host_runtime_searches_files_with_ignore_support() {
    let workspace = tempfile::tempdir().expect("workspace");
    fs::write(workspace.path().join(".gitignore"), "ignored.txt\n").expect("gitignore");
    fs::write(
        workspace.path().join("main.rs"),
        "fn main() { /* needle */ }\n",
    )
    .expect("main");
    fs::write(workspace.path().join("ignored.txt"), "needle\n").expect("ignored");

    let runtime = ToolRuntime::new(SessionStore::in_memory().expect("sessions"))
        .with_host_workspace(workspace.path())
        .expect("host runtime");
    let result = runtime
        .execute(ToolCall {
            name: "search_files".to_string(),
            input: json!({ "query": "needle" }),
        })
        .await
        .expect("search files");

    let hits = result.output["hits"].as_array().expect("hits");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0]["path"], "main.rs");
}

#[tokio::test]
async fn host_runtime_shell_executes_in_workspace() {
    let workspace = tempfile::tempdir().expect("workspace");
    let runtime = ToolRuntime::new(SessionStore::in_memory().expect("sessions"))
        .with_host_workspace(workspace.path())
        .expect("host runtime");
    let result = runtime
        .execute(ToolCall {
            name: "shell".to_string(),
            input: json!({ "command": "printf ok" }),
        })
        .await
        .expect("shell");

    assert_eq!(result.output["status"], 0);
    assert_eq!(result.output["stdout"], "ok");
}

#[tokio::test]
async fn host_runtime_git_status_runs_in_workspace() {
    let workspace = tempfile::tempdir().expect("workspace");
    let status = Command::new("git")
        .arg("init")
        .current_dir(workspace.path())
        .status()
        .expect("git init");
    assert!(status.success());
    fs::write(workspace.path().join("new.txt"), "new\n").expect("file");

    let runtime = ToolRuntime::new(SessionStore::in_memory().expect("sessions"))
        .with_host_workspace(workspace.path())
        .expect("host runtime");
    let result = runtime
        .execute(ToolCall {
            name: "git_status".to_string(),
            input: json!({}),
        })
        .await
        .expect("git status");

    assert_eq!(result.output["status"], 0);
    assert!(
        result.output["stdout"]
            .as_str()
            .expect("stdout")
            .contains("new.txt")
    );
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
