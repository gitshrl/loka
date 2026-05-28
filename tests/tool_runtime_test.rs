use httpmock::prelude::*;
use loka_agent::messages::Role;
use loka_agent::session::SessionStore;
use loka_agent::tool_runtime::{ToolCall, ToolRuntime};
use loka_agent::wiki::WikiClient;
use serde_json::json;
use std::fs;
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
async fn tool_runtime_executes_wiki_rag() {
    let wiki = MockServer::start();
    let rag = wiki.mock(|when, then| {
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
        .with_wiki(WikiClient::new(wiki.base_url()), "loka-agent");
    let result = runtime
        .execute(ToolCall {
            name: "wiki_rag".to_string(),
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
async fn tool_runtime_executes_wiki_add_note_in_proposal_mode() {
    let wiki = MockServer::start();
    let note = wiki.mock(|when, then| {
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
        .with_wiki(WikiClient::new(wiki.base_url()), "loka-agent");
    let result = runtime
        .execute(ToolCall {
            name: "wiki_add_note".to_string(),
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
