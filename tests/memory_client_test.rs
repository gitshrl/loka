use httpmock::prelude::*;
use loka_agent::memory::{
    MemoryClient, MemoryNoteInput, MemoryPrefetchInput, MemorySessionEndInput, MemoryShutdownInput,
    MemoryTurnInput,
};
use serde_json::json;

#[tokio::test]
async fn recall_posts_query_to_memory_api() {
    let server = MockServer::start();
    let rag = server.mock(|when, then| {
        when.method(POST).path("/api/rag").json_body(json!({
            "query": "next work",
            "limit": 6,
            "depth": 1
        }));

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "mode": "fts",
                "markdown": "# Memory Context\n- ship the platform spine"
            }));
    });

    let client = MemoryClient::new(server.base_url());
    let output = client
        .recall("next work", 6, 1)
        .await
        .expect("rag should succeed");

    rag.assert();
    assert_eq!(
        output.markdown,
        "# Memory Context\n- ship the platform spine"
    );
}

#[tokio::test]
async fn prefetch_posts_session_scoped_query_to_memory_api() {
    let server = MockServer::start();
    let prefetch = server.mock(|when, then| {
        when.method(POST)
            .path("/api/memory/prefetch")
            .json_body(json!({
                "query": "next work",
                "limit": 6,
                "depth": 1,
                "sessionId": "session-1"
            }));

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "mode": "prefetch",
                "markdown": "# Memory Context\n- prefetched"
            }));
    });

    let client = MemoryClient::new(server.base_url());
    let output = client
        .prefetch(MemoryPrefetchInput {
            query: "next work".to_string(),
            limit: 6,
            depth: 1,
            session_id: Some("session-1".to_string()),
        })
        .await
        .expect("prefetch should succeed");

    prefetch.assert();
    assert_eq!(output.markdown, "# Memory Context\n- prefetched");
}

#[tokio::test]
async fn sync_turn_posts_completed_turn_to_memory_api() {
    let server = MockServer::start();
    let sync = server.mock(|when, then| {
        when.method(POST)
            .path("/api/memory/turns")
            .json_body(json!({
                "sessionId": "session-1",
                "user": "what next",
                "assistant": "ship it",
                "agentId": "loka-agent"
            }));

        then.status(202);
    });

    let client = MemoryClient::new(server.base_url());
    client
        .sync_turn(MemoryTurnInput {
            session_id: Some("session-1".to_string()),
            user: "what next".to_string(),
            assistant: "ship it".to_string(),
            agent_id: "loka-agent".to_string(),
        })
        .await
        .expect("turn sync should succeed");

    sync.assert();
}

#[tokio::test]
async fn session_end_returns_optional_proposal_id() {
    let server = MockServer::start();
    let end = server.mock(|when, then| {
        when.method(POST)
            .path("/api/memory/session-end")
            .json_body(json!({
                "sessionId": "session-1",
                "agentId": "loka-agent"
            }));

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "proposalId": "proposal-1"
            }));
    });

    let client = MemoryClient::new(server.base_url());
    let proposal_id = client
        .end_session(MemorySessionEndInput {
            session_id: "session-1".to_string(),
            agent_id: "loka-agent".to_string(),
        })
        .await
        .expect("session end should succeed");

    end.assert();
    assert_eq!(proposal_id.as_deref(), Some("proposal-1"));
}

#[tokio::test]
async fn shutdown_posts_agent_id_to_memory_api() {
    let server = MockServer::start();
    let shutdown = server.mock(|when, then| {
        when.method(POST)
            .path("/api/memory/shutdown")
            .json_body(json!({
                "agentId": "loka-agent"
            }));

        then.status(204);
    });

    let client = MemoryClient::new(server.base_url());
    client
        .shutdown(MemoryShutdownInput {
            agent_id: "loka-agent".to_string(),
        })
        .await
        .expect("shutdown should succeed");

    shutdown.assert();
}

#[tokio::test]
async fn propose_note_defaults_to_proposal_mode() {
    let server = MockServer::start();
    let note = server.mock(|when, then| {
        when.method(POST).path("/api/notes").json_body(json!({
            "title": "Decision",
            "body": "Use Rust for the agent control plane.",
            "kind": "note",
            "agentId": "loka-agent",
            "tags": ["architecture"],
            "mode": "propose"
        }));

        then.status(201)
            .header("content-type", "application/json")
            .json_body(json!({
                "mode": "propose",
                "proposal": { "id": "proposal-1" }
            }));
    });

    let client = MemoryClient::new(server.base_url());
    let proposal_id = client
        .propose_note(MemoryNoteInput {
            title: "Decision".to_string(),
            body: "Use Rust for the agent control plane.".to_string(),
            kind: "note".to_string(),
            agent_id: "loka-agent".to_string(),
            tags: vec!["architecture".to_string()],
        })
        .await
        .expect("note proposal should succeed");

    note.assert();
    assert_eq!(proposal_id, "proposal-1");
}

#[tokio::test]
async fn propose_note_rejects_direct_write_response() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/api/notes");
        then.status(201)
            .header("content-type", "application/json")
            .json_body(json!({
                "mode": "direct",
                "page": { "id": "page-1" }
            }));
    });

    let client = MemoryClient::new(server.base_url());
    let error = client
        .propose_note(MemoryNoteInput {
            title: "Decision".to_string(),
            body: "Keep writes proposal-first.".to_string(),
            kind: "note".to_string(),
            agent_id: "loka-agent".to_string(),
            tags: vec![],
        })
        .await
        .expect_err("direct write should fail");

    assert!(error.to_string().contains("unexpected note write mode"));
}

#[tokio::test]
async fn list_pending_proposals_uses_status_and_limit_query() {
    let server = MockServer::start();
    let proposals = server.mock(|when, then| {
        when.method(GET)
            .path("/api/proposals")
            .query_param("status", "pending")
            .query_param("limit", "5");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "proposals": [
                    {
                        "id": "proposal-1",
                        "title": "Session learning: abc",
                        "kind": "note",
                        "tags": ["learning", "session"],
                        "createdAt": "2026-05-28T00:00:00Z"
                    }
                ]
            }));
    });

    let client = MemoryClient::new(server.base_url());
    let output = client
        .pending_proposals(5)
        .await
        .expect("pending proposals");

    proposals.assert();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].id, "proposal-1");
    assert_eq!(output[0].title, "Session learning: abc");
    assert_eq!(output[0].tags, vec!["learning", "session"]);
}
