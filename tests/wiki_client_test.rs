use httpmock::prelude::*;
use loka_agent::wiki::{NoteInput, WikiClient};
use serde_json::json;

#[tokio::test]
async fn rag_posts_query_to_personal_wiki() {
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
                "markdown": "# Wiki Context\n- ship the platform spine"
            }));
    });

    let client = WikiClient::new(server.base_url());
    let output = client
        .rag("next work", 6, 1)
        .await
        .expect("rag should succeed");

    rag.assert();
    assert_eq!(output.markdown, "# Wiki Context\n- ship the platform spine");
}

#[tokio::test]
async fn add_note_defaults_to_proposal_mode() {
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

    let client = WikiClient::new(server.base_url());
    let proposal_id = client
        .add_note(NoteInput {
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
async fn add_note_rejects_direct_write_response() {
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

    let client = WikiClient::new(server.base_url());
    let error = client
        .add_note(NoteInput {
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
