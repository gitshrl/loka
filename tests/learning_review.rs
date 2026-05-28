use httpmock::prelude::*;
use loka::learning::pending_learning_proposals;
use loka::memory::MemoryClient;
use serde_json::json;

#[tokio::test]
async fn pending_learning_review_filters_learning_proposals() {
    let memory = MockServer::start();
    memory.mock(|when, then| {
        when.method(GET)
            .path("/api/proposals")
            .query_param("status", "pending")
            .query_param("limit", "10");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "proposals": [
                    {
                        "id": "proposal-learning",
                        "title": "Session learning: abc",
                        "kind": "note",
                        "tags": ["learning", "session"],
                        "createdAt": "2026-05-28T00:00:00Z"
                    },
                    {
                        "id": "proposal-other",
                        "title": "Architecture note",
                        "kind": "note",
                        "tags": ["architecture"],
                        "createdAt": "2026-05-28T01:00:00Z"
                    }
                ]
            }));
    });

    let client = MemoryClient::new(memory.base_url());
    let proposals = pending_learning_proposals(&client, 10)
        .await
        .expect("learning proposals");

    assert_eq!(proposals.len(), 1);
    assert_eq!(proposals[0].id, "proposal-learning");
    assert_eq!(proposals[0].title, "Session learning: abc");
}
