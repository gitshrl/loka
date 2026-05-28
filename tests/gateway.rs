use futures::future::BoxFuture;
use httpmock::prelude::*;
use loka::config::AppConfig;
use loka::gateway::{
    GatewayAgent, GatewayRequest, GatewayResponse, GatewaySessionStore, LokaGatewayAgent,
    TelegramClient, TelegramGateway, TelegramGatewayOutcome, TelegramUpdate,
};
use serde_json::json;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[tokio::test]
async fn telegram_gateway_replies_to_text_message() {
    let telegram = MockServer::start();
    let send = telegram.mock(|when, then| {
        when.method(POST)
            .path("/botTOKEN/sendMessage")
            .json_body(json!({
                "chat_id": 123,
                "text": "agent answer"
            }));

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({ "ok": true }));
    });

    let agent = CaptureAgent::new("agent answer");
    let gateway = TelegramGateway::new(
        TelegramClient::with_base_url(telegram.base_url(), "TOKEN"),
        agent.clone(),
        true,
    );
    let update: TelegramUpdate = serde_json::from_value(json!({
        "update_id": 1,
        "message": {
            "message_id": 10,
            "chat": { "id": 123 },
            "text": "hello"
        }
    }))
    .expect("update");

    let outcome = gateway.handle_update(update).await.expect("update handled");

    send.assert();
    assert_eq!(outcome, TelegramGatewayOutcome::Replied { chat_id: 123 });
    assert_eq!(
        agent.requests(),
        vec![GatewayRequest {
            gateway: "telegram".to_string(),
            conversation_key: "123".to_string(),
            session_key: "telegram:123".to_string(),
            text: "hello".to_string(),
            recall: true,
        }]
    );
}

#[tokio::test]
async fn telegram_gateway_ignores_non_text_updates() {
    let telegram = MockServer::start();
    let agent = CaptureAgent::new("unused");
    let gateway = TelegramGateway::new(
        TelegramClient::with_base_url(telegram.base_url(), "TOKEN"),
        agent.clone(),
        false,
    );
    let update: TelegramUpdate = serde_json::from_value(json!({
        "update_id": 1,
        "message": {
            "message_id": 10,
            "chat": { "id": 123 }
        }
    }))
    .expect("update");

    let outcome = gateway.handle_update(update).await.expect("update handled");

    assert_eq!(outcome, TelegramGatewayOutcome::Ignored);
    assert!(agent.requests().is_empty());
}

#[test]
fn gateway_session_store_maps_conversation_to_session() {
    let store = GatewaySessionStore::in_memory().expect("store");
    assert!(
        store
            .session_id("telegram", "123")
            .expect("session lookup")
            .is_none()
    );

    store
        .upsert("telegram", "123", "session-1")
        .expect("upsert");

    assert_eq!(
        store.session_id("telegram", "123").expect("session lookup"),
        Some("session-1".to_string())
    );
}

#[tokio::test]
async fn strict_loka_gateway_prefetches_and_syncs_turn() {
    let memory = MockServer::start();
    let model_client = MockServer::start();
    let state = tempfile::tempdir().expect("state");

    let rag = memory.mock(|when, then| {
        when.method(POST).path("/api/rag");
        then.status(500);
    });
    let prefetch = memory.mock(|when, then| {
        when.method(POST)
            .path("/api/memory/prefetch")
            .body_includes("\"query\":\"hello gateway\"")
            .body_includes("\"limit\":6")
            .body_includes("\"depth\":1")
            .body_includes("\"sessionId\":");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "mode": "prefetch",
                "markdown": "# Memory Context\n- gateway context"
            }));
    });
    let turn = memory.mock(|when, then| {
        when.method(POST)
            .path("/api/memory/turns")
            .body_includes("\"user\":\"hello gateway\"")
            .body_includes("\"assistant\":\"gateway answer\"")
            .body_includes("\"agentId\":\"loka\"");

        then.status(202);
    });
    let completion = model_client.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .body_includes("gateway context")
            .body_includes("hello gateway");

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "content": "gateway answer" } }
                ]
            }));
    });

    let agent = LokaGatewayAgent::new(AppConfig {
        model_base_url: model_client.base_url(),
        model_api_key: "sk-test".to_string(),
        memory_base_url: memory.base_url(),
        model: "gpt-5.5".to_string(),
        agent_id: "loka".to_string(),
        model_protocol: loka::config::ModelProtocol::OpenAiCompatible,
        memory_lifecycle: loka::config::MemoryLifecycleMode::Strict,
        working_dir: PathBuf::from("/tmp"),
        state_dir: state.path().to_path_buf(),
    });

    let response = agent
        .respond(GatewayRequest {
            gateway: "telegram".to_string(),
            conversation_key: "123".to_string(),
            session_key: "telegram:123".to_string(),
            text: "hello gateway".to_string(),
            recall: true,
        })
        .await
        .expect("gateway response");

    prefetch.assert();
    completion.assert();
    turn.assert();
    assert_eq!(rag.calls(), 0);
    assert_eq!(response.text, "gateway answer");
}

#[derive(Clone)]
struct CaptureAgent {
    answer: String,
    requests: Arc<Mutex<Vec<GatewayRequest>>>,
}

impl CaptureAgent {
    fn new(answer: &str) -> Self {
        Self {
            answer: answer.to_string(),
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn requests(&self) -> Vec<GatewayRequest> {
        self.requests.lock().expect("requests").clone()
    }
}

impl GatewayAgent for CaptureAgent {
    fn respond(&self, request: GatewayRequest) -> BoxFuture<'_, anyhow::Result<GatewayResponse>> {
        Box::pin(async move {
            self.requests.lock().expect("requests").push(request);
            Ok(GatewayResponse {
                text: self.answer.clone(),
            })
        })
    }
}
