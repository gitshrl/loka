use futures::future::BoxFuture;
use httpmock::prelude::*;
use loka_agent::gateway::{
    GatewayAgent, GatewayRequest, GatewayResponse, GatewaySessionStore, TelegramClient,
    TelegramGateway, TelegramGatewayOutcome, TelegramUpdate,
};
use serde_json::json;
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
