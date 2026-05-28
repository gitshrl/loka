use httpmock::prelude::*;
use loka_agent::llm::{ChatRequest, LlmClient};
use loka_agent::messages::Message;
use serde_json::json;

#[tokio::test]
async fn chat_completion_sends_openai_compatible_request() {
    let server = MockServer::start();
    let completion = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .header("authorization", "Bearer sk-test")
            .json_body(json!({
                "model": "gpt-5.5",
                "messages": [
                    { "role": "user", "content": "ping" }
                ]
            }));

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "role": "assistant", "content": "pong" } }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7
                }
            }));
    });

    let client = LlmClient::new(server.base_url(), "sk-test".to_string());
    let output = client
        .chat(ChatRequest {
            model: "gpt-5.5".to_string(),
            messages: vec![Message::user("ping")],
        })
        .await
        .expect("chat should succeed");

    completion.assert();
    assert_eq!(output.content, "pong");
    assert_eq!(output.usage.expect("usage").total_tokens, 7);
}

#[tokio::test]
async fn chat_completion_reports_upstream_error_body() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/chat/completions");
        then.status(503).body("no provider accounts available");
    });

    let client = LlmClient::new(server.base_url(), "sk-test".to_string());
    let error = client
        .chat(ChatRequest {
            model: "gpt-5.5".to_string(),
            messages: vec![Message::user("ping")],
        })
        .await
        .expect_err("upstream 503 should fail");

    let message = error.to_string();
    assert!(message.contains("503"));
    assert!(message.contains("no provider accounts available"));
}

#[tokio::test]
async fn chat_completion_rejects_empty_assistant_content() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "role": "assistant", "content": "   " } }
                ]
            }));
    });

    let client = LlmClient::new(server.base_url(), "sk-test".to_string());
    let error = client
        .chat(ChatRequest {
            model: "gpt-5.5".to_string(),
            messages: vec![Message::user("ping")],
        })
        .await
        .expect_err("empty content should fail");

    assert!(error.to_string().contains("no assistant content"));
}

#[tokio::test]
async fn chat_stream_sends_stream_request_and_yields_deltas() {
    let server = MockServer::start();
    let stream = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .header("authorization", "Bearer sk-test")
            .json_body(json!({
                "model": "gpt-5.5",
                "messages": [
                    { "role": "user", "content": "ping" }
                ],
                "stream": true
            }));

        then.status(200)
            .header("content-type", "text/event-stream")
            .body(concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"po\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"ng\"}}]}\n\n",
                "data: [DONE]\n\n"
            ));
    });

    let client = LlmClient::new(server.base_url(), "sk-test".to_string());
    let mut deltas = Vec::new();
    let output = client
        .chat_stream(
            ChatRequest {
                model: "gpt-5.5".to_string(),
                messages: vec![Message::user("ping")],
            },
            |delta| {
                deltas.push(delta.to_string());
                Ok(())
            },
        )
        .await
        .expect("stream should succeed");

    stream.assert();
    assert_eq!(deltas, vec!["po", "ng"]);
    assert_eq!(output.content, "pong");
}
